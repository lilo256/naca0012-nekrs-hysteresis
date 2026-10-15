#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_series_montee.py -- series temporelles de la branche montante (section 4.3),
                            a lancer SUR LE CLUSTER, a la racine de la campagne.

  python3 extract_series_montee.py                 # racine = dossier courant
  python3 extract_series_montee.py --root ~/naca   # autre racine
  python3 extract_series_montee.py --only cold_a18.00,cold_a19.00

Pour chaque cas de CASES (dossier trouve automatiquement sous la racine) :
  1. EFFORTS  drag_results.dat : segments de restart recolles (on garde le plus recent en
     cas de recouvrement), lignes aberrantes retirees (|CL| ou |CD| > 5 apres mise a
     l'echelle), moyennes par tranches de DT_BIN CTU -> t, CL, CD, CDp, CDv.
  2. PEIGNE   probes_uvp.csv : ux moyen par tranche aux sondes A020..A092 (0.008c de la
     paroi) et B050, B070, B090 ; vitesse TANGENTIELLE u_t (ux, uy projetes sur la tangente
     locale de l'extrados) et fraction du temps avec u_t < 0 (gamma) -- un ux > 0 ne prouve
     pas un ecoulement vers l'aval si la composante normale est forte.
  3. PAROI    resultats_csv/cf_cp_time_*.csv (cas de WALL_CASES) : pour chaque instantane,
     Cp et Cf a l'extrados sur une grille commune de x/c, point de decollement instantane,
     Cp,min, et force normale de pression (integrale du Cp) comparee a celle des efforts
     au meme instant -> diagnostic du controle integral rate sur les etats decroches.
  4. STATS    sur les 20 derniers CTU de chaque cas (fenetre homogene pour le tableau 4.3) :
     moyennes et rms ponderees en temps, ecart entre moities et entre quarts, et, si la
     paroi est lue, x_s et Cp,min du Cf/Cp moyens sur la meme fenetre.
Sortie : series_montee.json (quelques Mo) + resume imprime. Rapatrier le JSON.

Controle d'echelle : si montee_data.json est present a la racine, la moyenne de CL sur la
fenetre du JSON est recalculee et comparee ; un facteur 2 (efforts bruts / coefficients)
est detecte et corrige automatiquement.

Dependances : numpy ; pandas conseille (lecture 10x plus rapide des gros fichiers).
"""
import argparse
import glob
import json
import os
import re
import sys
import time as _time

import numpy as np

try:
    import pandas as pd
except ImportError:          # pragma: no cover
    pd = None

# =============================================================================
# CONFIG
# =============================================================================
CASES = {                                  # cle : incidence (deg). Dossier = cle, cherche sous --root
    "prod_a15":    15.00,
    "up_a17.00":   17.00,
    "up_a17.25":   17.25,
    "up_a17.50":   17.50,
    "up_a17.75":   17.75,
    "up_a18.00":   18.00,
    "cold_a16.50": 16.50,
    "cold_a17.00": 17.00,
    "cold_a18.00": 18.00,
    "cold_a19.00": 19.00,
}
PATHS = {}               # chemins imposes si la recherche automatique echoue : {"cold_a18.00": "froid/cold_a18.00"}
WALL_CASES = ["up_a17.25", "cold_a17.00", "cold_a18.00", "cold_a19.00"]
DT_BIN = 0.2             # largeur des tranches [CTU]
LAST = 20.0              # fenetre des statistiques finales [CTU]
FORCE_COLS = dict(CD=2, CL=3, CDp=5, CDv=8)      # tstep time dragX dragY dragZ dragPX .. dragVX ..
FORCE_FACTOR = None      # None = auto (x2 si efforts bruts) ; sinon 1.0 ou 2.0
ABERRANT = 5.0           # |CL| ou |CD| au-dela : ligne ecartee
COMB = ["A020", "A035", "A050", "A065", "A080", "A092"]
COMB_X = [0.20, 0.35, 0.50, 0.65, 0.80, 0.92]
EXTRA = ["B050", "B070", "B090"]
NX_WALL = 160            # points de la grille x/c commune (extrados)
REF_JSON = "montee_data.json"
OUT = "series_montee.json"

ALIASES = dict(x=["x"], y=["y"], cp=["cp", "c_p"], cfx=["cf_x", "cfx"], cfy=["cf_y", "cfy"])
DEFAULT_ORDER = ["x", "y", "z", "cfx", "cfy", "cfz", "cp"]


# =============================================================================
# utilitaires
# =============================================================================
def find_case_dir(root, key):
    if key in PATHS:
        p = os.path.join(root, PATHS[key])
        return p if os.path.isdir(p) else None
    hits = []
    for dirpath, dirnames, _ in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > 4:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if os.path.basename(dirpath) == key and os.path.isfile(os.path.join(dirpath, "drag_results.dat")):
            hits.append(dirpath)
    if len(hits) > 1:
        hits.sort(key=lambda p: (p.count(os.sep), p))
        print(f"   ! {key} : plusieurs dossiers, retenu {hits[0]} (autres : {hits[1:]}) -> PATHS si besoin")
    return hits[0] if hits else None


def read_numeric(path, usecols=None):
    """Colonnes numeriques d'un fichier texte (espaces ou virgules), en-tetes ignores."""
    if pd is not None:
        try:
            with open(path) as fh:
                first = fh.readline()
            sep = "," if "," in first else r"\s+"
            df = pd.read_csv(path, sep=sep, header=None, comment="#", usecols=usecols,
                             engine="c" if sep == "," else "python" if usecols is None else "c",
                             on_bad_lines="skip", dtype=str)
            A = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            A = A[np.all(np.isfinite(A), axis=1)]
            if A.size:
                return A
        except Exception as e:                                   # repli numpy
            print(f"   (pandas : {e} -> lecture numpy)")
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s[0] in "#%!":
                continue
            parts = [p for p in re.split(r"[,\s;]+", s) if p]
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if usecols is not None:
                if len(vals) <= max(usecols):
                    continue
                vals = [vals[i] for i in usecols]
            rows.append(vals)
    if not rows:
        raise RuntimeError(f"aucune ligne numerique dans {path}")
    n = np.bincount([len(r) for r in rows]).argmax()
    return np.array([r for r in rows if len(r) == n])


def stitch(t, order=None):
    """Indices a garder pour une serie temporelle ecrite par segments (restarts) :
    a chaque recul du temps, on coupe ce qui precede et depasse le debut du nouveau segment."""
    brk = np.where(np.diff(t) <= 0)[0] + 1
    segs = np.split(np.arange(len(t)), brk)
    kept_t, kept_i, restarts = np.empty(0), np.empty(0, dtype=int), []
    for k, idx in enumerate(segs):
        if idx.size == 0:
            continue
        t0 = t[idx[0]]
        if k > 0:
            n = np.searchsorted(kept_t, t0, side="left")
            kept_t, kept_i = kept_t[:n], kept_i[:n]
            restarts.append(float(t0))
        kept_t = np.concatenate([kept_t, t[idx]])
        kept_i = np.concatenate([kept_i, idx])
    return kept_i, restarts


def bin_series(t, Y, dt):
    """Moyennes par tranches [t0 + k dt, t0 + (k+1) dt[ ; Y : dict nom -> tableau."""
    t0 = np.floor(t.min() / dt) * dt
    k = np.floor((t - t0) / dt + 1e-9).astype(int)
    nb = k.max() + 1
    cnt = np.bincount(k, minlength=nb).astype(float)
    ok = cnt > 0
    tc = t0 + (np.arange(nb) + 0.5) * dt
    out = {"t": tc[ok]}
    for name, y in Y.items():
        s = np.bincount(k, weights=y, minlength=nb)
        out[name] = s[ok] / cnt[ok]
    return out


def tw_stats(t, y, t0, t1):
    """Moyenne et rms ponderees en temps sur [t0, t1] (trapezes, trous > 50 dt ignores)."""
    m = (t >= t0) & (t <= t1)
    tt, yy = t[m], y[m]
    if tt.size < 3:
        return float("nan"), float("nan")
    d = np.diff(tt)
    d = np.where(d > 50 * np.median(d), 0.0, d)
    w = np.zeros_like(tt)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    mu = float(np.sum(w * yy) / np.sum(w))
    return mu, float(np.sqrt(np.sum(w * (yy - mu) ** 2) / np.sum(w)))


def r5(a, nd=5):
    """Liste JSON compacte (nd chiffres significatifs)."""
    a = np.asarray(a, dtype=float)
    return [None if not np.isfinite(v) else float(f"{v:.{nd}g}") for v in a]


# =============================================================================
# 1. efforts
# =============================================================================
def load_forces(case_dir, key, ref):
    path = os.path.join(case_dir, "drag_results.dat")
    ncol_needed = max(FORCE_COLS.values()) + 1
    A = read_numeric(path, usecols=list(range(ncol_needed)))
    tstep, t = A[:, 0], A[:, 1]
    keep, restarts = stitch(t)
    A = A[keep]
    t = A[:, 1]
    sig = {q: A[:, c] for q, c in FORCE_COLS.items()}
    # facteur d'echelle
    med = float(np.median(np.abs(sig["CL"][t >= t.min() + 0.5 * (t.max() - t.min())])))
    if FORCE_FACTOR is not None:
        fac, how = float(FORCE_FACTOR), "CONFIG"
    elif 0.30 < med < 0.72:
        fac, how = 2.0, f"auto (|dragY| median {med:.3f} -> efforts bruts, x2)"
    else:
        fac, how = 1.0, f"auto (|dragY| median {med:.3f} -> coefficients)"
    # controle contre montee_data.json
    if ref and key in ref:
        for wname, w in ref[key]["fenetres"].items():
            mu, _ = tw_stats(t, fac * sig["CL"], w["t0"], w["t1"])
            ratio = mu / w["CL"] if w["CL"] else float("nan")
            if np.isfinite(ratio) and abs(ratio - 0.5) < 0.02 and FORCE_FACTOR is None:
                fac *= 2.0
                how += " ; corrige x2 d'apres montee_data.json"
            elif np.isfinite(ratio) and abs(ratio - 2.0) < 0.04 and FORCE_FACTOR is None:
                fac /= 2.0
                how += " ; corrige /2 d'apres montee_data.json"
            mu, _ = tw_stats(t, fac * sig["CL"], w["t0"], w["t1"])
            how += f" ; controle {wname} [{w['t0']:.0f},{w['t1']:.0f}] CL {mu:.4f} vs JSON {w['CL']:.4f}"
            break
    for q in sig:
        sig[q] = fac * sig[q]
    bad = (np.abs(sig["CL"]) > ABERRANT) | (np.abs(sig["CD"]) > ABERRANT)
    for q in sig:
        sig[q] = sig[q][~bad]
    t = t[~bad]
    return t, sig, restarts, int(bad.sum()), how


# =============================================================================
# 2. peigne
# =============================================================================
def load_probes(case_dir):
    path = os.path.join(case_dir, "probes_uvp.csv")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        header = [h.strip().lower() for h in fh.readline().strip().lstrip("#").split(",")]
    want = {"time": header.index("time") if "time" in header else 0}
    for L in COMB + EXTRA:
        for comp in ("ux", "uy"):
            name = f"{comp}_{L.lower()}"
            if name in header:
                want[f"{comp}_{L}"] = header.index(name)
    cols = sorted(set(want.values()))
    A = read_numeric(path, usecols=cols)
    pos = {k: cols.index(v) for k, v in want.items()}
    t = A[:, pos["time"]]
    keep, _ = stitch(t)
    A, t = A[keep], t[keep]
    return t, {k: A[:, pos[k]] for k in want if k != "time"}


def comb_tangent(xb, alpha):
    """Tangente a l'extrados (vers le BF) en x/c = xb, repere global (profil tourne autour du BF)."""
    yt = lambda x: 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    h = 1e-4
    dy = (yt(xb + h) - yt(xb - h)) / (2 * h)
    n = np.hypot(1.0, dy)
    a = np.radians(alpha)
    tx, ty = 1.0 / n, dy / n
    return tx * np.cos(a) + ty * np.sin(a), -tx * np.sin(a) + ty * np.cos(a)


# =============================================================================
# 3. paroi
# =============================================================================
def parse_time(name):
    m = re.search(r"time_(\d+)_(\d+)(?=_rank|\.csv)", name)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"time_([0-9.eE+-]+?)(?:_rank_?\d+)?\.csv$", name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def list_snapshots(case_dir):
    files = glob.glob(os.path.join(case_dir, "resultats_csv", "cf_cp_time_*.csv"))
    files += glob.glob(os.path.join(case_dir, "cf_cp_time_*.csv"))
    groups = {}
    for f in files:
        tt = parse_time(os.path.basename(f))
        if tt is not None:
            groups.setdefault(round(tt, 6), []).append(f)
    return sorted(groups.items())


def read_table(path):
    with open(path) as fh:
        first = fh.readline()
    if not first.strip():
        return None, None
    toks = [s for s in re.split(r"[,\s;]+", first.strip().lstrip("#").strip()) if s]
    try:
        [float(s) for s in toks]
        header, skip = None, 0
    except ValueError:
        header, skip = [s.strip().lower() for s in toks], 1
    try:
        D = np.loadtxt(path, delimiter="," if "," in first else None, skiprows=skip, ndmin=2)
    except Exception:
        return header, None
    return header, (D if D.size else None)


def _col(header, D, key):
    if header is not None:
        for a in ALIASES[key]:
            if a in header and header.index(a) < D.shape[1]:
                return D[:, header.index(a)]
        return None
    if key in DEFAULT_ORDER and DEFAULT_ORDER.index(key) < D.shape[1]:
        return D[:, DEFAULT_ORDER.index(key)]
    return None


def read_snapshot(files):
    parts = []
    for f in files:
        h, D = read_table(f)
        if D is None:
            continue
        x, y, cp, cfx = (_col(h, D, k) for k in ("x", "y", "cp", "cfx"))
        if x is None or y is None or cp is None or cfx is None:
            raise RuntimeError(f"{f} : colonnes x, y, cp, cf_x introuvables (en-tete {h})")
        parts.append(np.column_stack([x, y, cp, cfx]))
    if not parts:
        return None
    P = np.vstack(parts)
    key = np.round(P[:, :2], 7)                     # noeuds confondus (plans z, aretes) fusionnes
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((cnt.size, 4))
    for j in range(4):
        out[:, j] = np.bincount(inv, weights=P[:, j]) / cnt
    return out


def profile_frame(X, Y):
    """Repere du profil : BA = point le plus loin du BF (milieu du culot), x/c de 0 a 1."""
    P = np.column_stack([X, Y])
    i_far = np.argmax(np.hypot(*(P - P.mean(axis=0)).T))
    i_oth = np.argmax(np.hypot(*(P - P[i_far]).T))
    le, te = (P[i_oth], P[i_far]) if P[i_oth, 0] < P[i_far, 0] else (P[i_far], P[i_oth])
    for _ in range(2):
        c = np.linalg.norm(te - le)
        ex = (te - le) / c
        xs = (P - le) @ ex
        te = P[xs > xs.max() - 2e-3 * c].mean(axis=0)
        le = P[np.argmax(np.hypot(*(P - te).T))]
    c = np.linalg.norm(te - le)
    ex = (te - le) / c
    ey = np.array([-ex[1], ex[0]])
    return (P - le) @ ex / c, (P - le) @ ey / c, float(np.degrees(np.arctan2(-(te - le)[1], (te - le)[0])))


XG = 0.5 * (1 - np.cos(np.linspace(0.02, np.pi * 0.985, NX_WALL)))      # grille x/c, resserree au BA


def wall_snapshot(S):
    xs, ys, aoa = profile_frame(S[:, 0], S[:, 1])
    up, lo = ys > 0, ys <= 0
    out = {"aoa": aoa}
    for side, m in (("up", up), ("lo", lo)):
        o = np.argsort(xs[m])
        xx, cp, cf = xs[m][o], S[m, 2][o], S[m, 3][o]
        X, Y = S[m, 0][o], S[m, 1][o]
        Tx = np.gradient(X)
        sg = np.where(Tx >= 0, 1.0, -1.0)
        cf = cf * sg                                   # cisaillement oriente BA -> BF
        for i in range(1, len(cf)):                    # continuite pres d'une tangente verticale
            nt = abs(Tx[i]) / (np.hypot(Tx[i], np.gradient(Y)[i]) + 1e-30)
            if nt < 0.1 and abs(-cf[i] - cf[i - 1]) < abs(cf[i] - cf[i - 1]):
                cf[i] = -cf[i]
        out[side] = (xx, cp, cf)
    xu, cpu, cfu = out["up"]
    xl, cpl, _ = out["lo"]
    g = np.unique(np.concatenate([xu, xl]))
    g = g[(g >= max(xu.min(), xl.min())) & (g <= min(xu.max(), xl.max()))]
    trapz = getattr(np, "trapezoid", None) or np.trapz
    out["CN_cp"] = float(trapz(np.interp(g, xl, cpl) - np.interp(g, xu, cpu), g))
    out["cp_up"] = np.interp(XG, xu, cpu)
    out["cf_up"] = np.interp(XG, xu, cfu)
    out["x_sep"] = first_sep(xu, cfu)
    j = int(np.argmin(cpu))
    out["cp_min"], out["x_cp_min"] = float(cpu[j]), float(xu[j])
    return out


def first_sep(x, cf, xmin=0.02, xmax=0.975):
    m = (x > xmin) & (x < xmax)
    xx, cc = x[m], cf[m]
    for i in range(len(cc) - 1):
        if cc[i] > 0 >= cc[i + 1]:
            return float(xx[i] + cc[i] / (cc[i] - cc[i + 1]) * (xx[i + 1] - xx[i]))
    return None


# =============================================================================
# principal
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--only", default="")
    ap.add_argument("--no-wall", action="store_true")
    args = ap.parse_args()
    root = os.path.abspath(os.path.expanduser(args.root))
    keys = [k for k in CASES if not args.only or k in args.only.split(",")]
    ref = None
    rp = os.path.join(root, REF_JSON)
    if os.path.isfile(rp):
        ref = json.load(open(rp)).get("cas")
    res = {"version": 1, "genere": _time.strftime("%Y-%m-%d %H:%M"), "dt_bin": DT_BIN,
           "comb_x": COMB_X, "x_wall": r5(XG, 5), "cas": {}}
    if os.path.isfile(os.path.join(root, OUT)):                       # mise a jour incrementale
        old = json.load(open(os.path.join(root, OUT)))
        res["cas"] = {k: v for k, v in old.get("cas", {}).items() if k not in keys}

    for key in keys:
        t_start = _time.time()
        d = find_case_dir(root, key)
        print(f"\n== {key} ({CASES[key]:.2f} deg) : {d}")
        if d is None:
            print("   dossier introuvable -> PATHS")
            continue
        entry = {"alpha": CASES[key], "dir": os.path.relpath(d, root)}
        # ---- efforts
        t, sig, restarts, nbad, how = load_forces(d, key, ref)
        B = bin_series(t, sig, DT_BIN)
        entry["forces"] = {k: r5(v, 6) for k, v in B.items()}
        entry["restarts"] = restarts
        entry["t_range"] = [float(t.min()), float(t.max())]
        print(f"   efforts : {t.size} lignes, t = {t.min():.2f} -> {t.max():.2f}, restarts {restarts},"
              f" {nbad} aberrantes ; {how}")
        # ---- stats des 20 derniers CTU
        t1 = float(t.max())
        t0 = max(float(t.min()), t1 - LAST)
        st = {"t0": t0, "t1": t1}
        for q in ("CL", "CD", "CDp", "CDv"):
            st[q], st[q + "_rms"] = tw_stats(t, sig[q], t0, t1)
        h1, _ = tw_stats(t, sig["CL"], t0, 0.5 * (t0 + t1))
        h2, _ = tw_stats(t, sig["CL"], 0.5 * (t0 + t1), t1)
        qs = [tw_stats(t, sig["CL"], t0 + i * (t1 - t0) / 4, t0 + (i + 1) * (t1 - t0) / 4)[0] for i in range(4)]
        st["drift_halves_pct"] = 100 * (h2 - h1) / st["CL"]
        st["spread_quarters_pct"] = 100 * (max(qs) - min(qs)) / st["CL"]
        # ---- peigne
        P = load_probes(d)
        if P is not None:
            tp, U = P
            Y = {}
            m = (tp >= t0) & (tp <= t1)
            st["comb"] = {}
            for L in COMB + EXTRA:
                if f"ux_{L}" not in U:
                    continue
                Y[L] = U[f"ux_{L}"]                                  # ux global (comme le peigne v2)
                if L in COMB and f"uy_{L}" in U:
                    tx, ty = comb_tangent(COMB_X[COMB.index(L)], CASES[key])
                    ut = U[f"ux_{L}"] * tx + U[f"uy_{L}"] * ty           # vitesse tangentielle (vers le BF)
                    Y[f"ut_{L}"] = ut
                    Y[f"g_{L}"] = (ut < 0).astype(float)
                    st["comb"][L] = dict(ux=float(np.mean(U[f"ux_{L}"][m])), ut=float(np.mean(ut[m])),
                                         gamma_t=float(np.mean(ut[m] < 0)),
                                         gamma_x=float(np.mean(U[f"ux_{L}"][m] < 0)))
            BP = bin_series(tp, Y, DT_BIN)
            entry["probes"] = {k: r5(v, 4) for k, v in BP.items()}
            print(f"   peigne : {tp.size} lignes ; 20 derniers CTU, u_t et fraction u_t < 0 : " +
                  " ".join(f"{L}:{v['ut']:+.2f}/{v['gamma_t']:.2f}" for L, v in st["comb"].items()))
        else:
            print("   peigne : probes_uvp.csv absent")
        # ---- paroi
        if key in WALL_CASES and not args.no_wall:
            snaps = list_snapshots(d)
            W = {"t": [], "cp_up": [], "cf_up": [], "x_sep": [], "cp_min": [], "CN_cp": [], "CN_F": [], "aoa": []}
            a = np.radians(CASES[key])
            CNF = sig["CL"] * np.cos(a) + sig["CD"] * np.sin(a)
            for ts, files in snaps:
                S = read_snapshot(files)
                if S is None or S.shape[0] < 50:
                    continue
                w = wall_snapshot(S)
                W["t"].append(ts)
                W["cp_up"].append(r5(w["cp_up"], 4))
                W["cf_up"].append(r5(w["cf_up"], 4))
                W["x_sep"].append(w["x_sep"])
                W["cp_min"].append(w["cp_min"])
                W["CN_cp"].append(w["CN_cp"])
                W["aoa"].append(w["aoa"])
                # force normale des efforts, moyenne sur +-0.05 CTU autour de l'instantane
                mm = np.abs(t - ts) <= 0.05
                W["CN_F"].append(float(np.mean(CNF[mm])) if mm.any() else None)
            entry["wall"] = W
            if W["t"]:
                tw = np.array(W["t"])
                m = (tw >= t0 - 1e-6) & (tw <= t1 + 1e-6)
                if m.sum() >= 2:
                    cfm = np.mean(np.array(W["cf_up"], dtype=float)[m], axis=0)
                    cpm = np.mean(np.array(W["cp_up"], dtype=float)[m], axis=0)
                    st["x_sep"] = first_sep(XG, cfm)
                    st["cp_min"] = float(np.min(cpm))
                    st["n_wall"] = int(m.sum())
                cn = np.array([np.nan if v is None else v for v in W["CN_F"]], dtype=float)
                ratio = np.array(W["CN_cp"]) / cn
                print(f"   paroi : {len(W['t'])} instantanes, t = {tw.min():.1f} -> {tw.max():.1f},"
                      f" alpha geo {np.median(W['aoa']):.3f} deg ; CN(Cp)/CN(efforts) : median "
                      f"{np.nanmedian(ratio):.3f}, min {np.nanmin(ratio):.3f}, max {np.nanmax(ratio):.3f}")
        entry["stats_last20"] = st
        res["cas"][key] = entry
        xs = st.get("x_sep")
        print(f"   20 derniers CTU [{t0:.1f}, {t1:.1f}] : CL {st['CL']:.4f} +- {st['CL_rms']:.4f}   "
              f"CD {st['CD']:.5f} +- {st['CD_rms']:.5f}   CDp {st['CDp']:.5f}   CDv {st['CDv']:.5f}   "
              f"moities {st['drift_halves_pct']:+.2f} %   quarts {st['spread_quarters_pct']:.2f} %"
              + (f"   x_s {xs:.3f}" if xs else "") + f"   ({_time.time() - t_start:.0f} s)")

    with open(os.path.join(root, OUT), "w") as fh:
        json.dump(res, fh, separators=(",", ":"))
    print(f"\n{OUT} ecrit ({os.path.getsize(os.path.join(root, OUT)) / 1e6:.1f} Mo) -> a rapatrier")


if __name__ == "__main__":
    main()

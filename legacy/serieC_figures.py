#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
serieC_figures.py -- Serie C (densite de points de surface) : chiffres du tableau
                     et figures du rapport, pour C1 / C0 / C+.

CE QUE FAIT LE SCRIPT
  1. Efforts (drag_results.dat)
     - recolle les segments de restart (tstep repart de 0, recouvrements retires) ;
     - moyennes sur t >= T_CUT en RETIRANT les impulsions de restart et une fenetre
       de garde RESTART_GUARD apres chaque restart (on donne aussi la moyenne brute,
       pour mesurer l'effet de ces impulsions) ;
     - erreur-type par moyennes de blocs, RMS, ecart entre demi-fenetres ;
     - pas de temps etabli dt = d(time)/d(tstep), par segment (un saut de dt au
       restart trahit un changement de CFL).
  2. Paroi (resultats_csv/cf_cp_time_*.csv + wall_distance_rank_*.csv)
     - moyenne temporelle NOEUD PAR NOEUD, SANS binning : les noeuds confondus
       (plans z, noeuds partages entre elements) sont fusionnes -> plus de dents de scie ;
     - repere du profil, abscisse curviligne s depuis le bord d'attaque (s<0 intrados) ;
     - Cf signe (tangente BA -> BF sur chaque face), Cp,min raffine par parabole,
       point d'arret, pic de Cf au BA, decollement eventuel ;
     - y+ = y1 sqrt(|Cf|/2) Re, y1 = distance paroi du 1er noeud interieur (KD-tree).
  3. Figures PDF pour LaTeX, resume JSON, lignes du tableau LaTeX.

v2 (apres le premier passage sur les donnees reelles)
  - dt de reference = PREMIER segment de la fenetre (CFL = 1) : les restarts de C0 et
    C+ ont tourne avec un pas 2.7x et 2.3x plus grand, et la mediane globale prenait
    ce pas-la pour C+ (figure cout faussee) ;
  - saut des coefficients a chaque restart (moyennes avant / apres, dt avant / apres) ;
  - point d'arret = zero de Cf le plus proche du maximum de Cp (le noeud de jonction du
    maillage au bord d'attaque donnait un faux zero pour C+) ;
  - alignement du frottement exporte (|Cf tangentiel| / |Cf|) : pres du nez le vecteur
    de l'UDF porte une composante non tangentielle ; y+ est donne avec le module
    (conservatif) ET avec la seule composante tangentielle ;
  - y1 geometrique (distance au 1er noeud interieur), compare a la distance du fichier ;
  - controle integral : CL, CD,p et CD,v recalcules sur le profil, compares aux efforts ;
  - Cp,min nodal (comme la serie A) + valeur raffinee ; --from-json pour refaire la
    figure cout sans relire les donnees.

v3 (diagnostics v2 sur les donnees reelles)
  - l'UDF n'exporte PAS un vecteur : cf_y est identiquement nul et cf_x contient le
    cisaillement tangentiel (scalaire), oriente vers x croissant. Le script v2 le
    projetait comme une composante x (Cf x T_x) : creux parasite la ou la tangente
    est verticale (s/c = -0.003), pic du nez deplace et attenue, frottement integre
    -2.7 %. Detection automatique (CF_EXPORT) : en mode scalaire, Cf(BA->BF) =
    cf_x * signe(T_x), y+ utilise |cf_x|, et le controle integral compare
    l'integrale de Cf T sur le contour a CD,v ;
  - valeur au noeud du bord d'attaque rapportee a part (pic isole a la jonction).

v3.1
  - signe du Cf pres de la tangente verticale : l'orientation de l'UDF et celle de la
    tangente discrete pouvaient differer d'un noeud (C+ : un noeud a -0.049 au lieu de
    +0.047). Continuite imposee dans la zone |T_x| < 0.1 ;
  - panneau (b) borne a la corde (le pic du nez est dans le panneau (d)) ;
  - donnees de paroi sauvegardees (surface_<cas>.npz) et option --replot pour refaire
    les figures paroi en quelques secondes, sans relire les CSV.

SORTIES (dans OUTDIR)
  fig_ch4_seriesC_surface.pdf      Cp, Cf sur la corde + zoom bord d'attaque   (rapport)
  fig_ch4_seriesC_cost.pdf         ecart relatif a C0 (+-1 %) et cout          (rapport)
  fig_ch4_seriesC_convergence.pdf  signaux + moyennes cumulees DEPUIS T_CUT     (annexe)
  fig_ch4_seriesC_yplus.pdf        y+ le long de la corde, non binne           (annexe)
  serieC_summary.json

USAGE
  python3 serieC_figures.py                  # racine des cas = dossier courant
  python3 serieC_figures.py --root ~/stage   # autre racine
  python3 serieC_figures.py --preview        # figure cout/precision depuis les valeurs
                                             # du tableau actuel (aucune donnee requise)
  python3 serieC_figures.py --from-json figures_serieC/serieC_summary.json
                                             # refait la figure cout depuis un resume
"""
import argparse
import glob
import json
import os
import re
import sys
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, FixedLocator

# =============================================================================
# CONFIG -- a verifier avant de lancer
# =============================================================================
CASES = [
    dict(key="C1", path="serieC/C1_dens_-25", n_elem=8370, pts=(26, 15),
         label="C1 (8 370 elements)", color="#c00000", ls="--", marker="s"),
    dict(key="C0", path="serieA/AoA10_reference/A0_baseline_CFL1", n_elem=10106, pts=(35, 20),
         label="C0 (10 106 elements, reference)", color="#1f4e79", ls="-", marker="o"),
    dict(key="C+", path="serieC/Cplus_dens_+25", n_elem=12648, pts=(50, 30),
         label="C+ (12 648 elements)", color="#2e7d32", ls="-.", marker="^"),
]
REF_KEY = "C0"          # maillage de reference pour les ecarts relatifs
T_CUT = 20.0            # fin du transitoire [CTU]
RESTART_GUARD = 1.0     # duree exclue apres chaque restart [CTU]
SPIKE_K = 12.0          # seuil impulsion : |y - mediane| > SPIKE_K * MAD
N_BATCH = 5             # nombre de blocs pour l'erreur-type
RE = 1.0e6
FORCE_FACTOR = None     # None = detection auto (efforts bruts x2 ou deja coefficients)
LE_ZOOM = 0.04          # demi-largeur du zoom bord d'attaque, en s/c
MERGE_TOL = 1e-8        # tolerance de fusion des noeuds confondus
ALIGN_FLAG = 0.5        # (mode vecteur) noeud signale si |Cf tangentiel| / |Cf| < ALIGN_FLAG
CF_EXPORT = "auto"      # "auto" | "scalar" (cf_x = cisaillement tangentiel) | "vector" (cf_x, cf_y)
OUTDIR = "figures_serieC"
FIG_PREFIX = "fig_ch4_seriesC"   # prefixe des figures (serieB_figures.py le change)
COST_FIGURE = True               # figure cout/precision (inutile pour la serie B : 2 points)

# Valeurs du tableau actuel -- utilisees UNIQUEMENT par --preview
TABLE = {
    "C1": dict(CL=1.0185, CD=0.02069, CDv=0.00823, cpmin=-5.108, dt=1.01e-4),
    "C0": dict(CL=1.0182, CD=0.02064, CDv=0.00819, cpmin=-5.143, dt=6.27e-5),
    "C+": dict(CL=1.0168, CD=0.02053, CDv=0.00818, cpmin=-5.114, dt=3.10e-5),
}

_trapz = getattr(np, "trapezoid", None) or np.trapz
import logging
logging.getLogger("fontTools").setLevel(logging.ERROR)


def pub_style():
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
        "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "axes.grid": True,
        "grid.alpha": 0.3, "grid.linewidth": 0.5, "lines.linewidth": 1.1,
        "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    })


# =============================================================================
# EFFORTS
# =============================================================================
FORCE_COLS = dict(CD=2, CL=3, CDp=5, CDv=8)   # tstep time dragX dragY dragZ dragPX .. dragVX ..


def read_numeric(path):
    try:
        A = np.loadtxt(path, comments=("#", "%"), ndmin=2)
        if A.size:
            return A
    except Exception:
        pass
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s[0] in "#%!":
                continue
            try:
                rows.append([float(p) for p in re.split(r"[,\s;]+", s) if p])
            except ValueError:
                continue                      # ligne d'en-tete
    if not rows:
        raise RuntimeError(f"aucune ligne numerique dans {path}")
    ncol = np.bincount([len(r) for r in rows]).argmax()
    return np.array([r for r in rows if len(r) == ncol])


def load_forces(case_dir):
    path = os.path.join(case_dir, "drag_results.dat")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    A = read_numeric(path)
    if A.shape[1] < 4:
        raise RuntimeError(f"{path} : {A.shape[1]} colonnes (attendu tstep time dragX dragY ...)")
    tstep, time = A[:, 0], A[:, 1]
    brk = np.where((np.diff(tstep) < 0) | (np.diff(time) <= 0))[0] + 1
    segs = np.split(np.arange(len(A)), brk)
    kept_t, kept_i, restarts = np.empty(0), np.empty(0, dtype=int), []
    for k, idx in enumerate(segs):
        t0 = time[idx[0]]
        if k > 0:
            n = np.searchsorted(kept_t, t0, side="left")      # recouvrement : on garde le plus recent
            kept_t, kept_i = kept_t[:n], kept_i[:n]
            if tstep[idx[0]] < tstep[idx[0] - 1] or t0 < time[idx[0] - 1]:
                restarts.append(float(t0))
        kept_t = np.concatenate([kept_t, time[idx]])
        kept_i = np.concatenate([kept_i, idx])
    B = A[kept_i]
    t = B[:, 1]
    # pas de temps par segment, sur les lignes conservees (un segment = entre deux restarts)
    rs = np.array(sorted(restarts))
    dstep, dtime = np.diff(B[:, 0]), np.diff(t)
    ok = dstep > 0
    t_row, dt_row = t[1:][ok], dtime[ok] / dstep[ok]
    seg_id = np.searchsorted(rs, t_row, side="right") if rs.size else np.zeros(t_row.size, dtype=int)
    dt_segments = []
    for sid in np.unique(seg_id):
        m = (seg_id == sid) & (t_row >= T_CUT)
        if m.sum() > 5:
            dt_segments.append((float(t_row[m][0]), float(np.median(dt_row[m]))))
    lift = B[:, 3]
    tail = lift[t >= T_CUT] if np.any(t >= T_CUT) else lift[len(lift) // 2:]
    med = float(np.median(np.abs(tail)))
    if FORCE_FACTOR is not None:
        fac, how = float(FORCE_FACTOR), "CONFIG"
    elif 0.35 < med < 0.75:
        fac, how = 2.0, f"auto : efforts bruts (|dragY| median = {med:.3f}) -> x2"
    elif 0.75 <= med < 1.6:
        fac, how = 1.0, f"auto : deja des coefficients (|dragY| median = {med:.3f})"
    else:
        fac, how = 1.0, f"auto : NON RECONNU (|dragY| median = {med:.3g}) -> fixer FORCE_FACTOR"
    sig = {q: fac * B[:, c] for q, c in FORCE_COLS.items() if c < B.shape[1]}
    # dt de reference = premier segment de la fenetre (CFL nominal) ; les suivants sont
    # rapportes a part (un restart relance a un autre CFL change le pas)
    dt_first = dt_segments[0][1] if dt_segments else float("nan")
    return dict(t=t, sig=sig, restarts=restarts, factor=fac, factor_how=how,
                dt=dt_first, dt_segments=dt_segments)


def window_mask(t, y, restarts):
    m = t >= T_CUT
    for tr in restarts:
        m &= ~((t >= tr) & (t < tr + RESTART_GUARD))
    if m.sum() < 20:
        return m, 0
    med = np.median(y[m])
    mad = 1.4826 * np.median(np.abs(y[m] - med))
    if mad <= 0:
        mad = np.std(y[m]) + 1e-30
    spk = np.zeros(len(y), dtype=bool)
    spk[m] = np.abs(y[m] - med) > SPIKE_K * mad
    nspk = int(spk.sum())
    if nspk:
        idx = np.where(spk)[0]
        for s in (-3, -2, -1, 1, 2, 3):
            spk[np.clip(idx + s, 0, len(y) - 1)] = True
    return m & ~spk, nspk


def _runs(mask):
    idx = np.where(mask)[0]
    return np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if idx.size else []


def tmean(t, y, mask):
    I = T = 0.0
    for r in _runs(mask):
        if r.size > 1:
            I += _trapz(y[r], t[r])
            T += t[r[-1]] - t[r[0]]
    return (I / T if T > 0 else float("nan")), T


def trms(t, y, mask, mean):
    I = T = 0.0
    for r in _runs(mask):
        if r.size > 1:
            I += _trapz((y[r] - mean) ** 2, t[r])
            T += t[r[-1]] - t[r[0]]
    return float(np.sqrt(I / T)) if T > 0 else float("nan")


def batch_means(t, y, mask, nb):
    tt = t[mask]
    if tt.size < 10 * nb:
        return []
    edges = np.linspace(tt[0], tt[-1], nb + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        mu, T = tmean(t, y, mask & (t >= a) & (t <= b))
        if T > 0:
            out.append(mu)
    return out


def running_mean(t, y, mask):
    tt, yy = t[mask], y[mask]
    if tt.size < 3:
        return tt, yy
    d = np.diff(tt)
    gap = d > 50 * np.median(d)
    inc = 0.5 * (yy[1:] + yy[:-1]) * d
    inc[gap], d = 0.0, np.where(gap, 0.0, d)
    cI, cT = np.concatenate([[0.0], np.cumsum(inc)]), np.concatenate([[0.0], np.cumsum(d)])
    with np.errstate(invalid="ignore", divide="ignore"):
        return tt, np.where(cT > 0, cI / np.where(cT > 0, cT, 1.0), yy[0])


def analyse_forces(case):
    F = load_forces(case["dir"])
    t = F["t"]
    out = dict(restarts=F["restarts"], factor_how=F["factor_how"], t_end=float(t.max()),
               dt=F["dt"], dt_segments=F["dt_segments"], window_nominal=float(t.max() - T_CUT))
    masks = {}
    for q, y in F["sig"].items():
        m, nspk = window_mask(t, y, F["restarts"])
        mean, T = tmean(t, y, m)
        raw, _ = tmean(t, y, t >= T_CUT)
        bm = batch_means(t, y, m, N_BATCH)
        hv = batch_means(t, y, m, 2)
        out[q] = dict(mean=mean, raw_mean=raw, rms=trms(t, y, m, mean),
                      se=float(np.std(bm, ddof=1) / np.sqrt(len(bm))) if len(bm) > 1 else float("nan"),
                      halves=hv, n_spikes=nspk, window_effective=T)
        masks[q] = m
    F["masks"] = masks
    # saut a chaque restart dans la fenetre : moyenne avant / apres (garde exclue)
    rs_in = [tr for tr in F["restarts"] if T_CUT < tr < t.max()]
    steps = []
    for i, tr in enumerate(rs_in):
        t_next = rs_in[i + 1] if i + 1 < len(rs_in) else np.inf
        rec = dict(t=tr)
        for a, b in zip(F["dt_segments"][:-1], F["dt_segments"][1:]):
            if a[0] < tr <= b[0] + 1e-6:
                rec["dt_before"], rec["dt_after"] = a[1], b[1]
        for q, y in F["sig"].items():
            mb, Tb = tmean(t, y, masks[q] & (t < tr))
            ma, Ta = tmean(t, y, masks[q] & (t >= tr) & (t < t_next))
            if Tb > 1.0 and Ta > 1.0:
                rec[q] = (mb, ma)
        steps.append(rec)
    out["restart_steps"] = steps
    return out, F


# =============================================================================
# PAROI
# =============================================================================
ALIASES = dict(x=["x"], y=["y"], z=["z"], cp=["cp", "c_p"], cfx=["cf_x", "cfx"],
               cfy=["cf_y", "cfy"], cfz=["cf_z", "cfz"], cf=["cf"])
DEFAULT_ORDER = ["x", "y", "z", "cfx", "cfy", "cfz", "cp"]     # schema UDF connu


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
    if not files:
        files = glob.glob(os.path.join(case_dir, "cf_cp_time_*.csv"))
    groups = {}
    for f in files:
        t = parse_time(os.path.basename(f))
        if t is not None:
            groups.setdefault(round(t, 9), []).append(f)
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
            continue                                   # fichier de rang vide
        c = {k: _col(h, D, k) for k in ("x", "y", "cp", "cfx", "cfy", "cfz", "cf")}
        if c["x"] is None or c["y"] is None or c["cp"] is None:
            raise RuntimeError(f"{f} : colonnes x, y, Cp introuvables (en-tete : {h})")
        n = len(c["x"])
        nan = np.full(n, np.nan)
        parts.append(np.column_stack([c["x"], c["y"], c["cp"],
                                      c["cfx"] if c["cfx"] is not None else nan,
                                      c["cfy"] if c["cfy"] is not None else nan,
                                      c["cfz"] if c["cfz"] is not None else np.zeros(n),
                                      c["cf"] if c["cf"] is not None else nan]))
    return np.vstack(parts) if parts else None


def merge_nodes(xy):
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(xy)
    pairs = cKDTree(xy).query_pairs(r=MERGE_TOL, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n), n
    G = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(G, directed=False)[::-1]


def profile_frame(X, Y):
    P = np.column_stack([X, Y])
    i_far = np.argmax(np.hypot(*(P - P.mean(axis=0)).T))
    i_oth = np.argmax(np.hypot(*(P - P[i_far]).T))
    le, te = (P[i_oth], P[i_far]) if P[i_oth, 0] < P[i_far, 0] else (P[i_far], P[i_oth])
    for _ in range(2):                  # BF = milieu du culot ; BA = point le plus loin du BF
        c = np.linalg.norm(te - le)
        ex = (te - le) / c
        xs = (P - le) @ ex
        te = P[xs > xs.max() - 2e-3 * c].mean(axis=0)
        le = P[np.argmax(np.hypot(*(P - te).T))]
    c = np.linalg.norm(te - le)
    ex = (te - le) / c
    ey = np.array([-ex[1], ex[0]])
    return dict(xs=(P - le) @ ex / c, ys=(P - le) @ ey / c, chord=float(c),
                aoa=float(np.degrees(np.arctan2(-(te - le)[1], (te - le)[0]))))


def _polyline(P, idx, start=None):
    Q = P[idx] if start is None else np.vstack([start, P[idx]])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])
    T = np.gradient(Q, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None] + 1e-300
    return (s, T) if start is None else (s[1:], T[1:])


def _weights(s):
    w = np.zeros_like(s)
    if len(s) > 2:
        w[1:-1] = 0.5 * (s[2:] - s[:-2])
        w[0], w[-1] = 0.5 * (s[1] - s[0]), 0.5 * (s[-1] - s[-2])
    return np.abs(w)


def _refine_min(s, v, i):
    if 0 < i < len(v) - 1:
        a, b, c = np.polyfit(s[i - 1:i + 2], v[i - 1:i + 2], 2)
        if a > 0:
            s0 = -b / (2 * a)
            if min(s[i - 1], s[i + 1]) <= s0 <= max(s[i - 1], s[i + 1]):
                return float(a * s0 ** 2 + b * s0 + c), float(s0)
    return float(v[i]), float(s[i])


def _crossing(x, v, rising):
    for i in range(len(v) - 1):
        if (rising and v[i] < 0 <= v[i + 1]) or (not rising and v[i] > 0 >= v[i + 1]):
            f = v[i] / (v[i] - v[i + 1])
            return float(x[i] + f * (x[i + 1] - x[i]))
    return None


def load_y1(case_dir, X, Y):
    from scipy.spatial import cKDTree
    files = sorted(glob.glob(os.path.join(case_dir, "wall_distance_rank_*.csv")) +
                   glob.glob(os.path.join(case_dir, "resultats_csv", "wall_distance_rank_*.csv")))
    chunks = []
    for f in files:
        h, D = read_table(f)
        if D is None:
            continue
        if h is not None:
            ix = h.index("x") if "x" in h else 0
            iy = h.index("y") if "y" in h else 1
            idd = next((h.index(a) for a in ("wall_distance", "walldistance", "wall_dist", "distance",
                                             "dist", "wd", "d", "y1") if a in h), D.shape[1] - 1)
        else:
            ix, iy, idd = 0, 1, D.shape[1] - 1
        dd = D[:, idd]
        keep = (dd > 1e-12) & (dd < 1e-3)               # noeuds interieurs proches paroi
        chunks.append(np.column_stack([D[keep, ix], D[keep, iy], dd[keep]]))
    if not chunks or sum(len(c) for c in chunks) == 0:
        return None, files
    W = np.vstack(chunks)
    dist, j = cKDTree(W[:, :2]).query(np.column_stack([X, Y]), k=2)   # 1er noeud hors paroi
    first = dist[:, 0] > 1e-10
    jj = np.where(first, j[:, 0], j[:, 1])
    dd = np.where(first, dist[:, 0], dist[:, 1])
    return dict(geom=dd, file=W[jj, 2]), files


def export_mode(V):
    if CF_EXPORT in ("scalar", "vector"):
        return CF_EXPORT
    if np.all(np.isfinite(V[:, 1])) and (np.all(~np.isfinite(V[:, 2])) or np.nanmax(np.abs(V[:, 2])) == 0.0):
        return "scalar"                     # cf_y absent ou identiquement nul : cf_x est un scalaire
    return "vector" if np.all(np.isfinite(V[:, 1])) else "column_cf"


def _fix_flip(cf, Tx, amb=0.1):
    """Pres d'une tangente verticale, l'orientation du scalaire exporte (vers x croissant)
    et celle deduite de la tangente discrete peuvent differer d'un noeud. Dans la zone
    |T_x| < amb, on parcourt les noeuds dans l'ordre et on garde le signe qui assure la
    continuite avec le noeud precedent (un mauvais signe cree un saut de 2|Cf|)."""
    cf = cf.copy()
    for i in range(1, len(cf)):
        if abs(Tx[i]) < amb and abs(-cf[i] - cf[i - 1]) < abs(cf[i] - cf[i - 1]):
            cf[i] = -cf[i]
    return cf


def surface_stats(G, up, lo, s_up, s_lo, T_up, T_lo, V, sign):
    """V : colonnes cp, cfx, cfy, cfz, cf (moyennes par noeud)."""
    out = {}
    mode = export_mode(V)
    for side, idx, s, T in (("up", up, s_up, T_up), ("lo", lo, s_lo, T_lo)):
        cp = V[idx, 0]
        if mode == "scalar":
            sg = np.where(T[:, 0] >= 0, 1.0, -1.0)
            cf = _fix_flip(sign * V[idx, 1] * sg, T[:, 0])                # cisaillement BA -> BF
            mag = tan = np.abs(V[idx, 1])
        elif mode == "vector":
            cf = sign * (V[idx, 1] * T[:, 0] + V[idx, 2] * T[:, 1])   # composante tangentielle signee
            mag = np.sqrt(V[idx, 1] ** 2 + V[idx, 2] ** 2 + V[idx, 3] ** 2)
            tan = np.sqrt(cf ** 2 + V[idx, 3] ** 2)
        else:
            cf, mag = sign * V[idx, 4], np.abs(V[idx, 4])
            tan = mag
        with np.errstate(divide="ignore", invalid="ignore"):
            align = np.where(mag > 1e-12, tan / mag, 1.0)
        out[side] = dict(xs=G["xs"][idx], s=s / G["chord"], cp=cp, cf=cf, cfmag=mag, cftan=tan,
                         align=align, idx=idx, T=T)
    out["mode"] = mode
    return out


def cpmin_upper(S):
    """(raffine, x/c du minimum raffine, nodal)"""
    u = S["up"]
    m = u["xs"] < 0.1
    i = int(np.argmin(np.where(m, u["cp"], np.inf)))
    val, s0 = _refine_min(u["s"], u["cp"], i)
    return val, float(np.interp(s0, u["s"], u["xs"])), float(u["cp"][i])


def stagnation(S):
    """Zero de Cf (projete sur s croissant) le plus proche du maximum de Cp."""
    lo, up = S["lo"], S["up"]
    s = np.concatenate([lo["s"][::-1], up["s"]])
    cf = np.concatenate([-lo["cf"][::-1], up["cf"]])
    cp = np.concatenate([lo["cp"][::-1], up["cp"]])
    xs = np.concatenate([lo["xs"][::-1], up["xs"]])
    near = np.abs(s) < 0.1
    s_cp = float(s[near][np.argmax(cp[near])])
    cands = []
    for i in range(len(s) - 1):
        if near[i] and near[i + 1] and cf[i] < 0 <= cf[i + 1]:
            f = cf[i] / (cf[i] - cf[i + 1])
            cands.append((float(s[i] + f * (s[i + 1] - s[i])), float(xs[i] + f * (xs[i + 1] - xs[i]))))
    if not cands:
        return None, None, s_cp
    sc, xc = min(cands, key=lambda c: abs(c[0] - s_cp))
    return sc, xc, s_cp


def section_integrals(S, P, V, sign, chord):
    """Coefficients recalcules par integration sur le contour ferme (trapezes aux noeuds)."""
    idx = np.concatenate([S["up"]["idx"], S["lo"]["idx"][::-1]])   # BA->BF extrados, BF->BA intrados
    Q = P[idx] / chord
    cp = V[idx, 0]
    cfv = sign * V[idx, 1:3]
    d = np.roll(Q, -1, axis=0) - Q
    L = np.hypot(d[:, 0], d[:, 1])
    dh = d / np.maximum(L, 1e-300)[:, None]
    N = np.column_stack([-dh[:, 1], dh[:, 0]])                  # normale sortante (sens horaire)
    cpm = 0.5 * (cp + np.roll(cp, -1))
    cfm = 0.5 * (cfv + np.roll(cfv, -1, axis=0))
    Fp = -(cpm[:, None] * N * L[:, None]).sum(axis=0)
    Fraw = (cfm * L[:, None]).sum(axis=0)
    Ftan = (((cfm * dh).sum(axis=1))[:, None] * dh * L[:, None]).sum(axis=0)
    # frottement reconstruit : Cf(BA->BF) x tangente BA->BF, aux noeuds, puis trapezes
    wv = np.vstack([S["up"]["cf"][:, None] * S["up"]["T"], (S["lo"]["cf"][:, None] * S["lo"]["T"])[::-1]])
    Fw = (0.5 * (wv + np.roll(wv, -1, axis=0)) * L[:, None]).sum(axis=0)
    return dict(CDp=float(Fp[0]), CLp=float(Fp[1]), CDv=float(Fw[0]), CLv=float(Fw[1]),
                CDv_tan=float(Ftan[0]), CLv_tan=float(Ftan[1]),
                CDv_raw=float(Fraw[0]), CLv_raw=float(Fraw[1]))


def analyse_surface(case):
    from scipy.spatial import cKDTree
    snaps = [(t, f) for t, f in list_snapshots(case["dir"]) if t >= T_CUT]
    if not snaps:
        raise RuntimeError("aucun cf_cp_time_*.csv avec t >= T_CUT")
    S0 = read_snapshot(snaps[0][1])
    lab, n0 = merge_nodes(S0[:, :2])
    cnt0 = np.bincount(lab, minlength=n0)
    X = np.bincount(lab, weights=S0[:, 0], minlength=n0) / cnt0
    Y = np.bincount(lab, weights=S0[:, 1], minlength=n0) / cnt0
    tree = cKDTree(np.column_stack([X, Y]))
    has = [bool(np.all(np.isfinite(S0[:, 2 + c]))) for c in range(5)]
    sums, nsnap = np.zeros((n0, 5)), 0
    edges = np.linspace(snaps[0][0], snaps[-1][0] + 1e-9, N_BATCH + 1)
    bsum, bcnt = np.zeros((N_BATCH, n0, 5)), np.zeros(N_BATCH)
    snap_cpmin = []
    for t, files in snaps:
        S = S0 if nsnap == 0 else read_snapshot(files)
        if S is None:
            continue
        d, j = tree.query(S[:, :2])
        if d.max() > 1e-6:
            print(f"   ! instantane t={t} : noeuds non reconnus (maillage different ?) -> ignore")
            continue
        c = np.maximum(np.bincount(j, minlength=n0), 1)
        vals = np.nan_to_num(S[:, 2:7])
        M = np.column_stack([np.bincount(j, weights=vals[:, k], minlength=n0) / c for k in range(5)])
        sums += M
        nsnap += 1
        b = min(int(np.searchsorted(edges, t, side="right")) - 1, N_BATCH - 1)
        bsum[b] += M
        bcnt[b] += 1
        snap_cpmin.append(M[:, 0])
    V = sums / nsnap
    for k in range(5):
        if not has[k]:
            V[:, k] = np.nan
    G = profile_frame(X, Y)
    theta = np.arctan2(G["ys"], G["xs"] - 0.5)
    up = np.where(G["ys"] >= 0)[0]
    up = up[np.argsort(-theta[up])]
    lo = np.where(G["ys"] < 0)[0]
    lo = lo[np.argsort(theta[lo])]
    P = np.column_stack([X, Y])
    s_up, T_up = _polyline(P, up)
    s_lo, T_lo = _polyline(P, lo, start=P[up[0]])
    s_lo = -s_lo
    Sd = surface_stats(G, up, lo, s_up, s_lo, T_up, T_lo, V, +1.0)
    u = Sd["up"]
    ref = (u["xs"] > 0.1) & (u["xs"] < 0.9)
    sign = +1.0
    if ref.any() and np.median(u["cf"][ref]) < 0:              # convention de signe du UDF
        sign = -1.0
        Sd = surface_stats(G, up, lo, s_up, s_lo, T_up, T_lo, V, sign)
    out = dict(n_snap=nsnap, t_snap=(snaps[0][0], snaps[-1][0]), n_nodes=int(n0),
               chord=G["chord"], aoa=G["aoa"], cf_sign=sign, data=Sd)
    # Cp,min (moyenne temporelle), incertitude par blocs, dispersion instantane par instantane
    out["cpmin"], out["cpmin_xs"], out["cpmin_nodal"] = cpmin_upper(Sd)
    bref, bnod = [], []
    for b in range(N_BATCH):
        if bcnt[b] > 0:
            Vb = bsum[b] / bcnt[b]
            r = cpmin_upper(surface_stats(G, up, lo, s_up, s_lo, T_up, T_lo, Vb, sign))
            bref.append(r[0])
            bnod.append(r[2])
    se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
    out["cpmin_se"], out["cpmin_nodal_se"] = se(bref), se(bnod)
    out["cpmin_snap_std"] = float(np.std([np.min(c[up][Sd["up"]["xs"] < 0.1]) for c in snap_cpmin]))
    l = Sd["lo"]
    out["stag_s"], out["stag_xs"], out["cpmax_s"] = stagnation(Sd)
    out["cpmax"] = float(np.max(np.concatenate([u["cp"], l["cp"]])))
    out["integrals"] = section_integrals(Sd, np.column_stack([X, Y]), V, sign, G["chord"])
    al = np.concatenate([u["align"], l["align"]])
    mg = np.concatenate([u["cfmag"], l["cfmag"]])
    xsa = np.concatenate([u["xs"], l["xs"]])
    sa = np.concatenate([u["s"], l["s"]])
    bad = (al < ALIGN_FLAG) & (mg > 1e-3) & (Sd["mode"] == "vector")
    out["misaligned"] = dict(n=int(bad.sum()), s=sorted(float(v) for v in sa[bad]),
                             xs_max=float(xsa[bad].max()) if bad.any() else None)
    out["cf_export"] = Sd["mode"]
    nose = (u["xs"] < 0.05) & (u["s"] > 0)                  # noeud du BA (s = 0) exclu
    i_pk = int(np.argmax(np.where(nose, u["cf"], -np.inf)))
    out["cf_nose_peak"], out["cf_nose_xs"] = float(u["cf"][i_pk]), float(u["xs"][i_pk])
    out["cf_nose_peak_s"] = float(u["s"][i_pk])
    out["cf_le_node"] = float(u["cfmag"][0])
    aft = u["xs"] > 0.05
    out["sep_xs"] = _crossing(u["xs"][aft], u["cf"][aft], rising=False)
    te = u["xs"] > 0.9
    out["cf_te_min"] = float(np.min(u["cf"][te])) if te.any() else float("nan")
    # y+
    y1, wfiles = load_y1(case["dir"], X, Y)
    out["yplus"] = out["yplus_tan"] = None
    if y1 is not None:
        for side in ("up", "lo"):
            D = Sd[side]
            D["y1"], D["y1_file"] = y1["geom"][D["idx"]], y1["file"][D["idx"]]
            D["yplus"] = D["y1"] * np.sqrt(D["cfmag"] / 2.0) * RE          # module : conservatif
            D["yplus_tan"] = D["y1"] * np.sqrt(D["cftan"] / 2.0) * RE      # composante tangentielle
        for key, name in (("yplus", "yplus"), ("yplus_tan", "yplus_tan")):
            yp = np.concatenate([Sd["up"][key], Sd["lo"][key]])
            ww = np.concatenate([_weights(Sd["up"]["s"]), _weights(Sd["lo"]["s"])])
            i = int(np.argmax(yp))
            sides = np.array(["up"] * len(Sd["up"][key]) + ["lo"] * len(Sd["lo"][key]))
            o = np.argsort(yp)
            cw = np.cumsum(ww[o]) / ww.sum()
            out[name] = dict(mean=float((ww * yp).sum() / ww.sum()), max=float(yp[i]), max_side=str(sides[i]),
                             max_xs=float(np.concatenate([Sd["up"]["xs"], Sd["lo"]["xs"]])[i]),
                             max_s=float(sa[i]), align_at_max=float(al[i]),
                             y1_at_max=float(np.concatenate([Sd["up"]["y1"], Sd["lo"]["y1"]])[i]),
                             frac_above_1=float(ww[yp > 1.0].sum() / ww.sum()),
                             p95=float(np.interp(0.95, cw, yp[o])),
                             max_excl_le=float(np.max(yp[np.abs(sa) > 0])),
                             max_outside_spike=float(np.max(yp[np.abs(sa) > 1e-3])))
        g, f = y1["geom"], y1["file"]
        out["y1"] = dict(median_geom=float(np.median(g)), median_file=float(np.median(f)),
                         max_rel_diff=float(np.max(np.abs(f / g - 1))))
    else:
        print(f"   ! wall_distance_rank_*.csv introuvable ou vide ({len(wfiles)} fichier(s)) -> pas de y+")
    return out


# =============================================================================
# FIGURES
# =============================================================================
def _case_handles():
    return [Line2D([], [], color=c["color"], ls=c["ls"], label=c["label"]) for c in CASES]


def fig_surface(res, path):
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 5.4))
    (a, b), (c, d) = axs
    for case in CASES:
        S = res[case["key"]]["surf"]["data"]
        kw = dict(color=case["color"], ls=case["ls"])
        for side, alpha in (("up", 1.0), ("lo", 0.5)):
            a.plot(S[side]["xs"], S[side]["cp"], lw=1.0, alpha=alpha, **kw)
            b.plot(S[side]["xs"], S[side]["cf"], lw=1.0, alpha=alpha, **kw)
        s = np.concatenate([S["lo"]["s"][::-1], S["up"]["s"]])
        cp = np.concatenate([S["lo"]["cp"][::-1], S["up"]["cp"]])
        cf = np.concatenate([-S["lo"]["cf"][::-1], S["up"]["cf"]])   # projete sur s croissant
        m = np.abs(s) <= LE_ZOOM
        for ax, v in ((c, cp), (d, cf)):
            ax.plot(s[m], v[m], lw=0.8, marker=case["marker"], ms=2.8, mfc="none", mew=0.7, **kw)
    for ax in (a, c):
        ax.invert_yaxis()
        ax.set_ylabel(r"$C_p$")
    for ax in (b, d):
        ax.axhline(0.0, color="k", lw=0.6)
    b.set_ylabel(r"$C_f$")
    d.set_ylabel(r"$C_f$ (along increasing $s$)")
    a.set_xlabel(r"$x/c$")
    b.set_xlabel(r"$x/c$")
    vals = np.concatenate([res[c_["key"]]["surf"]["data"][sd]["cf"][res[c_["key"]]["surf"]["data"][sd]["xs"] > 0.005]
                           for c_ in CASES for sd in ("up", "lo")])
    lo_, hi_ = min(np.percentile(vals, 0.5), -0.002), np.percentile(vals, 99.5)
    pad = 0.08 * (hi_ - lo_)
    b.set_ylim(lo_ - pad, hi_ + pad)
    for ax in (c, d):
        ax.axvline(0.0, color="0.5", lw=0.6, ls=":")
        ax.set_xlabel(r"$s/c$  (lower surface $<0<$ upper surface)")
        ax.set_xlim(-LE_ZOOM, LE_ZOOM)
    for ax, t in zip(axs.flat, "abcd"):
        ax.set_title(f"({t})", loc="left", fontsize=8.5)
    sides = [Line2D([], [], color="k", label="upper surface"),
             Line2D([], [], color="k", alpha=0.5, label="lower surface")]
    b.legend(handles=_case_handles() + sides, loc="upper right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_cost(rows, path, with_err=True):
    ref = next(r for r in rows if r["key"] == REF_KEY)
    Ns = np.array([r["Ns"] for r in rows], dtype=float)
    if len(set(Ns)) < len(Ns):
        print("   (figure cout ignoree : meme nombre de points de surface pour tous les cas)")
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.75))
    a.axhspan(-1, 1, color="0.88", lw=0, zorder=0)
    a.axhline(0, color="k", lw=0.6)
    quants = [("CL", r"$C_L$", "o"), ("CD", r"$C_D$", "s"),
              ("CDv", r"$C_{D,v}$", "D"), ("cpmin", r"$|C_{p,\min}|$", "^")]
    ymax = 1.2
    for (q, lab, mk), off in zip(quants, (-0.045, -0.015, 0.015, 0.045)):
        v0 = abs(ref[q])
        dev = np.array([100 * (abs(r[q]) - v0) / v0 for r in rows])
        err = np.array([100 * r.get("se_" + q, np.nan) / v0 for r in rows]) if with_err else None
        if err is not None and not np.all(np.isfinite(err)):
            err = np.where(np.isfinite(err), err, 0.0)
        a.errorbar(Ns * (1 + off), dev, yerr=err, marker=mk, ms=4, lw=0.9, capsize=2, label=lab)
        ymax = max(ymax, np.max(np.abs(dev) + (err if err is not None else 0)) * 1.25)
    a.set_ylim(-ymax, ymax)
    a.set_ylabel(r"deviation from C0 [%]")
    a.legend(ncol=2, loc="upper right", frameon=True, framealpha=0.9)
    steps = np.array([(1 / r["dt"]) / (1 / ref["dt"]) for r in rows])
    work = np.array([(r["E"] / r["dt"]) / (ref["E"] / ref["dt"]) for r in rows])
    for y, lab, mk in ((steps, "time steps per CTU", "o"), (work, r"work $E/\Delta t$", "s")):
        p = np.polyfit(np.log(Ns), np.log(y), 1)
        xx = np.linspace(Ns.min() * 0.95, Ns.max() * 1.05, 50)
        line, = b.plot(Ns, y, marker=mk, ls="none", ms=4.5, label=lab + rf"  ($\propto N_s^{{{p[0]:.1f}}}$)")
        b.plot(xx, np.exp(np.polyval(p, np.log(xx))), ls="--", lw=0.9, color=line.get_color())
    b.set_xscale("log")
    b.set_yscale("log")
    b.axhline(1.0, color="0.5", lw=0.6)
    b.set_ylabel("relative to C0")
    b.yaxis.set_major_locator(FixedLocator([0.5, 1, 2]))
    b.set_yticklabels(["0.5", "1", "2"])
    b.yaxis.set_minor_formatter(NullFormatter())
    b.legend(loc="upper left", frameon=True, framealpha=0.9)
    for ax in (a, b):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(Ns))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticklabels([f"{int(n)}\n{r['key']}" for n, r in zip(Ns, rows)])
        ax.set_xlabel(r"total surface points $N_s$")
    for ax, t in zip((a, b), "ab"):
        ax.set_title(f"({t})", loc="left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_convergence(res, path):
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 5.0), sharex=True)
    labels = dict(CL=r"$C_L$", CD=r"$C_D$", CDp=r"$C_{D,p}$", CDv=r"$C_{D,v}$")
    for ax, q in zip(axs.flat, ("CL", "CD", "CDp", "CDv")):
        lo, hi = [], []
        for case in CASES:
            F = res[case["key"]]["_F"]
            if q not in F["sig"]:
                continue
            t, y, m = F["t"], F["sig"][q], F["masks"][q]
            show = t >= T_CUT - 5
            k = max(1, int(show.sum() // 40000))
            ax.plot(t[show][::k], y[show][::k], color=case["color"], lw=0.3, alpha=0.3, rasterized=True)
            tt, rm = running_mean(t, y, m)
            keep = tt >= T_CUT + 0.5
            ax.plot(tt[keep], rm[keep], color=case["color"], ls=case["ls"], lw=1.4)
            for tr in F["restarts"]:
                if tr >= T_CUT - 5:
                    ax.axvline(tr, color=case["color"], ls=":", lw=0.8)
            if m.any():
                lo.append(np.percentile(y[m], 0.5))
                hi.append(np.percentile(y[m], 99.5))
        if lo:
            pad = 0.15 * (max(hi) - min(lo))
            ax.set_ylim(min(lo) - pad, max(hi) + pad)
        ax.axvline(T_CUT, color="k", ls=":", lw=0.8)
        ax.set_ylabel(labels[q])
    for ax in axs[1]:
        ax.set_xlabel(r"$t\,U_\infty/c$")
    axs[0, 0].legend(handles=_case_handles(), loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_yplus(res, path):
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    top = 1.05
    for case in CASES:
        S = res[case["key"]]["surf"]["data"]
        for side, alpha in (("up", 1.0), ("lo", 0.5)):
            ax.plot(S[side]["xs"], S[side]["yplus"], color=case["color"], ls=case["ls"], lw=0.9, alpha=alpha)
            top = max(top, 1.05 * np.max(S[side]["yplus"]))
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_ylim(0, top)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$y^+$")
    ax.legend(handles=_case_handles(), loc="upper right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================
def save_surface_npz(res):
    for c in CASES:
        S = res[c["key"]]["surf"]
        if not S:
            continue
        arrs = {}
        for side in ("up", "lo"):
            for f in ("xs", "s", "cp", "cf", "yplus"):
                if f in S["data"][side]:
                    arrs[f"{side}_{f}"] = S["data"][side][f]
        np.savez(os.path.join(OUTDIR, f"surface_{c['key'].replace('+', 'plus')}.npz"), **arrs)


def replot():
    res = {}
    for c in CASES:
        path = os.path.join(OUTDIR, f"surface_{c['key'].replace('+', 'plus')}.npz")
        if not os.path.isfile(path):
            print(f"introuvable : {path} (lancer d'abord le calcul complet)")
            return
        Z = np.load(path)
        data = {sd: {f: Z[f"{sd}_{f}"] for f in ("xs", "s", "cp", "cf", "yplus") if f"{sd}_{f}" in Z}
                for sd in ("up", "lo")}
        res[c["key"]] = dict(surf=dict(data=data, yplus=("yplus" in data["up"]) or None))
    fig_surface(res, os.path.join(OUTDIR, f"{FIG_PREFIX}_surface.pdf"))
    if all(res[k]["surf"]["yplus"] for k in res):
        fig_yplus(res, os.path.join(OUTDIR, f"{FIG_PREFIX}_yplus.pdf"))
    print(f"figures paroi refaites depuis {OUTDIR}/surface_*.npz")


def rows_from_json(path):
    with open(path) as fh:
        J = json.load(fh)
    rows = []
    for c in CASES:
        fr, s = J[c["key"]]["forces"], J[c["key"]].get("surf")
        dt = fr["dt_segments"][0][1] if fr.get("dt_segments") else fr["dt"]
        row = dict(key=c["key"], Ns=2 * sum(c["pts"]), E=c["n_elem"], dt=dt)
        for q in ("CL", "CD", "CDv"):
            row[q], row["se_" + q] = fr[q]["mean"], fr[q]["se"]
        if s:
            row["cpmin"] = s.get("cpmin_nodal", s["cpmin"])
            row["se_cpmin"] = s.get("cpmin_nodal_se", s["cpmin_snap_std"] / np.sqrt(s["n_snap"]))
        else:
            row["cpmin"], row["se_cpmin"] = float("nan"), float("nan")
        rows.append(row)
    return rows


def preview():
    rows = [dict(key=c["key"], Ns=2 * sum(c["pts"]), E=c["n_elem"], **TABLE[c["key"]]) for c in CASES]
    out = os.path.join(OUTDIR, f"{FIG_PREFIX}_cost_preview.pdf")
    fig_cost(rows, out, with_err=False)
    print(f"apercu (valeurs du tableau, sans barres d'erreur) : {out}")


def latex_sci(v, nd=2):
    e = int(np.floor(np.log10(abs(v))))
    return f"${v / 10 ** e:.{nd}f} \\times 10^{{{e}}}$"


def surface_diffs(res):
    """Ecarts locaux a C0, sans binning : C0 interpole sur les noeuds de chaque cas."""
    R0 = res[REF_KEY]["surf"]["data"]
    print("\n" + "=" * 78 + "\n ECARTS LOCAUX A C0 (paroi)\n" + "=" * 78)
    for c in CASES:
        if c["key"] == REF_KEY:
            continue
        R = res[c["key"]]["surf"]["data"]
        for side in ("up", "lo"):
            x0, o = R0[side]["xs"], np.argsort(R0[side]["xs"])
            for lo_, hi_ in ((0.0, 0.02), (0.02, 0.1), (0.1, 0.95)):
                m = (R[side]["xs"] >= lo_) & (R[side]["xs"] < hi_)
                if m.sum() < 2:
                    continue
                cp0 = np.interp(R[side]["xs"][m], x0[o], R0[side]["cp"][o])
                cf0 = np.interp(R[side]["xs"][m], x0[o], R0[side]["cf"][o])
                dcf = R[side]["cf"][m] - cf0
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = np.nanmedian(np.where(np.abs(cf0) > 1e-5, dcf / np.abs(cf0), np.nan))
                print(f"   {c['key']:3s} {side} x/c [{lo_:.2f},{hi_:.2f}) : max|dCp| = {np.max(np.abs(R[side]['cp'][m] - cp0)):.3g}"
                      f"   max|dCf| = {np.max(np.abs(dcf)):.2e}   dCf/|Cf| median = {100 * rel:+.1f} %")


def fmt_pm(d, key, nd=5):
    v = d[key]
    return f"{v['mean']:.{nd}f} +- {v['se']:.1e}  (brut {v['raw_mean']:.{nd}f}, rms {v['rms']:.2e}, " \
           f"demi-fenetres {', '.join(f'{h:.{nd}f}' for h in v['halves'])}, impulsions {v['n_spikes']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--from-json", default=None)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    pub_style()
    os.makedirs(OUTDIR, exist_ok=True)
    if args.preview:
        preview()
        return
    if args.replot:
        replot()
        return
    if args.from_json:
        out = os.path.join(OUTDIR, f"{FIG_PREFIX}_cost.pdf")
        fig_cost(rows_from_json(args.from_json), out)
        print(f"figure cout refaite depuis {args.from_json} : {out}")
        return
    res = {}
    for case in CASES:
        case["dir"] = os.path.join(os.path.expanduser(args.root), case["path"])
        print("=" * 78)
        print(f" {case['key']}  --  {case['dir']}")
        print("=" * 78)
        if not os.path.isdir(case["dir"]):
            print("   DOSSIER INTROUVABLE. Dossiers disponibles :")
            for p in sorted(glob.glob(os.path.join(os.path.expanduser(args.root), "serie*", "*"))):
                print("     ", p)
            sys.exit(1)
        R = {}
        R["forces"], R["_F"] = analyse_forces(case)
        fr = R["forces"]
        print(f"   efforts : {fr['factor_how']}")
        print(f"   restarts : {', '.join(f'{x:.2f}' for x in fr['restarts']) or 'aucun'}   "
              f"fin : {fr['t_end']:.2f} CTU   fenetre nominale : {fr['window_nominal']:.1f} CTU")
        print(f"   dt (1er segment de la fenetre) : {fr['dt']:.3e}   -> {1 / fr['dt']:.0f} pas/CTU")
        segs = fr["dt_segments"]
        if len(segs) > 1:
            print("   dt par segment : " + "; ".join(f"t>{a:.1f}: {v:.3e}" for a, v in segs))
            vals = [v for _, v in segs]
            if max(vals) / min(vals) > 1.2:
                print("   ! dt change de plus de 20 % entre segments : CFL different au restart")
        for st in fr["restart_steps"]:
            txt = f"   restart t = {st['t']:.2f}"
            if "dt_before" in st:
                txt += f" : dt x{st['dt_after'] / st['dt_before']:.2f}"
            for q in ("CL", "CD", "CDp", "CDv"):
                if q in st:
                    b, a = st[q]
                    txt += f" | {q} {100 * (a / b - 1):+.2f} %"
            print(txt + "   (moyenne apres / avant)")
        for q in ("CL", "CD", "CDp", "CDv"):
            if q in fr:
                print(f"   {q:4s} = {fmt_pm(fr, q)}")
        try:
            R["surf"] = analyse_surface(case)
            s = R["surf"]
            print(f"   paroi : {s['n_snap']} instantanes (t = {s['t_snap'][0]:.2f} -> {s['t_snap'][1]:.2f}), "
                  f"{s['n_nodes']} noeuds, corde {s['chord']:.4f}, AoA {s['aoa']:.2f} deg, signe Cf {s['cf_sign']:+.0f}")
            print(f"   Cp,min = {s['cpmin']:.4f} +- {s['cpmin_se']:.1e} a x/c = {s['cpmin_xs']:.4f} "
                  f"(nodal {s['cpmin_nodal']:.4f} ; dispersion par instantane {s['cpmin_snap_std']:.3f})")
            print(f"   point d'arret : s/c = {s['stag_s']}, x/c = {s['stag_xs']}   (max de Cp a s/c = {s['cpmax_s']:.4f})"
                  f"   Cp,max = {s['cpmax']:.4f}")
            I = s["integrals"]
            fr = R["forces"]
            print(f"   export du frottement : {s['cf_export']}")
            print(f"   controle integral : CL = {I['CLp'] + I['CLv']:.4f} (efforts {fr['CL']['mean']:.4f})"
                  f"  CD,p = {I['CDp']:.5f} (efforts {fr['CDp']['mean']:.5f})"
                  f"  CD,v = {I['CDv']:.5f} (efforts {fr['CDv']['mean']:.5f}, ecart {100 * (I['CDv'] / fr['CDv']['mean'] - 1):+.2f} %)"
                  f"   [brut {I['CDv_raw']:.5f}, projete {I['CDv_tan']:.5f}]")
            mis = s["misaligned"]
            if s["cf_export"] == "vector":
                print(f"   frottement non tangentiel (|Cf tan|/|Cf| < {ALIGN_FLAG}) : {mis['n']} noeud(s)"
                      + (f", s/c = {', '.join(f'{v:+.4f}' for v in mis['s'][:8])}" if mis["n"] else ""))
            print(f"   maximum de Cf au nez (hors noeud du BA) = {s['cf_nose_peak']:.4f} a s/c = {s['cf_nose_peak_s']:.4f}"
                  f"   | noeud du BA : {s['cf_le_node']:.4f}")
            print(f"   decollement extrados : {s['sep_xs'] if s['sep_xs'] is not None else 'aucun'}   "
                  f"Cf min pour x/c > 0.9 : {s['cf_te_min']:.2e}")
            for name, lab in (("yplus", "module"), ("yplus_tan", "tangentiel")):
                y = s.get(name)
                if y and not (name == "yplus_tan" and s["cf_export"] == "scalar"):
                    print(f"   y+ ({lab:10s}) moyen = {y['mean']:.3f}  p95 = {y['p95']:.3f}  max = {y['max']:.3f} "
                          f"(s/c = {y['max_s']:+.4f}, alignement {y['align_at_max']:.2f}, y1 = {y['y1_at_max']:.2e})"
                          f"  fraction > 1 = {100 * y['frac_above_1']:.2f} %  max hors noeud du BA = {y['max_excl_le']:.3f}"
                          f"  max pour |s|/c > 0.001 = {y['max_outside_spike']:.3f}")
            if s.get("y1"):
                print(f"   y1 median : geometrique {s['y1']['median_geom']:.3e}, fichier {s['y1']['median_file']:.3e}"
                      f", ecart max {100 * s['y1']['max_rel_diff']:.1f} %")
        except Exception as e:
            print(f"   ! analyse de paroi impossible : {e}")
            R["surf"] = None
        res[case["key"]] = R

    rows = []
    for c in CASES:
        fr, s = res[c["key"]]["forces"], res[c["key"]]["surf"]
        row = dict(key=c["key"], Ns=2 * sum(c["pts"]) if c.get("pts") else None, E=c["n_elem"], dt=fr["dt"])
        for q in ("CL", "CD", "CDv"):
            row[q], row["se_" + q] = fr[q]["mean"], fr[q]["se"]
        row["cpmin"] = s["cpmin_nodal"] if s else float("nan")
        row["se_cpmin"] = s["cpmin_nodal_se"] if s else float("nan")
        rows.append(row)
    if COST_FIGURE:
        fig_cost(rows, os.path.join(OUTDIR, f"{FIG_PREFIX}_cost.pdf"))
    fig_convergence(res, os.path.join(OUTDIR, f"{FIG_PREFIX}_convergence.pdf"))
    if all(res[c["key"]]["surf"] for c in CASES):
        save_surface_npz(res)
        fig_surface(res, os.path.join(OUTDIR, f"{FIG_PREFIX}_surface.pdf"))
        if all(res[c["key"]]["surf"]["yplus"] for c in CASES):
            fig_yplus(res, os.path.join(OUTDIR, f"{FIG_PREFIX}_yplus.pdf"))

    if all(res[c["key"]]["surf"] for c in CASES):
        surface_diffs(res)
    ref = next(r for r in rows if r["key"] == REF_KEY)
    print("\n" + "=" * 78 + "\n ECARTS\n" + "=" * 78)
    for r in rows:
        print(f"   {r['key']:3s} vs {REF_KEY} : CL {100 * (r['CL'] / ref['CL'] - 1):+.2f} %  CD {100 * (r['CD'] / ref['CD'] - 1):+.2f} %"
              f"  CDv {100 * (r['CDv'] / ref['CDv'] - 1):+.2f} %  |Cp,min| {100 * (abs(r['cpmin']) / abs(ref['cpmin']) - 1):+.2f} %"
              f"   | vs Martin Ferrer : CL {100 * (r['CL'] / 1.028 - 1):+.2f} %  CD {100 * (r['CD'] / 0.0205 - 1):+.2f} %")

    def cells(fn):
        return " & ".join(fn(c["key"]) for c in CASES)
    F = {k: res[k]["forces"] for k in res}
    S = {k: res[k]["surf"] for k in res}
    print("\n" + "=" * 78 + "\n LIGNES DU TABLEAU LaTeX\n" + "=" * 78)
    if all(S.values()) and all(S[k]["yplus"] for k in S):
        print(r"    $y^{+}$ mean & " + cells(lambda k: f"{S[k]['yplus']['mean']:.2f}") + r" \\")
        print(r"    $y^{+}$, 95th percentile & " + cells(lambda k: f"{S[k]['yplus']['p95']:.2f}") + r" \\")
        print(r"    $y^{+}$ max & " + cells(lambda k: f"{S[k]['yplus']['max']:.2f}") + r" \\")
        print(r"    $y^{+}$ max for $|s|/c > 0.001$ & " + cells(lambda k: f"{S[k]['yplus']['max_outside_spike']:.2f}") + r" \\")
        print(r"    Chord fraction with $y^{+} > 1$ & " + cells(lambda k: f"{100 * S[k]['yplus']['frac_above_1']:.1f}\\,\\%") + r" \\")
    if all(S.values()):
        print(r"    $C_{p,\min}$ (nodal) & " + cells(lambda k: f"${S[k]['cpmin_nodal']:.3f}$") + r" \\")
    print(r"    Friction share of $\CD$ & " + cells(lambda k: f"{100 * F[k]['CDv']['mean'] / F[k]['CD']['mean']:.1f}\\,\\%") + r" \\")
    print(r"    Time step $\Delta t$ [$c/U_\infty$] & " + cells(lambda k: latex_sci(F[k]['dt'])) + r" \\")
    print(r"    Time steps per CTU & " + cells(lambda k: f"{1 / F[k]['dt']:.0f}") + r" \\")
    dof = {c["key"]: c["n_elem"] * (c.get("N", 3) + 1) ** 3 for c in CASES}
    dofp = {c["key"]: c["n_elem"] * (c.get("N", 3) + 1) ** 2 for c in CASES}
    print(r"    DOF (3D) & " + cells(lambda k: f"${dof[k]:,}$".replace(",", r"\,")) + r" \\")
    print(r"    DOF per spanwise plane & " + cells(lambda k: f"${dofp[k]:,}$".replace(",", r"\,")) + r" \\")
    print(r"    Work relative to reference (DOF$/\Delta t$) & "
          + cells(lambda k: f"{(dof[k] / F[k]['dt']) / (dof[REF_KEY] / F[REF_KEY]['dt']):.2f}") + r" \\")
    print(r"    Averaging window [CTU] & " + cells(lambda k: f"{F[k]['window_nominal']:.1f}") + r" \\")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k not in ("_F", "data")}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    with open(os.path.join(OUTDIR, FIG_PREFIX.replace("fig_ch4_series", "serie") + "_summary.json"), "w") as fh:
        json.dump(clean(res), fh, indent=1)
    print(f"\nFigures et resume ecrits dans {OUTDIR}/")


if __name__ == "__main__":
    main()

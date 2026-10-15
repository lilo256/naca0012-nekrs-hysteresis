#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_spectra_44.py -- donnees de la section 4.4 (etude frequentielle), a lancer SUR LE CLUSTER.

  cd ~/cdf_runs && python3 -u extract_spectra_44.py | tee extract_spectra_44.log
  python3 -u extract_spectra_44.py --root ~/cdf_runs --only spec_a19.00,prod_a15

Ne relance aucun calcul : lit drag_results.dat, probes_uvp.csv et probes_positions.csv des cas
ci-dessous, calcule les spectres (Welch, Hann, recouvrement 50 %, moyenne retiree par segment,
amplitude = celle d'une sinusoide equivalente) et ecrit spectra_44.json (quelques Mo) a rapatrier.

Contenu (une entree par TACHE) :
  prod_a15     15 deg, N = 3, 50-100      spectres (segments 10 et 25 CTU), bandes, C_L(t) fin
  spec_a19     19 deg, pas fixe, 236-300  spectres (32 et 20 CTU), bandes, series lissees,
                                          cycle lent moyenne en phase, phases a f0, retour u_t < 0
  spec_a19_a   19 deg, 200-235            spectres (recoupement, pas 1.1e-5)
  N3 / N4      film_a15 / res_a15_N4, fenetre commune 100.30-101.499 : FFT de Hann unique (x8),
               series brutes (test de resolution)
  N4_late      res_a15_N4 104-fin         spectres 5 CTU (lacher a N = 4), C_L par tranches de 1 CTU
  a10          dom_a20_b30_v2, 42.5-fin   pic rapide a 10 deg sur le maillage de production
  up17         up_a17.00, 240-340         attache pres du decrochage (pas variable -> reechantillonne)
  cold17       cold_a17.00, 110-160       decroche (reechantillonne)
  jump13       jump_a13.50, 500-fin       decroche bas (reechantillonne) : St ou St_h constant ?
Dependances : numpy ; pandas conseille (lecture par blocs, memoire bornee).
"""
import argparse
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
# nom de tache : (dossier, incidence, t0, t1 (None = fin), segments Welch [CTU], options)
TASKS = {
    "prod_a15":   ("prod_a15",       15.00,  50.0, 100.0, [10.0, 25.0], {"cl_fine": (97.0, 100.0)}),
    "spec_a19":   ("spec_a19.00",    19.00, 236.0, 300.0, [32.0, 20.0], {"slow": True}),
    "spec_a19_a": ("spec_a19.00",    19.00, 200.0, 235.0, [10.0],       {}),
    "N3":         ("film_a15",       15.00, 100.30, 101.499, [],        {"single": True}),
    "N4":         ("res_a15_N4",     15.00, 100.30, 101.499, [],        {"single": True}),
    "N4_late":    ("res_a15_N4",     15.00, 104.0,  None,  [5.0],       {"slices": 102.0}),
    "a10":        ("dom_a20_b30_v2", 10.00,  42.5,  None,  [10.0],      {}),
    "up17":       ("up_a17.00",      17.00, 240.0, 340.0, [20.0],       {}),
    "cold17":     ("cold_a17.00",    17.00, 110.0, 160.0, [20.0],       {}),
    "jump13":     ("jump_a13.50",    13.50, 500.0,  None,  [20.0],      {}),
}
PATHS = {}               # chemins imposes si besoin : {"spec_a19.00": "spec/spec_a19.00"}
FORCE_COLS = dict(CD=2, CL=3)        # drag_results.dat : tstep time dragX dragY ...
PROBES = ["A020", "A035", "A050", "A065", "A080", "A092", "B050", "B070", "B090",
          "W015p", "W015m", "W050p", "W050m", "W100p", "W100m", "W100P", "W100M", "REF", "SPAN1", "SPAN2"]
COMB_X = {"A020": 0.20, "A035": 0.35, "A050": 0.50, "A065": 0.65, "A080": 0.80, "A092": 0.92}
BANDS = {"slow": (0.05, 0.8), "shed": (0.8, 2.6), "double": (2.6, 4.6), "f7_9": (6.5, 9.5),
         "f10_12": (9.5, 12.0), "above12": (12.0, 40.0)}
FMAX_STORE = 40.0        # spectres stockes jusqu'a St = 40
ABERRANT = 5.0
OUT = "spectra_44.json"


# =============================================================================
# lecture
# =============================================================================
def find_case_dir(root, key):
    if key in PATHS:
        p = os.path.join(root, PATHS[key])
        return p if os.path.isdir(p) else None
    direct = os.path.join(root, key)
    if os.path.isfile(os.path.join(direct, "drag_results.dat")):
        return direct
    hits = []
    for dirpath, dirnames, _ in os.walk(root):
        if dirpath[len(root):].count(os.sep) > 4:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if os.path.basename(dirpath) == key and os.path.isfile(os.path.join(dirpath, "drag_results.dat")):
            hits.append(dirpath)
    hits.sort(key=lambda p: (p.count(os.sep), p))
    return hits[0] if hits else None


def read_cols(path, cols, sep, t_col, t0, t1, skip_header=False):
    """Colonnes `cols` (indices) des lignes numeriques dont le temps est dans [t0 - 1, t1 + 1].
    Lecture par blocs (memoire bornee). Renvoie un tableau (n, len(cols)) dans l'ordre de cols."""
    lo, hi = t0 - 1.0, (np.inf if t1 is None else t1 + 1.0)
    out = []
    if pd is not None:
        chunk = max(20000, 2000000 // len(cols))             # memoire bornee
        for fast in (True, False):                           # lecture directe en flottants, sinon texte
            out = []
            try:
                it = pd.read_csv(path, sep=sep, header=None, usecols=cols, dtype=float if fast else str,
                                 comment="#", skiprows=1 if skip_header else 0, chunksize=chunk, engine="c",
                                 on_bad_lines="skip")
                for ch in it:
                    if fast:
                        A = ch[cols].to_numpy(dtype=float)
                    else:
                        A = ch[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                    A = A[np.all(np.isfinite(A), axis=1)]
                    tt = A[:, cols.index(t_col)]
                    A = A[(tt >= lo) & (tt <= hi)]
                    if A.size:
                        out.append(A)
                break
            except (ValueError, TypeError, pd.errors.ParserError):
                if not fast:
                    raise
    else:
        with open(path) as fh:
            if skip_header:
                fh.readline()
            for line in fh:
                parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
                if len(parts) <= max(cols):
                    continue
                try:
                    vals = [float(parts[c]) for c in cols]
                except ValueError:
                    continue
                if lo <= vals[cols.index(t_col)] <= hi:
                    out.append(np.array([vals]))
    if not out:
        return np.empty((0, len(cols)))
    return np.vstack(out)


def stitch(t):
    """Restarts : a chaque recul du temps, on garde le segment le plus recent."""
    brk = np.where(np.diff(t) <= 0)[0] + 1
    segs = np.split(np.arange(len(t)), brk)
    kept_t, kept_i = np.empty(0), np.empty(0, dtype=int)
    for k, idx in enumerate(segs):
        if idx.size == 0:
            continue
        if k > 0:
            n = np.searchsorted(kept_t, t[idx[0]], side="left")
            kept_t, kept_i = kept_t[:n], kept_i[:n]
        kept_t = np.concatenate([kept_t, t[idx]])
        kept_i = np.concatenate([kept_i, idx])
    return kept_i


def load_forces(d, t0, t1):
    path = os.path.join(d, "drag_results.dat")
    cols = [1, FORCE_COLS["CD"], FORCE_COLS["CL"]]
    A = read_cols(path, cols, r"\s+", 1, t0, t1)
    if A.size == 0:
        raise RuntimeError("aucune ligne d'efforts dans la fenetre")
    A = A[stitch(A[:, 0])]
    t, cd, cl = A[:, 0], A[:, 1], A[:, 2]
    med = float(np.median(np.abs(cl)))
    fac = 2.0 if 0.30 < med < 0.72 else 1.0          # efforts bruts -> coefficients
    cd, cl = fac * cd, fac * cl
    ok = (np.abs(cl) < ABERRANT) & (np.abs(cd) < ABERRANT)
    return t[ok], {"CL": cl[ok], "CD": cd[ok]}, fac, int((~ok).sum())


def load_probes(d, t0, t1):
    path = os.path.join(d, "probes_uvp.csv")
    if not os.path.isfile(path):
        return None, None
    with open(path) as fh:
        header = [h.strip() for h in fh.readline().strip().lstrip("#").split(",")]
    hl = [h.lower() for h in header]
    tcol = hl.index("time") if "time" in hl else 0
    want = {}
    for P in PROBES:
        for comp in ("ux", "uy"):
            name = f"{comp}_{P}"
            if name in header:                              # sensible a la casse (W100p / W100P)
                want[name] = header.index(name)
    cols = [tcol] + sorted(set(want.values()))
    A = read_cols(path, cols, ",", tcol, t0, t1, skip_header=True)
    if A.size == 0:
        return None, None
    A = A[stitch(A[:, 0])]
    t = A[:, 0]
    return t, {k: A[:, cols.index(v)] for k, v in want.items()}


def read_positions(d):
    for name in ("probes_positions.csv", "probes_positions.txt"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            with open(p) as fh:
                return fh.read()[:20000]
    return None


# =============================================================================
# traitement du signal
# =============================================================================
def window(t, y, t0, t1):
    m = (t >= t0) & (t <= (t.max() if t1 is None else t1))
    return t[m], y[m]


def uniform(t, y, tag=""):
    """Echantillonnage uniforme : tel quel si tous les pas sont a 1 % du median, sinon
    reechantillonnage lineaire au pas median. Renvoie (dt, y_uniforme, reech)."""
    d = np.diff(t)
    dm = float(np.median(d))
    if np.all(np.abs(d[:-1] / dm - 1.0) < 0.01) and abs(d[-1] / dm - 1.0) < 0.5:
        return dm, y, False
    tu = np.arange(t[0], t[-1], dm)
    return dm, np.interp(tu, t, y), True


def welch(y, dt, seg):
    """Welch (Hann, 50 %, moyenne retiree par segment). Renvoie f, amplitude (sinusoide
    equivalente), densite spectrale de puissance one-sided, nombre de segments."""
    L = int(round(seg / dt))
    L = min(L, y.size)
    step = max(1, L // 2)
    w = np.hanning(L)
    S1, S2 = w.sum(), (w ** 2).sum()
    acc, n = np.zeros(L // 2 + 1), 0
    for s in range(0, y.size - L + 1, step):
        x = y[s:s + L] - y[s:s + L].mean()
        acc += np.abs(np.fft.rfft(w * x)) ** 2
        n += 1
    P = acc / max(n, 1)
    f = np.fft.rfftfreq(L, dt)
    amp = 2.0 * np.sqrt(P) / S1
    psd = 2.0 * P * dt / S2            # variance = sum(psd) * df
    psd[0] /= 2.0
    return f, amp, psd, n


def single_fft(y, dt, pad=8):
    """FFT de Hann unique avec bourrage x pad (test N3/N4)."""
    w = np.hanning(y.size)
    x = (y - y.mean()) * w
    nfft = pad * y.size
    X = np.fft.rfft(x, nfft)
    f = np.fft.rfftfreq(nfft, dt)
    amp = 2.0 * np.abs(X) / w.sum()
    # bandes : sans bourrage
    X1 = np.fft.rfft(x)
    f1 = np.fft.rfftfreq(y.size, dt)
    psd1 = 2.0 * np.abs(X1) ** 2 * dt / (w ** 2).sum()
    psd1[0] /= 2.0
    return f, amp, f1, psd1


def bands(f, psd):
    df = f[1] - f[0]
    return {b: float(np.sqrt(np.sum(psd[(f >= lo) & (f < hi)]) * df)) for b, (lo, hi) in BANDS.items()}


def peaks(f, amp, fmin, fmax, nmax=8, lobe=None):
    """Maxima locaux (interpolation parabolique sur log A), >= 2 % du plus grand."""
    m = (f >= fmin) & (f <= fmax)
    idx = np.where(m)[0]
    if idx.size < 3:
        return []
    a = amp
    cand = [i for i in idx[1:-1] if a[i] >= a[i - 1] and a[i] >= a[i + 1] and a[i] > 0]
    if lobe:                                      # un pic par lobe
        cand.sort(key=lambda i: -a[i])
        kept = []
        for i in cand:
            if all(abs(f[i] - f[j]) > lobe for j in kept):
                kept.append(i)
        cand = kept
    cand.sort(key=lambda i: -a[i])
    out = []
    for i in cand[:nmax]:
        if a[i] < 0.02 * a[cand[0]]:
            break
        y0, y1, y2 = np.log(a[i - 1] + 1e-30), np.log(a[i]), np.log(a[i + 1] + 1e-30)
        den = y0 - 2 * y1 + y2
        p = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        out.append((float(f[i] + p * (f[1] - f[0])), float(np.exp(y1 - 0.25 * (y0 - y2) * p))))
    return out


def comb_tangent(xb, alpha):
    """Tangente a l'extrados (vers le BF) en x/c = xb, repere global (profil tourne autour du BF)."""
    yt = lambda x: 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    h = 1e-4
    dy = (yt(xb + h) - yt(xb - h)) / (2 * h)
    n = np.hypot(1.0, dy)
    a = np.radians(alpha)
    tx, ty = 1.0 / n, dy / n
    return tx * np.cos(a) + ty * np.sin(a), -tx * np.sin(a) + ty * np.cos(a)


def block(t, y, dt_out):
    """Moyennes par blocs de dt_out (sortie lissee pour les figures temporelles)."""
    k = np.floor((t - t[0]) / dt_out).astype(int)
    n = np.bincount(k)
    ok = n > 0
    tc = t[0] + (np.arange(n.size) + 0.5) * dt_out
    return tc[ok], np.bincount(k, weights=y)[ok] / n[ok]


def r4(a, nd=4):
    a = np.asarray(a, dtype=float)
    return [None if not np.isfinite(v) else float(f"{v:.{nd}g}") for v in a]


# =============================================================================
# cycle lent (spec_a19)
# =============================================================================
def slow_cycle(tu, dt, sig, f_lo=0.05, f_hi=0.16, nbin=24):
    """Phase du cycle lent par la transformee de Hilbert de C_L filtre (passe-bande FFT),
    puis moyenne en phase de chaque signal ; phases relatives a f0 par interspectre."""
    cl = sig["CL"] - sig["CL"].mean()
    n = cl.size
    F = np.fft.rfft(cl)
    f = np.fft.rfftfreq(n, dt)
    F[(f < f_lo) | (f > f_hi)] = 0.0
    xb = np.fft.irfft(F, n)
    # signal analytique
    Z = np.fft.fft(xb)
    h = np.zeros(n)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[1:(n + 1) // 2] = 2
    an = np.fft.ifft(Z * h)
    ph = np.mod(np.angle(an), 2 * np.pi)                 # 0 = maximum de C_L filtre
    edges = np.linspace(0, 2 * np.pi, nbin + 1)
    k = np.clip(np.digitize(ph, edges) - 1, 0, nbin - 1)
    out = {"phase": r4(0.5 * (edges[1:] + edges[:-1]))}
    for name, y in sig.items():
        cnt = np.bincount(k, minlength=nbin)
        out[name] = r4(np.bincount(k, weights=y, minlength=nbin) / np.maximum(cnt, 1))
    # nombre de cycles
    out["n_cycles"] = float(np.sum(np.diff(np.unwrap(np.angle(an))) ) / (2 * np.pi))
    return out


def cross_phase(tu, dt, a, b, seg, f0):
    """Phase de b par rapport a a (degres, b en retard si > 0) et coherence a f0 (Welch)."""
    L = int(round(seg / dt))
    w = np.hanning(L)
    Sab, Saa, Sbb, n = 0, 0, 0, 0
    for s in range(0, a.size - L + 1, L // 2):
        A = np.fft.rfft(w * (a[s:s + L] - a[s:s + L].mean()))
        B = np.fft.rfft(w * (b[s:s + L] - b[s:s + L].mean()))
        Sab, Saa, Sbb, n = Sab + A * np.conj(B), Saa + np.abs(A) ** 2, Sbb + np.abs(B) ** 2, n + 1
    f = np.fft.rfftfreq(L, dt)
    i = int(np.argmin(np.abs(f - f0)))
    coh = float(np.abs(Sab[i]) ** 2 / (Saa[i] * Sbb[i]))
    return float(np.degrees(np.angle(Sab[i]))), coh


# =============================================================================
# une tache
# =============================================================================
def run_task(name, root):
    key, alpha, t0, t1, segs, opt = TASKS[name]
    d = find_case_dir(root, key)
    if d is None:
        print(f"== {name:10s} : dossier {key} introuvable sous {root} -> ignore")
        return None
    tic = _time.time()
    entry = {"case": key, "alpha": alpha, "dir": d, "spectra": {}, "bands": {}, "peaks": {}}
    tF, F, fac, nbad = load_forces(d, t0, t1)
    tP, P = load_probes(d, t0, t1)
    t1_eff = float(tF.max()) if t1 is None else t1
    entry["window"] = [t0, t1_eff]
    sig, tsig = {}, {}
    for q in ("CL", "CD"):
        tt, yy = window(tF, F[q], t0, t1_eff)
        sig[q], tsig[q] = yy, tt
    if P is not None:
        for k, y in P.items():
            tt, yy = window(tP, y, t0, t1_eff)
            sig[k], tsig[k] = yy, tt
        for L_, xb in COMB_X.items():                     # vitesse tangente a l'extrados
            if f"ux_{L_}" in P and f"uy_{L_}" in P:
                tx, ty = comb_tangent(xb, alpha)
                tt, ux = window(tP, P[f"ux_{L_}"], t0, t1_eff)
                _, uy = window(tP, P[f"uy_{L_}"], t0, t1_eff)
                sig[f"ut_{L_}"], tsig[f"ut_{L_}"] = ux * tx + uy * ty, tt
    entry["means"] = {k: float(np.mean(v)) for k, v in sig.items()}
    entry["rms"] = {k: float(np.std(v)) for k, v in sig.items()}
    resamp = {}
    U = {}
    for k in sig:
        if sig[k].size < 50:
            continue
        dt, yu, rs = uniform(tsig[k], sig[k])
        U[k] = (dt, yu)
        resamp[k] = rs
    entry["dt"] = {"forces": U["CL"][0] if "CL" in U else None,
                   "probes": next((U[k][0] for k in U if k not in ("CL", "CD")), None)}
    entry["resampled"] = sorted({("forces" if k in ("CL", "CD") else "probes") for k, v in resamp.items() if v})
    # --- spectres
    if opt.get("single"):
        for k, (dt, yu) in U.items():
            f, amp, f1, psd1 = single_fft(yu, dt)
            m = f <= FMAX_STORE
            entry["spectra"][k] = {"df": float(f[1]), "amp": r4(amp[m])}
            entry["bands"][k] = bands(f1, psd1)
            entry["peaks"][k] = peaks(f, amp, 0.5, 40.0, 8, lobe=0.8)
        entry["series"] = {}
        for k in ("CL", "CD", "uy_W015m", "uy_W015p", "ux_A020", "ux_A050", "ux_A092", "uy_B070"):
            if k in U:
                dt, yu = U[k]
                stride = max(1, int(round(7.0e-4 / dt)))         # ~ pas des sondes
                entry["series"][k] = {"t0": float(tsig[k][0]), "dt": dt * stride, "y": r4(yu[::stride], 6)}
    else:
        for seg in segs:
            tag = f"seg{seg:g}"
            entry["spectra"][tag], entry["bands"][tag], entry["peaks"][tag] = {}, {}, {}
            for k, (dt, yu) in U.items():
                f, amp, psd, nseg = welch(yu, dt, seg)
                m = f <= FMAX_STORE
                entry["spectra"][tag][k] = {"df": float(f[1]), "amp": r4(amp[m])}
                entry["bands"][tag][k] = bands(f, psd)
                entry["peaks"][tag][k] = peaks(f, amp, 0.03, 40.0, 8, lobe=2.0 / seg)
            entry["spectra"][tag]["_nseg"] = nseg
    # --- options
    if "cl_fine" in opt:
        a, b = opt["cl_fine"]
        m = (tsig["CL"] >= a) & (tsig["CL"] <= b)
        tb, yb = block(tsig["CL"][m], sig["CL"][m], 1e-3)
        entry["cl_fine"] = {"t": r4(tb, 7), "CL": r4(yb, 6)}
    if "slices" in opt:
        tt, cl = window(tF, F["CL"], opt["slices"], None)
        _, cd = window(tF, F["CD"], opt["slices"], None)
        k = np.floor(tt).astype(int)
        entry["slices"] = [(int(s), float(cl[k == s].mean()), float(cd[k == s].mean())) for s in np.unique(k)]
    if opt.get("slow"):
        dt = U["CL"][0]
        ser = {}
        for k in ("CL", "CD", "ut_A050", "ut_A065", "ut_A080", "ut_A092", "uy_B070", "uy_REF", "uy_W100P"):
            if k in sig:
                tb, yb = block(tsig[k], sig[k], 0.02)
                ser[k] = {"t": r4(tb, 7), "y": r4(yb, 5)}
        entry["series"] = ser
        # cycle lent : tout sur une grille commune (pas des sondes)
        tg = np.arange(t0, t1_eff, 0.005)
        G = {k: np.interp(tg, tsig[k], sig[k]) for k in sig if k in ("CL", "CD") or k.startswith("ut_")
             or k in ("uy_REF", "uy_B070", "uy_B090")}
        entry["cycle"] = slow_cycle(tg, 0.005, G)
        f0 = 0.0967
        entry["phase_f0"] = {k: cross_phase(tg, 0.005, G["CL"], G[k], 32.0, f0) for k in G if k != "CL"}
        entry["reverse_fraction"] = {k: float(np.mean(sig[k] < 0)) for k in sig if k.startswith("ut_")}
    positions = read_positions(d)
    if positions:
        entry["positions_raw"] = positions
    print(f"== {name:10s} {key} ({alpha:.2f} deg) [{t0:.2f} ; {t1_eff:.2f}] efforts x{fac:g}, {nbad} aberrantes,"
          f" dt efforts {entry['dt']['forces']:.3g}, sondes {entry['dt']['probes'] if entry['dt']['probes'] else 0:.3g},"
          f" reechantillonne : {entry['resampled'] or 'non'} ({_time.time() - tic:.0f} s)")
    show = ["CL", "CD", "ux_A020", "ux_A050", "ut_A092", "uy_B070", "uy_W015p", "uy_W015m", "uy_REF"]
    for k in show:
        if k not in entry["rms"]:
            continue
        tag = next(iter(entry["bands"])) if not opt.get("single") else None
        bd = entry["bands"][tag][k] if tag else entry["bands"][k]
        pk = entry["peaks"][tag][k] if tag else entry["peaks"][k]
        print(f"   {k:9s} moy {entry['means'][k]:+.5f} rms {entry['rms'][k]:.3e} | lent {bd['slow']:.2e} "
              f"lacher {bd['shed']:.2e} double {bd['double']:.2e} 7-9 {bd['f7_9']:.2e} 10-12 {bd['f10_12']:.2e} "
              f">12 {bd['above12']:.2e} | pics " + "  ".join(f"{a:.3f} ({b:.1e})" for a, b in pk[:5]))
    if "slices" in entry:
        print("   C_L, C_D par tranches de 1 CTU : " + "  ".join(f"{s}:{a:.4f}/{b:.5f}" for s, a, b in entry["slices"]))
    if "cycle" in entry:
        print(f"   cycle lent : {entry['cycle']['n_cycles']:.1f} cycles ; fraction u_t < 0 : " +
              "  ".join(f"{k} {v:.2f}" for k, v in entry["reverse_fraction"].items()))
        print("   phase a f0 = 0.0967 par rapport a C_L (deg, coherence) : " +
              "  ".join(f"{k} {p:+.0f} ({c:.2f})" for k, (p, c) in entry["phase_f0"].items()))
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    root = os.path.abspath(os.path.expanduser(args.root))
    names = [n for n in TASKS if not args.only or n in args.only.split(",")]
    out = {"version": 1, "genere": _time.strftime("%Y-%m-%d %H:%M"), "root": root,
           "bands_def": BANDS, "taches": {}}
    for n in names:
        try:
            e = run_task(n, root)
        except Exception as ex:                                       # une tache ratee n'arrete pas les autres
            print(f"== {n:10s} ECHEC : {type(ex).__name__}: {ex}")
            e = None
        if e is not None:
            out["taches"][n] = e
    with open(OUT, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"\n{OUT} ecrit ({os.path.getsize(OUT) / 1e6:.1f} Mo) -> a rapatrier")


if __name__ == "__main__":
    main()

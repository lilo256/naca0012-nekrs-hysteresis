#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figures_44.py -- figures et chiffres de la section 4.4 (etude frequentielle).   Version du 04/10/2026.
A lancer SUR LE CLUSTER ; python3 + numpy + matplotlib seulement (teste avec python 3.9, numpy 1.26.4,
matplotlib 3.3.4). Remplace extract_spectra_44.py et fig_frequency_44.py.

  cd ~/cdf_runs && python3 -u ~/bin/figures_44.py | tee figures_44.log   # tout : 2 a 5 minutes
  python3 -u ~/bin/figures_44.py --only tests,cases,transient            # quelques figures seulement
  python3 -u ~/bin/figures_44.py --json figures_44.json                  # retracer sans relire les donnees brutes

Ne relance aucun calcul. Lit dans <cas>/ : drag_results.dat, probes_uvp.csv, probes_positions.csv.
Ecrit : figs_44/fig_freq_*.pdf et .png, figures_44.json (tout ce qui est trace), et imprime les chiffres.

Methode (celle du dossier 4.4) : Welch, fenetre de Hann, recouvrement 50 %, moyenne retiree par segment ;
amplitude = celle de la sinusoide equivalente (2 sqrt(P) / somme(w)) ; bande = rms du signal dans la bande.
Les pics sont lus sur le spectre bourre de zeros ; les bandes sur les raies non bourrees.
Un enregistrement est pris tel quel si tous ses pas sont a 1 % du pas median, sinon reechantillonne
lineairement au pas median (c'est indique dans la sortie).

Une fonction par figure :
  fig_overview    fig_freq_overview    C_L : etat attache (15 deg) et etat decroche (19 deg)        intro
  fig_probes      fig_freq_probes      position des sondes                                         4.4.1
  fig_bands       fig_freq_bands       part de variance par bande et par sonde, 15 et 19 deg       4.4.2 / 4.4.4
  fig_tests       fig_freq_tests       spectres : reference, filtre x4, N = 4 (C_L et u_x a A050)  4.4.3
  fig_cases       fig_freq_cases       les cas de test : rapports a la reference, moyennes         4.4.3
  fig_transient   fig_freq_transient   niveaux par bande de part et d'autre du changement (t=100)  4.4.3
  fig_stalled     fig_freq_stalled     etat decroche a 19 deg : series, spectres, cycle moyen      4.4.4
  fig_scaling     fig_freq_scaling     St_h = St sin(alpha), comparaison a la litterature          4.4.4
"""
import argparse
import json
import os
import re
import sys
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

# =============================================================================
# CONFIGURATION
# =============================================================================
# nom de tache : (dossier, incidence, t0, t1 (None = fin), segments de Welch [CTU], options)
TASKS = {
    "prod_a15":   ("prod_a15",       15.00,  50.0, 100.0,   [10.0],       {}),
    "ref_short":  ("prod_a15",       15.00,  96.0, 100.0,   [2.0],        {}),                 # meme estimateur que les tests
    "N3":         ("film_a15",       15.00, 100.30, 101.499, [],          {"single": True}),
    "N4":         ("res_a15_N4",     15.00, 100.30, 101.499, [],          {"single": True}),
    "N4_late":    ("res_a15_N4",     15.00, 104.0, 114.0,   [5.0],        {"slices": 100.0, "mean_win": (109.0, 114.0)}),
    "t15_ptol":   ("t15_ptol",       15.00, 102.0, 106.0,   [2.0],        {"mean_win": (101.0, 106.0)}),
    "t15_bdf3":   ("t15_bdf3",       15.00, 102.0, 106.0,   [2.0],        {"mean_win": (101.0, 106.0)}),
    "t15_hpf2m":  ("t15_hpf2m",      15.00, 102.0, 106.0,   [2.0],        {"mean_win": (101.0, 106.0)}),
    "t15_hpf40":  ("t15_hpf40",      15.00, 102.0, 106.0,   [2.0],        {"mean_win": (101.0, 106.0)}),
    "t15_gjp":    ("t15_gjp",        15.00, 102.0, 106.0,   [2.0],        {"mean_win": (101.0, 106.0)}),
    "spec_a19":   ("spec_a19.00",    19.00, 236.0, 300.0,   [32.0, 20.0], {"slow": True}),
    "spec_a19_a": ("spec_a19.00",    19.00, 200.0, 235.0,   [10.0],       {}),
    "a10":        ("dom_a20_b30_v2", 10.00,  42.5, None,    [10.0],       {}),
    "up17":       ("up_a17.00",      17.00, 240.0, 340.0,   [20.0],       {}),
    "cold17":     ("cold_a17.00",    17.00, 110.0, 160.0,   [20.0],       {}),
    "jump13":     ("jump_a13.50",    13.50, 500.0, None,    [20.0],       {}),
}
# tests du filtre sur les etats decroches (resultats attendus) : (test, parent suppose, incidence).
# Fenetres automatiques : les W derniers CTU du test et les W CTU du parent avant le depart du test.
STALL_TESTS = [("stall13_hpf40", "jump_a13.00", 13.0), ("stall10_hpf40", "jump_a10.00", 10.0)]
PATHS = {}                                   # chemins imposes si besoin : {"spec_a19.00": "spec/spec_a19.00"}
OUTDIR = "figs_44"
CACHE = "figures_44.json"
PROBES = ["A020", "A035", "A050", "A065", "A080", "A092", "B050", "B070", "B090",
          "W015p", "W015m", "W050p", "W050m", "W100p", "W100m", "W100P", "W100M", "REF", "SPAN1", "SPAN2"]
COMB_X = {"A020": 0.20, "A035": 0.35, "A050": 0.50, "A065": 0.65, "A080": 0.80, "A092": 0.92}
BANDS = {"slow": (0.05, 0.8), "shed": (0.8, 2.6), "double": (2.6, 4.6), "f7_9": (6.5, 9.5),
         "f10_12": (9.5, 12.0), "above12": (12.0, 40.0), "fast": (9.5, 20.0)}
FMAX_STORE = 40.0                            # spectres gardes jusqu'a St = 40
F0 = 0.0967                                  # fondamentale du cycle lent a 19 deg (dossier 4.4, §5.3)
T_CHANGE = 100.0062                          # instant du changement de reglage (champ initial des tests)

BLUE, DARK, ORANGE, GREY, LIGHT, GREEN, PURPLE = ("#2878d6", "#0d3a73", "#eb6834", "#8a8a82", "#9cc2f0",
                                                  "#1f9e89", "#8e6bbf")
# cas de test a 15 deg : cle -> (tache, libelle court, libelle long, couleur)
CASES = [
    ("t15_ptol",  "pressure\ntolerance", r"pressure tolerance $10^{-8}$", GREY),
    ("t15_bdf3",  "third-order\ntime stepping", "third-order time stepping", PURPLE),
    ("t15_hpf2m", "filter on\ntwo modes", "filter on two modes", LIGHT),
    ("t15_hpf40", "filter\n" + r"$\times 4$", r"filter $\times 4$", GREEN),
    ("t15_gjp",   "gradient-jump\npenalty", "gradient-jump penalty", "#c9a227"),
    ("N4_late",   r"$N = 4$", r"$N = 4$", ORANGE),
]

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "legend.fontsize": 7.8, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.edgecolor": "0.3",
    "savefig.bbox": "tight", "savefig.dpi": 300, "mathtext.fontset": "dejavusans",
})

# ------------------------------------------------------------------ tables du dossier 4.4 (secours)
ORDER = ["CL", "CD", "A020", "A035", "A050", "A065", "A080", "A092", "B050", "B070", "B090",
         "W015p", "W015m", "W050p", "W050m", "W100p", "W100m", "W100P", "W100M", "REF", "SPAN1", "SPAN2"]
# tableau M2 de la mise a jour du 04/10 (verifie contre les annexes A1, A10, A11, A12).
# te : C_L, bande 9,5-12 (N = 4 : bande > 12, sortie extract_spectra_44.log) ; ss : A050, bande 6,5-9,5 ;
# sh : C_L, bande 0,8-2,6 ; cl, cd : moyennes.
M2 = {
    "ref":       dict(f=10.66, te=3.52e-3, ss=3.71e-3, sh=9.43e-4, cl=1.3859, cd=0.03578),
    "t15_ptol":  dict(f=10.65, te=3.45e-3, ss=3.72e-3, sh=9.17e-4, cl=1.3854, cd=0.03575),
    "t15_bdf3":  dict(f=10.66, te=3.35e-3, ss=3.08e-3, sh=7.53e-4, cl=1.3884, cd=0.03541),
    "t15_hpf2m": dict(f=10.63, te=3.40e-3, ss=2.47e-3, sh=7.66e-4, cl=1.3864, cd=0.03585),
    "t15_hpf40": dict(f=10.26, te=4.03e-3, ss=3.04e-4, sh=7.97e-5, cl=1.3867, cd=0.03542),
    "N4_late":   dict(f=14.45, te=1.45e-3, ss=1.08e-4, sh=8.61e-5, cl=1.3677, cd=0.03564),
}
# Martin Ferrer (2025), fig. 4.34 (numerisation +-0.001) ; Mittal & Saxena (2002), fig. 4 (lecture +-0.03)
MF_A = [17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0]
MF_UP = [0.580, 0.576, 0.581, 0.557, 0.566, 0.566, 0.586, 0.602]
MF_DN = [0.551, 0.563, 0.540, 0.547, 0.552, 0.551, 0.585, 0.593]
MS_DEC = [(18.0, 0.55), (18.5, 0.52), (19.0, 0.50), (20.0, 0.48)]

J = {}                                       # resultats (taches), rempli par compute() ou lu dans --json
X = {}                                       # resultats annexes (transitoire, tests decroches)


# =============================================================================
# LECTURE DES FICHIERS (numpy seulement)
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


def read_table(path, cols, delimiter=None):
    """Colonnes `cols` de toutes les lignes numeriques d'un fichier texte, tableau (n, len(cols)).
    Les lignes d'en-tete (elles commencent par 't' : 'tstep ...', 'time,...') sont ignorees, meme au
    milieu du fichier (redemarrages). Lecture rapide par numpy ; lecture ligne a ligne en secours."""
    try:
        A = np.loadtxt(path, usecols=cols, delimiter=delimiter, comments="t", ndmin=2)
        return A[np.all(np.isfinite(A), axis=1)]
    except Exception as ex:                                        # ligne tronquee, texte inattendu...
        print(f"      (lecture ligne a ligne de {os.path.basename(path)} : {type(ex).__name__})")
    out, nmax = [], max(cols)
    with open(path) as fh:
        for line in fh:
            parts = line.split(delimiter) if delimiter else line.split()
            if len(parts) <= nmax:
                continue
            try:
                out.append([float(parts[c]) for c in cols])
            except ValueError:
                continue
    A = np.array(out, dtype=float).reshape(-1, len(cols))
    return A[np.all(np.isfinite(A), axis=1)]


def stitch(t):
    """Redemarrages : a chaque recul du temps, on garde le segment le plus recent."""
    brk = np.where(np.diff(t) <= 0)[0] + 1
    if brk.size == 0:
        return np.arange(t.size)
    segs = np.split(np.arange(t.size), brk)
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


_RAW = {}                                    # un seul dossier garde en memoire a la fois


def _raw(d):
    """(tF, CD, CL, n_aberrantes), (tP, {nom: serie}), positions brutes du dossier d."""
    if d in _RAW:
        return _RAW[d]
    _RAW.clear()
    # efforts : tstep time dragX dragY ...  (dragX = C_D, dragY = C_L ; deja des coefficients)
    A = read_table(os.path.join(d, "drag_results.dat"), (1, 2, 3))
    if A.shape[0] == 0:
        raise RuntimeError("drag_results.dat vide ou illisible")
    A = A[stitch(A[:, 0])]
    ok = (np.abs(A[:, 2]) < 2.0) & (A[:, 1] > 0.0) & (A[:, 1] < 2.0)        # filtre du dossier
    forces = (A[ok, 0], A[ok, 1], A[ok, 2], int((~ok).sum()))
    # sondes : time,ux_<S>,uy_<S>,p_<S>,...
    probes = (None, None)
    path = os.path.join(d, "probes_uvp.csv")
    if os.path.isfile(path):
        with open(path) as fh:
            header = [h.strip() for h in fh.readline().strip().lstrip("#").split(",")]
        want = {}
        for P in PROBES:
            for comp in ("ux", "uy"):
                name = f"{comp}_{P}"
                if name in header:                                  # sensible a la casse (W100p / W100P)
                    want[name] = header.index(name)
        tcol = header.index("time") if "time" in header else 0
        cols = [tcol] + sorted(set(want.values()))
        B = read_table(path, tuple(cols), delimiter=",")
        if B.shape[0] > 0:
            B = B[stitch(B[:, 0])]
            probes = (B[:, 0], {k: B[:, cols.index(v)] for k, v in want.items()})
    pos = None
    p = os.path.join(d, "probes_positions.csv")
    if os.path.isfile(p):
        with open(p) as fh:
            pos = fh.read()[:20000]
    _RAW[d] = (forces, probes, pos)
    return _RAW[d]


# =============================================================================
# TRAITEMENT DU SIGNAL
# =============================================================================
def cut(t, y, t0, t1):
    m = (t >= t0) & (t <= t1)
    return t[m], y[m]


def uniform(t, y):
    """Tel quel si tous les pas sont a 1 % du median (dernier pas ecarte), sinon reechantillonnage
    lineaire au pas median. Renvoie (dt, y, reechantillonne ?)."""
    d = np.diff(t)
    dm = float(np.median(d))
    if d.size > 2 and np.all(np.abs(d[:-1] / dm - 1.0) < 0.01):
        if abs(d[-1] / dm - 1.0) < 0.01:
            return dm, y, False
        return dm, y[:-1], False
    tu = np.arange(t[0], t[-1], dm)
    return dm, np.interp(tu, t, y), True


def welch(y, dt, seg, pad_max=4, nfft_max=2 ** 22):
    """Welch (Hann, 50 %, moyenne retiree par segment).
    Renvoie : f, psd (raies non bourrees ; variance = somme(psd) df), fp, amp (spectre bourre, amplitude
    de la sinusoide equivalente), nombre de segments."""
    L = int(min(round(seg / dt), y.size))
    step = max(1, L // 2)
    w = np.hanning(L)
    S1, S2 = w.sum(), (w ** 2).sum()
    pad = int(max(1, min(pad_max, nfft_max // L)))
    nfft = pad * L
    acc, n = np.zeros(nfft // 2 + 1), 0
    for s in range(0, y.size - L + 1, step):
        x = y[s:s + L]
        acc += np.abs(np.fft.rfft(w * (x - x.mean()), nfft)) ** 2
        n += 1
    P = acc / max(n, 1)
    fp = np.fft.rfftfreq(nfft, dt)
    amp = 2.0 * np.sqrt(P) / S1
    f = fp[::pad][:L // 2 + 1]
    psd = 2.0 * P[::pad][:L // 2 + 1] * dt / S2
    psd[0] /= 2.0
    return f, psd, fp, amp, n


def single_fft(y, dt, pad=8):
    """Une seule FFT de Hann, bourrage x pad (fenetre commune N = 3 / N = 4)."""
    w = np.hanning(y.size)
    x = (y - y.mean()) * w
    Xp = np.fft.rfft(x, pad * y.size)
    fp = np.fft.rfftfreq(pad * y.size, dt)
    amp = 2.0 * np.abs(Xp) / w.sum()
    f = fp[::pad][:y.size // 2 + 1]
    psd = 2.0 * np.abs(Xp[::pad][:y.size // 2 + 1]) ** 2 * dt / (w ** 2).sum()
    psd[0] /= 2.0
    return f, psd, fp, amp


def band_rms(f, psd):
    df = f[1] - f[0]
    return {b: float(np.sqrt(np.sum(psd[(f >= lo) & (f < hi)]) * df)) for b, (lo, hi) in BANDS.items()}


def peaks(f, amp, fmin, fmax, nmax=8, lobe=None):
    """Maxima locaux du spectre (un par lobe de Hann), >= 2 % du plus grand ; affinage parabolique."""
    idx = np.where((f >= fmin) & (f <= fmax))[0]
    if idx.size < 3:
        return []
    a = amp
    cand = [i for i in idx[1:-1] if a[i] >= a[i - 1] and a[i] >= a[i + 1] and a[i] > 0]
    cand.sort(key=lambda i: -a[i])
    if lobe:
        kept = []
        for i in cand:
            if all(abs(f[i] - f[j]) > lobe for j in kept):
                kept.append(i)
        cand = kept
    out = []
    for i in cand[:nmax]:
        if a[i] < 0.02 * a[cand[0]]:
            break
        y0, y1, y2 = np.log(a[i - 1] + 1e-300), np.log(a[i]), np.log(a[i + 1] + 1e-300)
        den = y0 - 2 * y1 + y2
        p = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        out.append((float(f[i] + p * (f[1] - f[0])), float(np.exp(y1 - 0.25 * (y0 - y2) * p))))
    return out


def naca_t(x):
    return 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)


def comb_tangent(xb, alpha):
    """Tangente a l'extrados (vers le bord de fuite) en x/c = xb, repere du maillage."""
    h = 1e-4
    dy = (naca_t(xb + h) - naca_t(xb - h)) / (2 * h)
    n = np.hypot(1.0, dy)
    a = np.radians(alpha)
    tx, ty = 1.0 / n, dy / n
    return tx * np.cos(a) + ty * np.sin(a), -tx * np.sin(a) + ty * np.cos(a)


def block(t, y, dt_out):
    k = np.floor((t - t[0]) / dt_out).astype(int)
    n = np.bincount(k)
    ok = n > 0
    tc = t[0] + (np.arange(n.size) + 0.5) * dt_out
    return tc[ok], np.bincount(k, weights=y)[ok] / n[ok]


def r4(a, nd=4):
    a = np.asarray(a, dtype=float)
    return [None if not np.isfinite(v) else float(f"{v:.{nd}g}") for v in a]


def lift_maxima(t, y, width=1.0, half=3.5):
    """Instants des maxima successifs d'un signal lisse par une moyenne glissante de `width` CTU.
    Un maximum est retenu s'il domine son voisinage de +/- `half` CTU ; les bords sont ecartes."""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    dt = float(np.median(np.diff(t)))
    w = max(1, int(round(width / dt))) | 1
    ys = np.convolve(np.pad(y, w // 2, mode="edge"), np.ones(w) / w, mode="valid")
    h = max(1, int(round(half / dt)))
    out = []
    for i in range(h, ys.size - int(round(1.0 / dt))):
        if ys[i] == ys[i - h:i + h + 1].max():
            out.append(float(t[i]))
    return out


def slow_cycle(dt, sig, f_lo=0.05, f_hi=0.16, nbin=24):
    """Phase du cycle lent : transformee de Hilbert de C_L filtre en bande (0 = maximum de C_L),
    puis moyenne de chaque signal par classe de phase."""
    cl = sig["CL"] - sig["CL"].mean()
    n = cl.size
    F = np.fft.rfft(cl)
    f = np.fft.rfftfreq(n, dt)
    F[(f < f_lo) | (f > f_hi)] = 0.0
    xb = np.fft.irfft(F, n)
    Z = np.fft.fft(xb)
    h = np.zeros(n)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[1:(n + 1) // 2] = 2
    an = np.fft.ifft(Z * h)
    ph = np.mod(np.angle(an), 2 * np.pi)
    edges = np.linspace(0, 2 * np.pi, nbin + 1)
    k = np.clip(np.digitize(ph, edges) - 1, 0, nbin - 1)
    cnt = np.maximum(np.bincount(k, minlength=nbin), 1)
    out = {"phase": r4(0.5 * (edges[1:] + edges[:-1]))}
    for name, y in sig.items():
        out[name] = r4(np.bincount(k, weights=y, minlength=nbin) / cnt)
    out["n_cycles"] = float(np.sum(np.diff(np.unwrap(np.angle(an)))) / (2 * np.pi))
    # maxima successifs de C_L (lisse sur 1 CTU) : durees des cycles
    out["t_max"] = r4(lift_maxima(np.arange(n) * dt, sig["CL"]), 6)
    return out


def cross_phase(dt, a, b, seg, f0):
    """Dephasage de b par rapport a a (degres ; > 0 : b en retard) et coherence a f0."""
    L = int(round(seg / dt))
    w = np.hanning(L)
    Sab, Saa, Sbb = 0, 0, 0
    for s in range(0, a.size - L + 1, L // 2):
        A = np.fft.rfft(w * (a[s:s + L] - a[s:s + L].mean()))
        B = np.fft.rfft(w * (b[s:s + L] - b[s:s + L].mean()))
        Sab, Saa, Sbb = Sab + A * np.conj(B), Saa + np.abs(A) ** 2, Sbb + np.abs(B) ** 2
    f = np.fft.rfftfreq(L, dt)
    i = int(np.argmin(np.abs(f - f0)))
    return float(np.degrees(np.angle(Sab[i]))), float(np.abs(Sab[i]) ** 2 / (Saa[i] * Sbb[i]))


def sliding_band(t, y, width, step, lo, hi):
    """rms du signal dans la bande [lo, hi[ sur une fenetre glissante (Hann, tendance lineaire retiree)."""
    dt = t[1] - t[0]
    L = int(round(width / dt))
    w = np.hanning(L)
    S2 = (w ** 2).sum()
    f = np.fft.rfftfreq(L, dt)
    m = (f >= lo) & (f < hi)
    k = np.arange(L) - 0.5 * (L - 1)
    kk = float(np.sum(k * k))
    tc, out = [], []
    for s in range(0, y.size - L + 1, max(1, int(round(step / dt)))):
        x = y[s:s + L]
        x = x - x.mean()
        x = x - k * (np.sum(k * x) / kk)
        psd = 2.0 * np.abs(np.fft.rfft(w * x)) ** 2 * dt / S2
        tc.append(t[s] + 0.5 * width)
        out.append(np.sqrt(psd[m].sum() * f[1]))
    return np.array(tc), np.array(out)


# =============================================================================
# CALCUL : une tache
# =============================================================================
def signals(d, alpha, t0, t1):
    """Signaux (efforts, sondes, vitesse tangente du peigne) du dossier d sur [t0, t1]."""
    (tF, cd, cl, nbad), (tP, P), pos = _raw(d)
    t1 = float(tF.max()) if t1 is None else t1
    sig, tsig = {}, {}
    tsig["CL"], sig["CL"] = cut(tF, cl, t0, t1)
    tsig["CD"], sig["CD"] = cut(tF, cd, t0, t1)
    if P is not None:
        for k, y in P.items():
            tsig[k], sig[k] = cut(tP, y, t0, t1)
        for name, xb in COMB_X.items():
            if f"ux_{name}" in P and f"uy_{name}" in P:
                tx, ty = comb_tangent(xb, alpha)
                tsig[f"ut_{name}"] = tsig[f"ux_{name}"]
                sig[f"ut_{name}"] = sig[f"ux_{name}"] * tx + sig[f"uy_{name}"] * ty
    return sig, tsig, t1, nbad, pos


def run_task(name, root, spec=None):
    key, alpha, t0, t1, segs, opt = spec or TASKS[name]
    d = find_case_dir(root, key)
    if d is None:
        print(f"== {name:10s} : dossier {key} introuvable sous {root} -> ignore")
        return None
    tic = _time.time()
    sig, tsig, t1, nbad, pos = signals(d, alpha, t0, t1)
    if sig["CL"].size < 100:
        print(f"== {name:10s} : moins de 100 points d'efforts dans [{t0} ; {t1}] -> ignore")
        return None
    e = {"case": key, "alpha": alpha, "window": [t0, t1], "spectra": {}, "bands": {}, "peaks": {}}
    e["t_data"] = [float(tsig["CL"][0]), float(tsig["CL"][-1])]      # ce que le fichier couvre vraiment
    e["partial"] = bool(e["t_data"][1] < t1 - 0.05)                  # calcul en cours ou arrete avant t1
    e["means"] = {k: float(np.mean(v)) for k, v in sig.items() if v.size}
    e["rms"] = {k: float(np.std(v)) for k, v in sig.items() if v.size}
    U, resamp, steps = {}, set(), {}
    for k in sig:
        if sig[k].size < 50:
            continue
        dt, yu, rs = uniform(tsig[k], sig[k])
        U[k] = (dt, yu)
        grp = "forces" if k in ("CL", "CD") else "probes"
        if rs:
            resamp.add(grp)
        if grp not in steps:
            dd = np.diff(tsig[k])
            steps[grp] = [float(dd.min()), float(np.median(dd)), float(dd.max())]
    e["dt"] = {g: v[1] for g, v in steps.items()}
    e["dt_range"] = steps
    e["resampled"] = sorted(resamp)
    if opt.get("single"):
        for k, (dt, yu) in U.items():
            f, psd, fp, amp = single_fft(yu, dt)
            e["spectra"][k] = {"df": float(fp[1]), "amp": r4(amp[fp <= FMAX_STORE])}   # 1 FFT : spectre bourre
            e["bands"][k] = band_rms(f, psd)
            e["peaks"][k] = peaks(fp, amp, 0.5, FMAX_STORE, 8, lobe=0.8)
    else:
        for seg in segs:
            tag = f"seg{seg:g}"
            e["spectra"][tag], e["bands"][tag], e["peaks"][tag] = {}, {}, {}
            nseg = 0
            for k, (dt, yu) in U.items():
                f, psd, fp, amp, nseg = welch(yu, dt, seg)
                pad = int(round(f[1] / fp[1]))
                e["spectra"][tag][k] = {"df": float(f[1]), "amp": r4(amp[::pad][:f.size][f <= FMAX_STORE])}
                e["bands"][tag][k] = band_rms(f, psd)
                e["peaks"][tag][k] = peaks(fp, amp, max(0.03, 0.5 / seg), FMAX_STORE, 8, lobe=2.0 / seg)
            e["spectra"][tag]["_nseg"] = nseg
    if "mean_win" in opt:                                            # moyennes des efforts (comme `etat`)
        a, b = opt["mean_win"]
        s2, ts2, _, _, _ = signals(d, alpha, a, b)
        if s2["CL"].size:
            e["mean_win"] = [a, b, float(np.mean(s2["CL"])), float(np.mean(s2["CD"]))]
    if "slices" in opt:                                              # moyennes par tranches de 1 CTU
        s2, ts2, _, _, _ = signals(d, alpha, opt["slices"], None)
        k = np.floor(ts2["CL"]).astype(int)
        e["slices"] = [(int(s), float(s2["CL"][k == s].mean()), float(s2["CD"][k == s].mean()),
                        int((k == s).sum())) for s in np.unique(k)]
    if opt.get("slow"):
        e["series"] = {}
        for k in ("CL", "CD", "ut_A020", "ut_A050", "ut_A092", "uy_REF"):
            if k in sig:
                tb, yb = block(tsig[k], sig[k], 0.02)
                e["series"][k] = {"t": r4(tb, 7), "y": r4(yb, 5)}
        tg = np.arange(t0, t1, 0.005)                                # grille commune
        G = {k: np.interp(tg, tsig[k], sig[k]) for k in sig
             if k in ("CL", "CD", "uy_REF", "uy_B070", "uy_B090") or k.startswith("ut_")}
        e["cycle"] = slow_cycle(0.005, G)
        e["cycle"]["t_max"] = [t0 + v for v in e["cycle"]["t_max"]]
        e["phase_f0"] = {k: cross_phase(0.005, G["CL"], G[k], 32.0, F0) for k in G if k != "CL"}
        e["reverse_fraction"] = {k: float(np.mean(sig[k] < 0)) for k in sig if k.startswith("ut_")}
    if pos:
        e["positions_raw"] = pos
    # ----- impression
    rs = (", reechantillonne : " + "+".join(e["resampled"])) if e["resampled"] else ""
    dtf = e["dt_range"].get("forces", [0, 0, 0])
    print(f"== {name:10s} {key} ({alpha:g} deg) [{t0:g} ; {t1:g}]  pas efforts {dtf[1]:.3g} "
          f"(min {dtf[0]:.3g}, max {dtf[2]:.3g}), sondes {e['dt'].get('probes', 0):.3g}, "
          f"{nbad} lignes aberrantes ecartees{rs}  ({_time.time() - tic:.0f} s)")
    if e["partial"]:
        print(f"   !! les donnees s'arretent a t = {e['t_data'][1]:.2f} : fenetre incomplete, resultat provisoire")
    tag = None if opt.get("single") else next(iter(e["bands"]))
    for k in ("CL", "CD", "ux_A020", "ux_A050", "ux_A092", "uy_B050", "uy_B090", "uy_W015p", "uy_W015m",
              "uy_REF", "uy_SPAN1", "uy_SPAN2"):
        if k not in e["rms"] or k not in (e["bands"][tag] if tag else e["bands"]):
            continue
        b = e["bands"][tag][k] if tag else e["bands"][k]
        p = e["peaks"][tag][k] if tag else e["peaks"][k]
        print(f"   {k:9s} moy {e['means'][k]:+.5f} rms {e['rms'][k]:.3e} | lent {b['slow']:.2e} lacher {b['shed']:.2e} "
              f"double {b['double']:.2e} 7-9 {b['f7_9']:.2e} 10-12 {b['f10_12']:.2e} >12 {b['above12']:.2e} | "
              + "  ".join(f"{a:.2f} ({c:.1e})" for a, c in p[:5]))
    if "ux_SPAN1" in e["means"] and "ux_SPAN2" in e["means"]:
        m = e["means"]
        print(f"   envergure : SPAN1 (z = 0,5) u = ({m['ux_SPAN1']:+.4f}, {m['uy_SPAN1']:+.4f})   "
              f"SPAN2 (z = 0,05) u = ({m['ux_SPAN2']:+.4f}, {m['uy_SPAN2']:+.4f})   "
              f"rms uy {e['rms']['uy_SPAN1']:.2e} / {e['rms']['uy_SPAN2']:.2e}")
    if "mean_win" in e:
        a, b, cl, cd = e["mean_win"]
        print(f"   moyennes sur [{a:g} ; {b:g}] : C_L {cl:.4f}  C_D {cd:.5f}")
    if "slices" in e:
        print("   C_L / C_D par tranche de 1 CTU : " + "  ".join(f"{s}:{a:.4f}/{b:.5f}" for s, a, b, n in e["slices"]))
    if "cycle" in e:
        tm = np.array(e["cycle"]["t_max"])
        print(f"   cycle lent : {e['cycle']['n_cycles']:.2f} cycles ; maxima de C_L (lisse sur 1 CTU) a t = "
              + ", ".join(f"{v:.1f}" for v in tm) + " ; ecarts " + ", ".join(f"{v:.1f}" for v in np.diff(tm)))
        print("   fraction du temps avec u_t < 0 : " + "  ".join(f"{k[3:]} {v:.2f}" for k, v in e["reverse_fraction"].items()))
        print(f"   retard sur C_L a f0 = {F0} (deg, coherence) : "
              + "  ".join(f"{k} {p:+.0f} ({c:.2f})" for k, (p, c) in e["phase_f0"].items()))
    return e


def run_stall_test(test, parent, alpha, root):
    """Test du filtre sur un etat decroche : W derniers CTU du test contre W CTU du parent avant le depart."""
    d = find_case_dir(root, test)
    if d is None:
        print(f"== {test:13s} : dossier introuvable -> ignore (resultat attendu)")
        return {}
    (tF, cd, cl, nbad), _, _ = _raw(d)
    ts, te = float(tF.min()), float(tF.max())
    W = float(min(40.0, np.floor(te - ts - 5.0)))
    if W < 10.0:
        print(f"== {test:13s} : {te - ts:.1f} CTU seulement depuis t = {ts:.1f} -> trop court, ignore")
        return {}
    seg = 20.0 if W >= 40.0 else W / 2.0
    out = {}
    e = run_task(test, root, (test, alpha, te - W, te, [seg], {}))
    if e:
        out[test] = e
    if find_case_dir(root, parent):
        e = run_task(test + "_parent", root, (parent, alpha, ts - W, ts, [seg], {}))
        if e:
            out[test + "_parent"] = e
    else:
        print(f"   parent {parent} introuvable : corriger STALL_TESTS en tete du script")
    return out


def transient_data(root):
    """Niveaux par bande sur fenetre glissante, du calcul de reference (avant t = 100) a chaque test."""
    dref = find_case_dir(root, "prod_a15")
    if dref is None:
        return {}
    t_pre = 94.0
    sig, tsig, _, _, _ = signals(dref, 15.0, t_pre, 100.0)
    pre = {k: (tsig[k].copy(), sig[k].copy()) for k in ("CL", "ux_A050", "ux_A092") if k in sig}
    out = {}
    for task, _, _, _ in CASES:
        key = TASKS[task][0]
        d = find_case_dir(root, key)
        if d is None:
            continue
        sig, tsig, t1, _, _ = signals(d, 15.0, T_CHANGE + 0.03, None)   # 0,03 CTU ecartes apres la reprise
        e = {"t_end": t1}
        for name, k, width, lo, hi in (("ss", "ux_A050", 1.0, 6.5, 9.5), ("sh", "CL", 2.0, 0.8, 2.6),
                                       ("shA", "ux_A092", 2.0, 0.8, 2.6), ("te", "CL", 1.0, 9.5, 20.0)):
            if k not in sig or k not in pre:
                continue
            t = np.concatenate([pre[k][0], tsig[k]])
            y = np.concatenate([pre[k][1], sig[k]])
            tu = np.arange(t_pre, t[-1], 1.0e-3)
            tc, v = sliding_band(tu, np.interp(tu, t, y), width, 0.1, lo, hi)
            e[name] = {"t": r4(tc, 6), "y": r4(v, 4), "width": width}
        out[task] = e
    return out


def compute(root, names):
    out = {}
    for n in names:
        try:
            e = run_task(n, root)
        except Exception as ex:                                      # une tache ratee n'arrete pas les autres
            print(f"== {n:10s} ECHEC : {type(ex).__name__}: {ex}")
            e = None
        if e is not None:
            out[n] = e
    extra = {}
    try:
        extra["transient"] = transient_data(root)
    except Exception as ex:
        print(f"== transitoire ECHEC : {type(ex).__name__}: {ex}")
    extra["stall"] = {}
    for test, parent, alpha in STALL_TESTS:
        try:
            extra["stall"].update(run_stall_test(test, parent, alpha, root))
        except Exception as ex:
            print(f"== {test:13s} ECHEC : {type(ex).__name__}: {ex}")
    return out, extra


# =============================================================================
# OUTILS DE TRACE
# =============================================================================
def sig_key(name):
    """Nom de sonde -> cle (composante analysee : ux pour le peigne A, uy sinon)."""
    if name in ("CL", "CD"):
        return name
    return ("ux_" if name.startswith("A") else "uy_") + name


def spec(task, tag, key):
    s = J[task]["spectra"][tag][key] if tag else J[task]["spectra"][key]
    a = np.array([np.nan if v is None else v for v in s["amp"]], dtype=float)
    return np.arange(a.size) * s["df"], a


def first_tag(task, prefer=None):
    tags = [t for t in J[task]["spectra"] if t.startswith("seg")]
    return prefer if prefer in tags else tags[0]


def mark(ax, x, txt, y=0.97, color="0.35", ha="center", **kw):
    ax.axvline(x, color=color, lw=0.6, ls=":", zorder=0)
    ax.text(x, y, txt, transform=ax.get_xaxis_transform(), ha=ha, va="top", fontsize=7, color=color, **kw)


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"))
    plt.close(fig)


def panel(ax, letter, text=""):
    ax.set_title(f"({letter}) {text}".strip(), loc="left", fontsize=8.8)


# =============================================================================
# FIGURE : vue d'ensemble (introduction de 4.4)
# =============================================================================
def fig_overview():
    """C_L : spectre d'amplitude de l'etat attache (15 deg) et de l'etat decroche (19 deg)."""
    if "prod_a15" not in J or "spec_a19" not in J:
        return print("fig_overview : il manque prod_a15 ou spec_a19 -> non tracee")
    t15, t19 = first_tag("prod_a15", "seg10"), first_tag("spec_a19", "seg32")
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    f19, a19 = spec("spec_a19", t19, "CL")
    f15, a15 = spec("prod_a15", t15, "CL")
    m19, m15 = f19 >= 0.03, f15 >= 0.09
    ax.axvspan(0.05, 0.8, color=DARK, alpha=0.06, lw=0)
    ax.loglog(f19[m19], a19[m19], color=DARK, lw=1.1, label=r"$19^\circ$, stalled")
    ax.loglog(f15[m15], a15[m15], color=BLUE, lw=1.0, label=r"$15^\circ$, attached")
    ax.set_xlim(0.03, 40)
    ax.set_ylim(1e-6, 0.3)
    ax.set_xlabel(r"Strouhal number $St = f c / U_\infty$")
    ax.set_ylabel(r"amplitude of $C_L$")
    for x, t in ((F0, r"$f_0$"), (1.75, "1.75"), (10.66, "10.66")):
        mark(ax, x, t)
    ax.legend(loc="lower left", frameon=False)
    save(fig, "fig_freq_overview")
    print(f"fig_freq_overview : segments {t15[3:]} CTU (15 deg) et {t19[3:]} CTU (19 deg)")


# =============================================================================
# FIGURE : sondes (4.4.1)
# =============================================================================
def parse_positions(raw):
    pos = {}
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                pos[parts[0]] = (float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError:
                continue
    return pos


def naca_outline(alpha, n=200):
    b = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(b))
    yt = naca_t(x)
    Xb = np.r_[x[::-1], x[1:]] - 1.0                      # bord de fuite a l'origine
    Yb = np.r_[yt[::-1], -yt[1:]]
    a = np.radians(alpha)
    return Xb * np.cos(a) + Yb * np.sin(a), -Xb * np.sin(a) + Yb * np.cos(a)


def fig_probes():
    """Position des sondes autour du profil (a 19 deg)."""
    task = next((k for k in ("spec_a19", "prod_a15") if k in J and J[k].get("positions_raw")), None)
    if task is None:
        return print("fig_probes : positions absentes -> non tracee")
    pos = parse_positions(J[task]["positions_raw"])
    alpha = J[task]["alpha"]
    fig, ax = plt.subplots(figsize=(6.3, 2.35))
    Xo, Yo = naca_outline(alpha)
    ax.fill(Xo, Yo, color="0.86", ec="0.45", lw=0.6, zorder=1)
    col = {"A": BLUE, "B": DARK, "W": ORANGE, "R": GREY, "S": GREEN}
    off = {"A": (0, -9, "center"), "B": (0, 5, "center"), "W": (5, -2.5, "left"), "R": (0, 5, "center")}
    for name, (x, y, z) in pos.items():
        c = col.get(name[0], "k")
        if name == "SPAN2":
            continue
        ax.plot(x, y, "o", ms=3.2, color=c, mec="white", mew=0.4, zorder=3)
        if name == "SPAN1":
            ax.annotate("SPAN1, SPAN2", (x, y), xytext=(5, -2.5), textcoords="offset points", fontsize=6.2,
                        color=c, ha="left")
            continue
        dx, dy, ha = off.get(name[0], (4, 3, "left"))
        dy = {"W015p": 1.0, "W015m": -6.0}.get(name, dy)
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.2, color=c, ha=ha)
    ax.set_aspect("equal")
    ax.set_xlim(-1.78, 1.42)
    ax.set_ylim(-0.42, 0.50)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$y/c$")
    save(fig, "fig_freq_probes")
    print(f"fig_freq_probes : positions de {J[task]['case']} ({alpha:g} deg)")
    a = np.radians(alpha)
    for name, (x, y, z) in pos.items():                   # repere du profil : x/c depuis le bord d'attaque
        dx, dy = x + np.cos(a), y - np.sin(a)
        xb, yb = dx * np.cos(a) - dy * np.sin(a), dx * np.sin(a) + dy * np.cos(a)
        h = f", {yb - naca_t(xb):.3f}c au-dessus de l'extrados" if 0 < xb < 1 else ""
        print(f"   {name:6s} maillage ({x:+.3f}, {y:+.3f}, z = {z:g})   profil x/c = {xb:+.3f}{h}")


# =============================================================================
# FIGURE : carte des bandes (4.4.2 et 4.4.4)
# =============================================================================
def fig_bands():
    """Part de la variance de chaque signal dans chaque bande de Strouhal, a 15 et 19 deg."""
    if "prod_a15" not in J or "spec_a19" not in J:
        return print("fig_bands : il manque prod_a15 ou spec_a19 -> non tracee")
    cols = ["slow", "shed", "double", "f7_9", "f10_12", "above12"]
    labs = ["0.05–0.8", "0.8–2.6", "2.6–4.6", "6.5–9.5", "9.5–12", "> 12"]
    tabs = []
    for task, tag in (("prod_a15", first_tag("prod_a15", "seg10")), ("spec_a19", first_tag("spec_a19", "seg20"))):
        T = {}
        for n in ORDER:
            k = sig_key(n)
            if k in J[task]["rms"] and k in J[task]["bands"][tag]:
                b, r = J[task]["bands"][tag][k], J[task]["rms"][k]
                T[n] = [100.0 * (b[c] / r) ** 2 for c in cols]
        tabs.append(T)
    rows = [n for n in ORDER if n in tabs[0] and n in tabs[1]]
    lab = {"CL": r"$C_L$", "CD": r"$C_D$"}
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 5.0), gridspec_kw=dict(wspace=0.05))
    norm, cmap = Normalize(0, 100), plt.get_cmap("Blues")
    for ax, T, title in zip(axs, tabs, (r"(a) $15^\circ$, attached", r"(b) $19^\circ$, stalled")):
        M = np.clip(np.array([T[n] for n in rows]), 0, 100)
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(labs, fontsize=7.0)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([lab.get(n, n) for n in rows], fontsize=7.5)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=8.8, loc="left", pad=17)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] >= 0.5:
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.6,
                            color="white" if M[i, j] > 55 else "0.15")
        for y in (1.5, 7.5, 10.5, 18.5):                   # efforts | peigne | couche | sillage | amont, envergure
            ax.axhline(y, color="white", lw=1.6)
    axs[1].set_yticklabels([])
    cax = fig.add_axes([0.30, 0.045, 0.42, 0.017])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("share of the variance of the signal in the band (%)", fontsize=7.8)
    cb.ax.tick_params(labelsize=7)
    save(fig, "fig_freq_bands")
    print("fig_freq_bands")


# =============================================================================
# FIGURE : spectres reference / filtre x4 / N = 4 (4.4.3)
# =============================================================================
def fig_tests():
    """Spectres de C_L et de u_x a A050 : reference (N = 3), filtre quatre fois plus fort, N = 4."""
    sel = [("prod_a15", "seg10", r"reference, $N = 3$", BLUE, 1.0),
           ("t15_hpf40", "seg2", r"filter $\times 4$, $N = 3$", GREEN, 1.1),
           ("N4_late", "seg5", r"$N = 4$", ORANGE, 1.1)]
    sel = [s for s in sel if s[0] in J]
    if len(sel) < 2:
        return print("fig_tests : il faut au moins deux cas parmi prod_a15, t15_hpf40, N4_late -> non tracee")
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw=dict(wspace=0.30))
    for ax, key, ylab in ((axs[0], "CL", r"amplitude of $C_L$"), (axs[1], "ux_A050", r"amplitude of $u_x/U_\infty$ at A050")):
        for task, tag, lab, col, lw in sel:
            tag = first_tag(task, tag)
            if key not in J[task]["spectra"][tag]:
                continue
            f, a = spec(task, tag, key)
            m = f >= 0.4
            ax.loglog(f[m], a[m], color=col, lw=lw, label=lab)
        ax.set_xlim(0.4, 40)
        ax.set_ylim(1e-6, 2e-2)
        ax.set_xlabel(r"$St$")
        ax.set_ylabel(ylab)
    for x, t in ((1.75, "1.75"), (10.66, "10.66"), (14.45, "14.45")):
        mark(axs[0], x, t, ha="right" if x == 10.66 else "left" if x == 14.45 else "center")
    for x, t in ((1.75, "1.75"), (7.8, "7.8")):
        mark(axs[1], x, t)
    panel(axs[0], "a")
    panel(axs[1], "b")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.94), ncol=3,
               frameon=False, fontsize=7.6)
    save(fig, "fig_freq_tests")
    print("fig_freq_tests : " + ", ".join(f"{t} ({first_tag(t, g)[3:]} CTU)" for t, g, _, _, _ in sel))


# =============================================================================
# FIGURE : les cas de test (4.4.3)
# =============================================================================
def case_table():
    """Pour chaque cas : frequence du mode du bord de fuite, rms des trois fluctuations, moyennes.
    Donnees recalculees si disponibles, sinon tableau M2 du dossier."""
    T, src = {}, {}

    def row(task, tag):
        b = J[task]["bands"][tag]
        pk = [p for p in J[task]["peaks"][tag].get("CL", []) if 9.0 < p[0] < 20.0]
        te = b["CL"].get("fast")
        if te is None:                                               # ancien JSON : pas de bande 9,5-20
            te = b["CL"]["above12"] if (pk and pk[0][0] > 12.0) else b["CL"]["f10_12"]
        mw = J[task].get("mean_win")
        return dict(f=pk[0][0] if pk else np.nan, a=pk[0][1] if pk else np.nan, te=te, te12=b["CL"]["f10_12"],
                    w15=b.get("uy_W015m", {}).get("fast", np.nan), a92=b.get("ux_A092", {}).get("shed", np.nan),
                    part=bool(J[task].get("partial")),
                    ss=b.get("ux_A050", {}).get("f7_9", np.nan), sh=b["CL"]["shed"],
                    cl=mw[2] if mw else J[task]["means"]["CL"], cd=mw[3] if mw else J[task]["means"]["CD"])

    if "prod_a15" in J:
        T["ref"], src["ref"] = row("prod_a15", first_tag("prod_a15", "seg10")), "calcul"
    else:
        T["ref"], src["ref"] = dict(M2["ref"], a=np.nan, te12=M2["ref"]["te"], w15=np.nan, a92=np.nan,
                                    part=False), "dossier"
    for task, _, _, _ in CASES:
        if task in J:
            T[task], src[task] = row(task, first_tag(task)), "calcul"
            if task == "N4_late" and "mean_win" not in J[task]:      # ancien JSON : moyennes sur 109-114 du dossier
                T[task]["cl"], T[task]["cd"] = M2[task]["cl"], M2[task]["cd"]
        elif task in M2:
            T[task], src[task] = dict(M2[task], a=np.nan, te12=np.nan, w15=np.nan, a92=np.nan, part=False), "dossier"
    return T, src


def fig_cases():
    """Les cas de test a 15 deg : niveau des trois fluctuations rapporte a la reference, et moyennes."""
    T, src = case_table()
    cases = [c for c in CASES if c[0] in T]
    if not cases:
        return print("fig_cases : aucun cas -> non tracee")
    ref = T["ref"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw=dict(width_ratios=[1.55, 1.0], wspace=0.34))
    x = np.arange(len(cases))
    for key, lab, col, mk, dx in (("te", r"trailing-edge mode ($C_L$, $St > 9.5$)", DARK, "s", -0.16),
                                  ("ss", "suction-side mode (A050, $6.5 < St < 9.5$)", BLUE, "o", 0.0),
                                  ("sh", r"shedding ($C_L$, $0.8 < St < 2.6$)", ORANGE, "^", 0.16)):
        y = [T[c[0]][key] / ref[key] for c in cases]
        a.semilogy(x + dx, y, mk, color=col, ms=5.2, label=lab, zorder=3)
        for xi, yi in zip(x + dx, y):
            a.plot([xi, xi], [1.0, yi], color=col, lw=0.8, alpha=0.55, zorder=2)
    a.axhline(1.0, color="0.4", lw=0.8)
    a.set_ylim(0.015, 2.5)
    a.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2])
    a.set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"])
    a.set_xticks(x)
    a.set_xticklabels([c[1] for c in cases], fontsize=6.6)
    a.set_xlim(-0.6, len(cases) - 0.4)
    a.set_ylabel("r.m.s. in the band / reference")
    a.grid(axis="x", alpha=0)
    panel(a, "a")
    dcl = [100.0 * (T[c[0]]["cl"] / ref["cl"] - 1.0) for c in cases]
    dcd = [100.0 * (T[c[0]]["cd"] / ref["cd"] - 1.0) for c in cases]
    b.axhline(0.0, color="0.4", lw=0.8)
    b.plot(x - 0.1, dcl, "o", color=DARK, ms=5, label=r"$C_L$")
    b.plot(x + 0.1, dcd, "D", color=ORANGE, ms=4.5, mfc="white", label=r"$C_D$")
    b.set_xticks(x)
    b.set_xticklabels([c[1] for c in cases], fontsize=6.6, rotation=90)
    b.set_xlim(-0.6, len(cases) - 0.4)
    b.set_ylim(-2.0, 2.0)
    b.set_ylabel("change of the mean load (%)")
    b.grid(axis="x", alpha=0)
    b.legend(frameon=False, loc="upper left", ncol=2, fontsize=7.3, handletextpad=0.2, columnspacing=0.8)
    panel(b, "b")
    plt.setp(a.get_xticklabels(), rotation=90)
    fig.legend(*a.get_legend_handles_labels(), frameon=False, fontsize=7.3, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=1)
    save(fig, "fig_freq_cases")
    # ----- tableau imprime (a comparer au tableau M2 du dossier)
    print("fig_freq_cases ; tableau des cas (rapports a la reference entre parentheses) :")
    print("   cas          source   St_bf    ampl.   C_L 9,5-12  C_L St>9,5 (bord de fuite)  A050 6,5-9,5      "
          "C_L 0,8-2,6       A092 0,8-2,6  W015m St>9,5   C_L       C_D")
    for name in ["ref"] + [c[0] for c in cases]:
        r = T[name]
        print(f"   {name:12s} {src[name]:8s} {r['f']:6.2f}  {r['a']:.1e}  {r['te12']:.2e}  "
              f"{r['te']:.2e} (x{r['te'] / ref['te']:.2f})          "
              f"{r['ss']:.2e} (x{r['ss'] / ref['ss']:.2f})  {r['sh']:.2e} (x{r['sh'] / ref['sh']:.2f})  "
              f"{r['a92']:.2e}      {r['w15']:.2e}     {r['cl']:.4f}  {r['cd']:.5f}"
              + ("   (fenetre incomplete)" if r.get("part") else ""))
    if "ref_short" in J and "prod_a15" in J:
        s, r = J["ref_short"]["bands"]["seg2"], J["prod_a15"]["bands"][first_tag("prod_a15", "seg10")]
        print("   controle de l'estimateur (reference sur 96-100, segments de 2 CTU, rapportee a 50-100) : "
              f"C_L 9,5-12 x{s['CL']['f10_12'] / r['CL']['f10_12']:.2f}, A050 6,5-9,5 "
              f"x{s['ux_A050']['f7_9'] / r['ux_A050']['f7_9']:.2f}, C_L 0,8-2,6 x{s['CL']['shed'] / r['CL']['shed']:.2f}")


# =============================================================================
# FIGURE : transitoire apres le changement de reglage (4.4.3)
# =============================================================================
def fig_transient():
    """Niveaux par bande (fenetre glissante) avant et apres le changement de reglage a t = 100."""
    Tr = X.get("transient") or {}
    cases = [c for c in CASES if c[0] in Tr and c[0] != "t15_bdf3"]
    if not cases:
        return print("fig_transient : donnees absentes (il faut les donnees brutes des tests) -> non tracee")
    rows = [("ss", r"$u_x$ at A050," + "\n" + r"$6.5 < St < 9.5$"), ("sh", r"$C_L$," + "\n" + r"$0.8 < St < 2.6$"),
            ("te", r"$C_L$," + "\n" + r"$9.5 < St < 20$")]
    fig, axs = plt.subplots(3, 1, figsize=(6.3, 5.2), sharex=True, gridspec_kw=dict(hspace=0.16))
    tmax = 100.0
    for ax, (key, ylab), letter in zip(axs, rows, "abc"):
        w = 1.0
        for task, _, lab, col in cases:
            if key not in Tr[task]:
                continue
            t, y = np.array(Tr[task][key]["t"], dtype=float), np.array(Tr[task][key]["y"], dtype=float)
            w = Tr[task][key]["width"]
            pre, post = t <= T_CHANGE - 0.5 * w, t >= T_CHANGE + 0.03 + 0.5 * w
            ax.semilogy(t[post], y[post], color=col, lw=1.2, label=lab)
            ax.semilogy(t[pre], y[pre], color="0.25", lw=1.2)
            tmax = max(tmax, t[-1])
        ax.axvspan(T_CHANGE - 0.5 * w, T_CHANGE + 0.03 + 0.5 * w, color="0.5", alpha=0.10, lw=0)
        ax.axvline(T_CHANGE, color="0.4", lw=0.8, ls="--")
        ax.set_ylabel("r.m.s. of " + ylab, fontsize=8.2)
        panel(ax, letter)
    axs[-1].set_xlabel(r"$t\,U_\infty/c$")
    axs[-1].set_xlim(96.0, np.ceil(tmax))
    h, l = axs[0].get_legend_handles_labels()
    h = [Line2D([], [], color="0.25", lw=1.2)] + h
    l = ["production run (before the change)"] + l
    fig.legend(h, l, frameon=False, fontsize=7.4, loc="lower center", bbox_to_anchor=(0.5, 0.925), ncol=3)
    save(fig, "fig_freq_transient")
    print("fig_freq_transient : " + ", ".join(c[0] for c in cases)
          + " ; fenetres glissantes de 1 CTU (bandes rapides) et 2 CTU (lacher)")
    for task, _, _, _ in cases:                                      # temps de decroissance du lacher
        if "sh" not in Tr[task]:
            continue
        t, y = np.array(Tr[task]["sh"]["t"], dtype=float), np.array(Tr[task]["sh"]["y"], dtype=float)
        y0 = np.median(y[t < 99.0])
        w = Tr[task]["sh"]["width"]
        post = t >= T_CHANGE + 0.03 + 0.5 * w                         # fenetres entierement apres le changement
        k = np.where(post & (y < 0.5 * y0))[0]
        k2 = np.where(post & (y < 0.2 * y0))[0]
        first = y[post][0] if post.any() else float("nan")
        print(f"   {task:10s} lacher dans C_L (fenetre de {w:g} CTU, instant = centre) : niveau avant {y0:.2e} ; "
              f"premiere fenetre apres le changement x{first / y0:.2f} ; sous 50 % a t = "
              f"{t[k[0]] if k.size else float('nan'):.1f}, sous 20 % a t = {t[k2[0]] if k2.size else float('nan'):.1f} ; "
              f"dernier niveau {y[-1]:.2e} (x{y[-1] / y0:.2f})")
        if "ss" in Tr[task]:
            t2, y2 = np.array(Tr[task]["ss"]["t"], dtype=float), np.array(Tr[task]["ss"]["y"], dtype=float)
            w2 = Tr[task]["ss"]["width"]
            p2 = t2 >= T_CHANGE + 0.03 + 0.5 * w2
            y20 = np.median(y2[t2 < T_CHANGE - 0.5 * w2])
            if p2.any():
                print(f"   {'':10s} mode d'extrados a A050 (fenetre de {w2:g} CTU) : niveau avant {y20:.2e} ; premiere "
                      f"fenetre apres x{y2[p2][0] / y20:.2f} ; dernier niveau x{y2[-1] / y20:.2f}")


# =============================================================================
# FIGURE : etat decroche a 19 deg (4.4.4)
# =============================================================================
def fig_stalled():
    """Etat decroche a 19 deg : C_L(t), C_D(t), spectres rapportes a la fondamentale, cycle moyen."""
    if "spec_a19" not in J or "series" not in J["spec_a19"]:
        return print("fig_stalled : il manque spec_a19 -> non tracee")
    S = J["spec_a19"]
    tag = first_tag("spec_a19", "seg32")
    fig = plt.figure(figsize=(6.6, 6.3))
    gt = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.75], hspace=0.32, left=0.10, right=0.98, top=0.97, bottom=0.58)
    gb = fig.add_gridspec(1, 2, wspace=0.34, left=0.10, right=0.98, top=0.44, bottom=0.07)
    a = fig.add_subplot(gt[0])
    a2 = fig.add_subplot(gt[1], sharex=a)
    c = fig.add_subplot(gb[0])
    d = fig.add_subplot(gb[1])
    t, y = np.array(S["series"]["CL"]["t"]), np.array(S["series"]["CL"]["y"])
    a.plot(t, y, color=DARK, lw=1.0)
    a.axhline(S["means"]["CL"], color="0.5", lw=0.6, ls="--")
    a.set_ylabel(r"$C_L$")
    t, y = np.array(S["series"]["CD"]["t"]), np.array(S["series"]["CD"]["y"])
    a2.plot(t, y, color=DARK, lw=0.9)
    a2.axhline(S["means"]["CD"], color="0.5", lw=0.6, ls="--")
    a2.set_ylabel(r"$C_D$")
    a2.set_xlabel(r"$t\,U_\infty/c$", labelpad=1)
    a.set_xlim(S["window"][0], S["window"][1])
    plt.setp(a.get_xticklabels(), visible=False)
    sel = [("CL", r"$C_L$", DARK, "-"), ("ux_A050", "A050", BLUE, "-"), ("ux_A092", "A092", LIGHT, "-"),
           ("uy_W100P", "W100P", ORANGE, "-"), ("uy_REF", "REF", GREY, "--")]
    for k, lab, col, ls in sel:
        if k not in S["spectra"][tag]:
            continue
        f, A = spec("spec_a19", tag, k)
        m = (f > 0.03) & (f < 1.5)
        i0 = np.argmax(np.where((f > 0.06) & (f < 0.14), A, 0))
        c.loglog(f[m], A[m] / A[i0], color=col, lw=1.0, ls=ls, label=lab)
    for n in range(1, 10):
        c.axvline(n * F0, color="0.6", lw=0.5, ls=":", zorder=0)
    for n in (1, 3, 6, 9):
        c.text(n * F0, 1.02, f"{n}" + r"$f_0$" if n > 1 else r"$f_0$", transform=c.get_xaxis_transform(),
               ha="center", va="bottom", fontsize=6.5, color="0.4")
    c.set_xlim(0.04, 1.5)
    c.set_ylim(3e-3, 3)
    c.set_xlabel(r"$St$")
    c.set_ylabel(r"amplitude / amplitude at $f_0$")
    c.legend(frameon=False, fontsize=7, loc="lower left", handlelength=1.6)
    C = S["cycle"]
    ph = np.degrees(np.array(C["phase"]))
    for k, lab, col, ls in (("CL", r"$C_L$", DARK, "-"), ("CD", r"$C_D$", ORANGE, "-"),
                            ("ut_A020", r"$u_t$ at A020", LIGHT, "-"), ("ut_A092", r"$u_t$ at A092", BLUE, "-"),
                            ("uy_REF", r"$u_y$ at REF", GREY, "--")):
        if k not in C:
            continue
        v = np.array(C[k], dtype=float)
        vn = (v - v.mean()) / (np.max(np.abs(v - v.mean())) + 1e-30)
        d.plot(np.r_[ph, ph[0] + 360], np.r_[vn, vn[0]], color=col, ls=ls, lw=1.1, label=lab)
    d.set_xlim(0, 360)
    d.set_xticks([0, 90, 180, 270, 360])
    d.set_ylim(-1.15, 1.15)
    d.set_xlabel("phase of the slow cycle (deg)")
    d.set_ylabel("phase average, normalised")
    d.legend(frameon=False, fontsize=6.8, loc="lower left", bbox_to_anchor=(0.10, 1.0), ncol=3,
             handlelength=1.5, columnspacing=0.9, borderaxespad=0.2)
    panel(a, "a")
    panel(a2, "b")
    c.set_title("(c)", loc="left", fontsize=8.8, pad=12)
    d.set_title("(d)", loc="left", fontsize=8.8, pad=12)
    save(fig, "fig_freq_stalled")
    print(f"fig_freq_stalled : segments de {tag[3:]} CTU ; {C.get('n_cycles', float('nan')):.2f} cycles")


# =============================================================================
# FIGURE : echelle St_h = St sin(alpha) (4.4.4)
# =============================================================================
def peak_in(task, key, lo, hi, tag=None):
    try:
        tag = first_tag(task, tag)
        pk = [p for p in J[task]["peaks"][tag][key] if lo < p[0] < hi]
        return max(pk, key=lambda p: p[1]) if pk else None
    except (KeyError, IndexError):
        return None


def fig_scaling():
    """St_h = f c sin(alpha) / U : cette etude (etats decroches), Martin Ferrer (2025), Mittal & Saxena (2002)."""
    s = lambda a: np.sin(np.radians(a))
    slow, wake = [], []
    for task in ("jump13", "cold17", "spec_a19"):
        if task not in J:
            continue
        al = J[task]["alpha"]
        p = peak_in(task, "CL", 0.04, 0.22, "seg20")
        if task == "spec_a19":
            slow.append((al, F0))                                    # valeur affinee par les harmoniques (dossier)
        elif p:
            slow.append((al, p[0]))
        q = peak_in(task, "uy_W100P", 0.22, 0.40, "seg20")
        if q:
            wake.append((al, q[0]))
    if not slow:
        slow = [(19.0, F0), (17.0, 0.100)]                           # dossier 4.4, §5.7
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ax.axhspan(0.15, 0.20, color=ORANGE, alpha=0.12, lw=0)
    ax.text(12.65, 0.173, "bluff-body shedding of deep stall", fontsize=7, color="0.3", va="center")
    ax.axhline(0.03, color="0.45", lw=0.8, ls="--")
    ax.text(14.4, 0.0262, "low-frequency oscillation near stall", fontsize=7, color="0.3", va="top")
    ax.semilogy(MF_A, [v * s(a) for a, v in zip(MF_A, MF_UP)], "D", color=ORANGE, ms=4.2, mfc=ORANGE,
                label="Martin Ferrer (2025), upstroke")
    ax.semilogy(MF_A, [v * s(a) for a, v in zip(MF_A, MF_DN)], "D", color=ORANGE, ms=4.2, mfc="white",
                label="Martin Ferrer (2025), downstroke")
    ax.semilogy([a for a, _ in MS_DEC], [v * s(a) for a, v in MS_DEC], "s", color=GREY, ms=4.6, mfc="white",
                label=r"Mittal & Saxena (2002), decreasing $\alpha$")
    if wake:
        ax.semilogy([a for a, _ in wake], [f * s(a) for a, f in wake], "o", color=BLUE, ms=5, mfc="white",
                    label="this study, strongest component in the upper wake")
    ax.semilogy([a for a, _ in slow], [f * s(a) for a, f in slow], "o", color=DARK, ms=5.5,
                label="this study, slow oscillation of the loads")
    ax.set_xlim(12.5, 20.5)
    ax.set_ylim(0.012, 0.6)
    ax.set_yticks([0.02, 0.03, 0.05, 0.1, 0.2, 0.5])
    ax.set_yticklabels(["0.02", "0.03", "0.05", "0.1", "0.2", "0.5"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"angle of attack $\alpha$ (deg)")
    ax.set_ylabel(r"$St_h = f\,c\,\sin\alpha / U_\infty$")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=7.3, loc="upper center",
               bbox_to_anchor=(0.5, -0.03), ncol=2)
    save(fig, "fig_freq_scaling")
    print("fig_freq_scaling ; (alpha, St, St_h) : cycle lent "
          + ", ".join(f"({a:g}, {f:.3f}, {f * s(a):.4f})" for a, f in slow)
          + " ; composante du sillage " + ", ".join(f"({a:g}, {f:.3f}, {f * s(a):.4f})" for a, f in wake))


# =============================================================================
# RAPPORT : chiffres utiles au texte
# =============================================================================
def report():
    print("\n" + "=" * 100 + "\n CHIFFRES POUR LE TEXTE\n" + "=" * 100)
    for task, tag in (("spec_a19", "seg20"), ("spec_a19", "seg32"), ("spec_a19_a", "seg10"),
                      ("cold17", "seg20"), ("jump13", "seg20")):
        if task not in J or tag not in J[task]["spectra"]:
            continue
        f, a = spec(task, tag, "CL")
        df_bin = 1.0 / float(tag[3:])                                # raies non bourrees
        step = max(1, int(round(df_bin / (f[1] - f[0]))))
        fu, au = f[::step], a[::step]
        tot = J[task]["rms"]["CL"] ** 2
        v = lambda lo, hi: float(np.nansum(au[(fu >= lo) & (fu < hi)] ** 2) / 3.0)   # Hann : variance par raie = A^2/3
        hi = (fu >= 12.0) & (fu <= 22.0)
        cen = float(np.nansum(fu[hi] * au[hi] ** 2) / np.nansum(au[hi] ** 2))
        cum = np.nancumsum(au[hi] ** 2) / np.nansum(au[hi] ** 2)
        print(f" {task:10s} {tag:5s} C_L : part de variance 0,05-0,8 : {100 * v(0.05, 0.8) / tot:.1f} % ; "
              f"au-dessus de 0,8 : {100 * v(0.8, 40.01) / tot:.2f} % (dont au-dessus de 12 : "
              f"{100 * v(12.0, 40.01) / tot:.2f} %) ; bande rapide centree a St = {cen:.1f} "
              f"(80 % de l'energie entre {fu[hi][np.searchsorted(cum, 0.1)]:.1f} et {fu[hi][np.searchsorted(cum, 0.9)]:.1f})")
    for task in ("jump13", "cold17", "spec_a19"):
        if task not in J:
            continue
        tag = first_tag(task, "seg20")
        P = J[task]["peaks"][tag]
        line = []
        for k in ("CL", "CD", "ux_A092", "uy_REF", "uy_W050p", "uy_W100p", "uy_W100P", "uy_SPAN1", "uy_SPAN2"):
            if k in P:
                line.append(f"{k} " + " ".join(f"{a:.3f}({c:.1e})" for a, c in P[k][:3]))
        print(f" {task:10s} pics lents ({tag}) : " + " | ".join(line))
    for task in ("spec_a19", "cold17", "jump13"):
        ser = (J.get(task) or {}).get("series", {}).get("CL")
        if ser:
            tm = np.array(lift_maxima(ser["t"], ser["y"]))
            print(f" {task:10s} maxima de C_L (lisse sur 1 CTU) a t = " + ", ".join(f"{v:.1f}" for v in tm)
                  + " ; ecarts " + ", ".join(f"{v:.1f}" for v in np.diff(tm)))
    if "a10" in J:
        tag = first_tag("a10")
        p = [q for q in J["a10"]["peaks"][tag]["CL"] if q[0] > 6]
        if p:
            print(f" pic rapide de C_L a 10 deg (maillage de production) : St = {p[0][0]:.3f}, amplitude {p[0][1]:.1e} ; "
                  f"rapport a Martin Ferrer (8,879) : {p[0][0] / 8.879:.3f}")
    if "prod_a15" in J:
        tag = first_tag("prod_a15", "seg10")
        p = [q for q in J["prod_a15"]["peaks"][tag]["CL"] if q[0] > 6]
        if p:
            print(f" pic rapide de C_L a 15 deg : St = {p[0][0]:.3f}, amplitude {p[0][1]:.1e} ; "
                  f"rapport a Martin Ferrer (7,704) : {p[0][0] / 7.704:.3f}")
    for name, e in (X.get("stall") or {}).items():
        tag = first_tag_of(e)
        b, pk = e["bands"][tag]["CL"], e["peaks"][tag]["CL"]
        print(f" {name:22s} [{e['window'][0]:.0f} ; {e['window'][1]:.0f}] C_L {e['means']['CL']:.4f} +- {e['rms']['CL']:.4f} "
              f"C_D {e['means']['CD']:.5f} | lent {b['slow']:.2e} >12 {b['above12']:.2e} | "
              + "  ".join(f"{a:.3f} ({c:.1e})" for a, c in pk[:4]))


def first_tag_of(e):
    return [t for t in e["spectra"] if t.startswith("seg")][0]


# =============================================================================
FIGS = {"overview": fig_overview, "probes": fig_probes, "bands": fig_bands, "tests": fig_tests,
        "cases": fig_cases, "transient": fig_transient, "stalled": fig_stalled, "scaling": fig_scaling}


def main():
    global J, X, OUTDIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.getcwd(), help="dossier qui contient les cas (defaut : dossier courant)")
    ap.add_argument("--json", default="", help="retracer a partir d'un JSON existant, sans relire les donnees")
    ap.add_argument("--only", default="", help="figures a tracer : " + ",".join(FIGS))
    ap.add_argument("--tasks", default="", help="taches a calculer (defaut : toutes) : " + ",".join(TASKS))
    ap.add_argument("--out", default=OUTDIR, help="dossier des figures")
    args = ap.parse_args()
    OUTDIR = args.out
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}, matplotlib {matplotlib.__version__}")
    if args.json:
        D = json.load(open(args.json))
        J, X = D.get("taches", {}), D.get("extra", {})
        print(f"lecture de {args.json} : " + ", ".join(J))
    else:
        root = os.path.abspath(os.path.expanduser(args.root))
        names = [n for n in TASKS if not args.tasks or n in args.tasks.split(",")]
        J, X = compute(root, names)
        with open(CACHE, "w") as fh:
            json.dump({"version": 2, "genere": _time.strftime("%Y-%m-%d %H:%M"), "root": root,
                       "bandes": BANDS, "taches": J, "extra": X}, fh, separators=(",", ":"))
        print(f"\n{CACHE} ecrit ({os.path.getsize(CACHE) / 1e6:.1f} Mo)")
    print()
    for name, fn in FIGS.items():
        if args.only and name not in args.only.split(","):
            continue
        try:
            fn()
        except Exception as ex:                                      # une figure ratee n'arrete pas les autres
            print(f"{fn.__name__} : ECHEC {type(ex).__name__}: {ex}")
    try:
        report()
    except Exception as ex:
        print(f"report : ECHEC {type(ex).__name__}: {ex}")
    print(f"\nfigures dans {OUTDIR}/ (pdf et png)")
    if not args.json:
        print(f"a rapatrier : figures_44.log, {CACHE} et {OUTDIR}/*.pdf")


if __name__ == "__main__":
    main()

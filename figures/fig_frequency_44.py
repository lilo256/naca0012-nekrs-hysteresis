#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_frequency_44.py -- figures de la section 4.4 (etude frequentielle).

  python3 fig_frequency_44.py [spectra_44.json]

Produit (chaque figure seulement si ses donnees existent) :
  fig_freq_overview.pdf    spectres de C_L, attache (15 deg) et decroche (19 deg)      <- JSON
  fig_freq_probes.pdf      position des sondes                                          <- JSON (probes_positions)
  fig_freq_bands.pdf       part de la variance par bande de Strouhal et par sonde, 15 et 19 deg
                           (JSON, sinon tables verifiees du dossier 4.4)
  fig_freq_resolution.pdf  test N = 3 / N = 4                                            <- JSON
  fig_freq_stalled.pdf     etat decroche a 19 deg : series, spectres normalises, cycle moyen <- JSON
  fig_freq_scaling.pdf     St_h = St sin(alpha) : cette etude, Martin Ferrer, Mittal & Saxena, ordres
                           de grandeur de la litterature (JSON pour nos points, sinon dossier)
Imprime les chiffres utiles au texte.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

SRC = sys.argv[1] if len(sys.argv) > 1 else "spectra_44.json"
J = json.load(open(SRC))["taches"] if os.path.isfile(SRC) else {}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "legend.fontsize": 7.8, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.edgecolor": "0.3",
    "savefig.bbox": "tight", "savefig.dpi": 300, "mathtext.fontset": "dejavusans",
})
BLUE, DARK, ORANGE, GREY, LIGHT = "#2878d6", "#0d3a73", "#eb6834", "#8a8a82", "#9cc2f0"
F0 = 0.0967                       # fondamentale du mode lent (espacement des harmoniques, dossier 4.4 §5.3)

# ------------------------------------------------------------------ tables verifiees (dossier 4.4)
ORDER = ["CL", "CD", "A020", "A035", "A050", "A065", "A080", "A092", "B050", "B070", "B090",
         "W015p", "W015m", "W050p", "W050m", "W100p", "W100m", "W100P", "W100M", "REF", "SPAN1", "SPAN2"]
# 15 deg, prod_a15 50-100, segments 10 CTU : total, lacher, double, 7-9, 10-12   (dossier §3.3)
T15 = {"CL": (3.824e-3, 9.431e-4, 3.160e-4, 8.866e-5, 3.521e-3), "CD": (9.549e-4, 2.427e-4, 8.270e-5, 2.216e-5, 8.770e-4),
       "A020": (1.256e-2, 7.350e-3, 5.563e-3, 5.833e-3, 2.051e-3), "A035": (7.954e-3, 4.107e-3, 3.430e-3, 3.973e-3, 1.395e-3),
       "A050": (6.372e-3, 3.084e-3, 2.624e-3, 3.710e-3, 1.334e-3), "A065": (5.155e-3, 2.434e-3, 2.692e-3, 2.587e-3, 4.730e-4),
       "A080": (4.102e-3, 2.924e-3, 2.379e-3, 7.965e-4, 2.151e-4), "A092": (3.623e-3, 3.131e-3, 1.595e-3, 2.117e-4, 2.426e-4),
       "B050": (6.621e-3, 1.626e-3, 2.126e-3, 4.465e-3, 1.751e-3), "B070": (4.513e-3, 1.954e-3, 2.314e-3, 2.436e-3, 6.365e-4),
       "B090": (4.229e-3, 2.942e-3, 2.454e-3, 1.030e-3, 2.774e-4), "W015p": (1.110e-2, 3.829e-3, 9.718e-3, 9.721e-4, 2.338e-3),
       "W015m": (3.047e-2, 4.578e-3, 9.229e-3, 5.268e-3, 2.771e-2), "W050p": (1.252e-3, 8.886e-4, 8.561e-4, 3.058e-5, 9.528e-6),
       "W050m": (4.383e-3, 3.000e-3, 3.130e-3, 5.784e-5, 2.801e-5), "W100p": (1.893e-3, 1.536e-3, 1.071e-3, 3.065e-5, 6.786e-6),
       "W100m": (1.873e-2, 6.054e-3, 1.747e-2, 1.222e-3, 1.931e-4), "W100P": (2.835e-4, 2.504e-4, 3.303e-5, 2.707e-5, 3.605e-6),
       "W100M": (9.890e-4, 8.186e-4, 5.003e-4, 3.849e-5, 5.756e-5), "REF": (1.670e-5, 4.579e-6, 7.613e-7, 8.897e-8, 2.530e-6),
       "SPAN1": (9.066e-4, 8.201e-4, 3.341e-4, 2.940e-5, 5.763e-6), "SPAN2": (1.261e-3, 1.081e-3, 4.690e-4, 3.592e-5, 8.302e-6)}
# 19 deg, spec_a19 236-300, segments 20 CTU : total, lent, lacher, double, 7-9, 10-12   (dossier §5.1)
T19 = {"CL": (6.517e-2, 6.071e-2, 4.045e-4, 3.215e-4, 6.095e-4, 1.049e-3), "CD": (2.022e-2, 1.946e-2, 1.591e-4, 1.145e-4, 1.984e-4, 3.407e-4),
       "A020": (2.565e-2, 2.321e-2, 1.974e-3, 2.236e-3, 2.404e-3, 1.948e-3), "A035": (3.415e-2, 3.145e-2, 1.331e-3, 7.212e-4, 8.286e-4, 7.606e-4),
       "A050": (3.111e-2, 3.082e-2, 1.298e-3, 3.451e-4, 3.044e-4, 2.569e-4), "A065": (6.203e-2, 6.205e-2, 8.268e-4, 9.256e-5, 6.813e-5, 2.978e-5),
       "A080": (8.074e-2, 7.761e-2, 6.103e-4, 1.076e-4, 8.966e-5, 5.119e-5), "A092": (8.807e-2, 8.148e-2, 9.588e-4, 1.554e-4, 1.839e-4, 5.096e-5),
       "B050": (1.043e-2, 1.099e-2, 3.461e-4, 3.027e-4, 4.795e-4, 4.147e-4), "B070": (2.632e-2, 2.701e-2, 2.541e-4, 1.520e-4, 1.495e-4, 4.335e-5),
       "B090": (2.115e-2, 2.112e-2, 3.886e-4, 7.081e-5, 2.556e-5, 2.346e-5), "W015p": (1.279e-2, 9.180e-3, 3.119e-3, 4.795e-3, 2.873e-3, 8.454e-4),
       "W015m": (3.427e-2, 1.580e-2, 1.577e-3, 4.705e-3, 1.020e-2, 5.467e-3), "W050p": (2.303e-2, 2.076e-2, 2.705e-3, 2.977e-4, 5.514e-5, 1.314e-5),
       "W050m": (1.759e-2, 1.687e-2, 1.234e-3, 5.194e-4, 5.692e-5, 5.129e-5), "W100p": (2.705e-2, 2.596e-2, 3.233e-3, 4.229e-4, 4.932e-5, 1.190e-5),
       "W100m": (1.833e-2, 1.739e-2, 2.737e-3, 1.425e-3, 1.741e-3, 1.274e-3), "W100P": (3.130e-2, 3.023e-2, 3.101e-3, 1.852e-4, 9.542e-6, 3.744e-6),
       "W100M": (1.265e-2, 1.237e-2, 7.551e-4, 8.508e-5, 3.282e-5, 3.639e-5), "REF": (3.167e-3, 2.711e-3, 8.926e-6, 8.017e-7, 5.602e-7, 7.807e-7),
       "SPAN1": (2.709e-2, 2.599e-2, 2.525e-3, 2.896e-4, 3.399e-5, 7.558e-6), "SPAN2": (2.131e-1, 2.023e-1, 1.389e-2, 2.219e-3, 2.425e-4, 5.126e-5)}
# Martin Ferrer (2025), fig. 4.34 (numerisation +-0.001) ; Mittal & Saxena (2002), fig. 4 (lecture +-0.03)
MF_A = [17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0]
MF_UP = [0.580, 0.576, 0.581, 0.557, 0.566, 0.566, 0.586, 0.602]
MF_DN = [0.551, 0.563, 0.540, 0.547, 0.552, 0.551, 0.585, 0.593]
MS_DEC = [(18.0, 0.55), (18.5, 0.52), (19.0, 0.50), (20.0, 0.48)]
MS_INC = [(18.0, 1.65)]


def sig_key(name):
    """Nom de sonde -> cle du JSON (composante analysee : ux pour A, uy sinon)."""
    if name in ("CL", "CD"):
        return name
    return ("ux_" if name.startswith("A") else "uy_") + name


def spec(task, tag, key):
    s = J[task]["spectra"][tag][key] if tag else J[task]["spectra"][key]
    a = np.array([np.nan if v is None else v for v in s["amp"]], dtype=float)
    return np.arange(a.size) * s["df"], a


def mark(ax, x, txt, y=0.97, color="0.35", ha="center", **kw):
    ax.axvline(x, color=color, lw=0.6, ls=":", zorder=0)
    ax.text(x, y, txt, transform=ax.get_xaxis_transform(), ha=ha, va="top", fontsize=7, color=color, **kw)


# =============================================================================
# A. vue d'ensemble : spectres de C_L, 15 et 19 deg
# =============================================================================
def fig_overview():
    if "prod_a15" not in J or "spec_a19" not in J:
        return print("overview : JSON incomplet, figure non produite")
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    f15, a15 = spec("prod_a15", "seg25", "CL")
    f19, a19 = spec("spec_a19", "seg32", "CL")
    m15, m19 = f15 > 0, f19 > 0
    ax.loglog(f19[m19], a19[m19], color=DARK, lw=1.2, label=r"$19^\circ$, stalled ($t = 236$–$300$)")
    ax.loglog(f15[m15], a15[m15], color=BLUE, lw=1.0, label=r"$15^\circ$, attached ($t = 50$–$100$)")
    ax.axvspan(0.05, 0.8, color=DARK, alpha=0.06, lw=0)
    ax.set_xlim(0.025, 40)
    ax.set_ylim(1e-6, 0.3)
    ax.set_xlabel(r"Strouhal number $St = f c / U_\infty$")
    ax.set_ylabel(r"amplitude of $C_L$")
    for x, t in ((F0, r"$f_0$"), (1.75, "1.75"), (10.66, "10.66"), (14.8, "14.8")):
        mark(ax, x, t)
    ax.legend(loc="lower left", frameon=False)
    fig.savefig("fig_freq_overview.pdf")
    plt.close(fig)
    print("fig_freq_overview.pdf")


# =============================================================================
# B. sondes
# =============================================================================
def parse_positions(raw):
    pos = {}
    for line in raw.splitlines():
        parts = [p.strip() for p in line.replace(";", ",").replace("\t", ",").split(",") if p.strip()]
        if len(parts) < 3:
            parts = line.split()
        if len(parts) >= 3:
            try:
                pos[parts[0]] = (float(parts[1]), float(parts[2]))
            except ValueError:
                continue
    return pos


def naca(alpha, n=200):
    b = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(b))
    yt = 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    X = np.r_[x[::-1], x[1:]] - 1.0                      # bord de fuite a l'origine
    Y = np.r_[yt[::-1], -yt[1:]]
    a = np.radians(alpha)
    return X * np.cos(a) + Y * np.sin(a), -X * np.sin(a) + Y * np.cos(a)


def fig_probes():
    raw = next((J[k].get("positions_raw") for k in ("spec_a19", "prod_a15") if k in J and J[k].get("positions_raw")), None)
    if raw is None:
        return print("probes : positions absentes du JSON, figure non produite")
    task = "spec_a19" if J.get("spec_a19", {}).get("positions_raw") else "prod_a15"
    pos = parse_positions(raw)
    alpha = J[task]["alpha"]
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    X, Y = naca(alpha)
    ax.fill(X, Y, color="0.8", ec="0.4", lw=0.6)
    col = {"A": BLUE, "B": DARK, "W": ORANGE, "R": GREY, "S": "0.5"}
    for name, (x, y) in pos.items():
        c = col.get(name[0], "k")
        ax.plot(x, y, "o", ms=3.5, color=c, mec="white", mew=0.4, zorder=3)
        ax.annotate(name, (x, y), xytext=(2, 3), textcoords="offset points", fontsize=6.3, color=c)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$y/c$")
    fig.savefig("fig_freq_probes.pdf")
    plt.close(fig)
    print("fig_freq_probes.pdf :", ", ".join(f"{k} ({x:+.3f}, {y:+.3f})" for k, (x, y) in pos.items()))


# =============================================================================
# C. carte des bandes
# =============================================================================
def band_table():
    """Fractions de variance (%) par bande, 15 et 19 deg. JSON si present, sinon dossier."""
    if "prod_a15" in J and "spec_a19" in J:
        t15, t19 = {}, {}
        for n in ORDER:
            k = sig_key(n)
            if k in J["prod_a15"]["rms"]:
                b, r = J["prod_a15"]["bands"]["seg10"][k], J["prod_a15"]["rms"][k]
                t15[n] = (r, b["shed"], b["double"], b["f7_9"], b["f10_12"])
            if k in J["spec_a19"]["rms"]:
                b, r = J["spec_a19"]["bands"]["seg20"][k], J["spec_a19"]["rms"][k]
                t19[n] = (r, b["slow"], b["shed"], b["double"], b["f7_9"], b["f10_12"])
        src = "JSON"
    else:
        t15, t19, src = T15, T19, "dossier 4.4 (tables 3.3 et 5.1)"
    return t15, t19, src


def fig_bands():
    t15, t19, src = band_table()
    rows = [n for n in ORDER if n in t15 and n in t19]
    lab = {"CL": r"$C_L$", "CD": r"$C_D$"}
    M15 = np.array([[100 * (v / t15[n][0]) ** 2 for v in t15[n][1:]] for n in rows])
    M19 = np.array([[100 * (v / t19[n][0]) ** 2 for v in t19[n][1:]] for n in rows])
    M15, M19 = np.clip(M15, 0, 100), np.clip(M19, 0, 100)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 5.0), gridspec_kw=dict(width_ratios=[4, 5], wspace=0.06))
    norm = Normalize(0, 100)
    cmap = plt.get_cmap("Blues")
    for ax, M, cols, title in ((a1, M15, ["0.8–2.6", "2.6–4.6", "6.5–9.5", "9.5–12"], r"(a) $15^\circ$, attached"),
                               (a2, M19, ["0.05–0.8", "0.8–2.6", "2.6–4.6", "6.5–9.5", "9.5–12"], r"(b) $19^\circ$, stalled")):
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, fontsize=7.5)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([lab.get(n, n) for n in rows], fontsize=7.5)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=8.5, loc="left", pad=16)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if v >= 0.5:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.6,
                            color="white" if v > 55 else "0.15")
        for y in (1.5, 7.5, 10.5, 18.5):                       # efforts | peigne | couche | sillage | REF, SPAN
            ax.axhline(y, color="white", lw=1.6)
    a2.set_yticklabels([])
    cax = fig.add_axes([0.30, 0.04, 0.42, 0.018])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("share of the variance of the signal in the band (%)", fontsize=7.8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig("fig_freq_bands.pdf")
    plt.close(fig)
    print(f"fig_freq_bands.pdf (source : {src})")


# =============================================================================
# D. test de resolution N = 3 / N = 4
# =============================================================================
def series(task, key):
    s = J[task]["series"][key]
    y = np.array(s["y"], dtype=float)
    return s["t0"] + np.arange(y.size) * s["dt"], y, s["dt"]


def highpass(y, dt, fc):
    F = np.fft.rfft(y - y.mean())
    f = np.fft.rfftfreq(y.size, dt)
    F[f < fc] = 0
    return np.fft.irfft(F, y.size)


def fig_resolution():
    if "N3" not in J or "N4" not in J:
        return print("resolution : JSON incomplet, figure non produite")
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 4.8), gridspec_kw=dict(hspace=0.45, wspace=0.3))
    (a, b), (c, d) = axs
    for task, col, lab in (("N3", BLUE, "$N = 3$"), ("N4", ORANGE, "$N = 4$")):
        t, y, dt = series(task, "CL")
        a.plot(t, y, color=col, lw=0.9, label=lab)
        t, y, dt = series(task, "uy_W015m")
        b.plot(t, highpass(y, dt, 6.0), color=col, lw=0.8, label=lab)
        f, A = spec(task, None, "CL")
        c.semilogy(f, A, color=col, lw=1.0, label=lab)
        f, A = spec(task, None, "ux_A050")
        d.semilogy(f, A, color=col, lw=1.0, label=lab)
    a.set_ylabel(r"$C_L$")
    b.set_ylabel(r"$u_y$ at W015m, $St > 6$")
    for ax in (a, b):
        ax.set_xlabel(r"$t\,U_\infty/c$")
    for ax, lab in ((c, r"amplitude of $C_L$"), (d, r"amplitude of $u_x$ at A050")):
        ax.set_xlim(0.5, 32)
        ax.set_xlabel(r"$St$")
        ax.set_ylabel(lab)
        mark(ax, 10.65, "10.65")
        mark(ax, 14.5, "14.5")
    mark(d, 7.8, "7.8", ha="right")
    c.set_ylim(1e-6, 2e-2)
    d.set_ylim(1e-6, 2e-2)
    for ax, t in zip((a, b, c, d), "abcd"):
        ax.set_title(f"({t})", loc="left", fontsize=9)
    a.legend(frameon=False, loc="lower right", ncol=2)
    fig.savefig("fig_freq_resolution.pdf")
    plt.close(fig)
    pk = {k: J[k]["peaks"]["CL"][:3] for k in ("N3", "N4")}
    print("fig_freq_resolution.pdf ; pics de C_L :", pk)


# =============================================================================
# E. etat decroche a 19 deg
# =============================================================================
def fig_stalled():
    if "spec_a19" not in J:
        return print("stalled : JSON incomplet, figure non produite")
    S = J["spec_a19"]
    fig = plt.figure(figsize=(6.6, 5.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.7, 1.6], hspace=0.55, wspace=0.32)
    a = fig.add_subplot(gs[0, :])
    a2 = fig.add_subplot(gs[1, :], sharex=a)
    c = fig.add_subplot(gs[2, 0])
    d = fig.add_subplot(gs[2, 1])
    t, y = np.array(S["series"]["CL"]["t"]), np.array(S["series"]["CL"]["y"])
    a.plot(t, y, color=DARK, lw=1.0)
    a.axhline(np.mean(y), color="0.5", lw=0.6, ls="--")
    a.set_ylabel(r"$C_L$")
    t, y = np.array(S["series"]["CD"]["t"]), np.array(S["series"]["CD"]["y"])
    a2.plot(t, y, color=DARK, lw=0.9)
    a2.set_ylabel(r"$C_D$")
    a2.set_xlabel(r"$t\,U_\infty/c$")
    plt.setp(a.get_xticklabels(), visible=False)
    # spectres normalises par la fondamentale
    sel = [("CL", r"$C_L$", DARK, "-"), ("ux_A050", "A050", BLUE, "-"), ("ux_A092", "A092", LIGHT, "-"),
           ("uy_B090", "B090", ORANGE, "-"), ("uy_REF", "REF", GREY, "--")]
    for k, lab, col, ls in sel:
        if k not in S["spectra"]["seg32"]:
            continue
        f, A = spec("spec_a19", "seg32", k)
        m = (f > 0.02) & (f < 3)
        i0 = np.argmax(np.where((f > 0.06) & (f < 0.14), A, 0))
        c.loglog(f[m], A[m] / A[i0], color=col, lw=1.0, ls=ls, label=lab)
    for n in range(1, 10):
        c.axvline(n * F0, color="0.6", lw=0.5, ls=":", zorder=0)
        if n in (1, 3, 6, 9):
            c.text(n * F0, 1.02, f"{n}$f_0$", transform=c.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5,
                   color="0.4")
    c.set_xlim(0.03, 3)
    c.set_ylim(1e-3, 2)
    c.set_xlabel(r"$St$")
    c.set_ylabel("amplitude / amplitude at $f_0$")
    c.legend(frameon=False, fontsize=7, loc="lower left")
    # cycle moyen en phase
    C = S["cycle"]
    ph = np.degrees(np.array(C["phase"]))
    for k, lab, col, ls in (("CL", r"$C_L$", DARK, "-"), ("CD", r"$C_D$", ORANGE, "-"),
                            ("ut_A092", r"$u_t$ at A092", BLUE, "-"), ("uy_REF", r"$u_y$ at REF", GREY, "--")):
        if k not in C:
            continue
        y = np.array(C[k], dtype=float)
        yn = (y - y.mean()) / (np.max(np.abs(y - y.mean())) + 1e-30)
        d.plot(np.r_[ph, ph[0] + 360], np.r_[yn, yn[0]], color=col, ls=ls, lw=1.1, label=lab)
    d.set_xlim(0, 360)
    d.set_xticks([0, 90, 180, 270, 360])
    d.set_xlabel("phase of the slow cycle (deg)")
    d.set_ylabel("phase average, normalised")
    d.legend(frameon=False, fontsize=7, loc="lower left")
    for ax, t in zip((a, c, d), "acd"):
        ax.set_title(f"({t})", loc="left", fontsize=9)
    a2.set_title("(b)", loc="left", fontsize=9)
    fig.savefig("fig_freq_stalled.pdf")
    plt.close(fig)
    print("fig_freq_stalled.pdf ; cycles :", round(C.get("n_cycles", np.nan), 1), "; fraction u_t < 0 :",
          {k: round(v, 2) for k, v in S.get("reverse_fraction", {}).items()},
          "; phases a f0 :", {k: (round(p), round(co, 2)) for k, (p, co) in S.get("phase_f0", {}).items()})


# =============================================================================
# F. echelle : St_h = St sin(alpha)
# =============================================================================
def slow_peak(task, tag):
    try:
        pk = [p for p in J[task]["peaks"][tag]["CL"] if 0.04 < p[0] < 0.3]
        return max(pk, key=lambda p: p[1])[0] if pk else None
    except KeyError:
        return None


def fig_scaling():
    s = lambda a: np.sin(np.radians(a))
    ours = [(19.0, F0, r"$19^\circ$")]
    for task, tag in (("cold17", "seg20"), ("jump13", "seg20")):
        f = slow_peak(task, tag) if task in J else None
        if f is not None:
            ours.append((J[task]["alpha"], f, ""))
    if not any(abs(a - 17.0) < 1e-6 for a, _, _ in ours):
        ours.append((17.0, 0.100, ""))                         # depart a froid a 17 deg (dossier 4.4 §5.6)
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ax.axhspan(0.15, 0.20, color=ORANGE, alpha=0.12, lw=0)
    ax.text(13.15, 0.173, "bluff-body shedding of deep stall", fontsize=7, color="0.3", va="center")
    ax.axhline(0.03, color="0.45", lw=0.8, ls="--")
    ax.text(14.1, 0.0335, "low-frequency oscillation near stall", fontsize=7, color="0.3", va="bottom")
    ax.semilogy(MF_A, [v * s(a) for a, v in zip(MF_A, MF_UP)], "D", color=ORANGE, ms=4.2, mfc=ORANGE,
                label="Martin Ferrer (2025), upstroke")
    ax.semilogy(MF_A, [v * s(a) for a, v in zip(MF_A, MF_DN)], "D", color=ORANGE, ms=4.2, mfc="white",
                label="Martin Ferrer (2025), downstroke")
    ax.semilogy([a for a, _ in MS_DEC], [v * s(a) for a, v in MS_DEC], "s", color=GREY, ms=4.6, mfc="white",
                label="Mittal & Saxena (2002), decreasing $\\alpha$")
    ax.semilogy([a for a, _ in MS_INC], [v * s(a) for a, v in MS_INC], "s", color=GREY, ms=4.6,
                label="Mittal & Saxena (2002), increasing $\\alpha$")
    ax.semilogy([15.0], [1.75 * s(15.0)], "o", color=BLUE, ms=5, mfc="white", label="this study, attached shedding")
    ax.semilogy([a for a, _, _ in ours], [f * s(a) for a, f, _ in ours], "o", color=DARK, ms=5.5,
                label="this study, slow oscillation of the stalled state")
    ax.set_xlim(13, 20.5)
    ax.set_ylim(0.01, 1.0)
    ax.set_xlabel(r"angle of attack $\alpha$ (deg)")
    ax.set_ylabel(r"$St_h = f\,c\,\sin\alpha / U_\infty$")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=7.3, loc="upper center",
               bbox_to_anchor=(0.5, -0.03), ncol=2)
    fig.savefig("fig_freq_scaling.pdf")
    plt.close(fig)
    print("fig_freq_scaling.pdf ; nos points (alpha, St, St_h) :",
          [(a, round(f, 4), round(f * s(a), 4)) for a, f, _ in ours])


if __name__ == "__main__":
    for fn in (fig_overview, fig_probes, fig_bands, fig_resolution, fig_stalled, fig_scaling):
        try:
            fn()
        except Exception as e:                       # une figure ratee n'arrete pas les autres
            print(f"{fn.__name__} : ECHEC {type(e).__name__}: {e}")

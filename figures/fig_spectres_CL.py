#!/usr/bin/env python3
"""
fig_spectres_CL.py -- spectres de C_L : etat attache (15 deg) et etat decroche (19 deg), memes axes
USAGE (sur le cluster, depuis n'importe quel dossier) :
    python3 fig_spectres_CL.py
Sorties : fig_spectres_CL.png et fig_spectres_CL.pdf
Methode : celle de spectre_sondes.py (Welch, fenetre de Hann, recouvrement 50 %, amplitude d'une sinusoide).
Colonnes de drag_results.dat (une ligne d'en-tete) : 1 = temps, 2 = C_D, 3 = C_L (comme dans spectre_sondes.py).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.expanduser("~/cdf_runs")
CAS = [  # dossier, t0, t1, segment (CTU), legende, couleur
    ("prod_a15",  50.0, 100.0, 10.0, r"attached, $\alpha=15^\circ$ (50-100 CTU)", "#1f77b4"),
    ("spec_a19.00", 236.0, 300.0, 32.0, r"stalled, $\alpha=19^\circ$ (236-300 CTU)", "#d62728"),
]


def charge(dossier, t0, t1):
    a = np.loadtxt(os.path.join(R, dossier, "drag_results.dat"), skiprows=1)
    a = a[(np.abs(a[:, 3]) < 2) & (a[:, 2] > 0) & (a[:, 2] < 2)]
    t, cl = a[:, 1], a[:, 3]
    k = np.concatenate(([True], np.diff(t) > 0))
    t, cl = t[k], cl[k]
    m = (t >= t0) & (t <= t1)
    t, cl = t[m], cl[m]
    dt = np.median(np.diff(t))
    tu = np.arange(t[0], t[-1], dt)           # grille reguliere (utile si le pas varie un peu)
    return tu, np.interp(tu, t, cl), dt


def spectre(y, dt, seg):
    y = y - y.mean()
    L = min(len(y), int(round(seg / dt)))
    pas = max(1, L // 2)
    segs = [y[i:i + L] for i in range(0, len(y) - L + 1, pas)] or [y]
    L = len(segs[0])
    w = np.hanning(L)
    nfft = L * max(1, min(8, 2 ** 18 // L))
    P = np.mean([np.abs(np.fft.rfft((s - s.mean()) * w, nfft)) ** 2 for s in segs], axis=0)
    return np.fft.rfftfreq(nfft, dt), 2.0 * np.sqrt(P) / w.sum()


def fig_spectres():
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for dossier, t0, t1, seg, leg, col in CAS:
        t, cl, dt = charge(dossier, t0, t1)
        f, A = spectre(cl, dt, seg)
        m = (f > 0.03) & (f < 40)
        ax.loglog(f[m], A[m], color=col, lw=1.1, label=leg)
    for x, txt in [(0.097, "0.097"), (1.75, "1.75"), (10.66, "10.66")]:
        ax.axvline(x, color="0.6", lw=0.7, ls=":")
        ax.text(x, 1.4e-1, txt, rotation=90, va="top", ha="right", fontsize=8, color="0.35")
    ax.set_xlabel(r"$St = f\,c/U_\infty$")
    ax.set_ylabel(r"amplitude of $C_L$")
    ax.set_xlim(0.03, 40)
    ax.set_ylim(1e-7, 2e-1)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig("fig_spectres_CL.png", dpi=200)
    fig.savefig("fig_spectres_CL.pdf")
    print("ecrit : fig_spectres_CL.png et fig_spectres_CL.pdf")


if __name__ == "__main__":
    fig_spectres()

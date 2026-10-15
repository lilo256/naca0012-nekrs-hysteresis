#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_upstroke_profiles.py -- profils normaux a l'extrados exportes de ParaView (Plot Over Line,
                            File > Save Data en CSV) : etat attache contre etat decroche.

  python3 fig_upstroke_profiles.py profil_*.csv

Nom des fichiers : profil_<etat>_x<NNN>.csv, par exemple
  profil_a17.25_plat_x050.csv    (17.25 deg, plateau attache, x/c = 0.50)
  profil_a17.25_stall_x080.csv   (17.25 deg, decroche, x/c = 0.80)
L'incidence est lue dans <etat> (aNN.NN), la station dans xNNN (x/c = NNN/100).
La ligne doit partir 0.003c SOUS la paroi (points hors fluide = NaN) : n = 0 au premier point
fluide. u_t = vitesse tangente a l'extrados, positive vers le bord de fuite.

Produit fig_upstroke_profiles.pdf (un panneau par station, hauteur du peigne 0.008c en pointille)
et imprime, pour chaque profil, l'epaisseur de la couche de retour contre la paroi et u_t a 0.008c.
"""
import csv
import math
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def yt(x):
    return 0.6 * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)


def tangent(xb, alpha):
    h = 1e-4
    dy = (yt(xb + h) - yt(xb - h)) / (2 * h)
    n = math.hypot(1.0, dy)
    a = math.radians(alpha)
    tx, ty = 1.0 / n, dy / n
    return tx * math.cos(a) + ty * math.sin(a), -tx * math.sin(a) + ty * math.cos(a)


def read(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    head = [h.strip().strip('"').lower() for h in rows[0]]
    D = np.array([[float(v) if v.strip() not in ("", "nan", "-nan") else np.nan for v in r]
                  for r in rows[1:] if r], dtype=float)
    col = lambda *names: next(D[:, head.index(n)] for n in names if n in head)
    ux, uy = col("velocity:0", "velocity_x"), col("velocity:1", "velocity_y")
    s = col("arc_length")
    ok = np.isfinite(ux) & np.isfinite(uy)
    if "vtkvalidpointmask" in head:
        ok &= col("vtkvalidpointmask") > 0
    return s[ok] - s[ok][0], ux[ok], uy[ok]


groups = {}
for f in sys.argv[1:]:
    m = re.search(r"profil_(?P<etat>.+)_x(?P<x>\d{3})\.csv$", f)
    a = re.search(r"a(\d+\.\d+)", f)
    if not m or not a:
        print(f"nom non reconnu (profil_aNN.NN_<etat>_xNNN.csv) : {f}")
        continue
    xb, alpha = int(m.group("x")) / 100.0, float(a.group(1))
    n, ux, uy = read(f)
    tx, ty = tangent(xb, alpha)
    ut = ux * tx + uy * ty
    groups.setdefault(m.group("etat"), {})[xb] = (n, ut)
    neg = np.where(ut < 0)[0]
    thick = n[neg[np.argmax(np.diff(np.r_[neg, 10 ** 9]) > 1)]] if neg.size and neg[0] <= 2 else 0.0
    print(f"{m.group('etat'):16s} x/c = {xb:.2f} : couche de retour contre la paroi {thick:.4f}c ; "
          f"u_t(0.008c) = {np.interp(0.008, n, ut):+.3f} ; min u_t = {ut.min():+.3f} a n = {n[np.argmin(ut)]:.4f}c")

if groups:
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "savefig.bbox": "tight", "savefig.dpi": 300})
    stations = sorted({x for g in groups.values() for x in g})
    fig, axs = plt.subplots(1, len(stations), figsize=(7.0, 2.8), sharey=True, squeeze=False)
    sty = [("-", "#0d3a73"), ("--", "k"), (":", "#8a8a82"), ("-.", "#eb6834")]
    for ax, xs in zip(axs[0], stations):
        for k, (etat, g) in enumerate(sorted(groups.items())):
            if xs in g:
                n, ut = g[xs]
                ax.plot(ut, n, sty[k % 4][0], color=sty[k % 4][1], lw=1.1, label=etat)
        ax.axhline(0.008, color="0.4", lw=0.7, ls=":")
        ax.axvline(0, color="0.3", lw=0.6)
        ax.set_title(f"x/c = {xs:.2f}", fontsize=8)
        ax.set_xlabel(r"$u_t/U_\infty$")
    axs[0][0].set_ylabel(r"$n/c$ (distance to the wall)")
    axs[0][0].set_ylim(0, 0.12)
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(groups), frameon=False)
    fig.tight_layout()
    fig.savefig("fig_upstroke_profiles.pdf")
    print("fig_upstroke_profiles.pdf ecrit")

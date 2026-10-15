#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
montage_vorticity_44.py -- figure fig_freq_vorticity (section 4.4.3) a partir de deux captures ParaView.

  python3 montage_vorticity_44.py profil_loin.png view_bord_de_fuite_claire.png

Chaque capture montre N = 3 (moitie gauche) et N = 4 (moitie droite), meme camera et meme echelle de
couleurs. Le script coupe chaque capture en deux, retire le bandeau du haut (etiquette « Text ... » de
ParaView) et assemble quatre panneaux : (a, b) profil entier, (c, d) bord de fuite.
Sortie : fig_freq_vorticity.pdf et .png. Dependances : numpy, matplotlib, pillow.
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

WIDE = sys.argv[1] if len(sys.argv) > 1 else "profil_loin.png"
ZOOM = sys.argv[2] if len(sys.argv) > 2 else "view_bord_de_fuite_claire.png"
T_N3, T_N4 = "101.5", "104"          # instants des deux champs (etiquettes des captures)
TOP = 0.30                           # fraction de la hauteur retiree en haut (etiquette, zone vide)


def halves(path):
    a = np.asarray(Image.open(path).convert("RGB"))
    h, w, _ = a.shape
    mid = w // 2
    # separateur vertical clair entre les deux vues : colonne la plus claire pres du milieu
    k = mid - 20 + int(np.argmax(a[:, mid - 20:mid + 20, :].astype(float).mean(axis=(0, 2))))
    y0 = int(round(TOP * h))
    return a[y0:, :k - 1, :], a[y0:, k + 2:, :]


rows = [halves(WIDE), halves(ZOOM)]
h, w, _ = rows[0][0].shape
fig, axs = plt.subplots(2, 2, figsize=(6.6, 6.6 * h / w + 0.55), gridspec_kw=dict(wspace=0.03, hspace=0.16))
titles = [(rf"(a) $N = 3$, $t = {T_N3}$", rf"(b) $N = 4$, $t = {T_N4}$"),
          (r"(c) $N = 3$, trailing edge", r"(d) $N = 4$, trailing edge")]
for i in range(2):
    for j in range(2):
        ax = axs[i, j]
        ax.imshow(rows[i][j], interpolation="lanczos")
        ax.set_title(titles[i][j], loc="left", fontsize=8.8, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
            s.set_color("0.4")
fig.savefig("fig_freq_vorticity.pdf", dpi=300, bbox_inches="tight")
fig.savefig("fig_freq_vorticity.png", dpi=200, bbox_inches="tight")
print("fig_freq_vorticity.pdf et .png ecrits a partir de", WIDE, "et", ZOOM)

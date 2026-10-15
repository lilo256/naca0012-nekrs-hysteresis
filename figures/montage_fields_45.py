#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
montage_fields_45.py -- les trois figures de vorticite de la section 4.5, a partir des captures ParaView.

  python3 montage_fields_45.py [dossier_des_png]

Captures (PNG, meme camera, echelle -5..5, carte "Cool to Warm", legende de couleur MASQUEE : le script
ajoute une barre commune). Liste et instants : CAPTURES_ParaView_4.5.md. Une capture absente est sautee ;
une figure est faite des qu'elle a au moins un panneau.
  fig_down_fields    la descente seule : d20 d18p5 d17p5 d17 d16 d15p5 d14 d10 (+ d12 facultatif)   4.5.2
  fig_updown_fields  montee (gauche) contre descente (droite) : 16, 17, 18.5 et 10 deg                4.5.3
  fig_hyst_fields    les etats a 13 deg : attache, decroche, filtre x4, melange gamma = 0.7 (+ N = 4)  4.5.4
Les bords uniformes de chaque capture sont retires automatiquement ; TOP retire en plus une bande en haut
(etiquette de ParaView), en fraction de la hauteur. Dependances : numpy, matplotlib, pillow.
Les C_L des legendes viennent des series de boucle_data.json (tranches de 0,1 CTU) a l'instant de la capture.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image

DIR = sys.argv[1] if len(sys.argv) > 1 else "."
TOP = 0.0                # fraction de la hauteur retiree en haut (0 si pas d'etiquette)
BLEND = "565"            # instant du melange gamma = 0.7 retenu pour la figure 3 : "565" ou "570"
T_N4 = "600"             # instant de la capture a N = 4 (si elle existe)

DOWN = [  # (fichier, titre)
    ("d20.png", r"$20^\circ$, $t = 65$, $C_L = 0.959$"),
    ("d18p5.png", r"$18.5^\circ$, $t = 250$, $C_L = 0.867$"),
    ("d17p5.png", r"$17.5^\circ$, $t = 405$, $C_L = 0.841$"),
    ("d17.png", r"$17^\circ$, $t = 465$, $C_L = 0.829$"),
    ("d16.png", r"$16^\circ$, $t = 275$, $C_L = 0.752$"),
    ("d15p5.png", r"$15.5^\circ$, $t = 345$, $C_L = 0.769$"),
    ("d14.png", r"$14^\circ$, $t = 570$, $C_L = 0.752$"),
    ("d12.png", r"$12^\circ$, $t = 590$, $C_L = 0.704$"),
    ("d10.png", r"$10^\circ$, $t = 659.9$, $C_L = 0.588$"),
]
UPDOWN = [  # (angle, (montee, titre), (descente, titre))
    (r"$16^\circ$", ("u16.png", r"upstroke, $t = 148$, $C_L = 1.397$"), ("d16.png", r"downstroke, $t = 275$, $C_L = 0.752$")),
    (r"$17^\circ$", ("u17.png", r"upstroke, $t = 340$, $C_L = 1.376$"), ("d17.png", r"downstroke, $t = 465$, $C_L = 0.829$")),
    (r"$18.5^\circ$", ("u18p5.png", r"upstroke, $t = 540$, $C_L = 0.901$"),
     ("d18p5.png", r"downstroke, $t = 250$, $C_L = 0.867$")),
    (r"$10^\circ$", ("u10.png", r"attached, $t = 80$, $C_L = 1.025$"), ("d10.png", r"stalled, $t = 659.9$, $C_L = 0.588$")),
]
BLEND_TITLE = {"565": r"blend $\gamma = 0.7$, $t = 565$, $C_L = 1.151$", "570": r"blend $\gamma = 0.7$, $t = 570$, $C_L = 1.059$"}
STATES13 = [
    ("a13_attached.png", r"attached, $t = 130$, $C_L = 1.277$"),
    ("a13_stalled.png", r"stalled, $t = 560$, $C_L = 0.738$"),
    ("a13_filter4.png", r"filter $\times 4$, $t = 625$, $C_L = 0.916$"),
    ("a13_blend70_%s.png" % BLEND, BLEND_TITLE[BLEND]),
    ("a13_N4.png", r"$N = 4$, $t = %s$" % T_N4),
]
LETTERS = "abcdefghijklmnop"


def load(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    a = a[int(round(TOP * a.shape[0])):]
    bg = a[0, 0]
    diff = np.abs(a - bg).sum(axis=2) > 30
    rows, cols = np.where(diff.any(axis=1))[0], np.where(diff.any(axis=0))[0]
    if rows.size and cols.size:
        m = 4
        a = a[max(rows[0] - m, 0):rows[-1] + m + 1, max(cols[0] - m, 0):cols[-1] + m + 1]
    return a.astype(np.uint8)


def present(name):
    p = os.path.join(DIR, name)
    if not os.path.exists(p):
        print("   absente (sautee) :", name)
        return None
    return p


def colourbar(fig, y=0.03):
    cax = fig.add_axes([0.3, y, 0.4, 0.014])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-5, 5), cmap="coolwarm"), cax=cax, orientation="horizontal")
    cb.set_ticks([-5, -2.5, 0, 2.5, 5])
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label(r"$\omega_z c/U_\infty$", fontsize=8.5)


def frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)
        s.set_color("0.4")


def grid(name, panels, ncol):
    imgs = [(load(p), t) for p, t in panels]
    nrow = int(np.ceil(len(imgs) / ncol))
    h, w, _ = imgs[0][0].shape
    fig, axs = plt.subplots(nrow, ncol, figsize=(6.8, ((6.8 / ncol) * h / w + 0.3) * nrow / 0.86), squeeze=False,
                            gridspec_kw=dict(wspace=0.04, hspace=0.2, left=0.01, right=0.99, top=0.96, bottom=0.1))
    for i, ax in enumerate(axs.flat):
        frame(ax)
        if i >= len(imgs):
            ax.axis("off")
            continue
        ax.imshow(imgs[i][0], interpolation="lanczos")
        ax.set_title("(%s) %s" % (LETTERS[i], imgs[i][1]), loc="left", fontsize=8.2, pad=2.5)
    colourbar(fig, y=0.035 if nrow > 2 else 0.05)
    save(fig, name, len(imgs))


def save(fig, name, n):
    fig.savefig(name + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(name + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("%s.pdf et .png ecrits (%d panneaux)" % (name, n))


def fig_down():
    P = [(present(f), t) for f, t in DOWN]
    P = [(p, t) for p, t in P if p]
    if P:
        grid("fig_down_fields", P, 3 if len(P) == 9 else 2)


def fig_updown():
    rows = []
    for ang, (fu, tu), (fd, td) in UPDOWN:
        pu, pd = present(fu), present(fd)
        if pu or pd:
            rows.append((ang, (pu, tu), (pd, td)))
    if not rows:
        return
    ref = load(next(p for r in rows for p, t in r[1:] if p))
    h, w, _ = ref.shape
    fig, axs = plt.subplots(len(rows), 2, figsize=(6.8, (3.4 * h / w + 0.3) * len(rows) / 0.86), squeeze=False,
                            gridspec_kw=dict(wspace=0.04, hspace=0.22, left=0.01, right=0.99, top=0.96, bottom=0.1))
    k = 0
    for i, (ang, up, down) in enumerate(rows):
        for j, (p, t) in enumerate((up, down)):
            ax = axs[i, j]
            frame(ax)
            if p is None:
                ax.axis("off")
                continue
            ax.imshow(load(p), interpolation="lanczos")
            ax.set_title("(%s) %s, %s" % (LETTERS[k], ang, t), loc="left", fontsize=8.2, pad=2.5)
            k += 1
    colourbar(fig, y=0.035 if len(rows) > 2 else 0.05)
    save(fig, "fig_updown_fields", k)


def fig_13():
    P = [(present(f), t) for f, t in STATES13]
    P = [(p, t) for p, t in P if p]
    if P:
        grid("fig_hyst_fields", [(p, r"$13^\circ$, " + t) for p, t in P], 2 if len(P) <= 4 else 3)


if __name__ == "__main__":
    fig_down()
    fig_updown()
    fig_13()

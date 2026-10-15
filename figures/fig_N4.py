#!/usr/bin/env python3
"""
fig_N4.py  --  test de resolution (N = 3 -> N = 4) sur les etats decroches a 13 et 16 deg
=========================================================================================
Deux figures pour le rapport (sections 4.4 et 4.5.4), et les valeurs de controle a l'ecran.

 fig_N4_histories.pdf/.png : C_L(t) et C_D(t)
     (a,c) 13 deg : etat decroche de production (jump_a13.00, N = 3) jusqu'a t = 560, puis, depuis
           le MEME champ : N = 4 (stall13_N4, recolle) et filtre x4 a N = 3 (stall13_hpf40 + b).
           Reference : etat attache a N = 3 (att_a13.00, 115-130).
     (b,d) 16 deg : descente courte (down_a16.00, N = 3) jusqu'a t = 280, puis N = 4 (stall16_N4).
           References : etat attache de la montee (up_a16.00, 2e moitie) et descente de
           Martin Ferrer (2025) a 16 deg (1.057 / 0.1839).
 fig_N4_spectra.pdf/.png : spectres d'amplitude, N = 3 contre N = 4
     (a) 13 deg attache, C_L      (b) 13 deg attache, uy_W015m (sillage proche)
     (c) 16 deg decroche, C_L     (d) 16 deg decroche, uy_B050 (couche cisaillee)
     Methode de spectre_sondes.py (Welch, Hann, recouvrement 50 %, amplitude d'une sinusoide),
     memes fenetres que spectres_rapides.txt : les valeurs imprimees doivent y correspondre.

USAGE (sur cdf)
   cd ~/cdf_runs && python3 -u ~/bin/fig_N4.py | tee fig_N4.txt
SORTIES (dans ~/cdf_runs) : fig_N4_histories.pdf/.png, fig_N4_spectra.pdf/.png, fig_N4.txt (via tee)
Colonnes de drag_results.dat (une ligne d'en-tete) : 1 = temps, 2 = C_D, 3 = C_L (comme spectre_sondes.py).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.expanduser("~/cdf_runs")
C3, C4, CF = "#2a78d6", "#eb6834", "#5f5e58"     # N = 3 bleu, N = 4 orange, filtre x4 gris fonce
GRIS, ENCRE = "#8a8983", "#1f1f1e"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 8,
                     "legend.frameon": False, "savefig.dpi": 200})

BANDES = [("lent", 0.05, 0.8), ("lacher", 0.8, 2.6), ("double", 2.6, 4.6),
          ("7-9", 6.5, 9.5), ("10-12", 9.5, 12.0)]


# ------------------------------------------------------------------ lecture (comme spectre_sondes.py)
def charger(p, delim=None):
    try:
        a = np.loadtxt(p, delimiter=delim, skiprows=1)
        return a if a.ndim == 2 else a.reshape(1, -1)
    except Exception:
        pass
    with open(p, errors="ignore") as f:
        ncol = len(f.readline().strip().split(delim))
        bons = []
        for l in f:
            s = l.strip().split(delim)
            if len(s) == ncol:
                try:
                    bons.append([float(x) for x in s])
                except ValueError:
                    pass
    return np.array(bons, float).reshape(-1, ncol)


def monotone(t):
    keep = np.ones(len(t), bool)
    for r in np.where(np.diff(t) < 0)[0] + 1:
        keep[:r] &= t[:r] < t[r]
    return keep


_cache = {}


def forces(cas):
    """(t, CL, CD) de drag_results.dat, lignes aberrantes et retours en arriere retires."""
    if ("f", cas) not in _cache:
        p = os.path.join(R, cas, "drag_results.dat")
        if not os.path.isfile(p):
            _cache[("f", cas)] = None
        else:
            a = charger(p)
            a = a[(np.abs(a[:, 3]) < 2) & (a[:, 2] > 0) & (a[:, 2] < 2)]
            a = a[monotone(a[:, 1])]
            _cache[("f", cas)] = (a[:, 1], a[:, 3], a[:, 2])
    return _cache[("f", cas)]


def sonde(cas, nom):
    p = os.path.join(R, cas, "probes_uvp.csv")
    if ("s", cas) not in _cache:
        if not os.path.isfile(p):
            _cache[("s", cas)] = None
        else:
            h = [c.strip() for c in open(p).readline().strip().split(",")]
            s = charger(p, ",")
            s = s[monotone(s[:, 0])]
            _cache[("s", cas)] = (h, s)
    if _cache[("s", cas)] is None:
        return None
    h, s = _cache[("s", cas)]
    if nom not in h:
        print(f"   ! {cas} : sonde {nom} absente de probes_uvp.csv")
        return None
    return s[:, 0], s[:, h.index(nom)]


def signal(cas, nom):
    if nom in ("CL", "CD"):
        f = forces(cas)
        if f is None:
            return None
        return f[0], (f[1] if nom == "CL" else f[2])
    return sonde(cas, nom)


def regulier(t, y):
    d = np.diff(t)
    med = np.median(d)
    if np.all(np.abs(d[:-1] - med) <= 0.01 * med):
        if abs(d[-1] - med) > 0.01 * med:
            t, y = t[:-1], y[:-1]
        return t, y, med
    tr = np.arange(t[0], t[-1], med)
    return tr, np.interp(tr, t, y), med


def spectre(y, dt, seg):
    """amplitude d'une sinusoide (Hann, Welch 50 %) et rms^2 par raie, comme spectre_sondes.py."""
    y = y - y.mean()
    n = len(y)
    L = min(n, int(round(seg / dt)))
    pas = max(1, L // 2)
    segs = [y[i:i + L] for i in range(0, n - L + 1, pas)] or [y]
    L = len(segs[0])
    w = np.hanning(L)
    nfft = L * max(1, min(8, 2 ** 18 // L))
    P = np.mean([np.abs(np.fft.rfft((s - s.mean()) * w, nfft)) ** 2 for s in segs], axis=0)
    f = np.fft.rfftfreq(nfft, dt)
    A = 2.0 * np.sqrt(P) / w.sum()
    pw = 2.0 * P / (L ** 2 * np.mean(w ** 2)) * (L / nfft)
    return f, A, pw


def fenetre(cas, nom, t0, t1):
    s = signal(cas, nom)
    if s is None:
        return None
    t, y = s
    m = (t >= t0) & (t <= t1)
    if m.sum() < 100:
        print(f"   ! {cas} {nom} : fenetre [{t0}, {t1}] vide ou trop courte (donnees : {t[0]:.1f}-{t[-1]:.1f})")
        return None
    return regulier(t[m], y[m])


def tranches(t, y, larg=0.05):
    """moyenne par tranches (allege les PDF sans changer l'allure)."""
    k = np.floor((t - t[0]) / larg).astype(int)
    n = np.bincount(k)
    ok = n > 0
    return (np.bincount(k, t)[ok] / n[ok]), (np.bincount(k, y)[ok] / n[ok])


def moyenne(cas, t0=None, t1=None):
    f = forces(cas)
    if f is None:
        return None
    t, cl, cd = f
    t0 = t[0] + 0.5 * (t[-1] - t[0]) if t0 is None else t0
    t1 = t[-1] if t1 is None else t1
    m = (t >= t0) & (t <= t1)
    if m.sum() < 10:
        return None
    w = np.gradient(t[m])                       # moyenne ponderee en temps (pas variable)
    return (float(np.sum(cl[m] * w) / w.sum()), float(np.sum(cd[m] * w) / w.sum()),
            float(t[m][0]), float(t[m][-1]))


# ------------------------------------------------------------------ figure 1 : historiques
def trace(ax_cl, ax_cd, cas, t0, t1, col, lab, lw=0.9, z=2):
    f = forces(cas)
    if f is None:
        print(f"   ! {cas} : drag_results.dat absent")
        return
    t, cl, cd = f
    m = (t >= t0) & (t <= t1)
    if m.sum() < 2:
        print(f"   ! {cas} : rien entre {t0} et {t1}")
        return
    tt, yl = tranches(t[m], cl[m])
    _, yd = tranches(t[m], cd[m])
    ax_cl.plot(tt, yl, color=col, lw=lw, label=lab, zorder=z)
    ax_cd.plot(tt, yd, color=col, lw=lw, zorder=z)


def fig_historiques():
    fig, ax = plt.subplots(2, 2, figsize=(6.6, 4.6), sharex="col")
    print("\n=== HISTORIQUES : moyennes de controle (ponderees en temps)")

    # --- 13 deg
    a_cl, a_cd = ax[0, 0], ax[1, 0]
    trace(a_cl, a_cd, "jump_a13.00", 530, 560, C3, r"$N=3$, production")
    trace(a_cl, a_cd, "stall13_hpf40", 560, 600, CF, r"$N=3$, filter $\times4$", lw=0.7, z=1)
    trace(a_cl, a_cd, "stall13_hpf40b", 600, 620, CF, None, lw=0.7, z=1)
    trace(a_cl, a_cd, "stall13_N4", 560, 600, C4, r"$N=4$")
    ref = moyenne("att_a13.00", 115.0, 130.0)
    if ref:
        a_cl.axhline(ref[0], color=C3, ls="--", lw=0.8, label=r"attached, $N=3$")
        a_cd.axhline(ref[1], color=C3, ls="--", lw=0.8)
        print(f"   att_a13.00   [{ref[2]:.1f};{ref[3]:.1f}]  CL {ref[0]:.4f}  CD {ref[1]:.5f}  (attache N = 3)")
    for c, t0, t1 in [("jump_a13.00", 530, 560), ("stall13_N4", 595, 600),
                      ("stall13_hpf40b", 600, None)]:
        r = moyenne(c, t0, t1)
        if r:
            print(f"   {c:13s}[{r[2]:.1f};{r[3]:.1f}]  CL {r[0]:.4f}  CD {r[1]:.5f}")
    for a_ in (a_cl, a_cd):
        a_.axvline(560, color=GRIS, lw=0.6, ls=":")
    a_cl.set_title(r"(a) $\alpha=13^\circ$", loc="left")
    a_cd.set_title(r"(c) $\alpha=13^\circ$", loc="left")
    a_cd.set_xlim(530, 620)

    # --- 16 deg
    b_cl, b_cd = ax[0, 1], ax[1, 1]
    trace(b_cl, b_cd, "down_a16.00", 245, 280, C3, r"$N=3$, downstroke")
    trace(b_cl, b_cd, "stall16_N4", 280, 295.5, C4, r"$N=4$")
    ref = moyenne("up_a16.00")
    if ref:
        b_cl.axhline(ref[0], color=C3, ls="--", lw=0.8, label=r"attached, $N=3$ (upstroke)")
        b_cd.axhline(ref[1], color=C3, ls="--", lw=0.8)
        print(f"   up_a16.00    [{ref[2]:.1f};{ref[3]:.1f}]  CL {ref[0]:.4f}  CD {ref[1]:.5f}  (attache N = 3, 2e moitie)")
    b_cl.axhline(1.057, color=GRIS, ls=":", lw=0.9, label="Martin Ferrer (2025), downstroke")
    b_cd.axhline(0.1839, color=GRIS, ls=":", lw=0.9)
    for c, t0, t1 in [("down_a16.00", 260, 280), ("stall16_N4", 283, 295)]:
        r = moyenne(c, t0, t1)
        if r:
            print(f"   {c:13s}[{r[2]:.1f};{r[3]:.1f}]  CL {r[0]:.4f}  CD {r[1]:.5f}")
    for a_ in (b_cl, b_cd):
        a_.axvline(280, color=GRIS, lw=0.6, ls=":")
        a_.axvspan(283, 295, color="#d9d8d3", alpha=0.35, lw=0, zorder=0)
    b_cl.set_title(r"(b) $\alpha=16^\circ$", loc="left")
    b_cd.set_title(r"(d) $\alpha=16^\circ$", loc="left")
    b_cd.set_xlim(245, 296)

    ax[0, 0].set_ylabel(r"$C_L$")
    ax[1, 0].set_ylabel(r"$C_D$")
    for a_ in ax[1]:
        a_.set_xlabel(r"$t\,U_\infty/c$")
    for a_ in ax.flat:
        a_.grid(True, alpha=0.25)
    # legendes dans les zones vides des vraies donnees (verifie sur la figure du 14/10) :
    # 13 deg : C_L 0.45-0.70 apres t = 575 (panneau a) ; 16 deg : C_D 0.05-0.14 avant t = 275 (panneau d)
    ax[0, 0].legend(loc="lower right", fontsize=7, framealpha=0.9, frameon=True, edgecolor="none")
    h, l = ax[0, 1].get_legend_handles_labels()
    ax[0, 1].legend(h, l, loc="upper left", bbox_to_anchor=(0.0, 0.94), fontsize=7)   # C_L 1.1-1.37 avant t = 278 : vide
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig_N4_histories.pdf"))
    fig.savefig(os.path.join(R, "fig_N4_histories.png"))
    print("   ecrit : fig_N4_histories.pdf / .png")


# ------------------------------------------------------------------ figure 2 : spectres
PANNEAUX = [  # titre, signal, [(cas, t0, t1, segment, couleur, legende)]
    (r"(a) $13^\circ$, attached: $C_L$", "CL",
     [("att_a13.00", 115.0, 130.0, 5.0, C3, r"$N=3$"), ("stall13_N4", 585.0, 600.0, 5.0, C4, r"$N=4$")]),
    (r"(b) $13^\circ$, attached: $u_y$ near wake (W015m)", "uy_W015m",
     [("att_a13.00", 115.0, 130.0, 5.0, C3, r"$N=3$"), ("stall13_N4", 585.0, 600.0, 5.0, C4, r"$N=4$")]),
    (r"(c) $16^\circ$, stalled: $C_L$", "CL",
     [("down_a16.00", 268.0, 280.0, 4.0, C3, r"$N=3$"), ("stall16_N4", 283.0, 295.0, 4.0, C4, r"$N=4$")]),
    (r"(d) $16^\circ$, stalled: $u_y$ shear layer (B050)", "uy_B050",
     [("down_a16.00", 268.0, 280.0, 4.0, C3, r"$N=3$"), ("stall16_N4", 283.0, 295.0, 4.0, C4, r"$N=4$")]),
]
REPERES = [(7.8, "7.8"), (10.66, "10.66"), (13.9, "13.9"), (15.4, "15.4"), (17.3, "17.3")]


def pic(f, A, fc, demi=0.6):
    m = (f >= fc - demi) & (f <= fc + demi)
    if not m.any():
        return float("nan"), float("nan")
    i = np.argmax(A[m])
    return float(f[m][i]), float(A[m][i])


def fig_spectres():
    fig, ax = plt.subplots(2, 2, figsize=(6.6, 5.0))
    print("\n=== SPECTRES : valeurs de controle (a comparer a spectres_rapides.txt)")
    for a_, (titre, nom, courbes) in zip(ax.flat, PANNEAUX):
        a_.axvspan(6.5, 9.5, color="#d9d8d3", alpha=0.35, lw=0, zorder=0)
        a_.axvspan(9.5, 12.0, color="#d9d8d3", alpha=0.6, lw=0, zorder=0)
        for fx, txt in REPERES:
            a_.axvline(fx, color=GRIS, lw=0.6, ls=":", zorder=1)
        print(f" -- {titre}")
        for cas, t0, t1, seg, col, lab in courbes:
            r = fenetre(cas, nom, t0, t1)
            if r is None:
                continue
            t, y, dt = r
            f, A, pw = spectre(y, dt, seg)
            m = (f > 0.3) & (f < 25)
            a_.loglog(f[m], A[m], color=col, lw=1.0, label=lab, zorder=3)
            bandes = {b: float(np.sqrt(pw[(f >= lo) & (f < hi)].sum())) for b, lo, hi in BANDES}
            p = "  ".join(f"pic[{fc:g}] {pic(f, A, fc)[0]:5.2f} ({pic(f, A, fc)[1]:.1e})"
                          for fc in (7.8, 10.66, 13.9, 14.3))
            print(f"    {cas:12s} [{t[0]:.1f};{t[-1]:.1f}] seg {seg:g}  rms {np.std(y):.3e}  "
                  f"7-9 {bandes['7-9']:.3e}  10-12 {bandes['10-12']:.3e}  lacher {bandes['lacher']:.3e}")
            print(f"    {'':12s} {p}")
        a_.set_title(titre, loc="left")
        a_.set_xlim(0.3, 25)
        a_.grid(True, which="both", alpha=0.2)
    for a_ in ax[1]:
        a_.set_xlabel(r"$St = f\,c/U_\infty$")
    for a_ in ax[:, 0]:
        a_.set_ylabel("amplitude")
    # etiquettes : modes de l'attache sur (a), modes du decroche sur (c)
    for a_, liste in ((ax[0, 0], (7.8, 10.66, 13.9)), (ax[1, 0], (15.4, 17.3))):
        y0 = a_.get_ylim()[1]
        for fx, txt in REPERES:
            if fx in liste:
                a_.text(fx, y0, txt, rotation=90, va="top", ha="right", fontsize=7, color=GRIS)
    ax[0, 0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig_N4_spectra.pdf"))
    fig.savefig(os.path.join(R, "fig_N4_spectra.png"))
    print("   ecrit : fig_N4_spectra.pdf / .png")


if __name__ == "__main__":
    fig_historiques()
    fig_spectres()

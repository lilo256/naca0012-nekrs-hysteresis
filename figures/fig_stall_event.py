#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_stall_event.py -- l'evenement de decrochage vu dans le temps (section 4.3.4), a partir
                      de series_montee.json (sortie de extract_series_montee.py).

  python3 fig_stall_event.py [series_montee.json]

Produit :
  fig_upstroke_stall_event.pdf  (a) C_L(t - t_pic) : chaine a 17.25 deg et departs a froid
                                (b) exces de trainee sur le plateau, echelle log -> taux de depart
                                (c) portrait de phase C_L - C_D de la chaine a 17.25 deg
                                (d) peigne a 0.008c (vitesse tangentielle u_t) pendant l'evenement, chaine a 17.25 deg
  fig_upstroke_stall_wall.pdf   diagramme espace-temps a l'extrados pendant le decrochage du
                                depart a froid a 17 deg : (a) succion en plus du plateau,
                                -(Cp - Cp_plateau)(x, t), (b) Cf(x, t) et x_s(t)
Imprime les chiffres de la chronologie (plateau, depart, pic, effondrement, temps de
croissance) pour le texte.
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

SRC = sys.argv[1] if len(sys.argv) > 1 else "series_montee.json"
S = json.load(open(SRC))
CAS = S["cas"]

EVENTS = [  # cle, etiquette, style, debut de recherche du pic [CTU]
    ("up_a17.25", r"chain, $17.25^\circ$ ($t_0=340$)", dict(color="#0d3a73", lw=1.3), 350.0),
    ("cold_a17.00", r"cold start, $17^\circ$", dict(color="#2878d6", lw=1.1), 15.0),
    ("cold_a18.00", r"cold start, $18^\circ$", dict(color="#6ba3e6", lw=1.0, ls="--"), 10.0),
    ("cold_a19.00", r"cold start, $19^\circ$", dict(color="#8a8a82", lw=1.0, ls=":"), 5.0),
]
WALL_CASE = "cold_a17.00"
COLLAPSE = 1.10            # C_L lisse sous ce seuil = effondrement
PLATEAU_WIN = (-30.0, -20.0)   # fenetre du plateau, relative au pic [CTU]
EX_FIT = (0.004, 0.02)         # plage de l'exces de trainee (lisse) pour le temps de croissance

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "legend.fontsize": 7.8, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.edgecolor": "0.3",
    "savefig.bbox": "tight", "savefig.dpi": 300, "mathtext.fontset": "dejavusans",
})


def arr(case, grp, name):
    return np.array(CAS[case][grp][name], dtype=float)


def smooth(y, n):
    if n <= 1:
        return y
    k = np.ones(n) / n
    return np.convolve(np.pad(y, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def event(case, t_search):
    t, cl, cd = arr(case, "forces", "t"), arr(case, "forces", "CL"), arr(case, "forces", "CD")
    dt = S["dt_bin"]
    cls = smooth(cl, max(1, int(round(1.0 / dt))))
    m = (t >= t_search)
    hi = np.where(m & (cls > 1.2))[0]
    if hi.size == 0:
        return None
    lo = np.where((np.arange(t.size) > hi[0]) & (cls < COLLAPSE))[0]
    if lo.size == 0:
        return None
    t_c = t[lo[0]]
    w = (t >= t_c - 12) & (t <= t_c)
    i_pk = np.where(w)[0][np.argmax(cl[w])]
    t_pk = t[i_pk]
    wp = (t >= t_pk + PLATEAU_WIN[0]) & (t <= t_pk + PLATEAU_WIN[1])
    cl_pl, cd_pl = (float(np.mean(cl[wp])), float(np.mean(cd[wp]))) if wp.any() else (np.nan, np.nan)
    # temps de croissance : exces de trainee LISSE sur 1 CTU (les modes rapides numeriques
    # dominent le signal brut sous 0.004), ajuste entre EX_FIT[0] et EX_FIT[1], avant le pic
    ex = cd - cd_pl
    exs = smooth(ex, max(1, int(round(1.0 / dt))))
    g = (t < t_pk) & (t > t_pk - 25) & (exs > EX_FIT[0]) & (exs < EX_FIT[1])
    tau = np.nan
    if g.sum() >= 4:
        p = np.polyfit(t[g], np.log(exs[g]), 1)
        tau = 1.0 / p[0] if p[0] > 0 else np.nan
    # depart : dernier instant avant le pic ou l'exces lisse est < 0.002
    b = np.where((t < t_pk) & (exs < 0.002))[0]
    t_dep = t[b[-1]] if b.size else np.nan
    # fond de l'effondrement et etat decroche (20 derniers CTU)
    wa = (t > t_pk) & (t < t_pk + 15)
    i_min = np.where(wa)[0][np.argmin(cl[wa])] if wa.any() else i_pk
    return dict(t=t, cl=cl, cd=cd, exs=exs, t_pk=t_pk, cl_pk=cl[i_pk], cd_pk=cd[i_pk], t_c=t_c, cl_pl=cl_pl,
                cd_pl=cd_pl, tau=tau, t_dep=t_dep, t_min=t[i_min], cl_min=cl[i_min])


EV = {}
print("CHRONOLOGIE (tranches de %.2f CTU)" % S["dt_bin"])
for case, lab, sty, ts in EVENTS:
    if case not in CAS:
        print(f"  {case:12s} absent du JSON")
        continue
    e = event(case, ts)
    if e is None:
        print(f"  {case:12s} pas d'evenement detecte (C_L jamais > 1.2 puis < {COLLAPSE})")
        continue
    EV[case] = e
    st = CAS[case]["stats_last20"]
    print(f"  {case:12s} plateau [{e['t_pk'] + PLATEAU_WIN[0]:.0f},{e['t_pk'] + PLATEAU_WIN[1]:.0f}] "
          f"CL {e['cl_pl']:.4f} CD {e['cd_pl']:.5f} | depart t={e['t_dep']:.1f} | pic t={e['t_pk']:.1f} "
          f"CL {e['cl_pk']:.4f} CD {e['cd_pk']:.4f} (+{100 * (e['cl_pk'] / e['cl_pl'] - 1):.0f} % sur le plateau)"
          f" | CL<{COLLAPSE} a t={e['t_c']:.1f} | creux t={e['t_min']:.1f} CL {e['cl_min']:.3f}"
          f" | tau(C_D) = {e['tau']:.2f} CTU | final CL {st['CL']:.4f}+-{st['CL_rms']:.4f}")

# =================================================================== figure : evenement
fig = plt.figure(figsize=(7.0, 5.6))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
ax, bx, cx, dx = (fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2))
for case, lab, sty, _ in EVENTS:
    if case not in EV:
        continue
    e = EV[case]
    tt = e["t"] - e["t_pk"]
    m = (tt > -45) & (tt < 35)
    ax.plot(tt[m], e["cl"][m], label=lab, **sty)
    mm = m & (tt <= 3)
    bx.semilogy(tt[mm], np.clip(e["exs"][mm], 1e-4, None), **sty)      # lisse sur 1 CTU
ax.axvline(0, color="0.5", lw=0.6)
ax.set_xlabel(r"$t - t_{\mathrm{peak}}$ (CTU)")
ax.set_ylabel(r"$C_L$")
ax.set_xlim(-45, 35)
ax.set_title("(a)", loc="left", fontsize=9)
bx.set_xlabel(r"$t - t_{\mathrm{peak}}$ (CTU)")
bx.set_ylabel(r"$C_D - C_{D,\mathrm{plateau}}$")
bx.set_xlim(-30, 3)
bx.set_ylim(1e-3, 0.4)
bx.set_title("(b)", loc="left", fontsize=9)
if "up_a17.25" in EV and np.isfinite(EV["up_a17.25"]["tau"]):
    e = EV["up_a17.25"]
    pre = (e["t"] < e["t_pk"]) & (e["t"] > e["t_pk"] - 25)
    tp_, ep_ = (e["t"] - e["t_pk"])[pre], np.clip(e["exs"], 1e-6, None)[pre]
    k0 = np.where(ep_ < EX_FIT[0])[0]
    t_lo = tp_[k0[-1]] if k0.size else tp_[0]                  # entree dans la plage d'ajustement
    tl = np.linspace(t_lo - 4, t_lo + e["tau"] * np.log(EX_FIT[1] / EX_FIT[0]) + 2, 20)
    bx.semilogy(tl, EX_FIT[0] * np.exp((tl - t_lo) / e["tau"]), color="0.35", lw=0.9, ls="-.")
    bx.text(-28, 0.12, rf"$\propto e^{{t/\tau}}$, $\tau \approx {e['tau']:.1f}$ CTU", fontsize=7.5, color="0.3")
# (c) portrait de phase
if "up_a17.25" in EV:
    e = EV["up_a17.25"]
    tt = e["t"] - e["t_pk"]
    m = tt > -30
    sc = cx.scatter(e["cd"][m], e["cl"][m], c=tt[m], cmap="viridis", s=3, lw=0, zorder=3)
    cx.plot(e["cd"][m], e["cl"][m], color="0.75", lw=0.5, zorder=2)
    cax = cx.inset_axes([0.06, 0.15, 0.30, 0.035])                # coin vide du plan (C_D, C_L)
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label(r"$t - t_{\mathrm{peak}}$ (CTU)", fontsize=7)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=6.5)
    cx.annotate("attached\nplateau", xy=(e["cd_pl"], e["cl_pl"]), xytext=(0.09, 1.15), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="0.3"))
    cx.annotate("peak", xy=(e["cd_pk"], e["cl_pk"]), xytext=(0.17, 1.62), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="0.3"))
    st_ = CAS["up_a17.25"]["stats_last20"]
    cx.text(st_["CD"] + 0.012, st_["CL"] - 0.13, "stalled", fontsize=7.5, color="0.2")
if "cold_a17.00" in EV:
    e = EV["cold_a17.00"]
    tt = e["t"] - e["t_pk"]
    m = tt > -30
    cx.plot(e["cd"][m], e["cl"][m], color="#2878d6", lw=0.5, alpha=0.5, zorder=1)
cx.set_xlabel(r"$C_D$")
cx.set_ylabel(r"$C_L$")
cx.set_title("(c)", loc="left", fontsize=9)
# (d) peigne
if "up_a17.25" in EV and "probes" in CAS["up_a17.25"]:
    e = EV["up_a17.25"]
    tp = arr("up_a17.25", "probes", "t") - e["t_pk"]
    m = (tp > -45) & (tp < 35)
    PR = CAS["up_a17.25"]["probes"]
    for L, col in (("A050", "#9cc2f0"), ("A065", "#2878d6"), ("A080", "#0d3a73"), ("A092", "k")):
        name = f"ut_{L}" if f"ut_{L}" in PR else L            # vitesse tangentielle si disponible
        if name in PR:
            dx.plot(tp[m], arr("up_a17.25", "probes", name)[m], color=col, lw=1.0,
                    label=f"x/c = {int(L[1:]) / 100:.2f}")
    dx.axhline(0, color="0.3", lw=0.7)
    dx.axvline(0, color="0.5", lw=0.6)
    dx.set_xlim(-45, 35)
    dx.legend(loc="upper left", ncol=2, frameon=False, fontsize=7)
dx.set_xlabel(r"$t - t_{\mathrm{peak}}$ (CTU)")
dx.set_ylabel(r"$u_t/U_\infty$ at $0.008c$")
dx.set_title("(d)", loc="left", fontsize=9)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)
fig.savefig("fig_upstroke_stall_event.pdf")
fig.savefig("fig_upstroke_stall_event.png", dpi=110)
plt.close(fig)

# =================================================================== figure : paroi espace-temps
if WALL_CASE in CAS and "wall" in CAS[WALL_CASE] and WALL_CASE in EV:
    W = CAS[WALL_CASE]["wall"]
    e = EV[WALL_CASE]
    xg = np.array(S["x_wall"])
    tw = np.array(W["t"]) - e["t_pk"]
    o = np.argsort(tw)
    tw = tw[o]
    cp = np.array(W["cp_up"], dtype=float)[o]
    cf = np.array(W["cf_up"], dtype=float)[o]
    xs = np.array([np.nan if v is None else v for v in W["x_sep"]], dtype=float)[o]
    m = (tw > -25) & (tw < 35)
    # (a) variation de la succion par rapport au plateau attache : -(Cp - Cp_plateau) > 0 = plus de
    # succion. Le passage d'un tourbillon sur l'extrados y laisse une trace oblique (x croissant avec t).
    wp = (tw >= PLATEAU_WIN[0]) & (tw <= PLATEAU_WIN[1])
    if wp.sum() < 3:
        wp = (tw < -8)
    cp_pl = np.nanmean(cp[wp], axis=0)
    dsuc = -(cp - cp_pl)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    pc = ax.pcolormesh(xg, tw[m], dsuc[m], cmap="RdBu_r", norm=TwoSlopeNorm(0, -1.5, 1.5), shading="nearest")
    c1 = fig.colorbar(pc, ax=ax, pad=0.02, extend="both")
    c1.set_label(r"$-(C_p - C_{p,\mathrm{plateau}})$, suction side", fontsize=8)
    print(f"\nPAROI ({WALL_CASE}) : {m.sum()} instantanes entre -25 et +35 CTU, plateau = {wp.sum()} instantanes")
    for x_ in (0.02, 0.2, 0.4, 0.6, 0.8, 0.95):
        j = int(np.argmin(np.abs(xg - x_)))
        mm = m & (tw > -6) & (tw < 15)
        if mm.any():
            k = np.where(mm)[0][np.argmax(dsuc[mm, j])]
            print(f"   x/c = {x_:.2f} : succion max en plus {dsuc[k, j]:+.2f} a t-t_pic = {tw[k]:+.1f} CTU")
    pf = bx.pcolormesh(xg, tw[m], 1e3 * cf[m], cmap="RdBu_r", norm=TwoSlopeNorm(0, -1.5, 6.0), shading="nearest")
    c2 = fig.colorbar(pf, ax=bx, pad=0.02)
    c2.set_label(r"$C_f \times 10^3$, suction side", fontsize=8)
    bx.plot(xs[m], tw[m], "k.", ms=3)                       # x_s instantane (explique dans la legende)
    for a_ in (ax, bx):
        a_.set_xlabel(r"$x/c$")
        a_.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_ylabel(r"$t - t_{\mathrm{peak}}$ (CTU)")
    ax.set_title("(a)", loc="left", fontsize=9)
    bx.set_title("(b)", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig("fig_upstroke_stall_wall.pdf")
    fig.savefig("fig_upstroke_stall_wall.png", dpi=110)
    plt.close(fig)
    cn = np.array([np.nan if v is None else v for v in W["CN_F"]], dtype=float)
    r = np.array(W["CN_cp"], dtype=float) / cn
    tt = np.array(W["t"])
    print(f"\nCONTROLE INTEGRAL par instantane ({WALL_CASE}) : CN(Cp)/CN(efforts)")
    for lo_, hi_, lab in ((-1e9, e["t_pk"] - 5, "avant le pic"), (e["t_pk"] + 20, 1e9, "decroche")):
        mm = (tt > lo_) & (tt < hi_) & np.isfinite(r)
        if mm.any():
            cc = np.corrcoef(np.array(W["CN_cp"])[mm], cn[mm])[0, 1] if mm.sum() > 2 and np.std(cn[mm]) > 0 and np.std(np.array(W["CN_cp"])[mm]) > 0 else np.nan
            print(f"   {lab:12s} : n={mm.sum():3d}  median {np.median(r[mm]):.3f}  "
                  f"[{np.min(r[mm]):.3f}, {np.max(r[mm]):.3f}]  correlation CN(Cp)-CN(F) {cc:+.2f}")
print("\nfig_upstroke_stall_event.pdf" + (" et fig_upstroke_stall_wall.pdf" if WALL_CASE in EV else "") + " ecrits")

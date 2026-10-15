#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_upstroke_43.py -- figures de la section 4.3 (branche montante jusqu'a 18 deg)
                      a partir de montee_data.json (v2, 28/09/2026).

  python3 fig_upstroke_43.py [montee_data.json] [series_montee.json]

Le second fichier est optionnel : sortie de extract_series_montee.py (cluster). S'il est
present, les departs a froid a 18 et 19 deg sont traces avec leurs statistiques sur les
20 derniers CTU (fenetre fiable) au lieu des valeurs du dossier du 26/09.

Produit (PDF pour LaTeX + PNG d'apercu) :
  fig_upstroke_pressure.pdf   (a) Cp extrados 15 -> 17.25 deg (plateau attache)
                              (b) force normale de pression par troncon, relative a 15 deg
  fig_upstroke_friction.pdf   (a) Cf extrados, branche attachee + apres decrochage
                              (b) point de decollement x_s(alpha) jusqu'a 19 deg, etats
                                  attaches et decroches, Martin Ferrer (2025), Mittal & Saxena (2002)
  fig_upstroke_bodyaxes.pdf   (a) force normale C_N, (b) succion de bord d'attaque -C_T,
                                  avec les modeles de Kirchhoff et de Leishman-Beddoes
                                  alimentes par le point de decollement mesure
  fig_upstroke_twostates.pdf  meme incidence (17.25 deg), deux etats : (a) Cp, (b) Cf
Imprime tous les chiffres cites dans le texte (a recopier dans le dossier si besoin).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SRC = sys.argv[1] if len(sys.argv) > 1 else "montee_data.json"
SERIES = sys.argv[2] if len(sys.argv) > 2 else "series_montee.json"
D = json.load(open(SRC))
C = D["cas"]
MF = {float(k): v for k, v in D["marina"].items()}            # alpha -> [CL, CD]
TRAPZ = getattr(np, "trapezoid", None) or np.trapz

# ------------------------------------------------------------------ references
# Martin Ferrer (2025), sec. 4.2.1 : separation (C_f = 0, Fig. 4.8), upstroke ;
# "from alpha = 19 deg onwards, the flow separates right at the leading edge"
MF_SEP = [(16.0, 0.76), (17.0, 0.58), (17.5, 0.50), (18.0, 0.37), (18.5, 0.19), (19.0, 0.001)]
# Mittal & Saxena (2002), sec. 4.6, increasing incidence ; LE reached at 19 deg (stall)
MS_SEP = [(10.0, 0.95), (15.0, 0.64), (16.0, 0.52), (17.0, 0.38), (18.0, 0.25), (18.5, 0.09), (19.0, 0.0)]
# Departs a froid 18 et 19 deg : 20 derniers CTU (dossier 26/09, boucle.py). La fenetre
# [40, 80] du JSON contient encore le transitoire (derive 25 % et 16 %) : ne pas l'utiliser.
COLD_LAST20 = {18.0: dict(CL=0.8427, CD=0.2008), 19.0: dict(CL=0.8768, CD=0.2169)}

ATTACHED = [("prod_a15", "regime_permanent", 15.00), ("up_a15.25", "regime_permanent", 15.25),
            ("up_a15.50", "regime_permanent", 15.50), ("up_a15.75", "regime_permanent", 15.75),
            ("up_a16.00", "regime_permanent", 16.00), ("up_a16.25", "regime_permanent", 16.25),
            ("up_a16.50", "regime_permanent", 16.50), ("up_a16.75", "regime_permanent", 16.75),
            ("up_a17.00", "regime_permanent", 17.00)]
PLATEAU = ("up_a17.25", "plateau_attache", 17.25)
STALLED = [("up_a17.25", "decroche", 17.25), ("up_a17.50", "regime_permanent", 17.50),
           ("up_a17.75", "regime_permanent", 17.75), ("up_a18.00", "regime_permanent", 18.00)]
LOW = [("prod_a00", "regime_permanent", 0.0), ("prod_a5", "regime_permanent", 5.0),
       ("dom_a20_b30_v2", "regime_permanent", 10.0)]
COLD_ATT = [("cold_a16.50", "regime_permanent", 16.5), ("cold_a17.00", "plateau_attache", 17.0)]
COLD_ST = [("cold_a17.00", "decroche", 17.0)]

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "legend.fontsize": 8, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.edgecolor": "0.3",
    "savefig.bbox": "tight", "savefig.dpi": 300, "mathtext.fontset": "dejavusans",
})
BLUE, DARK, LIGHT = "#2878d6", "#0d3a73", "#9cc2f0"
ORANGE, GREY = "#eb6834", "#8a8a82"
SHADES = {15.00: "#9cc2f0", 15.75: "#6ba3e6", 16.50: "#2878d6", 17.00: "#1c59a8", 17.25: "#0d3a73"}


def win(case, w):
    return C[case]["fenetres"][w]


def profiles(case, w):
    p = win(case, w)["paroi"]
    cp, cf = p["profil_cp"], p["profil_cf_extrados"]
    return (np.array(cp["x_haut"]), np.array(cp["cp_haut"]), np.array(cp["x_bas"]),
            np.array(cp["cp_bas"]), np.array(cf["x"]), np.array(cf["cf"]))


def body(CL, CD, alpha):
    """C_N (normale a la corde) et -C_T (succion : force vers le bord d'attaque)."""
    a = np.radians(alpha)
    return CL * np.cos(a) + CD * np.sin(a), CL * np.sin(a) - CD * np.cos(a)


def segment_loads(case, w):
    xu, cu, xl, cl, _, _ = profiles(case, w)
    g = np.unique(np.concatenate([xu, xl]))
    g = g[(g >= max(xu.min(), xl.min())) & (g <= min(xu.max(), xl.max()))]
    dcp = np.interp(g, xl, cl) - np.interp(g, xu, cu)

    def part(a, b):
        m = (g >= a) & (g <= b)
        return float(TRAPZ(dcp[m], g[m]))
    return part(0, 0.1), part(0.1, 0.5), part(0.5, 1.0), float(TRAPZ(dcp, g))


def load_series_stats():
    """Stats des 20 derniers CTU des departs a froid (extract_series_montee.py), si dispo."""
    if not os.path.isfile(SERIES):
        return {}
    S = json.load(open(SERIES))
    out = {}
    for key, v in S.get("cas", {}).items():
        st = v.get("stats_last20")
        if st and key.startswith("cold_"):
            out[float(v["alpha"])] = st
    return out


LAST20 = load_series_stats()
for a, st in LAST20.items():
    if a in COLD_LAST20:
        COLD_LAST20[a] = dict(CL=st["CL"], CD=st["CD"], x_sep=st.get("x_sep"))

# ================================================================== numbers for the text
print("=" * 78)
print("CHARGES EN AXES DU PROFIL")
rows = []
for case, w, a in LOW + ATTACHED + [PLATEAU] + STALLED + COLD_ATT + COLD_ST:
    r = win(case, w)
    cn, s = body(r["CL"], r["CD"], a)
    xs = (r.get("paroi") or {}).get("x_sep")
    f = 1.0 if a <= 10 else xs
    rows.append(dict(case=case, w=w, a=a, CL=r["CL"], CD=r["CD"], CN=cn, S=s, f=f,
                     cpmin=(r.get("paroi") or {}).get("cp_min")))
    print(f"  {case:14s} {w:16s} a={a:5.2f}  CL={r['CL']:.4f} CD={r['CD']:.5f}  CN={cn:.4f}  -CT={s:.4f}"
          f"  x_s={xs if xs is None else round(xs, 3)}")
for a, v in COLD_LAST20.items():
    cn, s = body(v["CL"], v["CD"], a)
    print(f"  cold {a:4.1f} (20 derniers CTU)            CL={v['CL']:.4f} CD={v['CD']:.4f}  CN={cn:.4f}  -CT={s:.4f}")
print("  Martin Ferrer (2025), montee :")
for a in sorted(MF):
    CL_, CD_ = MF[a]
    if CD_ is not None:
        cn, s = body(CL_, CD_, a)
        print(f"     a={a:5.2f}  CL={CL_:.3f} CD={CD_:.4f}  CN={cn:.4f}  -CT={s:.4f}")

# ---- Kirchhoff (C_N = a ((1+sqrt f)/2)^2 alpha) and Leishman-Beddoes (C_C = eta a alpha^2 sqrt f)
FIT_SET = [(c, w) for c, w, _ in LOW[1:] + ATTACHED + [PLATEAU]]     # 5, 10, 15 -> 17.25 plateau
fit = [r for r in rows if (r["case"], r["w"]) in FIT_SET]
al = np.radians([r["a"] for r in fit])
K = np.array([((1 + np.sqrt(r["f"])) / 2) ** 2 for r in fit])
CN = np.array([r["CN"] for r in fit])
CLf = np.array([r["CL"] for r in fit])
S = np.array([r["S"] for r in fit])
A_N = float(np.sum(CN * al * K) / np.sum((al * K) ** 2))
A_L = float(np.sum(CLf * al * K) / np.sum((al * K) ** 2))
eN = 100 * (A_N * al * K / CN - 1)
eL = 100 * (A_L * al * K / CLf - 1)
m10 = np.degrees(al) >= 10 - 1e-9
XC = A_N * al ** 2 * np.sqrt([r["f"] for r in fit])
ETA = float(np.sum(S[m10] * XC[m10]) / np.sum(XC[m10] ** 2))
eta_i = S / XC
print("\nMODELES (branche attachee 5 -> 17.25 plateau, f = x_s mesure, f = 1 a 5 et 10 deg)")
print(f"  pente lineaire mesuree 0-5 deg : CL/alpha = {rows[1]['CL'] / np.radians(5):.3f} /rad")
print(f"  Kirchhoff sur C_N : a = {A_N:.3f} /rad ; ecart max {np.max(np.abs(eN)):.1f} %, rms "
      f"{np.sqrt(np.mean(eN ** 2)):.1f} %")
print(f"  Kirchhoff sur C_L : a = {A_L:.3f} /rad ; ecart max {np.max(np.abs(eL)):.1f} %, rms "
      f"{np.sqrt(np.mean(eL ** 2)):.1f} %")
print("     ecarts C_N par angle : " + " ".join(f"{r['a']:.2f}:{e:+.1f}" for r, e in zip(fit, eN)))
print(f"  Leishman-Beddoes, eta (10 -> 17.25) = {ETA:.3f} ; eta par angle : " +
      " ".join(f"{r['a']:.2f}:{e:.3f}" for r, e in zip(fit, eta_i)))
i15 = [i for i, r in enumerate(fit) if abs(r["a"] - 15) < 1e-6][0]
ipl = [i for i, r in enumerate(fit) if r["case"] == "up_a17.25"][0]
mod = A_L * al * K
print(f"  perte de C_L 15 -> 17.25 plateau : modele {100 * (mod[ipl] / mod[i15] - 1):+.1f} %, calcul "
      f"{100 * (CLf[ipl] / CLf[i15] - 1):+.1f} %")
Kneed = CLf[ipl] / (A_L * al[ipl])
print(f"  f necessaire pour le plateau (C_L, a = {A_L:.3f}) : {(2 * np.sqrt(Kneed) - 1) ** 2:.3f} "
      f"(mesure {fit[ipl]['f']:.3f})")
print("  etats decroches, modeles alimentes par leur propre x_s :")
for r in rows:
    if (r["case"], r["w"]) in [(c, w) for c, w, _ in STALLED + COLD_ST]:
        a_ = np.radians(r["a"])
        k_ = ((1 + np.sqrt(r["f"])) / 2) ** 2
        cnm, sm = A_N * a_ * k_, ETA * A_N * a_ ** 2 * np.sqrt(r["f"])
        need = (2 * np.sqrt(r["CN"] / (A_N * a_)) - 1) ** 2
        need_s = (r["S"] / (ETA * A_N * a_ ** 2)) ** 2
        print(f"     {r['case']:12s} a={r['a']:5.2f} f={r['f']:.3f}  C_N mod {cnm:.3f} / calc {r['CN']:.3f}"
              f" ({100 * (cnm / r['CN'] - 1):+.0f} %)   -C_T mod {sm:.3f} / calc {r['S']:.3f}"
              f" (x{sm / r['S']:.1f})   f equivalent : C_N {need:.3f}, -C_T {need_s:.3f}")

pl = [r for r in rows if r["case"] == "up_a17.25" and r["w"] == "plateau_attache"][0]
st = [r for r in rows if r["case"] == "up_a17.25" and r["w"] == "decroche"][0]
print(f"\nDECROCHAGE A 17.25 : C_N {pl['CN']:.3f} -> {st['CN']:.3f} ({100 * (st['CN'] / pl['CN'] - 1):+.0f} %),"
      f" -C_T {pl['S']:.3f} -> {st['S']:.3f} ({100 * (st['S'] / pl['S'] - 1):+.0f} %)")
dCL = st["CL"] - pl["CL"]
a_ = np.radians(17.25)
share_N = (st["CN"] - pl["CN"]) * np.cos(a_) / dCL
print(f"  part de la perte de portance venant de C_N : {100 * share_N:.0f} %")
imax = int(np.argmax([r["S"] for r in rows if r["w"] != "decroche" and r["a"] <= 17.25 and r["case"] != "cold_a17.00"]))
print(f"  -C_T maximal sur la branche attachee : {max(r['S'] for r in rows if r['a'] <= 17.25 and r['w'] != 'decroche'):.4f}")
print(f"  C_N maximal : {max(r['CN'] for r in rows if r['a'] <= 17.25 and r['w'] != 'decroche'):.4f}")

print("\nPOINT D'ARRET (max du Cp intrados) ET CP_MIN")
for case, w, a in [("prod_a15", "regime_permanent", 15.0), PLATEAU, ("up_a17.25", "decroche", 17.25),
                   ("up_a18.00", "regime_permanent", 18.0)]:
    xu, cu, xl, cl, xf, cf = profiles(case, w)
    i = int(np.argmax(cl))
    j = np.argmin(np.where((xu > 0.4) & (xu < 0.9), cu, 9))
    print(f"  {case:12s} {w:16s} arret x/c={xl[i]:.4f} (Cp {cl[i]:.3f})  Cp_min {win(case, w)['paroi']['cp_min']:.2f}"
          f"  creux arriere Cp={cu[j]:.3f} a x/c={xu[j]:.2f}  Cf(0.01)={np.interp(0.01, xf, cf):.4f}")

xs_att = np.array([win(c, w)["paroi"]["x_sep"] for c, w, _ in ATTACHED + [PLATEAU]])
al_att = np.array([a for _, _, a in ATTACHED + [PLATEAU]])
p = np.polyfit(al_att, xs_att, 1)
print(f"\nDECOLLEMENT : pente {p[0]:+.3f} c/deg (15.00 -> 17.25 plateau) ; decroches : " +
      " ".join(f"{a:.2f}:{win(c, w)['paroi']['x_sep']:.3f}" for c, w, a in STALLED + COLD_ST))

# ================================================================== figure: pressure (inchangee)
CHECK_TOL = 1.0
ok = []
for case, w, a in ATTACHED + [PLATEAU]:
    tot = segment_loads(case, w)[3]
    r = win(case, w)
    cnf = body(r["CL"], r["CD"], a)[0]
    if abs(100 * (tot / cnf - 1)) <= CHECK_TOL:
        ok.append((case, w, a))
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
for case, w, a in ATTACHED + [PLATEAU]:
    if a not in SHADES:
        continue
    xu, cu, _, _, _, _ = profiles(case, w)
    m = (xu >= 0.1) & (xu <= 0.99)
    lab = rf"{a:.2f}$^\circ$" + (", attached plateau" if a == 17.25 else "")
    ax.plot(xu[m], cu[m], color=SHADES[a], lw=1.3, label=lab)
ax.invert_yaxis()
ax.set_xlabel(r"$x/c$")
ax.set_ylabel(r"$C_p$, suction side")
ax.set_xlim(0.1, 1.0)
ax.set_title("(a)", loc="left", fontsize=9)
al_ok = np.array([a for _, _, a in ok])
seg = np.array([segment_loads(c, w) for c, w, _ in ok])
d = seg - seg[0]
bx.axhline(0, color="0.3", lw=0.7)
bx.plot(al_ok, d[:, 0], "o-", color="#6ba3e6", ms=3.5, lw=1.1, label=r"nose, $x/c<0.1$")
bx.plot(al_ok, d[:, 1], "s-", color=DARK, ms=3.5, lw=1.1, label=r"mid-chord, $0.1<x/c<0.5$")
bx.plot(al_ok, d[:, 2], "^-", color=ORANGE, ms=3.8, lw=1.1, label=r"rear, $x/c>0.5$")
bx.plot(al_ok, d[:, 3], "-", color="k", lw=1.6, label="whole section")
bx.set_xlabel(r"angle of attack $\alpha$ (deg)")
bx.set_ylabel("change in normal-force\ncontribution since $15^\\circ$")
bx.set_title("(b)", loc="left", fontsize=9)
bx.set_xlim(14.9, 17.35)
h1, l1 = ax.get_legend_handles_labels()
fig.legend(h1, l1, loc="upper center", bbox_to_anchor=(0.29, 0.0), ncol=2, frameon=False)
h2, l2 = bx.get_legend_handles_labels()
fig.legend(h2, l2, loc="upper center", bbox_to_anchor=(0.77, 0.0), ncol=2, frameon=False)
fig.tight_layout()
fig.savefig("fig_upstroke_pressure.pdf")
fig.savefig("fig_upstroke_pressure.png", dpi=110)
plt.close(fig)
print("\nsegments (nez, milieu, arriere, total) relatifs a 15 deg :")
for (c, w, a), dd in zip(ok, d):
    print(f"   {a:5.2f}  " + "  ".join(f"{v:+.4f}" for v in dd))

# ================================================================== figure: friction + separation
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
for case, w, a in ATTACHED + [PLATEAU]:
    if a not in SHADES:
        continue
    _, _, _, _, xf, cf = profiles(case, w)
    m = (xf >= 0.3) & (xf <= 0.97)
    lab = rf"{a:.2f}$^\circ$" + (", attached plateau" if a == 17.25 else "")
    ax.plot(xf[m], cf[m] * 1e3, color=SHADES[a], lw=1.3, label=lab)
_, _, _, _, xf, cf = profiles("up_a17.25", "decroche")
m = (xf >= 0.3) & (xf <= 0.97)
ax.plot(xf[m], cf[m] * 1e3, "k--", lw=1.2, label=r"17.25$^\circ$, stalled")
ax.axhline(0, color="0.3", lw=0.7)
ax.set_xlabel(r"$x/c$")
ax.set_ylabel(r"mean $C_f \times 10^{3}$, suction side")
ax.set_xlim(0.3, 1.0)
ax.set_title("(a)", loc="left", fontsize=9)

for (c, w, a) in ATTACHED + [PLATEAU]:
    xr = (win(c, w).get("peigne") or {}).get("x_sep")
    if xr:
        bx.plot([a, a], xr, color=LIGHT, lw=3.2, solid_capstyle="butt", zorder=1)
bx.plot([10.0], [1.0], "o", mfc="white", mec=BLUE, ms=6, mew=1.4, zorder=3)
bx.plot(al_att[:-1], xs_att[:-1], "o-", color=BLUE, ms=4.5, lw=1.3, zorder=3)
bx.plot(al_att[-2:], xs_att[-2:], ":", color=BLUE, lw=1.1, zorder=3)
bx.plot([17.25], [xs_att[-1]], "o", mfc="white", mec=BLUE, ms=5.5, mew=1.3, zorder=4)
al_st = np.array([a for _, _, a in STALLED])
xs_st = np.array([win(c, w)["paroi"]["x_sep"] for c, w, _ in STALLED])
bx.plot(al_st, xs_st, "s-", color=DARK, ms=4.2, lw=1.2, zorder=3)
for c, w, a in COLD_ATT:
    bx.plot([a + 0.07], [win(c, w)["paroi"]["x_sep"]], "^", mfc="white", mec=BLUE, ms=6, mew=1.3, zorder=4)
for c, w, a in COLD_ST:
    bx.plot([a + 0.07], [win(c, w)["paroi"]["x_sep"]], "^", mfc="white", mec=DARK, ms=6, mew=1.3, zorder=4)
for a, v in COLD_LAST20.items():
    if v.get("x_sep") is not None:
        bx.plot([a + 0.07], [v["x_sep"]], "^", mfc="white", mec=DARK, ms=6, mew=1.3, zorder=4)
bx.plot(*zip(*MF_SEP), "D--", color=ORANGE, ms=4, lw=1.1, zorder=2)
bx.plot(*zip(*MS_SEP), "s:", color=GREY, mfc="white", ms=4.5, lw=1.0, zorder=2)
bx.annotate("stall", xy=(17.12, 0.41), xytext=(17.6, 0.62), fontsize=7.5, color=DARK,
            arrowprops=dict(arrowstyle="->", color=DARK, lw=0.8))
bx.set_xlabel(r"angle of attack $\alpha$ (deg)")
bx.set_ylabel(r"separation point $x_s/c$")
bx.set_xlim(9.5, 19.4)
bx.set_ylim(-0.03, 1.05)
bx.set_title("(b)", loc="left", fontsize=9)
h1, l1 = ax.get_legend_handles_labels()
fig.legend(h1, l1, loc="upper center", bbox_to_anchor=(0.29, 0.0), ncol=2, frameon=False)
h2 = [Line2D([], [], color=BLUE, marker="o", ms=4.5, lw=1.3),
      Line2D([], [], color=DARK, marker="s", ms=4.2, lw=1.2),
      Line2D([], [], color=BLUE, marker="^", mfc="white", ms=6, lw=0),
      Line2D([], [], color=LIGHT, lw=3.2),
      Line2D([], [], color=ORANGE, marker="D", ms=4, ls="--", lw=1.1),
      Line2D([], [], color=GREY, marker="s", mfc="white", ms=4.5, ls=":", lw=1.0)]
l2 = [r"this study, attached ($C_f=0$ at the wall)", "this study, stalled",
      "this study, cold starts", r"first mean reverse flow at $0.008c$",
      "Martin Ferrer (2025), upstroke", "Mittal & Saxena (2002), upstroke"]
fig.legend(h2, l2, loc="upper center", bbox_to_anchor=(0.77, 0.0), ncol=1, frameon=False)
fig.tight_layout()
fig.savefig("fig_upstroke_friction.pdf")
fig.savefig("fig_upstroke_friction.png", dpi=110)
plt.close(fig)

# ================================================================== figure: body axes
def model_pts(rs):
    out = []
    for r in rs:
        a_ = np.radians(r["a"])
        k_ = ((1 + np.sqrt(r["f"])) / 2) ** 2
        out.append((r["a"], A_N * a_ * k_, ETA * A_N * a_ ** 2 * np.sqrt(r["f"])))
    return np.array(out)


att_rows = [r for r in rows if r["case"] in [c for c, _, _ in ATTACHED]]
pl_row = [pl]
st_rows = [r for r in rows if (r["case"], r["w"]) in [(c, w) for c, w, _ in STALLED]]
cold_att = [r for r in rows if (r["case"], r["w"]) in [(c, w) for c, w, _ in COLD_ATT]]
cold_st = [r for r in rows if (r["case"], r["w"]) in [(c, w) for c, w, _ in COLD_ST]]
M_att = model_pts(att_rows + pl_row)
M_st = model_pts(st_rows)

fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
for k, (ax, key, lab) in enumerate(zip(axs, ("CN", "S"),
                                       (r"normal force $C_N$", r"leading-edge suction $-C_T$"))):
    a1 = [r["a"] for r in att_rows]
    ax.plot(a1, [r[key] for r in att_rows], "o-", color=BLUE, ms=4.2, lw=1.3, zorder=4)
    ax.plot([a1[-1], 17.25], [att_rows[-1][key], pl[key]], ":", color=BLUE, lw=1.1, zorder=3)
    ax.plot([17.25], [pl[key]], "o", mfc="white", mec=BLUE, mew=1.3, ms=5.5, zorder=5)
    ax.plot([r["a"] for r in st_rows], [r[key] for r in st_rows], "s-", color=DARK, ms=4.0, lw=1.2, zorder=4)
    ax.plot([17.0, 17.25], [att_rows[-1][key], st_rows[0][key]], "--", color=DARK, lw=0.8, zorder=2)
    for r in cold_att:
        ax.plot([r["a"] + 0.07], [r[key]], "^", mfc="white", mec=BLUE, ms=6, mew=1.2, zorder=5)
    for r in cold_st:
        ax.plot([r["a"] + 0.07], [r[key]], "^", mfc="white", mec=DARK, ms=6, mew=1.2, zorder=5)
    for a, v in COLD_LAST20.items():
        cn, s = body(v["CL"], v["CD"], a)
        ax.plot([a + 0.07], [cn if key == "CN" else s], "^", mfc="white", mec=DARK, ms=6, mew=1.2, zorder=5)
    mfa = [a for a in sorted(MF) if 14.9 <= a <= 19.1 and MF[a][1] is not None]
    mfv = [body(MF[a][0], MF[a][1], a)[0 if key == "CN" else 1] for a in mfa]
    ax.plot(mfa, mfv, "D--", color=ORANGE, ms=3.8, lw=1.0, zorder=2)
    j = 1 if key == "CN" else 2
    ax.plot(M_att[:, 0], M_att[:, j], "-", color="0.45", lw=0.9, zorder=1)
    ax.plot(M_att[:, 0], M_att[:, j], "x", color="0.2", ms=4.5, mew=1.0, zorder=6)
    ax.plot(M_st[:, 0], M_st[:, j], "x", color="0.2", ms=4.5, mew=1.0, zorder=6)
    ax.set_xlim(14.75, 19.35)
    ax.set_xlabel(r"angle of attack $\alpha$ (deg)")
    ax.set_ylabel(lab)
    ax.set_title("(a)" if k == 0 else "(b)", loc="left", fontsize=9)
axs[0].set_ylim(0.7, 1.5)
axs[1].set_ylim(0.0, 0.40)
axs[0].annotate("model, $f = x_s \\approx 0.34$", xy=(17.55, M_st[1, 1]), xytext=(17.9, 1.32), fontsize=7.5,
                color="0.2", arrowprops=dict(arrowstyle="->", color="0.3", lw=0.7))
axs[1].annotate("model, $f = x_s \\approx 0.34$", xy=(17.55, M_st[1, 2]), xytext=(17.75, 0.36), fontsize=7.5,
                color="0.2", arrowprops=dict(arrowstyle="->", color="0.3", lw=0.7))
hh = [Line2D([], [], color=BLUE, marker="o", ms=4.2, lw=1.3),
      Line2D([], [], color=BLUE, marker="o", mfc="white", ms=5.5, lw=0),
      Line2D([], [], color=DARK, marker="s", ms=4.0, lw=1.2),
      Line2D([], [], color=DARK, marker="^", mfc="white", ms=6, lw=0),
      Line2D([], [], color=ORANGE, marker="D", ms=3.8, ls="--", lw=1.0),
      Line2D([], [], color="0.2", marker="x", ms=4.5, lw=0.9, mew=1.0)]
ll = ["this study, attached (chained)", r"attached plateau at $17.25^\circ$", "this study, stalled (chained)",
      "cold starts", "Martin Ferrer (2025), upstroke",
      "trailing-edge separation model fed with the measured $x_s$"]
fig.legend(hh, ll, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
fig.tight_layout()
fig.savefig("fig_upstroke_bodyaxes.pdf")
fig.savefig("fig_upstroke_bodyaxes.png", dpi=110)
plt.close(fig)

# ================================================================== figure: two states at 17.25
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(width_ratios=[1, 1]))
sty = [(PLATEAU, dict(color=DARK, lw=1.4, ls="-"), r"17.25$^\circ$, attached plateau"),
       (("up_a17.25", "decroche", 17.25), dict(color="k", lw=1.2, ls="--"), r"17.25$^\circ$, stalled"),
       (("up_a18.00", "regime_permanent", 18.0), dict(color=GREY, lw=1.1, ls=":"), r"18.00$^\circ$, stalled")]
for (case, w, a), kw, lab in sty:
    xu, cu, xl, cl, xf, cf = profiles(case, w)
    ax.plot(xu, cu, label=lab, **kw)
    ax.plot(xl, cl, **kw)
    m = (xf >= 0.1) & (xf <= 0.97)
    bx.plot(xf[m], cf[m] * 1e3, **kw)
    xs = win(case, w)["paroi"]["x_sep"]
    bx.plot([xs], [0], "o", ms=4, mfc="white", mec=kw["color"], mew=1.1, zorder=5)
ax.invert_yaxis()
ax.set_xlim(-0.01, 1.0)
ax.set_xlabel(r"$x/c$")
ax.set_ylabel(r"mean $C_p$")
ax.set_title("(a)", loc="left", fontsize=9)
ax.annotate("suction peak\n$-10.2 \\rightarrow -3.2$", xy=(0.012, -3.3), xytext=(0.18, -7.2), fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="0.3", lw=0.7))
bx.axhline(0, color="0.3", lw=0.7)
bx.set_xlim(0.1, 1.0)
bx.set_ylim(-1.3, 6.5)
bx.set_xlabel(r"$x/c$")
bx.set_ylabel(r"mean $C_f \times 10^{3}$, suction side")
bx.set_title("(b)", loc="left", fontsize=9)
h1, l1 = ax.get_legend_handles_labels()
fig.legend(h1, l1, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
fig.tight_layout()
fig.savefig("fig_upstroke_twostates.pdf")
fig.savefig("fig_upstroke_twostates.png", dpi=110)
plt.close(fig)
print("\nfig_upstroke_pressure / friction / bodyaxes / twostates .pdf ecrits")

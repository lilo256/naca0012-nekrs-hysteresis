#!/usr/bin/env python3
"""
fig_polaire_montee.py  --  figures de la branche montante (rapport, section 4.3)
===============================================================================
Lit montee_data.json (produit par extract_montee.py sur le cluster) et trace :
  fig_polar_full.pdf/.png         CL(alpha) et CD(alpha) de 0 a 20 deg
  fig_upstroke_zoom.pdf/.png      zoom 14.75-18.25 deg sur la montee
  fig_upstroke_separation.pdf/.png  (si le JSON contient les donnees de paroi)
        (a) Cf moyen a l'extrados, (b) point de decollement en fonction de alpha

Barres d'erreur = rms du regime permanent (demande de F. Mellibovsky).
Superpose Martin Ferrer (2025) et, s'ils sont presents dans ./tmr/, les
donnees experimentales de la NASA TMR (a telecharger une fois) :
  https://tmbwg.github.io/turbmodels/NACA0012_validation/CLCD_Ladson_expdata.dat
  https://tmbwg.github.io/turbmodels/NACA0012_validation/CL_Gregory_expdata.dat
  https://tmbwg.github.io/turbmodels/NACA0012_validation/0012.abbottdata.cl.dat
  https://tmbwg.github.io/turbmodels/NACA0012_validation/0012.mccroskeydata.cl.dat
ATTENTION : ces essais sont a Re = 3 a 6 x 10^6 (contre 10^6 ici). Ils servent
de reference pour la pente et la tendance, pas pour CLmax ni l'angle de
decrochage, qui augmentent avec Re.

Les cas encore EN COURS (marques par extract_montee.py) sont ecartes ;
option --tout pour les inclure quand meme.

USAGE :  python3 fig_polaire_montee.py [montee_data.json] [--tout]
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
TOUT = "--tout" in sys.argv
SRC = ARGS[0] if ARGS else "montee_data.json"
if not os.path.isfile(SRC):
    sys.exit(f"{SRC} introuvable : lancer extract_montee.py sur le cluster puis\n"
             f"  rsync -avz USER@cdf:~/cdf_runs/montee_data.json .")
D = json.load(open(SRC))

# palette validee (2 teintes, toutes paires) + gris neutre pour les references
BLEU, ORANGE, GRIS, ENCRE, ENCRE2 = "#2a78d6", "#eb6834", "#8a8983", "#0b0b0b", "#52514e"
RAMPE = ["#9cc3f0", "#6aa3e6", "#2a78d6", "#1b5aa8", "#0d3a73"]   # sequentielle (angles croissants)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": ENCRE2, "axes.labelcolor": ENCRE,
                     "xtick.color": ENCRE2, "ytick.color": ENCRE2, "axes.linewidth": 0.8,
                     "legend.frameon": False, "savefig.dpi": 200})

# --- cas en cours ---------------------------------------------------------------
CAS = {}
for nom, ent in D["cas"].items():
    if ent.get("en_cours") and not TOUT:
        s = ent["segment"]
        print(f" (ecarte, en cours : {nom}  t = {s[1]:.1f} / {ent.get('t_fin_prevue')})")
        continue
    CAS[nom] = ent


def point(ent, pref=("regime_permanent", "decroche")):
    for lab in pref:
        if lab in ent["fenetres"]:
            return ent["fenetres"][lab]
    return None


# --- series de cette etude --------------------------------------------------
polaire, chaine, froid, meta = [], [], [], []
for nom, ent in CAS.items():
    a, role = ent["alpha"], ent["role"]
    if role == "polaire":
        p = point(ent)
        if p: polaire.append((a, p))
    elif role in ("chaine", "chaine_decroche"):
        p = point(ent)
        if p: chaine.append((a, p))
        if "plateau_attache" in ent["fenetres"]:
            meta.append((a, ent["fenetres"]["plateau_attache"]))
    elif role == "froid":
        for lab, p in ent["fenetres"].items():
            froid.append((a, p, lab))
chaine.sort(key=lambda x: x[0]); polaire.sort(key=lambda x: x[0])
montee = polaire + chaine                     # trajet complet 0 -> 18.00

# --- Martin Ferrer (2025) ----------------------------------------------------
MF = sorted((float(k), v) for k, v in D["marina"].items())


# --- donnees experimentales NASA TMR (optionnelles) --------------------------
def lire_dat(p):
    rows = []
    for ligne in open(p, errors="ignore"):
        s = ligne.replace(",", " ").split()
        try:
            v = [float(x) for x in s]
        except ValueError:
            continue
        if len(v) >= 2:
            rows.append(v[:3])
    return rows

EXP = []   # (etiquette, marqueur, rows, trace_en_ligne, fichier)
for f, lab, mk, ligne in [("CLCD_Ladson_expdata.dat", r"Ladson (1988), $Re=6\times10^6$", "o", False),
                          ("CL_Gregory_expdata.dat", r"Gregory & O'Reilly, $Re=3\times10^6$", "s", False),
                          ("0012.abbottdata.cl.dat", r"Abbott & von Doenhoff", "^", False),
                          ("0012.mccroskeydata.cl.dat", r"McCroskey best fit", None, True)]:
    p = os.path.join("tmr", f)
    if os.path.isfile(p):
        EXP.append((lab, mk, lire_dat(p), ligne, f))


def tracer(ax, champ, xlim, zoom):
    cle, crms = champ, champ + "_rms"
    # references experimentales (recessives)
    for lab, mk, rows, ligne, f in EXP:
        col = 1 if champ == "CL" else 2
        pts = [(r[0], r[col]) for r in rows if len(r) > col]
        if champ == "CD" and not f.startswith("CLCD"):
            continue
        pts = [q for q in pts if xlim[0] - 0.5 <= q[0] <= xlim[1] + 0.5]
        if not pts:
            continue
        x, y = zip(*pts)
        if ligne:
            ax.plot(x, y, "--", color=GRIS, lw=1.0, label=lab, zorder=1)
        else:
            ax.plot(x, y, mk, ms=4, mfc="none", mec=GRIS, mew=0.9, ls="none", label=lab, zorder=1)
    # Martin Ferrer (2025)
    i = 0 if champ == "CL" else 1
    mf = [(a, v[i]) for a, v in MF if v[i] is not None and xlim[0] - 0.5 <= a <= xlim[1] + 0.5]
    if mf:
        x, y = zip(*mf)
        ax.plot(x, y, "D--", color=ORANGE, ms=5, lw=1.1, mec="white", mew=0.6,
                label="Martin Ferrer (2025), upstroke", zorder=3)
    # cette etude : trajet de montee
    m = [(a, p) for a, p in montee if xlim[0] - 0.5 <= a <= xlim[1] + 0.5]
    if m:
        x = [a for a, _ in m]; y = [p[cle] for _, p in m]; e = [p[crms] for _, p in m]
        ax.errorbar(x, y, yerr=e, fmt="o-", color=BLEU, ms=5, lw=1.5, capsize=2.5,
                    elinewidth=1.0, mec="white", mew=0.6, label="This study, chained upstroke", zorder=5)
    # etat attache metastable (17.25) et departs a froid
    for a, p in meta:
        if xlim[0] <= a <= xlim[1]:
            ax.errorbar([a], [p[cle]], yerr=[p[crms]], fmt="o", mfc="white", mec=BLEU, mew=1.3,
                        ms=6, color=BLEU, capsize=2.5, elinewidth=1.0,
                        label="This study, metastable attached plateau", zorder=6)
            if zoom and champ == "CL":
                ax.annotate("held ~25 CTU,\nthen stalls", (a, p[cle]), xytext=(a + 0.18, p[cle] + 0.02),
                            fontsize=8, color=ENCRE2, arrowprops=dict(arrowstyle="-", color=ENCRE2, lw=0.7))
    premier = True
    for a, p, lab in froid:
        if xlim[0] <= a <= xlim[1]:
            ax.errorbar([a], [p[cle]], yerr=[p[crms]], fmt="^", mfc="white", mec=BLEU, mew=1.3,
                        ms=6, color=BLEU, capsize=2.5, elinewidth=1.0,
                        label="This study, cold starts" if premier else None, zorder=6)
            premier = False
    ax.set_xlim(*xlim)
    ax.set_xlabel(r"angle of attack $\alpha$ (deg)")
    ax.set_ylabel(r"$C_L$" if champ == "CL" else r"$C_D$")
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def legende_commune(fig, axs, rect_bas=0.13):
    h, l = [], []
    for ax in axs:
        hh, ll = ax.get_legend_handles_labels()
        h += hh; l += ll
    vu = {}
    for hh, ll in zip(h, l):
        vu.setdefault(ll, hh)
    fig.legend(vu.values(), vu.keys(), loc="lower center", ncol=2, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, rect_bas + 0.035 * max(0, (len(vu) - 4) // 2), 1, 1))


def sauver(fig, nom):
    for ext in ("pdf", "png"):
        fig.savefig(f"{nom}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f" -> {nom}.pdf / .png")


def figure(nom, xlim, zoom):
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.1))
    tracer(axs[0], "CL", xlim, zoom); tracer(axs[1], "CD", xlim, zoom)
    for ax, t in zip(axs, "ab"):
        ax.text(0.02, 0.97, f"({t})", transform=ax.transAxes, va="top", fontsize=9, color=ENCRE)
    legende_commune(fig, axs[:1])
    sauver(fig, nom)


figure("fig_polar_full", (-0.5, 20.5), zoom=False)
figure("fig_upstroke_zoom", (14.75, 18.25), zoom=True)


# --- decollement (si donnees de paroi) --------------------------------------------
def paroi_de(nom, lab):
    ent = CAS.get(nom)
    if not ent or lab not in ent["fenetres"]:
        return None
    return ent["fenetres"][lab].get("paroi")


# Mittal & Saxena (2002), Re = 1e6, Baldwin-Lomax : point de decollement a l'extrados
MS02 = [(5.0, 0.99), (10.0, 0.95), (15.0, 0.64)]
# Martin Ferrer (2025), montee, Cf = 0 (memoire, 4.2.1) ; bord d'attaque des 19 deg
MF_SEP = [(16.0, 0.76), (17.0, 0.58), (17.5, 0.50), (18.0, 0.37), (18.5, 0.19)]

COURBES = [("prod_a15", "regime_permanent", r"$15.00^\circ$"),
           ("up_a15.75", "regime_permanent", r"$15.75^\circ$"),
           ("up_a16.50", "regime_permanent", r"$16.50^\circ$"),
           ("up_a17.00", "regime_permanent", r"$17.00^\circ$"),
           ("up_a17.25", "plateau_attache", r"$17.25^\circ$, attached plateau")]
DECR = ("up_a17.25", "decroche", r"$17.25^\circ$, after stall")

if any(paroi_de(n, l) and "profil_cf_extrados" in paroi_de(n, l) for n, l, _ in COURBES):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.1))
    # (a) Cf moyen a l'extrados
    ymin, ymax = 0.0, 0.0
    k = 0
    for (n, l, lab) in COURBES + [DECR]:
        p = paroi_de(n, l)
        if not p or "profil_cf_extrados" not in p:
            continue
        x = np.array(p["profil_cf_extrados"]["x"]); cf = np.array(p["profil_cf_extrados"]["cf"])
        m = x >= 0.3
        if (n, l, lab) == DECR:
            a1.plot(x[m], cf[m], "--", color=ENCRE, lw=1.3, label=lab, zorder=4)   # orange = Martin Ferrer
        else:
            a1.plot(x[m], cf[m], "-", color=RAMPE[min(k, len(RAMPE) - 1)], lw=1.4, label=lab, zorder=3)
            k += 1
        ymin, ymax = min(ymin, cf[m].min()), max(ymax, cf[m].max())
    a1.axhline(0, color=ENCRE2, lw=0.7)
    pad = 0.08 * (ymax - ymin)
    a1.set_ylim(ymin - pad, ymax + pad)
    a1.set_xlim(0.3, 1.0)
    a1.set_xlabel(r"$x/c$"); a1.set_ylabel(r"mean $C_f$, suction side")
    # (b) point de decollement vs alpha
    xs_cf, xs_att, peig = [], [], []
    for nom, ent in CAS.items():
        if ent["role"] not in ("polaire", "chaine"):
            continue
        lab = "plateau_attache" if "plateau_attache" in ent["fenetres"] else "regime_permanent"
        st = ent["fenetres"].get(lab)
        if not st:
            continue
        p, g = st.get("paroi"), st.get("peigne")
        if p and p.get("n", 0) >= 1:
            if p.get("x_sep") is not None:
                xs_cf.append((ent["alpha"], p["x_sep"]))
            elif "profil_cf_extrados" in p:
                xs_att.append(ent["alpha"])
        if g and g.get("x_sep") and g["x_sep"][0] is not None:
            peig.append((ent["alpha"], g["x_sep"][0], g["x_sep"][1]))
    if xs_cf:
        xs_cf.sort()
        a2.plot(*zip(*xs_cf), "o-", color=BLEU, ms=5, lw=1.4, mec="white", mew=0.6,
                label=r"This study, $C_f = 0$ (wall)", zorder=5)
    if xs_att:
        a2.plot(xs_att, [1.0] * len(xs_att), "o", mfc="white", mec=BLEU, mew=1.2, ms=5,
                label="This study, attached to the trailing edge", zorder=5)
    for i, (a, x0, x1) in enumerate(sorted(peig)):
        a2.plot([a + 0.09, a + 0.09], [x0, x1], "-", color=RAMPE[0], lw=2.2, solid_capstyle="butt",
                label=r"This study, first reverse flow at $0.008c$" if i == 0 else None, zorder=4)
    a2.plot(*zip(*MF_SEP), "D--", color=ORANGE, ms=5, lw=1.1, mec="white", mew=0.6,
            label="Martin Ferrer (2025), upstroke", zorder=3)
    a2.plot(*zip(*MS02), "s", mfc="none", mec=GRIS, mew=1.0, ms=5,
            label="Mittal & Saxena (2002)", zorder=2)
    a2.set_ylim(0.0, 1.05); a2.set_xlim(9.5, 19.0)
    a2.set_xlabel(r"angle of attack $\alpha$ (deg)"); a2.set_ylabel(r"separation point $x_s/c$")
    for ax, t in zip((a1, a2), "ab"):
        ax.text(0.0, 1.03, f"({t})", transform=ax.transAxes, va="bottom", fontsize=9, color=ENCRE)
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    legende_commune(fig, (a1, a2), rect_bas=0.18)
    sauver(fig, "fig_upstroke_separation")
    # le point du stalle, en clair (pas trace en (b) : branche differente)
    p = paroi_de(*DECR[:2])
    if p:
        print(f"   17.25 apres decrochage (paroi) : {p.get('texte')}")
else:
    print(" (pas de donnees de paroi dans le JSON : figure du decollement non tracee)")

# --- chiffres clefs --------------------------------------------------------------
att = [(a, p) for a, p in chaine if a <= 17.0 + 1e-9]
if att:
    amax, pmax = max(att, key=lambda x: x[1]["CL"])
    print(f"\n CLmax (branche attachee) = {pmax['CL']:.4f} a alpha = {amax:.2f} deg")
    print(f"   Martin Ferrer (2025) : 1.477 a 15.5 deg -> ecart {(pmax['CL'] - 1.477) / 1.477 * 100:+.1f} %")
print(f" references experimentales trouvees dans ./tmr/ : {len(EXP)}")

#!/usr/bin/env python3
"""
check_yplus.py -- y+, Cf et Cp le long de la corde  (NACA0012 / NekRS)

Gere les sorties MULTI-RANGS : tous les fichiers d'un meme instant sont fusionnes.

ENTREES (dans le meme dossier) :
    resultats_csv/cf_cp_time_<t>_rank_<r>.csv   -> x,y,z,cf_x,cf_y,cf_z,Cp
    wall_distance_rank_<r>.csv                  -> x,y,z,wall_distance

SORTIES :
    yplus_report.txt   -> statistiques y+ (extrados / intrados)
    yplus_cf_cp.png    -> 3 graphes : y+, Cf, Cp  vs  x/c

USAGE :
    python3 check_yplus.py            # dernier instant disponible
    python3 check_yplus.py 2.0        # instant le plus proche de t = 2.0
"""

import sys, glob, re, os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# PARAMETRES PHYSIQUES  (doivent correspondre au .par)
# ----------------------------------------------------------------------
RHO  = 1.0        # density = 1.0
UINF = 1.0        # vitesse amont
NU   = 1.0e-6     # viscosity = 1/1000000   -> Re = 1e6

Z_TOL     = 1e-9    # plan z = 0
D_MIN     = 1e-12   # au-dela : point hors paroi
D_MAX     = 1e-2    # on ignore le champ lointain
N_VOISINS = 12      # voisins interroges pour trouver le 1er point GLL

# ----------------------------------------------------------------------
# 1. CHOIX DE L'INSTANT  (tri NUMERIQUE, pas alphabetique)
# ----------------------------------------------------------------------
fichiers = glob.glob("resultats_csv/cf_cp_time_*_rank_*.csv")
if not fichiers:
    sys.exit("ERREUR : aucun fichier dans resultats_csv/")

rgx = re.compile(r"cf_cp_time_([0-9.eE+-]+)_rank_(\d+)\.csv")
par_temps = {}
for f in fichiers:
    m = rgx.search(os.path.basename(f))
    if m:
        par_temps.setdefault(float(m.group(1)), []).append(f)

temps_dispo = sorted(par_temps)
if len(sys.argv) > 1:
    cible = float(sys.argv[1])
    t = min(temps_dispo, key=lambda x: abs(x - cible))
else:
    t = temps_dispo[-1]

lots = sorted(par_temps[t])
print("Instants disponibles : %s" % ", ".join("%.4f" % x for x in temps_dispo))
print("Instant retenu       : t = %.6f   (%d fichier(s) rang)" % (t, len(lots)))

cf = pd.concat([pd.read_csv(f) for f in lots], ignore_index=True)
cf.columns = [c.strip() for c in cf.columns]

# ----------------------------------------------------------------------
# 2. DISTANCE A LA PAROI  (tous les rangs fusionnes)
# ----------------------------------------------------------------------
fwd = sorted(glob.glob("wall_distance_rank_*.csv"))
if not fwd:
    sys.exit("ERREUR : aucun wall_distance_rank_*.csv trouve.\n"
             "  -> rsync -avz USER@nofx:~/naca0012_v2/wall_distance_rank_\\*.csv .")
print("Distance paroi       : %d fichier(s) rang" % len(fwd))

wd = pd.concat([pd.read_csv(f) for f in fwd], ignore_index=True)
wd.columns = [c.strip() for c in wd.columns]

# ----------------------------------------------------------------------
# 3. y1 = DISTANCE DU 1er POINT GLL A LA PAROI
# ----------------------------------------------------------------------
m = (np.abs(wd["z"]) < Z_TOL) & (wd["wall_distance"] > D_MIN) & (wd["wall_distance"] < D_MAX)
proches = wd[m]
if len(proches) == 0:
    sys.exit("ERREUR : aucun point hors-paroi. Verifie Z_TOL / D_MAX.")

arbre = cKDTree(proches[["x", "y"]].values)
d_par = proches["wall_distance"].values

pts = cf[["x", "y"]].values
dist_xy, idx = arbre.query(pts, k=N_VOISINS)
y1 = d_par[idx].min(axis=1)

suspect = dist_xy.min(axis=1) > 1e-3
if suspect.sum() > 0.05 * len(pts):
    print("\n!! ATTENTION : %d/%d points de paroi sans voisin proche." % (suspect.sum(), len(pts)))
    print("   Il manque probablement des fichiers wall_distance_rank_*.csv.\n")

# ----------------------------------------------------------------------
# 4. y+ = y1 * u_tau / nu ,  u_tau = sqrt(tau_w/rho) ,  tau_w = 0.5*rho*U^2*Cf
# ----------------------------------------------------------------------
Cf    = cf["cf_x"].values                 # cf_x = le Cf scalaire (cf_y = cf_z = 0)
tau_w = 0.5 * RHO * UINF**2 * np.abs(Cf)
u_tau = np.sqrt(tau_w / RHO)
yplus = y1 * u_tau / NU

# ----------------------------------------------------------------------
# 5. ABSCISSE x/c  (profil incline : projection sur la corde)
# ----------------------------------------------------------------------
P  = cf[["x", "y"]].values
c0 = P.mean(0)
A  = P[np.argmax(((P - c0) ** 2).sum(1))]
B  = P[np.argmax(((P - A) ** 2).sum(1))]
TE, LE = (A, B) if A[0] > B[0] else (B, A)

corde = np.linalg.norm(TE - LE)
u     = (TE - LE) / corde
rel   = P - LE
xc    = (rel @ u) / corde
haut  = (u[0] * rel[:, 1] - u[1] * rel[:, 0]) > 0

# ----------------------------------------------------------------------
# 6. RAPPORT
# ----------------------------------------------------------------------
def st(nom, msk):
    if msk.sum() == 0:
        return "%-9s : (aucun point)\n" % nom
    y = yplus[msk]
    return ("%-9s : moy=%.3f  med=%.3f  min=%.3f  max=%.3f   (%d pts)\n"
            % (nom, y.mean(), np.median(y), y.min(), y.max(), msk.sum()))

r  = "=" * 68 + "\n"
r += " y+ SUR LE PROFIL     t = %.6f     (%d points de paroi)\n" % (t, len(cf))
r += "=" * 68 + "\n"
r += " Corde = %.4f   BA = (%+.4f, %+.4f)   BF = (%+.4f, %+.4f)\n" % (
     corde, LE[0], LE[1], TE[0], TE[1])
r += " AoA deduit du maillage = %.2f deg\n\n" % np.degrees(np.arctan2(-u[1], u[0]))
r += st("EXTRADOS", haut)
r += st("INTRADOS", ~haut)
r += st("GLOBAL", np.ones(len(yplus), bool))
r += "\n y1 (1er pt GLL) : min=%.3e  max=%.3e  moy=%.3e\n" % (y1.min(), y1.max(), y1.mean())
r += " Cf              : min=%+.4e  max=%+.4e\n" % (Cf.min(), Cf.max())
r += " Cp              : min=%+.4e  max=%+.4e\n" % (cf["Cp"].min(), cf["Cp"].max())
r += "\n" + "-" * 68 + "\n"
ym = yplus.max()
if ym < 1.0:
    r += " OK : y+ < 1 partout  ->  k-tau resolu a la paroi, maillage VALIDE.\n"
elif ym < 3.0:
    r += " ATTENTION : y+ max = %.2f.  Acceptable, mais plus lache que Marina.\n" % ym
else:
    r += " PROBLEME : y+ max = %.2f (>> 1). Couche limite NON resolue.\n" % ym
    r += "            -> resserrer xfirstu dans le .geo.\n"
r += "-" * 68 + "\n"
r += " Reference Marina (Mesh 7) : y+ moyen ~ 0.66,  y+ max ~ 1.75\n"
r += " NB : si t est petit (transitoire), ce y+ n'est qu'une ESTIMATION.\n"
r += "=" * 68 + "\n"

print(r)
open("yplus_report.txt", "w").write(r)

# ----------------------------------------------------------------------
# 7. GRAPHES
# ----------------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
Cp = cf["Cp"].values

for msk, lab, col in ((haut, "extrados", "tab:red"), (~haut, "intrados", "tab:blue")):
    o = np.argsort(xc[msk])
    ax[0].plot(xc[msk][o], yplus[msk][o], ".", ms=4, color=col, label=lab)
    ax[1].plot(xc[msk][o], Cf[msk][o],    ".", ms=4, color=col, label=lab)
    ax[2].plot(xc[msk][o], Cp[msk][o],    ".", ms=4, color=col, label=lab)

ax[0].axhline(1.0, ls="--", c="k", lw=1, label="y+ = 1")
ax[0].set_ylabel("$y^+$"); ax[0].set_title("y+ le long de la corde   (t = %.3f)" % t)
ax[1].axhline(0.0, ls="--", c="k", lw=1)
ax[1].set_ylabel("$C_f$"); ax[1].set_title("Frottement   (passage a 0 = decollement)")
ax[2].set_ylabel("$C_p$"); ax[2].set_title("Pression")
ax[2].invert_yaxis(); ax[2].set_xlabel("$x/c$")
for a in ax:
    a.grid(alpha=.3); a.legend()

plt.tight_layout()
plt.savefig("yplus_cf_cp.png", dpi=130)
print("-> yplus_report.txt\n-> yplus_cf_cp.png")

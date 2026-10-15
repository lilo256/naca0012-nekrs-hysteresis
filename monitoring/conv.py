#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 CONVERGENCE  --  suivi rapide sur le cluster
================================================================================
 Repond a UNE question : est-ce que je peux enchainer sur l'angle suivant ?

 USAGE
   python3 conv.py                 # dossier courant
   python3 conv.py <cas>           # un cas precis
   python3 conv.py -a              # TOUS les cas de ~/cdf_runs (tableau de bord)

 CRITERE (adapte a l'hysteresis, pas a la mesure de precision) :
   dispersion du CL entre fenetres emboitees < 0.5 %
   -> les deux branches different de ~30 %, une derive de 0.4 % est invisible.
================================================================================
"""

import os
import sys
import glob
import numpy as np

SEUIL = 0.15         # % de dispersion CL acceptable pour enchainer
                     # (0.15 sur un PAS de chaine : l'ajustement doit
                     #  etre fini avant de perturber a nouveau)
RACINE = os.path.expanduser("~/cdf_runs")


def lire(dossier):
    """Renvoie (t, CL, CD, CDp, CDv) ou None.
    Filtre les lignes aberrantes ecrites par NekRS aux premiers pas
    apres un restart (efforts non physiques le temps que la pression
    se retablisse : |CL| ~ 1e6). Sans ce filtre, 10 lignes sur 325000
    suffisent a fausser completement les moyennes."""
    for p in (os.path.join(dossier, "drag_results.dat"),
              os.path.join(dossier, "Paraview", "drag_results.dat")):
        if os.path.isfile(p):
            try:
                d = np.loadtxt(p, skiprows=1)
            except Exception:
                return None
            if d.ndim != 2 or d.shape[0] < 100:
                return None
            # --- filtre des lignes aberrantes (transitoire de restart) ---
            ok = (np.abs(d[:, 3]) < 2.0) & (d[:, 2] > 0) & (d[:, 2] < 0.2)
            nrej = int((~ok).sum())
            d = d[ok]
            if d.shape[0] < 100:
                return None
            globals()["_NREJ"] = nrej
            CDp = d[:, 5] if d.shape[1] >= 11 else None
            CDv = d[:, 8] if d.shape[1] >= 11 else None
            return d[:, 1], d[:, 3], d[:, 2], CDp, CDv
    return None


def dispersion(t, y, fracs=(0.3, 0.5, 0.7, 0.9)):
    """Moyennes sur fenetres emboitees + dispersion relative en %.
    CORRECTION : les fenetres partent du DEBUT DU SEGMENT (t[0]) et non
    de zero. Sur un restart demarrant a t=112, l'ancienne version testait
    des fenetres [62,124], [74,124]... toutes identiques car couvrant tout
    le fichier -> dispersion nulle et faux verdict CONVERGE."""
    t0, tf = t[0], t[-1]
    vals = []
    for f in fracs:
        m = t >= t0 + f * (tf - t0)
        if m.sum() > 50:
            vals.append(y[m].mean())
    if len(vals) < 2:
        return None, []
    mo = abs(np.mean(vals))
    return (max(vals) - min(vals)) / max(mo, 1e-30) * 100, vals


def diagnostic(dossier, verbeux=True):
    r = lire(dossier)
    if r is None:
        return None
    t, CL, CD, CDp, CDv = r
    m = t >= t[0] + 0.5 * (t[-1] - t[0])
    dCL, vCL = dispersion(t, CL)
    dCD, _ = dispersion(t, CD)
    res = dict(t=t[-1], CL=CL[m].mean(), CD=CD[m].mean(),
               sCL=CL[m].std() / max(abs(CL[m].mean()), 1e-30) * 100,
               sCD=CD[m].std() / max(abs(CD[m].mean()), 1e-30) * 100,
               dCL=dCL, dCD=dCD, ok=(dCL is not None and dCL < SEUIL))
    if CDp is not None:
        res["CDp"] = CDp[m].mean()
        res["CDv"] = CDv[m].mean()

    if verbeux:
        print("=" * 70)
        print(" %s" % os.path.basename(os.path.abspath(dossier)))
        print("=" * 70)
        print(" segment       : %.1f -> %.1f CTU   (duree %.1f)"
              % (t[0], t[-1], t[-1] - t[0]))
        print(" fenetre       : %.1f -> %.1f CTU" % (t[m][0], t[-1]))
        nrej = globals().get("_NREJ", 0)
        if nrej:
            print(" lignes rejetees (transitoire de restart) : %d" % nrej)
        print()
        print("   CL = %.6f    fluctuation %.2f %%" % (res["CL"], res["sCL"]))
        print("   CD = %.6f    fluctuation %.2f %%" % (res["CD"], res["sCD"]))
        if "CDp" in res:
            print("   CDp = %.6f   CDv = %.6f" % (res["CDp"], res["CDv"]))
        print()
        print(" Derive du CL sur fenetres emboitees :")
        for f, v in zip((0.3, 0.5, 0.7, 0.9), vCL):
            print("   [%6.1f , %6.1f]  ->  CL = %.6f"
                  % (t[0] + f * (t[-1] - t[0]), t[-1], v))
        print()
        print("   dispersion CL = %.3f %%   (seuil %.1f %%)" % (dCL, SEUIL))
        print("   dispersion CD = %.3f %%" % dCD)
        print()
        # controle de plausibilite physique
        if abs(res["CL"]) > 2.0 or not (0 < res["CD"] < 0.2):
            print("   >>> !!! VALEURS NON PHYSIQUES — le calcul a diverge.")
            print("       NE PAS ENCHAINER. Verifier divUErr dans le log.")
        elif res["ok"]:
            print("   >>> CONVERGE : tu peux enchainer sur l'angle suivant.")
        else:
            tendance = "monte" if vCL[-1] > vCL[0] else "descend"
            print("   >>> PAS ENCORE : le CL %s toujours." % tendance)
            manque = max(10, int(t[-1] * 0.3))
            print("       Prolonger d'environ %d CTU puis recontroler." % manque)
        print()
        print(" Pour le pas suivant :  startFrom au dernier .f0*")
        print("                        endTime = %.1f + duree_voulue" % t[-1])
    return res


def tableau():
    """Vue d'ensemble de tous les cas."""
    dirs = sorted(d for d in glob.glob(os.path.join(RACINE, "*"))
                  if os.path.isdir(d))
    print("=" * 78)
    print(" TABLEAU DE BORD  --  %s" % RACINE)
    print("=" * 78)
    print(" %-22s %8s %11s %11s %8s %8s  %s"
          % ("cas", "t (CTU)", "CL", "CD", "disp CL", "disp CD", "etat"))
    print("-" * 78)
    for d in dirs:
        r = diagnostic(d, verbeux=False)
        if r is None:
            continue
        etat = "PRET" if r["ok"] else "en cours"
        print(" %-22s %8.1f %11.6f %11.6f %7.3f%% %7.3f%%  %s"
              % (os.path.basename(d), r["t"], r["CL"], r["CD"],
                 r["dCL"], r["dCD"], etat))
    print("-" * 78)
    print(" PRET = dispersion CL < %.1f %% -> on peut enchainer." % SEUIL)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-a", "--all"):
        tableau()
    else:
        cible = sys.argv[1] if len(sys.argv) > 1 else "."
        if not os.path.isdir(cible):
            cible = os.path.join(RACINE, cible)
        if diagnostic(cible) is None:
            print("Pas de drag_results.dat exploitable dans %s" % cible)

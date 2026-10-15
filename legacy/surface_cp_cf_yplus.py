#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
surface_cp_cf_yplus.py -- analyse de paroi (y+, Cf, Cp) statistiquement propre

CE QUI CHANGE PAR RAPPORT A LA V1 :

 1. MOYENNE TEMPORELLE SUR PLUSIEURS INSTANTANES
    La v1 lisait UN seul fichier CSV (un instant). Or Cf et Cp oscillent avec
    le lacher tourbillonnaire. Ici on moyenne tous les instantanes disponibles
    au-dela d'un temps de depart, et on donne l'ecart-type temporel.
    -> Cp,min devient une valeur MOYENNE, pas un tirage au hasard.

 2. MOYENNE SPATIALE PONDEREE PAR LA LONGUEUR D'ARC
    Les points de paroi sont concentres au bord d'attaque (maillage raffine).
    Une moyenne brute sur les points sur-pondere donc le BA, la ou y+ est le
    plus grand. On calcule  <y+> = int(y+ ds) / int(ds).

 3. PERCENTILES AU LIEU DU SEUL MAX
    y+max vient d'UN point, parfois la singularite du bord de fuite aigu.
    On donne p50 / p95 / p99 / max : bien plus robuste pour juger un maillage.

 4. DETECTION DU POINT DE DECOLLEMENT
    Passage de Cf par zero sur l'extrados (hors zone du point d'arret), avec
    interpolation lineaire. C'est l'objectif physique de la suite du stage.

 5. FRACTION DE SURFACE AU-DESSUS DE y+=1
    Le vrai critere de validite : quelle PROPORTION de la paroi viole y+<1,
    pas seulement le pic.

USAGE :
    python3 surface_cp_cf_yplus.py [dossier_cas] [t_min]

    dossier_cas : defaut = dossier courant
    t_min       : ne moyenne que les instantanes avec t >= t_min (defaut : 0)
"""

import numpy as np
import glob
import os
import re
import sys


class Tee:
    """Duplique tout ce qui est ecrit vers plusieurs flux (terminal + fichier)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


RE_C = 1.0e6          # nombre de Reynolds
U_INF = 1.0
CHORD = 1.0


# ------------------------------------------------------------------
#  LECTURE
# ------------------------------------------------------------------

def lire_csv(chemin):
    """Lit un CSV a en-tete et renvoie un dict {nom_colonne: array}."""
    with open(chemin) as f:
        entete = f.readline().strip().lstrip("#").strip()
    noms = [c.strip() for c in entete.split(",")]
    data = np.loadtxt(chemin, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return {n: data[:, i] for i, n in enumerate(noms) if i < data.shape[1]}


def trouver_colonne(d, motifs):
    """Trouve une colonne dont le nom correspond a l'un des motifs."""
    for m in motifs:
        for k in d:
            if re.fullmatch(m, k, flags=re.I):
                return d[k]
    return None


def charger_instantanes(dossier, t_min):
    """Charge tous les cf_cp_time_*.csv avec t >= t_min."""
    motif = os.path.join(dossier, "resultats_csv", "cf_cp_time_*.csv")
    fichiers = sorted(glob.glob(motif))
    if not fichiers:
        motif = os.path.join(dossier, "cf_cp_time_*.csv")
        fichiers = sorted(glob.glob(motif))
    retenus = []
    for f in fichiers:
        m = re.search(r"time_([0-9.eE+-]+)(?:_rank_\d+)?\.csv", os.path.basename(f))
        if not m:
            continue
        t = float(m.group(1))
        if t >= t_min:
            retenus.append((t, f))
    return sorted(retenus)


# ------------------------------------------------------------------
#  GEOMETRIE
# ------------------------------------------------------------------

def repere_profil(x, y):
    """
    Trouve bord d'attaque / bord de fuite, deduit l'AoA, et renvoie
    l'abscisse curviligne normalisee x/c le long de la corde.
    """
    d2 = x ** 2 + y ** 2
    i_bf = np.argmin(d2)                 # BF suppose proche de l'origine
    xbf, ybf = x[i_bf], y[i_bf]
    d_bf = (x - xbf) ** 2 + (y - ybf) ** 2
    i_ba = np.argmax(d_bf)               # BA = le plus loin du BF
    xba, yba = x[i_ba], y[i_ba]

    corde = np.hypot(xbf - xba, ybf - yba)
    aoa = np.degrees(np.arctan2(yba - ybf, -(xba - xbf)))

    # projection sur l'axe de corde -> x/c dans [0,1], 0 au BA
    ex = np.array([xbf - xba, ybf - yba]) / corde
    xs = ((x - xba) * ex[0] + (y - yba) * ex[1]) / corde
    ys = -(x - xba) * ex[1] + (y - yba) * ex[0]     # signe : + = extrados
    return xs, ys, corde, aoa, (xba, yba), (xbf, ybf)


def longueurs_arc(xc, sig):
    """
    Poids de longueur d'arc ds pour chaque point d'un cote (extrados ou
    intrados), apres tri par x/c. Utilise pour la moyenne spatiale ponderee.
    """
    ordre = np.argsort(xc)
    xs = xc[ordre]
    ds = np.zeros_like(xs)
    if len(xs) > 1:
        ds[1:-1] = 0.5 * (xs[2:] - xs[:-2])
        ds[0] = xs[1] - xs[0]
        ds[-1] = xs[-1] - xs[-2]
    return ordre, np.abs(ds)


def moyenne_ponderee(val, poids):
    s = poids.sum()
    return float((val * poids).sum() / s) if s > 0 else np.nan


# ------------------------------------------------------------------
#  PHYSIQUE
# ------------------------------------------------------------------

def calc_yplus(y1, cf):
    """
    y+ = y1 * u_tau / nu,  avec u_tau = U*sqrt(|Cf|/2)  et  nu = U*c/Re.
    """
    u_tau = U_INF * np.sqrt(np.abs(cf) / 2.0)
    nu = U_INF * CHORD / RE_C
    return y1 * u_tau / nu


def point_decollement(xc, cf, x_min=0.05):
    """
    Cherche le passage de Cf de + a - sur l'extrados, au-dela de x_min
    (on ignore la zone du point d'arret ou Cf change de signe pour une
    raison purement geometrique).
    Interpolation lineaire entre les deux points encadrants.
    """
    o = np.argsort(xc)
    xs, cs = xc[o], cf[o]
    m = xs >= x_min
    xs, cs = xs[m], cs[m]
    for i in range(len(xs) - 1):
        if cs[i] > 0 >= cs[i + 1]:
            f = cs[i] / (cs[i] - cs[i + 1])
            return xs[i] + f * (xs[i + 1] - xs[i])
    return None


# ------------------------------------------------------------------
#  PROGRAMME PRINCIPAL
# ------------------------------------------------------------------

def analyse(dossier=".", t_min=0.0):
    snaps = charger_instantanes(dossier, t_min)
    if not snaps:
        print(f"Aucun fichier cf_cp_time_*.csv trouve dans {dossier} "
              f"(avec t >= {t_min}).")
        return

    # --- distance a la paroi (independante du temps) ---
    wd_files = sorted(glob.glob(os.path.join(dossier, "wall_distance_rank_*.csv")))
    if not wd_files:
        print("wall_distance_rank_*.csv introuvable -> y+ impossible.")
        return
    xs_wd, ys_wd, y1_wd = [], [], []
    for f in wd_files:
        d = lire_csv(f)
        xs_wd.append(trouver_colonne(d, [r"x"]))
        ys_wd.append(trouver_colonne(d, [r"y"]))
        y1_wd.append(trouver_colonne(d, [r"wall_?distance", r"dist.*"]))
    xw = np.concatenate(xs_wd); yw = np.concatenate(ys_wd)
    y1w = np.concatenate(y1_wd)
    # on ne garde que les points de paroi (distance quasi nulle exclue : ce sont
    # les noeuds sur la paroi elle-meme ; on veut le 1er point interieur)
    ok = y1w > 0
    xw, yw, y1w = xw[ok], yw[ok], y1w[ok]

    print("=" * 68)
    print(" ANALYSE DE PAROI  (y+, Cf, Cp)   --  moyenne temporelle")
    print("=" * 68)
    print(f" Dossier      : {os.path.abspath(dossier)}")
    print(f" Instantanes  : {len(snaps)}  "
          f"(t = {snaps[0][0]:.2f} -> {snaps[-1][0]:.2f})")
    if len(snaps) == 1:
        print("   ATTENTION : un seul instantane -> pas de moyenne temporelle.")
        print("   Les valeurs portent la fluctuation du lacher tourbillonnaire.")
    print()

    # --- accumulation temporelle ---
    acc_cf, acc_cp, cpmin_t, xsep_t = [], [], [], []
    xc_ref = ysgn_ref = x_ref = y_ref = None

    for t, f in snaps:
        d = lire_csv(f)
        x = trouver_colonne(d, [r"x"]); y = trouver_colonne(d, [r"y"])
        cp = trouver_colonne(d, [r"cp", r"Cp"])
        cfx = trouver_colonne(d, [r"cf_?x"]); cfy = trouver_colonne(d, [r"cf_?y"])
        cf = trouver_colonne(d, [r"cf"]) if cfx is None else None

        xc, ysg, corde, aoa, ba, bf = repere_profil(x, y)
        if xc_ref is None:
            xc_ref, ysgn_ref = xc, ysg
            x_ref, y_ref = x, y
            corde_ref, aoa_ref, ba_ref, bf_ref = corde, aoa, ba, bf

        # Cf signe.
        # MEMO §9 : dans ce pipeline cf_x est DEJA le Cf scalaire signe
        # (streamwise, + oriente vers le BF) et cf_y = cf_z = 0 (remplissage).
        # On l'utilise donc directement. Reprojeter (cf_x*tx + cf_y*ty) revient
        # a multiplier le scalaire par tx -> corrompt signe ET amplitude, et
        # avec un tri melangeant extrados/intrados en prime : c'est l'origine
        # du faux "decollement a x/c=0.0511" (ecart-type 0.0000).
        if cf is not None:
            cfs = cf
        elif cfy is not None and \
                np.nanmax(np.abs(cfy)) > 1e-12 * (np.nanmax(np.abs(cfx)) + 1e-30):
            # cas d'un VRAI vecteur (cf_x, cf_y) non nul : on projette sur la
            # tangente, mais COTE PAR COTE (sinon argsort melange les surfaces).
            cfs = np.empty_like(xc)
            for msk in (ysg > 0, ysg <= 0):
                idx = np.where(msk)[0]
                o = idx[np.argsort(xc[idx])]
                tx = np.gradient(x[o]); ty = np.gradient(y[o])
                nn = np.hypot(tx, ty); nn[nn == 0] = 1
                cfs[o] = cfx[o] * (tx / nn) + cfy[o] * (ty / nn)
        else:
            cfs = cfx

        acc_cf.append(cfs)
        acc_cp.append(cp)
        cpmin_t.append(np.nanmin(cp))
        haut = ysg > 0
        xsep = point_decollement(xc[haut], cfs[haut])
        xsep_t.append(xsep if xsep is not None else np.nan)

    CF = np.mean(acc_cf, axis=0)
    CP = np.mean(acc_cp, axis=0)
    CF_sd = np.std(acc_cf, axis=0)
    CP_sd = np.std(acc_cp, axis=0)

    xc, ysg = xc_ref, ysgn_ref
    print(f" Corde = {corde_ref:.4f}   AoA deduit = {aoa_ref:.2f} deg")
    print(f" BA = ({ba_ref[0]:+.4f}, {ba_ref[1]:+.4f})   "
          f"BF = ({bf_ref[0]:+.4f}, {bf_ref[1]:+.4f})")
    print()

    # --- y1 (hauteur du 1er noeud interieur) pour chaque point de paroi ---
    # BUG v2 : l'ancien code interpolait y1 en fonction de x/c sur TOUS les
    # noeuds du volume. Pour un x/c donne il existe des noeuds de la paroi
    # jusqu'au champ lointain (distance ~ rayon du domaine), donc l'interp
    # renvoyait des distances enormes -> y+ ~ 1e5-1e6. Plus le domaine est
    # grand, pire c'est : d'ou l'explosion a 6.5e5 sur a20_b30 (rayon 20c).
    # CORRECTION : pour chaque point de paroi on prend la distance a la paroi
    # du noeud interieur le PLUS PROCHE en (x,y) = le 1er point hors-paroi.
    eps = 1e-9 * max(1.0, float(np.nanmax(np.abs(y1w))))   # seuil "hors-paroi"
    off = y1w > eps
    surf_xy = np.column_stack([x_ref, y_ref])
    vol_xy = np.column_stack([xw[off], yw[off]])
    try:
        from scipy.spatial import cKDTree
        _, inn = cKDTree(vol_xy).query(surf_xy, k=1)
        y1_surf = y1w[off][inn]
    except ImportError:
        # repli sans scipy : on isole la 1ere couche de noeuds (percentile bas)
        # puis on interpole par cote -- correct car la bande est mono-valuee.
        cut = np.nanpercentile(y1w[off], 2.0)
        near = off & (y1w <= cut)
        xcw, ysgw, *_ = repere_profil(xw[near], yw[near])
        y1n = y1w[near]
        y1_surf = np.empty_like(xc)
        for msk_s, msk_w in ((ysg > 0, ysgw > 0), (ysg <= 0, ysgw <= 0)):
            if msk_w.sum() < 2:
                y1_surf[msk_s] = np.median(y1n)
                continue
            o = np.argsort(xcw[msk_w])
            y1_surf[msk_s] = np.interp(xc[msk_s], xcw[msk_w][o], y1n[msk_w][o])

    YP = calc_yplus(y1_surf, CF)

    # --- statistiques ponderees par la longueur d'arc ---
    print("-" * 68)
    print(" y+   (moyennes PONDEREES par la longueur d'arc)")
    print("-" * 68)
    print(f" {'':10s} {'moy':>8s} {'p50':>8s} {'p95':>8s} {'p99':>8s} "
          f"{'max':>8s}  {'% > 1':>7s}   pts")
    for nom, msk in (("EXTRADOS", ysg > 0), ("INTRADOS", ysg <= 0),
                     ("GLOBAL", np.ones_like(ysg, bool))):
        if msk.sum() == 0:
            continue
        o, ds = longueurs_arc(xc[msk], None)
        v = YP[msk][o]
        moy = moyenne_ponderee(v, ds)
        # percentiles ponderes
        cum = np.cumsum(ds) / ds.sum()
        iv = np.argsort(v); vs = v[iv]; cs = np.cumsum(ds[iv]) / ds.sum()
        p = lambda q: float(np.interp(q, cs, vs))
        frac1 = float(ds[v > 1.0].sum() / ds.sum() * 100)
        print(f" {nom:10s} {moy:8.3f} {p(0.50):8.3f} {p(0.95):8.3f} "
              f"{p(0.99):8.3f} {v.max():8.3f}  {frac1:6.2f}%  {msk.sum():5d}")
    print()
    yp_glob = YP
    o, ds = longueurs_arc(xc, None)
    frac_tot = ds[yp_glob[o] > 1].sum() / ds.sum() * 100
    if frac_tot == 0:
        print(" OK : y+ < 1 sur TOUTE la paroi -> k-tau resolu, maillage VALIDE.")
    elif frac_tot < 2:
        print(f" ACCEPTABLE : y+ > 1 sur seulement {frac_tot:.2f} % de la corde")
        print("   (typiquement le pic au bord d'attaque, effet negligeable).")
    else:
        print(f" ATTENTION : y+ > 1 sur {frac_tot:.2f} % de la corde -> "
              f"sous-resolution notable.")
    print(" Reference Marina (Mesh 7) : y+ moyen ~ 0.66, y+ max ~ 1.75")
    print()

    # --- Cp ---
    print("-" * 68)
    print(" Cp  (pic de succion : indicateur cle de la densite de maillage)")
    print("-" * 68)
    i_min = int(np.nanargmin(CP))
    cpmin_t = np.array(cpmin_t)
    print(f" Cp,min moyen  = {CP[i_min]:+.4f}   a x/c = {xc[i_min]:.4f}")
    if len(cpmin_t) > 1:
        print(f" Cp,min par instantane : min {cpmin_t.min():+.4f}  "
              f"max {cpmin_t.max():+.4f}  "
              f"ecart-type {cpmin_t.std():.4f}")
        print(f"   -> fluctuation temporelle de "
              f"{cpmin_t.std()/abs(cpmin_t.mean())*100:.2f} % : "
              f"{'faible, valeur fiable' if cpmin_t.std()/abs(cpmin_t.mean())<0.01 else 'notable, moyenner davantage'}")
    print(f" Cp,max        = {np.nanmax(CP):+.4f}  "
          f"(doit valoir ~ +1.0 au point d'arret)")
    print()

    # --- Cf et decollement ---
    print("-" * 68)
    print(" Cf  et  point de decollement")
    print("-" * 68)
    print(f" Cf  : min {np.nanmin(CF):+.4e}   max {np.nanmax(CF):+.4e}")
    xsep_t = np.array(xsep_t, dtype=float)
    haut = ysg > 0
    xsep_moy = point_decollement(xc[haut], CF[haut])
    if xsep_moy is not None:
        print(f" DECOLLEMENT extrados detecte a x/c = {xsep_moy:.4f}")
        if np.isfinite(xsep_t).sum() > 1:
            print(f"   par instantane : {np.nanmin(xsep_t):.4f} -> "
                  f"{np.nanmax(xsep_t):.4f}  "
                  f"(ecart-type {np.nanstd(xsep_t):.4f})")
    else:
        cf_h = CF[haut]; xc_h = xc[haut]
        o = np.argsort(xc_h)
        cf_bf = cf_h[o][-max(3, len(o)//50):].mean()
        print(" Pas de decollement detecte sur l'extrados.")
        print(f"   Cf pres du bord de fuite = {cf_bf:+.3e}", end="")
        if cf_bf < 5e-4:
            print("  -> proche de zero : decollement IMMINENT.")
        else:
            print("  -> couche limite bien attachee.")
    print()
    print(" Note : le Cf negatif de l'INTRADOS pres du bord d'attaque n'est PAS")
    print(" un decollement -- c'est la zone avant le point d'arret, ou le fluide")
    print(" remonte vers l'avant. Verifiable : Cp = +1 au meme endroit.")
    print("=" * 68)


if __name__ == "__main__":
    dossier = sys.argv[1] if len(sys.argv) > 1 else "."
    tmin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    base = os.path.basename(os.path.abspath(dossier))
    out_path = f"{base}_surface_analyse.txt"
    stdout_orig = sys.stdout
    with open(out_path, "w") as fout:
        sys.stdout = Tee(stdout_orig, fout)
        try:
            analyse(dossier, tmin)
        finally:
            sys.stdout = stdout_orig
    print(f"\nResultats ecrits dans : {out_path}")

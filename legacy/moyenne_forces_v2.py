#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
moyenne_forces_v2.py -- moyennage statistiquement rigoureux des forces NekRS

CE QUI CHANGE PAR RAPPORT A LA V1 :

 1. MOYENNE PONDEREE PAR LE TEMPS (integrale trapezoidale)
    Le dt varie (targetCFL), donc une moyenne arithmetique biaise le resultat.
    On calcule  <f> = int(f dt) / int(dt)  au lieu de  sum(f)/N.

 2. VRAIE BARRE D'ERREUR (via le temps d'autocorrelation)
    Les echantillons sont correles : 60 000 points ne sont PAS 60 000 mesures
    independantes. On estime le temps d'autocorrelation integre tau_int, puis
    N_eff = T_fenetre / (2*tau_int), et l'erreur standard = sigma / sqrt(N_eff).
    C'est CETTE valeur qui dit si deux cas different vraiment.

 3. TEST DE SENSIBILITE A LA FENETRE
    On recalcule la moyenne pour plusieurs debuts de fenetre. Si la valeur
    bouge, c'est que le transitoire n'est pas fini -> pas converge.

 4. METHODE DES BLOCS (verification croisee)
    Le signal est decoupe en blocs ; la dispersion des moyennes de blocs
    donne une seconde estimation de l'incertitude, independante de (2).

USAGE :
    python3 moyenne_forces_v2.py <drag_results.dat> [T_fenetre_CTU]

    T_fenetre_CTU : longueur de la fenetre de moyenne (defaut : moitie du run)
"""

import numpy as np
import sys
import os


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

# ----- cibles de reference (Marina, Mesh 7, AoA=10, Re=1e6) -----
CL_REF = 1.028
CD_REF = 0.0205


# ============================================================
#  OUTILS STATISTIQUES
# ============================================================

def moyenne_temporelle(t, f):
    """
    Moyenne ponderee par le temps : <f> = int(f dt) / int(dt).
    Utilise la regle du trapeze. Corrige le biais du dt variable.
    """
    if len(t) < 2:
        return float(f[0]) if len(f) else np.nan
    return float(np.trapezoid(f, t) / (t[-1] - t[0]))


def variance_temporelle(t, f):
    """Variance ponderee par le temps (meme logique que la moyenne)."""
    m = moyenne_temporelle(t, f)
    return moyenne_temporelle(t, (f - m) ** 2)


def temps_autocorrelation(t, f, nlags=None):
    """
    Estime le temps d'autocorrelation integre tau_int, EN UNITES DE TEMPS (CTU).

    Methode : on reechantillonne sur une grille reguliere (le signal a un dt
    variable), on calcule la fonction d'autocorrelation, et on l'integre
    jusqu'au premier passage sous zero (fenetre auto-tronquee, standard pour
    eviter que le bruit de queue ne pollue l'integrale).

    Retourne (tau_int, rho, lags_ctu) pour pouvoir tracer si besoin.
    """
    T = t[-1] - t[0]
    n = min(len(t), 20000)                 # suffisant, et rapide
    tg = np.linspace(t[0], t[-1], n)
    fg = np.interp(tg, t, f)
    dt = tg[1] - tg[0]

    x = fg - fg.mean()
    var = np.mean(x * x)
    if var <= 0:
        return 0.0, np.array([1.0]), np.array([0.0])

    if nlags is None:
        nlags = n // 4                     # on ne regarde pas au-dela de T/4

    # autocorrelation par FFT (rapide)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    F = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(F * np.conjugate(F), nfft)[:nlags]
    acf /= (np.arange(n, n - nlags, -1) * var)   # normalisation non biaisee

    # troncature au premier passage sous zero
    neg = np.where(acf < 0)[0]
    kmax = neg[0] if len(neg) else nlags

    # tau_int = dt * (1/2 + somme des rho)  -> en unites de temps
    tau = dt * (0.5 + acf[1:kmax].sum()) if kmax > 1 else dt * 0.5
    return max(tau, dt), acf[:kmax], np.arange(kmax) * dt


def erreur_standard(t, f):
    """
    Erreur standard sur la moyenne, CORRIGEE de l'autocorrelation.

    N_eff = T / (2 * tau_int)   = nombre d'echantillons vraiment independants
    SE    = sigma / sqrt(N_eff)

    C'est la barre d'erreur a utiliser pour comparer deux cas.
    """
    T = t[-1] - t[0]
    sigma = np.sqrt(variance_temporelle(t, f))
    tau, _, _ = temps_autocorrelation(t, f)
    n_eff = T / (2.0 * tau) if tau > 0 else 1.0
    n_eff = max(n_eff, 1.0)
    return sigma / np.sqrt(n_eff), n_eff, tau, sigma


def methode_blocs(t, f, nblocs=8):
    """
    Verification croisee : on coupe la fenetre en blocs, on moyenne chaque bloc,
    et l'ecart-type des moyennes de blocs / sqrt(nblocs) donne l'incertitude.
    Independant de l'estimation par autocorrelation -> bon test de coherence.
    """
    bornes = np.linspace(t[0], t[-1], nblocs + 1)
    moyennes = []
    for i in range(nblocs):
        m = (t >= bornes[i]) & (t <= bornes[i + 1])
        if m.sum() > 2:
            moyennes.append(moyenne_temporelle(t[m], f[m]))
    moyennes = np.array(moyennes)
    if len(moyennes) < 2:
        return np.nan, moyennes
    return float(moyennes.std(ddof=1) / np.sqrt(len(moyennes))), moyennes


def sensibilite_fenetre(t, f, t_fin, fractions=(1.0, 0.8, 0.6, 0.4)):
    """
    Recalcule la moyenne en partant de differents debuts de fenetre.
    Si la valeur derive quand on rallonge la fenetre -> transitoire non purge.
    """
    res = []
    for fr in fractions:
        t0 = t_fin - fr * (t_fin - t[0])
        m = t >= t0
        if m.sum() > 10:
            res.append((t0, t_fin, moyenne_temporelle(t[m], f[m]), int(m.sum())))
    return res


# ============================================================
#  PROGRAMME PRINCIPAL
# ============================================================

def analyse(fichier, T_fen=None):
    d = np.loadtxt(fichier, skiprows=1)
    t = d[:, 1]
    CD, CL = d[:, 2], d[:, 3]
    CDp, CLp = d[:, 5], d[:, 6]    # composantes pression
    CDv, CLv = d[:, 8], d[:, 9]    # composantes visqueuses

    # tri + suppression des doublons temporels (restarts)
    ordre = np.argsort(t, kind="stable")
    t, CD, CL = t[ordre], CD[ordre], CL[ordre]
    CDp, CDv = CDp[ordre], CDv[ordre]
    garde = np.concatenate(([True], np.diff(t) > 0))
    t, CD, CL, CDp, CDv = t[garde], CD[garde], CL[garde], CDp[garde], CDv[garde]

    t_fin = t[-1]
    if T_fen is None:
        T_fen = 0.5 * (t_fin - t[0])
    t_deb = max(t_fin - T_fen, t[0])
    msk = t >= t_deb
    tw, CLw, CDw = t[msk], CL[msk], CD[msk]
    CDpw, CDvw = CDp[msk], CDv[msk]

    print("=" * 66)
    print(f" MOYENNAGE STATISTIQUE  --  {fichier}")
    print("=" * 66)
    print(f" Run complet      : {t[0]:.2f} -> {t_fin:.2f} CTU   ({len(t)} echantillons)")
    print(f" Fenetre analysee : {t_deb:.2f} -> {t_fin:.2f} CTU   ({msk.sum()} echantillons)")
    dt_med = np.median(np.diff(tw))
    print(f" dt median        : {dt_med:.3e}  (min {np.diff(tw).min():.2e}, "
          f"max {np.diff(tw).max():.2e})")
    if np.diff(tw).max() / max(np.diff(tw).min(), 1e-30) > 1.2:
        print("   -> dt VARIABLE : la moyenne ponderee par le temps est necessaire.")
    print()

    resultats = {}
    for nom, sig, ref in (("CL", CLw, CL_REF), ("CD", CDw, CD_REF)):
        moy_t = moyenne_temporelle(tw, sig)
        moy_a = float(sig.mean())                       # ancienne methode
        se, n_eff, tau, sigma = erreur_standard(tw, sig)
        se_bl, blocs = methode_blocs(tw, sig)

        resultats[nom] = (moy_t, se)

        print("-" * 66)
        print(f" {nom}")
        print("-" * 66)
        print(f"   Moyenne ponderee temps : {moy_t:.6f}")
        print(f"   (moyenne arithmetique  : {moy_a:.6f}   ecart : "
              f"{abs(moy_t-moy_a)/abs(moy_t)*100:.3f} %)")
        print()
        print(f"   Fluctuation du signal  sigma = {sigma:.3e}   "
              f"({sigma/abs(moy_t)*100:.2f} % de la moyenne)")
        print(f"   Temps d'autocorrelation tau  = {tau:.3f} CTU")
        print(f"   Echantillons INDEPENDANTS    = {n_eff:.1f}   "
              f"(et non {len(tw)} !)")
        print()
        print(f"   >>> {nom} = {moy_t:.6f}  +/-  {se:.2e}   "
              f"(+/- {se/abs(moy_t)*100:.3f} %)")
        print(f"   Verification par blocs : +/- {se_bl:.2e}", end="")
        if not np.isnan(se_bl):
            rap = se_bl / se if se > 0 else np.nan
            print(f"   (rapport {rap:.2f} -> "
                  f"{'coherent' if 0.4 < rap < 2.5 else 'INCOHERENT, prudence'})")
        else:
            print()
        print()

        ecart = (moy_t - ref) / ref * 100
        n_sig = abs(moy_t - ref) / se if se > 0 else np.inf
        print(f"   Reference Marina : {ref}")
        print(f"   Ecart            : {ecart:+.2f} %   "
              f"({n_sig:.0f} sigma)")
        print()

        # sensibilite a la fenetre
        print("   Sensibilite a la fenetre de moyennage :")
        for (a, b, m, n) in sensibilite_fenetre(tw, sig, t_fin):
            print(f"      [{a:6.2f} , {b:6.2f}] CTU  ->  {m:.6f}")
        spread = max(x[2] for x in sensibilite_fenetre(tw, sig, t_fin)) - \
                 min(x[2] for x in sensibilite_fenetre(tw, sig, t_fin))
        print(f"      dispersion = {spread:.2e}", end="")
        if spread < se:
            print("   -> STABLE (< barre d'erreur), fenetre bien choisie.")
        elif spread < 3 * se:
            print("   -> acceptable, mais rallonger le run serait mieux.")
        else:
            print("   -> DERIVE : transitoire non purge, NE PAS conclure.")
        print()

    # decomposition de la trainee
    cdp = moyenne_temporelle(tw, CDpw)
    cdv = moyenne_temporelle(tw, CDvw)
    tot = cdp + cdv
    print("-" * 66)
    print(" Decomposition de CD")
    print("-" * 66)
    print(f"   pression (CDp) : {cdp:.6f}   ({cdp/tot*100:.1f} %)")
    print(f"   visqueux (CDv) : {cdv:.6f}   ({cdv/tot*100:.1f} %)")
    print()

    # aide a la comparaison entre cas
    print("=" * 66)
    print(" POUR COMPARER DEUX CAS")
    print("=" * 66)
    print(" Deux cas different SIGNIFICATIVEMENT si :")
    print("     |valeur_A - valeur_B|  >  2 * sqrt(SE_A^2 + SE_B^2)")
    print()
    for nom, (m, se) in resultats.items():
        print(f"   {nom} = {m:.6f} +/- {se:.2e}"
              f"   -> seuil de significativite ~ {2*np.sqrt(2)*se:.2e}"
              f"  ({2*np.sqrt(2)*se/abs(m)*100:.3f} %)")
    print("=" * 66)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    f = sys.argv[1]
    T = float(sys.argv[2]) if len(sys.argv) > 2 else None

    base = os.path.splitext(os.path.basename(f))[0]
    out_path = f"{base}_analyse.txt"
    stdout_orig = sys.stdout
    with open(out_path, "w") as fout:
        sys.stdout = Tee(stdout_orig, fout)
        try:
            analyse(f, T)
        finally:
            sys.stdout = stdout_orig
    print(f"\nResultats ecrits dans : {out_path}")



#python3 ~/Bureau/stage_docs/code/mesh_modif/a15_b10/moyenne_forces_v2.py drag_results.dat 20

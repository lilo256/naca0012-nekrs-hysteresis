#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ANALYSE DES SONDES  --  probes_uvp.csv
================================================================================
 Exploite les 20 sondes placees dans airfoil.udf. Repond a 4 questions :

  1. OU l'ecoulement decolle-t-il ?          -> peigne parietal A020..A092
  2. Les oscillations sont-elles PHYSIQUES ? -> temoin amont REF
  3. L'ecoulement est-il bien 2D ?           -> paire SPAN1 / SPAN2
  4. QUELLE frequence, et NEE OU ?           -> spectres par region + coherence

 USAGE :
     python3 analyse_sondes.py                      # dossier courant
     python3 analyse_sondes.py <dossier> [t_min]

 t_min : temps a partir duquel analyser (purge du transitoire).
         Defaut = 50 % du run.
================================================================================
"""

import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

PEIGNE = ["A020", "A035", "A050", "A065", "A080", "A092"]
XC_PEIGNE = [0.20, 0.35, 0.50, 0.65, 0.80, 0.92]
CISAIL = ["B050", "B070", "B090"]
SILLAGE = ["W015p", "W015m", "W050p", "W050m", "W100p", "W100m"]


# =============================================================================
#  LECTURE
# =============================================================================

def lire(chemin, tmin=None):
    """Lit probes_uvp.csv -> (t, dict{label: {ux,uy,p}})."""
    with open(chemin) as f:
        entete = f.readline().strip().split(",")
    d = np.loadtxt(chemin, delimiter=",", skiprows=1)
    if d.ndim == 1:
        d = d[None, :]
    t = d[:, 0]
    if tmin is None:
        tmin = t[0] + 0.5 * (t[-1] - t[0])
    m = t >= tmin
    S = {}
    for j, nom in enumerate(entete[1:], start=1):
        if "_" not in nom:
            continue
        grand, lab = nom.split("_", 1)
        S.setdefault(lab, {})[grand] = d[m, j]
    return t[m], S, t[-1], tmin


def spectre(t, y, fenetrage=True, detrend=True):
    """FFT sur grille reguliere. Renvoie (freq, densite)."""
    n = len(y)
    if n < 64:
        return None, None
    tr = np.linspace(t[0], t[-1], n)          # re-interpolation (dt variable)
    yr = np.interp(tr, t, y)
    if detrend:                                # retire la derive lineaire
        yr = yr - np.polyval(np.polyfit(tr, yr, 1), tr)
    if fenetrage:
        yr = yr * np.hanning(n)               # limite la fuite spectrale
    dt = tr[1] - tr[0]
    Y = np.fft.rfft(yr)
    f = np.fft.rfftfreq(n, dt)
    P = np.abs(Y) ** 2
    return f[1:], P[1:]


def pics(f, P, n=3, fmin=0.05):
    """n plus grands pics locaux au-dessus de fmin."""
    m = f >= fmin
    f, P = f[m], P[m]
    if len(f) < 3:
        return []
    loc = np.where((P[1:-1] > P[:-2]) & (P[1:-1] > P[2:]))[0] + 1
    if len(loc) == 0:
        return []
    ordre = loc[np.argsort(P[loc])[::-1]][:n]
    return [(f[i], P[i]) for i in ordre]


def dephasage(t, a, b, f0):
    """Dephasage entre deux signaux a la frequence f0, via correlation croisee."""
    n = len(a)
    tr = np.linspace(t[0], t[-1], n)
    A = np.interp(tr, t, a - np.mean(a))
    B = np.interp(tr, t, b - np.mean(b))
    if A.std() == 0 or B.std() == 0:
        return None, None
    c = np.correlate(A, B, mode="full") / (n * A.std() * B.std())
    lags = (np.arange(len(c)) - (n - 1)) * (tr[1] - tr[0])
    i = np.argmax(np.abs(c))
    tau = lags[i]
    phi = (tau * f0 * 360.0) % 360.0
    return c[i], phi


# =============================================================================
#  PROGRAMME
# =============================================================================

def main():
    dossier = sys.argv[1] if len(sys.argv) > 1 else "."
    tmin = float(sys.argv[2]) if len(sys.argv) > 2 else None
    chemin = os.path.join(dossier, "probes_uvp.csv")
    if not os.path.isfile(chemin):
        print("Fichier introuvable : %s" % chemin)
        sys.exit(1)

    t, S, tfin, tmin = lire(chemin, tmin)
    T = t[-1] - t[0]
    print("=" * 78)
    print(" ANALYSE DES SONDES")
    print("=" * 78)
    print(" Dossier   : %s" % os.path.abspath(dossier))
    print(" Run       : 0 -> %.1f CTU   |  fenetre analysee : %.1f -> %.1f CTU"
          % (tfin, t[0], t[-1]))
    print(" Sondes    : %d   |  echantillons : %d" % (len(S), len(t)))
    print(" Resolution frequentielle : df = 1/T = %.4f" % (1.0 / T))

    # ---------------------------------------------------------------- 1. 2D ?
    print("\n" + "-" * 78)
    print(" 1. CONTROLE DE BIDIMENSIONNALITE   (SPAN1 vs SPAN2, memes x,y, z different)")
    print("-" * 78)
    if "SPAN1" in S and "SPAN2" in S:
        a, b = S["SPAN1"]["ux"], S["SPAN2"]["ux"]
        ec = np.max(np.abs(a - b))
        rel = ec / max(np.max(np.abs(a)), 1e-30)
        print("   ecart max sur ux : %.3e   (%.2e en relatif)" % (ec, rel))
        if rel < 1e-8:
            print("   -> ecoulement STRICTEMENT 2D. Hypothese quasi-2D validee.")
        elif rel < 1e-3:
            print("   -> quasi-2D (ecart negligeable).")
        else:
            print("   -> ATTENTION : variation en envergure detectee.")
            print("      L'hypothese 2D ne tient plus. Critique avant la DDES.")
    else:
        print("   sondes SPAN absentes.")

    # ------------------------------------------------- 2. physique ou artefact
    print("\n" + "-" * 78)
    print(" 2. LES OSCILLATIONS SONT-ELLES PHYSIQUES ?   (temoin amont REF)")
    print("-" * 78)
    if "REF" in S:
        f, P = spectre(t, S["REF"]["ux"])
        sig = S["REF"]["ux"]
        rms = sig.std() / max(abs(sig.mean()), 1e-30)
        print("   REF : ux moyen = %.6f   fluctuation RMS = %.2e (%.4f %%)"
              % (sig.mean(), sig.std(), rms * 100))
        if f is not None:
            pk = pics(f, P, 1)
            if pk:
                print("   pic dominant du temoin : St = %.3f" % pk[0][0])
        if rms < 1e-3:
            print("   -> spectre du temoin PLAT : l'oscillation nait pres du profil.")
            print("      = PHYSIQUE, et non un artefact numerique global (ni MPI).")
        else:
            print("   -> le temoin amont FLUCTUE : suspect.")
            print("      Verifier s'il n'est pas dans le sillage ou trop pres du profil.")
    else:
        print("   sonde REF absente.")

    # ------------------------------------------------------- 3. decollement
    print("\n" + "-" * 78)
    print(" 3. POSITION DU DECOLLEMENT   (peigne parietal, ux < 0 = ecoulement inverse)")
    print("-" * 78)
    print("   %-7s %6s %12s %12s %10s   %s"
          % ("sonde", "x/c", "ux moyen", "ux RMS", "% du temps", "etat"))
    xsep = None
    for lab, xc in zip(PEIGNE, XC_PEIGNE):
        if lab not in S:
            continue
        u = S[lab]["ux"]
        frac = np.mean(u < 0) * 100
        if frac > 50:
            etat = "DECOLLE"
        elif frac > 5:
            etat = "intermittent"
        else:
            etat = "attache"
        if frac > 50 and xsep is None:
            xsep = xc
        print("   %-7s %6.2f %12.5f %12.3e %9.1f %%   %s"
              % (lab, xc, u.mean(), u.std(), frac, etat))
    print()
    if xsep is not None:
        print("   >>> DECOLLEMENT en amont de x/c = %.2f" % xsep)
        print("       (le peigne borne la position ; le Cf parietal la precise)")
    else:
        print("   >>> Aucune sonde du peigne en ecoulement inverse.")
        print("       Decollement absent, ou situe en aval de x/c = 0.92.")
    print("\n   RAPPEL : c'est CE tableau qui distingue les deux branches de")
    print("   l'hysteresis. A angle IDENTIQUE, la branche descendante decolle")
    print("   bien plus en amont que la montante.")

    # --------------------------------------------------------- 4. spectres
    print("\n" + "-" * 78)
    print(" 4. SPECTRES PAR REGION   (ou nait la frequence ?)")
    print("-" * 78)
    print("   %-8s %10s %10s %10s   %s"
          % ("sonde", "St #1", "St #2", "St #3", "L_eff = 0.2/St"))
    Stw = None
    for lab in PEIGNE + CISAIL + SILLAGE:
        if lab not in S:
            continue
        f, P = spectre(t, S[lab]["uy" if "uy" in S[lab] else "ux"])
        if f is None:
            continue
        pk = pics(f, P, 3)
        if not pk:
            continue
        sts = [p[0] for p in pk]
        leff = 0.2 / sts[0] if sts[0] > 0 else np.nan
        print("   %-8s %10.3f %10s %10s   %.4f c"
              % (lab, sts[0],
                 "%.3f" % sts[1] if len(sts) > 1 else "-",
                 "%.3f" % sts[2] if len(sts) > 2 else "-",
                 leff))
        if lab.startswith("W0") and Stw is None:
            Stw = sts[0]
    if Stw:
        print("\n   Strouhal dominant du sillage : St = %.3f" % Stw)
        print("   -> longueur effective L = 0.2/St = %.4f c" % (0.2 / Stw))
        print("      ~0.02 c  = bord de fuite (ecoulement attache)")
        print("      ~0.3  c  = profil entier comme corps epais (DECROCHE)")
        ncyc = T * Stw
        print("   -> %.0f cycles dans la fenetre  |  df/f = %.2f %%"
              % (ncyc, 100.0 / max(ncyc, 1e-9)))
        if ncyc < 20:
            print("      ATTENTION : moins de 20 cycles, resolution insuffisante.")
            print("      Rallonger le run (il faut ~50 cycles).")

    # ---------------------------------------------- 5. lacher alterne ?
    print("\n" + "-" * 78)
    print(" 5. NATURE DU LACHER   (paires haut/bas : dephasage)")
    print("-" * 78)
    for hp, hm in (("W015p", "W015m"), ("W050p", "W050m"), ("W100p", "W100m")):
        if hp in S and hm in S and Stw:
            c, phi = dephasage(t, S[hp]["uy"], S[hm]["uy"], Stw)
            if c is None:
                continue
            if c < 0 and (phi < 60 or phi > 300):
                nat = "ANTIPHASE -> lacher ALTERNE (type Karman)"
            elif 120 < phi < 240:
                nat = "ANTIPHASE -> lacher ALTERNE (type Karman)"
            elif c > 0 and (phi < 60 or phi > 300):
                nat = "EN PHASE -> lacher symetrique"
            else:
                nat = "intermediaire"
            print("   %s / %s : correlation %+.3f , dephasage %6.1f deg  -> %s"
                  % (hp, hm, c, phi, nat))

    # ------------------------------------- 6. la couche de cisaillement alimente ?
    print("\n" + "-" * 78)
    print(" 6. LA COUCHE DE CISAILLEMENT ALIMENTE-T-ELLE LE SILLAGE ?")
    print("    (question laissee ouverte par Marina)")
    print("-" * 78)
    if "B090" in S and "W015p" in S and Stw:
        c, phi = dephasage(t, S["B090"]["uy"], S["W015p"]["uy"], Stw)
        if c is not None:
            print("   B090 (cisaillement) <-> W015p (sillage)")
            print("   correlation max = %+.3f   dephasage = %.1f deg" % (c, phi))
            if abs(c) > 0.3:
                print("   -> signaux CORRELES : les structures de la couche de")
                print("      cisaillement se retrouvent dans le sillage.")
                print("      FILIATION ETABLIE (Kelvin-Helmholtz -> sillage).")
            else:
                print("   -> faible correlation : mecanismes independants,")
                print("      ou fenetre trop courte.")
    else:
        print("   sondes B090 / W015p absentes.")

    # ------------------------------------------------------------- figures
    if HAS_PLT:
        fig, axes = plt.subplots(3, 1, figsize=(11, 11))

        ax = axes[0]
        for lab, xc in zip(PEIGNE, XC_PEIGNE):
            if lab in S:
                ax.plot(t, S[lab]["ux"], lw=0.8, label="%s (x/c=%.2f)" % (lab, xc))
        ax.axhline(0, color="k", ls="--", lw=1.2)
        ax.set_xlabel("t (CTU)"); ax.set_ylabel(r"$u_x$")
        ax.set_title("Peigne parietal : $u_x < 0$ = ecoulement inverse (decolle)",
                     fontweight="bold")
        ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.25)

        ax = axes[1]
        for lab in ["B090", "W015p", "REF"]:
            if lab in S:
                f, P = spectre(t, S[lab]["uy"])
                if f is not None:
                    ax.loglog(f, P / P.max(), lw=0.9, label=lab)
        ax.set_xlabel("St"); ax.set_ylabel("densite spectrale (normalisee)")
        ax.set_title("Spectres : cisaillement / sillage / temoin amont",
                     fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")

        ax = axes[2]
        for lab in ("W015p", "W015m"):
            if lab in S:
                ax.plot(t, S[lab]["uy"], lw=0.8, label=lab)
        ax.set_xlabel("t (CTU)"); ax.set_ylabel(r"$u_y$")
        ax.set_title("Paire de sillage : antiphase = lacher alterne",
                     fontweight="bold")
        if len(t) > 400:
            ax.set_xlim(t[0], t[0] + 0.15 * T)
        ax.legend(fontsize=8); ax.grid(alpha=0.25)

        plt.tight_layout()
        sortie = os.path.join(dossier, "sondes.png")
        plt.savefig(sortie, dpi=130, bbox_inches="tight")
        print("\n [figure] %s" % sortie)

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
spectre_film.py  --  spectre a PAS CONSTANT du run film_a15 (15 deg, dt = 3.5e-5 fixe)
=====================================================================================
Pour chaque signal (CL, CD, et chaque sonde), donne :
  - le rms total,
  - le rms dans trois bandes de Strouhal (base corde) :
        [0.8 ; 2.6]   lacher du bord de fuite (St ~ 1.76, vu en portance)
        [2.6 ; 4.6]   son double (St ~ 3.5, vu au centre du sillage)
        [5.5 ; 10.5]  mode rapide (St ~ 7.7 ; Martin Ferrer 2025 : 7.704 a 15 deg)
  - la frequence du pic dans la bande rapide.
Le rms de bande dit OU le mode rapide est le plus fort : c'est ce qui permet de
le situer (mi-corde, bord de fuite, sillage) sans interpretation.

USAGE (sur le cluster)
   cd ~/cdf_runs/film_a15
   python3 ~/bin/spectre_film.py | tee spectre_film.txt

LIMITE : 1.5 CTU de signal -> resolution en frequence ~0.7. Les pics sont donnes
a +/- 0.35 pres ; les rms de bande sont fiables pour COMPARER les sondes entre elles.
"""
import numpy as np, os, sys

BANDES = [("lacher", 0.8, 2.6), ("double", 2.6, 4.6), ("rapide", 5.5, 10.5)]
T_SKIP = 0.05          # CTU ecartes au debut (redemarrage)


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


def uniforme(t):
    """plus long bloc initial a pas d'echantillonnage constant (a 1 % pres)."""
    d = np.diff(t)
    med = np.median(d)
    bad = np.where(np.abs(d - med) > 0.01 * med)[0]
    fin = bad[0] + 1 if len(bad) else len(t)
    return slice(0, fin), med


def analyse(t, y):
    """rms total, rms par bande (Hann, puissance corrigee), pic de la bande rapide."""
    y = y - y.mean()
    n = len(y)
    w = np.hanning(n)
    X = np.fft.rfft(y * w)
    f = np.fft.rfftfreq(n, t[1] - t[0])
    p = 2.0 * np.abs(X) ** 2 / (n ** 2 * np.mean(w ** 2))   # rms^2 par raie (une face)
    out = {"rms": float(y.std())}
    for nom, a, b in BANDES:
        m = (f >= a) & (f < b)
        out[nom] = float(np.sqrt(p[m].sum()))
    # pic de la bande rapide : bourrage x8 + interpolation parabolique
    Z = np.abs(np.fft.rfft(y * w, 8 * n))
    fz = np.fft.rfftfreq(8 * n, t[1] - t[0])
    m = np.where((fz >= BANDES[2][1]) & (fz < BANDES[2][2]))[0]
    k = m[np.argmax(Z[m])]
    if 0 < k < len(Z) - 1:
        a_, b_, c_ = np.log(Z[k - 1] + 1e-300), np.log(Z[k] + 1e-300), np.log(Z[k + 1] + 1e-300)
        dk = 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_) if (a_ - 2 * b_ + c_) != 0 else 0.0
    else:
        dk = 0.0
    out["pic_rapide"] = float(fz[k] + dk * (fz[1] - fz[0]))
    return out


def ligne(nom, r):
    frac = r["rapide"] / r["rms"] * 100 if r["rms"] > 0 else 0.0
    return (f" {nom:12s} {r['rms']:10.3e}  {r['lacher']:10.3e} {r['double']:10.3e} "
            f"{r['rapide']:10.3e}  {frac:5.1f} %   {r['pic_rapide']:6.2f}")


print("=" * 96)
print(" SPECTRE A PAS CONSTANT — film_a15 (15 deg)   rms par bande de St (base corde)")
print("=" * 96)
print(f" {'signal':12s} {'rms total':>10s}  {'[0.8-2.6]':>10s} {'[2.6-4.6]':>10s} "
      f"{'[5.5-10.5]':>10s}  {'rapide/':>7s}   {'pic St':>6s}")
print(f" {'':12s} {'':10s}  {'lacher':>10s} {'double':>10s} {'rapide':>10s}  {'total':>7s}")
print("-" * 96)

# --- efforts (chaque pas) -------------------------------------------------------
if os.path.isfile("drag_results.dat"):
    a = charger("drag_results.dat")
    a = a[(np.abs(a[:, 3]) < 2) & (a[:, 2] > 0) & (a[:, 2] < 2)]
    t = a[:, 1]
    m = t >= t[0] + T_SKIP
    t, a = t[m], a[m]
    sl, dts = uniforme(t)
    print(f" (efforts : {sl.stop} echantillons, pas {dts:.2e}, duree {t[sl.stop-1]-t[0]:.3f} CTU)")
    for nom, j in (("CL", 3), ("CD", 2)):
        print(ligne(nom, analyse(t[sl], a[sl, j])))
else:
    print(" drag_results.dat absent")

# --- sondes ---------------------------------------------------------------------
if os.path.isfile("probes_uvp.csv"):
    h = [c.strip() for c in open("probes_uvp.csv").readline().strip().split(",")]
    s = charger("probes_uvp.csv", ",")
    t = s[:, 0]
    m = t >= t[0] + T_SKIP
    t, s = t[m], s[m]
    sl, dts = uniforme(t)
    print(f" (sondes : {sl.stop} echantillons, pas {dts:.2e}, duree {t[sl.stop-1]-t[0]:.3f} CTU)")
    groupes = [("A", "ux"), ("B", "uy"), ("W", "uy"), ("REF", "uy"), ("SPAN", "uy")]
    for pref, comp in groupes:
        noms = [n for n in h if n.startswith(comp + "_" + pref)]
        for n in noms:
            print(ligne(n, analyse(t[sl], s[sl, h.index(n)])))
else:
    print(" probes_uvp.csv absent")
print("-" * 96)
print(" rms de bande : Hann, puissance corrigee ; resolution ~ 1/duree (~0.7) -> pics a +/- 0.35")
print(" rapide/total : part du mode rapide dans le rms du signal")

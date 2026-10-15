#!/usr/bin/env python3
"""
compare_N3_N4.py  --  test de resolution a 15 deg : N = 3 (film_a15) contre N = 4 (res_a15_N4)
=========================================================================================
Les deux runs partent du MEME champ (film_a15/airfoil0.f00000, t = 100.0062) avec le MEME
pas de temps (3.5e-5). Seul l'ordre polynomial change. Si le mode rapide du sillage proche
(St ~ 10.6) est physique, sa frequence et son amplitude doivent se conserver a N = 4. S'il
est lie a la grille, il change fortement ou disparait.

Comme l'etat initial est identique, les grandes echelles (lacher a St ~ 1.75) doivent rester
tres proches pendant le premier CTU : c'est le controle que la comparaison est equitable.

USAGE (sur le cluster)
   python3 -u ~/bin/compare_N3_N4.py                              # dossiers par defaut
   python3 -u ~/bin/compare_N3_N4.py DOSSIER_N3 DOSSIER_N4 [t0]  # t0 defaut = 100.3
   (t0 ecarte le transitoire de l'interpolation N3 -> N4 : 0.3 CTU = 3 periodes du mode rapide)

SORTIES
   tableau a l'ecran (a copier avec | tee compare_N3_N4.txt)
   compare_N3_N4.png dans DOSSIER_N4
"""
import os, sys
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

D3 = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser("~/cdf_runs/film_a15")
D4 = os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else os.path.expanduser("~/cdf_runs/res_a15_N4")
T0 = float(sys.argv[3]) if len(sys.argv) > 3 else 100.3

BANDES = [("lacher", 0.8, 2.6), ("double", 2.6, 4.6), ("7-9", 6.5, 9.5),
          ("10-12", 9.5, 12.0), ("> 12", 12.0, 60.0)]
SIGNAUX = ["CL", "uy_W015m", "uy_W015p", "uy_W050m", "ux_A020", "ux_A050", "ux_A092",
           "uy_B050", "uy_B070", "uy_B090", "ux_B090"]   # B090 : couche cisaillee decollee (0.9c)
C3, C4 = "#2a78d6", "#eb6834"          # N = 3 bleu, N = 4 orange (palette validee)
GRIS, ENCRE, ENCRE2 = "#d9d8d3", "#1f1f1e", "#5f5e58"


# ------------------------------------------------------------------ lecture
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


def lire(d):
    """{nom: (t, y)} pour CL, CD et toutes les sondes."""
    out = {}
    p = os.path.join(d, "drag_results.dat")
    if os.path.isfile(p):
        a = charger(p)
        a = a[(np.abs(a[:, 3]) < 2) & (a[:, 2] > 0) & (a[:, 2] < 2)]
        a = a[monotone(a[:, 1])]
        out["CL"], out["CD"] = (a[:, 1], a[:, 3]), (a[:, 1], a[:, 2])
    p = os.path.join(d, "probes_uvp.csv")
    if os.path.isfile(p):
        h = [c.strip() for c in open(p).readline().strip().split(",")]
        s = charger(p, ",")
        s = s[monotone(s[:, 0])]
        for j, n in enumerate(h[1:], start=1):
            out[n] = (s[:, 0], s[:, j])
    return out


# ------------------------------------------------------------------ spectres
def grille(t, y, t0, t1, dt):
    tr = np.arange(t0, t1, dt)
    return tr, np.interp(tr, t, y)


def spectre(y, dt, pad=8):
    """amplitude d'une sinusoide (Hann corrige) et puissance rms^2 par raie."""
    y = y - y.mean()
    n = len(y)
    w = np.hanning(n)
    X = np.fft.rfft(y * w, pad * n)
    f = np.fft.rfftfreq(pad * n, dt)
    A = 2.0 * np.abs(X) / w.sum()
    Xb = np.fft.rfft(y * w)
    fb = np.fft.rfftfreq(n, dt)
    pw = 2.0 * np.abs(Xb) ** 2 / (n ** 2 * np.mean(w ** 2))
    return f, A, fb, pw


def pic(f, A, a, b):
    m = np.where((f >= a) & (f <= b))[0]
    if len(m) == 0:
        return np.nan, np.nan
    k = m[np.argmax(A[m])]
    # un vrai pic est un maximum local ; un maximum colle au bord de la bande n'est
    # que le pied d'un pic voisin (fuite spectrale) -> pas de pic dans cette bande
    if k in (m[0], m[-1]) or not (0 < k < len(A) - 1) or A[k] < A[k - 1] or A[k] < A[k + 1]:
        return np.nan, np.nan
    dk = 0.0
    if min(A[k - 1], A[k], A[k + 1]) > 0:
        a_, b_, c_ = np.log(A[k - 1]), np.log(A[k]), np.log(A[k + 1])
        den = a_ - 2 * b_ + c_
        dk = float(np.clip(0.5 * (a_ - c_) / den, -1, 1)) if den != 0 else 0.0
    return f[k] + dk * (f[1] - f[0]), A[k]


def passe_haut(y, dt, fc=6.0):
    Y = np.fft.rfft(y - y.mean())
    Y[np.fft.rfftfreq(len(y), dt) < fc] = 0
    return np.fft.irfft(Y, len(y))


# ------------------------------------------------------------------ calcul
S3, S4 = lire(D3), lire(D4)
if not S3 or not S4:
    sys.exit(f"donnees manquantes : {D3 if not S3 else D4}")

communs = [n for n in SIGNAUX if n in S3 and n in S4]
if "uy_W015m" not in communs:
    print(" ATTENTION : uy_W015m absent d'un des deux runs ; sondes communes :", communs)

t1 = min(S3[n][0][-1] for n in communs if n != "CL")
t1 = min(t1, S3["CL"][0][-1], S4["CL"][0][-1]) if "CL" in communs else t1
t1 = min(t1, min(S4[n][0][-1] for n in communs))
duree = t1 - T0
print("=" * 96)
print(f" TEST DE RESOLUTION  N = 3 : {D3}")
print(f"                     N = 4 : {D4}")
print(f" fenetre commune [{T0:.3f} ; {t1:.3f}] = {duree:.3f} CTU  ->  resolution en St ~ {1/max(duree,1e-9):.2f}")
print("=" * 96)
if duree < 0.3:
    sys.exit(" fenetre trop courte (< 0.3 CTU) : attendre que le run N = 4 avance.")
if duree < 1.5:
    print(f" NB : avec {duree:.2f} CTU, les bandes voisines (7-9 et 10-12) se contaminent par fuite")
    print("      spectrale ; comparer surtout les RAPPORTS N4/N3, qui restent valables.")

RES = {}
for n in communs:
    dt = max(np.median(np.diff(S3[n][0])), np.median(np.diff(S4[n][0])))
    t, y3 = grille(*S3[n], T0, t1, dt)
    _, y4 = grille(*S4[n], T0, t1, dt)
    r = {"t": t, "y3": y3, "y4": y4, "dt": dt}
    for tag, y in (("3", y3), ("4", y4)):
        f, A, fb, pw = spectre(y, dt)
        r["f" + tag], r["A" + tag] = f, A
        r["rms" + tag] = float(np.std(y))
        r["b" + tag] = {nom: float(np.sqrt(pw[(fb >= a) & (fb < b)].sum())) for nom, a, b in BANDES}
        r["p10_" + tag] = pic(f, A, 9.5, 12.0)
        r["p8_" + tag] = pic(f, A, 6.5, 9.5)
        r["p12_" + tag] = pic(f, A, 12.0, 60.0)
        r["moy" + tag] = float(np.mean(y))
        r["rapide" + tag] = float(np.sqrt(pw[fb >= 6.0].sum()))
    RES[n] = r

print(f"\n {'signal':10s} {'bande':>8s} {'rms N=3':>11s} {'rms N=4':>11s} {'N4/N3':>7s}")
print("-" * 96)
for n, r in RES.items():
    for nom, _, _ in BANDES:
        a3, a4 = r["b3"][nom], r["b4"][nom]
        rat = a4 / a3 if a3 > 0 else np.nan
        print(f" {n:10s} {nom:>8s} {a3:11.3e} {a4:11.3e} {rat:7.2f}")
    print(f" {'':10s} {'pic 10-12':>8s}  St {r['p10_3'][0]:6.2f} (N=3)   St {r['p10_4'][0]:6.2f} (N=4)")
    print(f" {'':10s} {'pic 7-9':>8s}  St {r['p8_3'][0]:6.2f} (N=3)   St {r['p8_4'][0]:6.2f} (N=4)")
    print(f" {'':10s} {'pic > 12':>8s}  St {r['p12_3'][0]:6.2f} (N=3)   St {r['p12_4'][0]:6.2f} (N=4)")
    rr = r["rapide4"] / r["rapide3"] if r["rapide3"] > 0 else np.nan
    print(f" {'':10s} {'St > 6':>8s} {r['rapide3']:11.3e} {r['rapide4']:11.3e} {rr:7.2f}   (tout le contenu rapide)")
    print("-" * 96)

# moyennes : la resolution change-t-elle les valeurs du rapport ?
for n in ("CL", "CD"):
    if n in S3 and n in S4:
        dtn = max(np.median(np.diff(S3[n][0])), np.median(np.diff(S4[n][0])))
        m3 = grille(*S3[n], T0, t1, dtn)[1].mean()
        m4 = grille(*S4[n], T0, t1, dtn)[1].mean()
        print(f" {n} moyen sur la fenetre : N=3 {m3:.5f}   N=4 {m4:.5f}   ecart {100*(m4-m3)/m3:+.2f} %")
print("-" * 96)

# lecture automatique, prudente
if "uy_W015m" in RES:
    r = RES["uy_W015m"]
    rat = r["b4"]["10-12"] / r["b3"]["10-12"] if r["b3"]["10-12"] > 0 else np.nan
    df = r["p10_4"][0] - r["p10_3"][0]
    ctrl = RES["CL"]["b4"]["lacher"] / RES["CL"]["b3"]["lacher"] if "CL" in RES and RES["CL"]["b3"]["lacher"] > 0 else np.nan
    print(" LECTURE (uy_W015m, bande 10-12) :")
    print(f"   rapport d'amplitude N4/N3 = {rat:.2f} ; decalage du pic = {df:+.2f} en St")
    print(f"   controle : lacher dans CL, rapport N4/N3 = {ctrl:.2f} (doit rester proche de 1)")
    if 0.7 <= rat <= 1.3 and abs(df) < 0.5:
        print("   -> mode CONSERVE a N = 4 : compatible avec un mode physique resolu.")
    elif rat < 0.5 or abs(df) > 1.0:
        print("   -> mode FORTEMENT MODIFIE a N = 4 : signature d'un mode lie a la discretisation.")
    else:
        print("   -> cas intermediaire : allonger la fenetre avant de conclure.")
print("=" * 96)


# ------------------------------------------------------------------ figure
if not HAS_MPL:
    print(" matplotlib absent sur cette machine : tableaux seulement, pas de figures.")
    sys.exit(0)
def habiller(ax, titre, xl, yl):
    ax.set_title(titre, loc="left", fontsize=10, color=ENCRE)
    ax.set_xlabel(xl, color=ENCRE2); ax.set_ylabel(yl, color=ENCRE2)
    ax.grid(True, color=GRIS, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ENCRE2)
    ax.tick_params(colors=ENCRE2, labelsize=8)


fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
fig.patch.set_facecolor("#fcfcfb")

if "uy_W015m" in RES:
    r = RES["uy_W015m"]
    ax = axs[0, 0]
    ax.plot(r["t"], passe_haut(r["y3"], r["dt"]), color=C3, lw=1.5, label="N = 3")
    ax.plot(r["t"], passe_haut(r["y4"], r["dt"]), color=C4, lw=1.5, label="N = 4")
    habiller(ax, f"(a) u_y at W015m (0.15c behind TE), content above St = 6\n15°, same initial field, window {T0:.2f}–{t1:.2f}",
             "t U∞/c", "u_y' / U∞")
    ax.legend(frameon=False, fontsize=9)

if "CL" in RES:
    r = RES["CL"]
    ax = axs[0, 1]
    ax.plot(r["t"], r["y3"], color=C3, lw=1.5, label="N = 3")
    ax.plot(r["t"], r["y4"], color=C4, lw=1.5, label="N = 4")
    habiller(ax, "(b) Lift coefficient, unfiltered\n(large scales should coincide)", "t U∞/c", "C_L")
    ax.legend(frameon=False, fontsize=9)

for ax, n, titre in ((axs[1, 0], "uy_W015m", "(c) Spectrum of u_y at W015m (near wake)"),
                     (axs[1, 1], "ux_A050", "(d) Spectrum of u_x at A050 (suction side, mid-chord)")):
    if n not in RES:
        ax.set_visible(False); continue
    r = RES[n]
    m3, m4 = (r["f3"] > 0.3) & (r["f3"] < 30), (r["f4"] > 0.3) & (r["f4"] < 30)
    ax.semilogy(r["f3"][m3], r["A3"][m3], color=C3, lw=1.5, label="N = 3")
    ax.semilogy(r["f4"][m4], r["A4"][m4], color=C4, lw=1.5, label="N = 4")
    for fr in (1.75, 3.49, 7.8, 10.66):
        ax.axvline(fr, color=GRIS, lw=1, ls=":")
    habiller(ax, titre, "St = f c / U∞", "amplitude")
    ax.legend(frameon=False, fontsize=9)

out = os.path.join(D4, "compare_N3_N4.png")
fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
print(f" figure : {out}")

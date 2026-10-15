#!/usr/bin/env python3
"""
paroi_film.py  --  frottement et pression a la paroi, moyens ET instantanes (run film_a15, 15 deg)
===================================================================================================
Lit tous les dumps de paroi resultats_csv/cf_cp_time_*.csv (film_a15 : un dump par sortie, soit
239 instantanes sur 1.5 CTU, pas 0.0063). Repond aux trois questions de F. Mellibovsky :

 1. C_f SIGNE (+ du bord d'attaque vers le bord de fuite) sur l'extrados et l'intrados :
    decollement et recollement sur le champ moyen, etendue du decollement sur les instantanes,
    fraction du temps ou C_f < 0 en chaque point.
 2. Les bandes de courte longueur d'onde (intrados, pres du BF) sont-elles CALEES SUR LA GRILLE ?
    Dans chaque element, la fluctuation de paroi est projetee sur les polynomes de Legendre aux
    points GLL. Part d'energie dans le mode le plus haut (N) :
      - onde de 2 elements de long ou plus            -> moins de 1 %
      - alternance d'un noeud GLL a l'autre (grille)  -> 50 a 75 %
    Le script imprime la table d'etalonnage (valeur attendue selon la longueur d'onde, en
    longueurs d'element) pour lire le resultat sans interpretation.
 3. Ce qui se propage : vitesse de phase et longueur d'onde du mode dominant de chaque face,
    a partir de la pression parietale, avec les frontieres d'elements sur les diagrammes
    espace-temps.

USAGE
   cd ~/cdf_runs/film_a15 && python3 -u ~/bin/paroi_film.py | tee paroi_film.txt
   cd ~/cdf_runs/res_a15_N4 && python3 -u ~/bin/paroi_film.py . 102 | tee paroi_N4.txt
   (arguments optionnels : le dossier du cas, puis le temps a partir duquel lire les dumps)
SORTIES (dans le dossier du cas)
   paroi_film.txt (via tee), paroi_film_cf.png, paroi_film_xt.png

Convention (Tutoriel 4.2) : colonnes x,y,z,cf_x,cf_y,cf_z,Cp ; cf_x = C_f scalaire DEJA signe
(+ vers le bord de fuite). Bord de fuite a l'origine du maillage.
"""
import glob, os, re, sys
import numpy as np
from numpy.polynomial import legendre as L
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

D = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else "."
T0 = float(sys.argv[2]) if len(sys.argv) > 2 else -np.inf     # ne garder que les dumps t >= T0
RX_CF = re.compile(r"cf_cp_time_([0-9.eE+-]+?)(?:_rank_\d+)?\.csv$")
CE, CI = "#2a78d6", "#eb6834"                  # extrados bleu, intrados orange (palette validee)
GRIS, ENCRE, ENCRE2 = "#d9d8d3", "#1f1f1e", "#5f5e58"
DIV = (LinearSegmentedColormap.from_list("div", ["#104281", "#f0efec", "#a3201f"])
       if HAS_MPL else None)
REG = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 0.97), (0.97, 1.001)]


# =============================================================================== lecture
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


def index_paroi(d):
    out = {}
    for f in glob.glob(os.path.join(d, "resultats_csv", "cf_cp_time_*.csv")):
        m = RX_CF.search(os.path.basename(f))
        if m:
            try:
                out.setdefault(round(float(m.group(1)), 6), []).append(f)
            except ValueError:
                pass
    return out


def lire_dump(fichiers):
    """lignes (x, y, z, cf, cp) dans l'ordre des fichiers (rang par rang)."""
    blocs = []
    for f in sorted(fichiers):
        with open(f, errors="ignore") as fh:
            h = [c.strip().lower() for c in fh.readline().strip().lstrip("#").split(",")]
        a = charger(f, ",")
        if a.ndim != 2 or len(a) == 0:
            continue
        col = lambda *c: next((h.index(n) for n in c if n in h), None)
        ix, iy, iz, icf, icp = col("x"), col("y"), col("z"), col("cf_x", "cfx", "cf"), col("cp")
        if None in (ix, iy, icf, icp):
            raise ValueError(f"colonnes inattendues : {h}")
        z = a[:, iz] if iz is not None else np.zeros(len(a))
        blocs.append(np.column_stack([a[:, ix], a[:, iy], z, a[:, icf], a[:, icp]]))
    return np.vstack(blocs) if blocs else None


# =============================================================================== geometrie
def geometrie(x, y):
    """x/c le long de la corde (0 au BA), distance normale a la corde (> 0 : extrados)."""
    i_bf = int(np.argmin(x ** 2 + y ** 2))
    xb, yb = x[i_bf], y[i_bf]
    i_ba = int(np.argmax((x - xb) ** 2 + (y - yb) ** 2))
    xa, ya = x[i_ba], y[i_ba]
    c = np.hypot(xb - xa, yb - ya)
    ex, ey = (xb - xa) / c, (yb - ya) / c
    return ((x - xa) * ex + (y - ya) * ey) / c, (-(x - xa) * ey + (y - ya) * ex) / c, c


def cle(x, y):
    return np.round(x, 7) + 1j * np.round(y, 7)


def detecter_N(x, y):
    """ordre polynomial : le dump ecrit face par face (N+1)^2 points, dont N+1 positions (x, y)
    distinctes le long de la paroi, chacune repetee sur N+1 plans z."""
    k = cle(x, y)
    for N in range(1, 11):
        B = (N + 1) ** 2
        if len(k) % B:
            continue
        ok = True
        for f in range(min(40, len(k) // B)):
            u, c = np.unique(k[f * B:(f + 1) * B], return_counts=True)
            if len(u) != N + 1 or np.any(c != N + 1):
                ok = False
                break
        if ok:
            return N
    return None


def structure(a, N):
    """pour chaque ligne du dump : indice global de noeud g = face*(N+1) + position le long
    de la face (ordonnee d'un bout a l'autre). Renvoie g, et X, Y de chaque noeud."""
    B, n1 = (N + 1) ** 2, N + 1
    nf = len(a) // B
    g = np.empty(len(a), int)
    X, Y = np.empty(nf * n1), np.empty(nf * n1)
    k = cle(a[:, 0], a[:, 1])
    for f in range(nf):
        sl = slice(f * B, (f + 1) * B)
        u, inv = np.unique(k[sl], return_inverse=True)
        px, py = u.real, u.imag
        d = np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :])
        i0 = np.unravel_index(np.argmax(d), d.shape)[0]
        ordre = np.argsort(d[i0])                       # d'une extremite a l'autre
        rang = np.empty(n1, int); rang[ordre] = np.arange(n1)
        g[sl] = f * n1 + rang[inv]
        X[f * n1 + rang], Y[f * n1 + rang] = px, py
    return g, X, Y, nf


# =============================================================================== outils
def gll(N):
    r = np.sort(L.Legendre.basis(N).deriv().roots().real)
    return np.concatenate(([-1.0], r, [1.0]))


def etalonnage(N):
    """part d'energie dans le mode N pour une sinusoide de longueur d'onde lambda = r * L_elem."""
    xi = gll(N)
    Vi = np.linalg.inv(L.legvander(xi, N))
    nrm = 2.0 / (2 * np.arange(N + 1) + 1)
    out = []
    for r in (2.0 / N, 0.5, 1.0, 1.5, 2.0, 4.0):
        E = np.zeros(N + 1)
        for ph in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            a = Vi @ np.sin(np.pi * xi / r + ph)
            E += a ** 2 * nrm
        out.append((r, E[N] / E.sum()))
    return out


def croisements(xs, cf, x_min=0.005, x_max=0.995):
    """zones ou Cf < 0 entre x_min et x_max, bornes interpolees."""
    m = (xs >= x_min) & (xs <= x_max)
    xs, cf = xs[m], cf[m]
    zones, debut = [], (float(xs[0]) if len(xs) and cf[0] < 0 else None)
    for i in range(len(xs) - 1):
        xi = float(xs[i] + cf[i] / (cf[i] - cf[i + 1]) * (xs[i + 1] - xs[i])) if cf[i] != cf[i + 1] else float(xs[i])
        if debut is None and cf[i] > 0 >= cf[i + 1]:
            debut = xi
        elif debut is not None and cf[i] <= 0 < cf[i + 1]:
            zones.append((debut, xi)); debut = None
    if debut is not None:
        zones.append((debut, float(xs[-1])))
    return zones


def habiller(ax, titre, xl, yl):
    ax.set_title(titre, loc="left", fontsize=10, color=ENCRE)
    ax.set_xlabel(xl, color=ENCRE2); ax.set_ylabel(yl, color=ENCRE2)
    ax.grid(True, color=GRIS, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ENCRE2)
    ax.tick_params(colors=ENCRE2, labelsize=8)


# =============================================================================== chargement
idx = index_paroi(D)
ts = sorted(t for t in idx if t >= T0)
if not ts:
    sys.exit(f" aucun dump de paroi (t >= {T0}) dans {os.path.join(D, 'resultats_csv')}")
# Un fichier peut contenir plusieurs blocs : si deux runs ont ecrit au meme instant, l'udf ouvre
# le fichier en mode ajout. On lit tout, on prend la plus petite taille comme taille d'un dump,
# et on ne garde que le DERNIER bloc de chaque fichier (le run le plus recent).
DUMPS = {t: lire_dump(idx[t]) for t in ts}
DUMPS = {t: a for t, a in DUMPS.items() if a is not None and len(a)}
ts = sorted(DUMPS)
n_ref = min(len(a) for a in DUMPS.values())
doubles = []
for t, a in DUMPS.items():
    if len(a) != n_ref:
        if len(a) % n_ref:
            sys.exit(f" dump a t = {t} : {len(a)} lignes, pas un multiple de {n_ref}")
        doubles.append((t, len(a) // n_ref))
        DUMPS[t] = a[-n_ref:]
a0 = DUMPS[ts[0]]
N = detecter_N(a0[:, 0], a0[:, 1])
print("=" * 100)
print(f" PAROI — {os.path.abspath(D)}")
print(f" {len(ts)} instantanes, t = {ts[0]:.4f} -> {ts[-1]:.4f} ({ts[-1]-ts[0]:.3f} CTU)"
      + (f", pas median {np.median(np.diff(ts)):.4f}" if len(ts) > 1 else ""))
if N is None:
    sys.exit(" structure face par face non reconnue dans le dump : analyse modale impossible.")
g, X, Y, nf = structure(a0, N)
n1 = N + 1
cnt = np.bincount(g, minlength=nf * n1)
print(f" ordre polynomial detecte : N = {N} ; {nf} faces d'element a la paroi ; {len(a0)} lignes par dump")
for t, k in doubles:
    print(f" NB : le dump a t = {t} contient {k} blocs (plusieurs runs) : seul le dernier est garde")

nt = len(ts)
CF = np.empty((nt, nf * n1)); CP = np.empty((nt, nf * n1))
ref = a0[:, :3]
for i, t in enumerate(ts):
    a = DUMPS[t]
    if not np.allclose(a[:, :3], ref, atol=1e-9):          # ordre different : on realigne
        o_ref = np.lexsort((ref[:, 2], ref[:, 1], ref[:, 0]))
        o_a = np.lexsort((a[:, 2], a[:, 1], a[:, 0]))
        b = np.empty_like(a); b[o_ref] = a[o_a]; a = b
    CF[i] = np.bincount(g, a[:, 3], minlength=nf * n1) / cnt
    CP[i] = np.bincount(g, a[:, 4], minlength=nf * n1) / cnt
T = np.array(ts)

xc, yn, corde = geometrie(X, Y)
cote_face = np.where(yn.reshape(nf, n1).mean(1) > 0, 1, -1)          # +1 extrados, -1 intrados
Lel = np.hypot(X.reshape(nf, n1)[:, -1] - X.reshape(nf, n1)[:, 0],
               Y.reshape(nf, n1)[:, -1] - Y.reshape(nf, n1)[:, 0]) / corde
xf = xc.reshape(nf, n1).mean(1)


# =============================================================================== stations
def stations(cote):
    """points distincts d'une face (fusion des noeuds partages aux interfaces), tries en x/c."""
    faces = np.where(cote_face == cote)[0]
    gl = (faces[:, None] * n1 + np.arange(n1)[None, :]).ravel()
    k = cle(X[gl], Y[gl])
    u, inv, c = np.unique(k, return_inverse=True, return_counts=True)
    M = np.zeros((len(u), len(gl)))
    M[inv, np.arange(len(gl))] = 1.0
    M /= M.sum(1, keepdims=True)
    xs = M @ xc[gl]
    o = np.argsort(xs)
    interf = (c >= 2)[o]
    return dict(x=xs[o], cf=(CF[:, gl] @ M.T)[:, o], cp=(CP[:, gl] @ M.T)[:, o], interf=interf,
                X=(M @ X[gl])[o], Y=(M @ Y[gl])[o])


S = {+1: stations(+1), -1: stations(-1)}
NOM = {+1: "EXTRADOS", -1: "INTRADOS"}
LAB = {+1: "suction side", -1: "pressure side"}

# =============================================================================== 1. Cf signe
print("\n" + "=" * 100)
print(" 1. FROTTEMENT PARIETAL SIGNE (+ du BA vers le BF)")
print("=" * 100)
for s in (+1, -1):
    st = S[s]
    cfm = st["cf"].mean(0)
    x = st["x"]
    ext = (x > 0.05) & (x < 0.995)
    print(f" {NOM[s]} : Cf moyen de {cfm[ext].min():+.2e} a {cfm[ext].max():+.2e} sur 0.05 < x/c < 0.995")
    if s == -1:
        m = x < 0.2
        stag = croisements(x[m], -cfm[m], 0.0, 0.2)       # passage de - a + vers l'aval
        if (cfm[m] < 0).any():
            neg = x[m][cfm[m] < 0]
            print(f"   point d'arret (Cf change de signe pres du BA) : x/c ~ {neg.max():.4f}")
        zones = croisements(x, cfm, 0.2, 0.995)
    else:
        zones = croisements(x, cfm, 0.005, 0.995)
    if zones:
        for z0, z1 in zones:
            print(f"   zone de Cf moyen < 0 : x/c {z0:.4f} -> {z1:.4f}"
                  + ("  (jusqu'au BF)" if z1 >= x[x <= 0.995].max() - 1e-6 else "  (recollement)"))
    else:
        print("   aucune zone de Cf moyen < 0 (hors voisinage du BA pour l'intrados)")
    fneg = (st["cf"] < 0).mean(0)
    for x0, x1 in ((0.5, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.995)):
        m = (x >= x0) & (x < x1)
        if m.any():
            print(f"   x/c {x0:.2f}-{x1:.3f} : Cf < 0 pendant {100*fneg[m].mean():5.1f} % du temps "
                  f"(max {100*fneg[m].max():5.1f} %)")
    if s == +1 and nt > 1:
        sep = []
        for i in range(nt):
            z = croisements(x, st["cf"][i], 0.005, 0.995)
            sep.append(z[0][0] if z else np.nan)
        sep = np.array(sep)
        ok = np.isfinite(sep)
        if ok.any():
            print(f"   decollement instantane : present sur {100*ok.mean():.0f} % des instantanes, "
                  f"x/c min {np.nanmin(sep):.3f}  mediane {np.nanmedian(sep):.3f}  max {np.nanmax(sep):.3f}")
        else:
            print("   decollement instantane : aucun instantane avec Cf < 0 avant x/c = 0.995")

# =============================================================================== 2. grille
print("\n" + "=" * 100)
print(" 2. LES FLUCTUATIONS SONT-ELLES A L'ECHELLE DE LA GRILLE ?")
print("=" * 100)
xi = gll(N)
Vi = np.linalg.inv(L.legvander(xi, N))
nrm = 2.0 / (2 * np.arange(N + 1) + 1)


def modal(V):                                        # V : (nt, nf*n1) -> energie (nt, nf, N+1)
    u = V.reshape(len(V), nf, n1)
    a = np.einsum("jk,tfk->tfj", Vi, u)
    return a ** 2 * nrm


Ep = {}
for nom, V in (("Cp", CP), ("Cf", CF)):
    Vf = V - V.mean(0)
    E = modal(Vf).mean(0)                            # (nf, N+1) energie moyenne de la fluctuation
    Em = modal(V.mean(0)[None])[0]                   # energie du champ moyen
    Ep[nom] = dict(fl=E, tot=E.sum(1), R=E[:, N] / np.maximum(E.sum(1), 1e-300),
                   Rm=Em[:, N] / np.maximum(Em[:, 1:].sum(1), 1e-300))

cal = etalonnage(N)
print(f" ETALONNAGE (N = {N}) : part d'energie dans le mode {N} pour une onde reguliere de longueur d'onde")
print("   " + "   ".join(f"lambda = {r:.2g} L_el : {100*v:4.1f} %" for r, v in cal))
print(f" (L_el = longueur de l'element le long de la paroi. lambda = {2/N:.2g} L_el = alternance d'un noeud GLL")
print("  a l'autre : c'est la signature d'une oscillation de grille. Une onde de 2 elements ou plus")
print("  met moins de 1 % dans le mode N : ce critere ne voit que ce qui est plus court qu'un element.)\n")
print(f" {'face':9s} {'x/c':>11s} {'L_el/c':>7s} {'rms Cp_':>9s} {'mode N Cp_':>10s} {'mode N Cf_':>10s} {'mode N Cp moy':>13s}")
print("   (Cp_, Cf_ = fluctuations autour de la moyenne temporelle ; '--' = fluctuation negligeable)")
print("-" * 100)
for s in (+1, -1):
    for x0, x1 in REG:
        m = (cote_face == s) & (xf >= x0) & (xf < x1)
        if not m.any():
            continue
        w = Ep["Cp"]["tot"][m]
        wf = Ep["Cf"]["tot"][m]
        rc = np.sum(Ep["Cp"]["R"][m] * w) / max(w.sum(), 1e-300)
        rf = np.sum(Ep["Cf"]["R"][m] * wf) / max(wf.sum(), 1e-300)
        rm = np.median(Ep["Cp"]["Rm"][m])
        st = S[s]; mm = (st["x"] >= x0) & (st["x"] < x1)
        rms = st["cp"][:, mm].std(0).mean() if mm.any() and nt > 1 else np.nan
        seuil = 1e-3 * max(S[+1]["cp"].std(0).max(), S[-1]["cp"].std(0).max()) if nt > 1 else 0
        txt = (f"{100*rc:9.1f}% {100*rf:9.1f}%" if (np.isfinite(rms) and rms > seuil)
               else f"{'--':>10s} {'--':>10s}")
        print(f" {NOM[s][:9]:9s} {x0:5.2f}-{min(x1,1):5.2f} {np.median(Lel[m]):7.4f} {rms:9.2e} "
              f"{txt} {100*rm:12.1f}%")
    print("-" * 100)
top = np.argsort(Ep["Cp"]["tot"])[::-1][:10] if nt > 1 else []
if nt > 1:
    print(" les 10 faces ou la fluctuation de Cp est la plus forte :")
for f in top:
    print(f"   {NOM[cote_face[f]][:8]:8s} x/c {xc[f*n1:(f+1)*n1].min():.4f}-{xc[f*n1:(f+1)*n1].max():.4f} "
          f"L_el {Lel[f]:.4f}c  rms Cp' {np.sqrt(Ep['Cp']['tot'][f]/2):.2e}  mode {N} : Cp' {100*Ep['Cp']['R'][f]:4.1f} %"
          f"  Cf' {100*Ep['Cf']['R'][f]:4.1f} %")

# =============================================================================== 3. propagation
print("\n" + "=" * 100)
print(" 3. CE QUI SE PROPAGE LE LONG DE LA PAROI (pression parietale, 0.5 < x/c < 0.995)")
print("=" * 100)
PROP = {}
if nt < 16:
    print(" trop peu d'instantanes pour une analyse en frequence.")
else:
    dt = np.median(np.diff(T))
    f = np.fft.rfftfreq(nt, dt)
    w = np.hanning(nt)[:, None]
    for s in (+1, -1):
        st = S[s]
        m = (st["x"] > 0.5) & (st["x"] < 0.995)
        V = st["cp"][:, m] - st["cp"][:, m].mean(0)
        Z = np.fft.rfft(V * w, axis=0)
        Pm = (np.abs(Z) ** 2).mean(1)
        k_ok = np.where((f > 2.5) & (f < 40))[0]
        kk = [k for k in k_ok if Pm[k] >= Pm[k - 1] and Pm[k] >= Pm[min(k + 1, len(Pm) - 1)]]
        kk = sorted(kk, key=lambda k: -Pm[k])[:3]
        print(f" {NOM[s]} : pics du spectre de Cp' moyen sur la zone : "
              + ", ".join(f"St {f[k]:.2f}" for k in kk) + f"  (resolution {1/(nt*dt):.2f})")
        xs = st["x"][m]
        sarc = np.r_[0, np.cumsum(np.hypot(np.diff(st["X"][m]), np.diff(st["Y"][m])))] / corde
        for k in kk[:2]:
            A = np.abs(Z[k]); ph = np.unwrap(np.angle(Z[k]))
            gm = A > 0.3 * A.max()
            bl = np.split(np.arange(len(gm)), np.where(np.diff(gm.astype(int)) != 0)[0] + 1)
            b = max([b for b in bl if gm[b[0]]], key=len)
            if len(b) < 4:
                continue
            p = np.polyfit(sarc[b], ph[b], 1)
            res = float(np.std(ph[b] - np.polyval(p, sarc[b])))
            c = 2 * np.pi * f[k] / (-p[0]) if p[0] != 0 else np.inf
            lam = 2 * np.pi / abs(p[0]) if p[0] != 0 else np.inf
            coh = "propagation coherente" if res < 0.8 else "PAS de propagation coherente (phase erratique)"
            print(f"   St {f[k]:5.2f} : x/c {xs[b[0]]:.3f}-{xs[b[-1]]:.3f}, lambda {lam:.3f} c, "
                  f"c_phase {c:+.2f} U_inf, residu de phase {res:.2f} rad -> {coh}")
            PROP[(s, k)] = (f[k], lam, c, res)

# =============================================================================== figures
if not HAS_MPL:
    print(" matplotlib absent sur cette machine : tableaux seulement, pas de figures.")
    sys.exit(0)
fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
fig.patch.set_facecolor("#fcfcfb")
ax = axs[0, 0]
lo, hi = 0.0, 0.0
for s, c in ((+1, CE), (-1, CI)):
    st = S[s]; x = st["x"]; cfm = st["cf"].mean(0)
    ax.fill_between(x, st["cf"].min(0), st["cf"].max(0), color=c, alpha=0.18, lw=0)
    ax.plot(x, cfm, color=c, lw=2, label=LAB[s] + " (mean; band = min–max over snapshots)")
    m = (x > 0.05) & (x < 0.995)
    lo, hi = min(lo, st["cf"][:, m].min()), max(hi, cfm[m].max())
ax.axhline(0, color=ENCRE2, lw=1)
ax.set_xlim(0, 1); ax.set_ylim(1.2 * lo if lo < 0 else -0.1 * hi, 1.15 * hi)
habiller(ax, f"(a) Signed skin friction (+ towards the trailing edge)\nN = {N}, {nt} snapshots, t = {ts[0]:.3f}–{ts[-1]:.3f}", "x/c", "C_f")
ax.legend(frameon=False, fontsize=8, loc="upper right")

ax = axs[0, 1]
for s, c in ((+1, CE), (-1, CI)):
    ax.plot(S[s]["x"], 100 * (S[s]["cf"] < 0).mean(0), color=c, lw=2, label=LAB[s])
ax.set_xlim(0, 1); ax.set_ylim(-2, 102)
habiller(ax, "(b) Fraction of time with C_f < 0", "x/c", "% of snapshots")
ax.legend(frameon=False, fontsize=8)

ax = axs[1, 0]
if nt > 1:
    top_ = 0.0
    for s, c in ((+1, CE), (-1, CI)):
        r_ = S[s]["cp"].std(0); top_ = max(top_, r_.max())
        ax.semilogy(S[s]["x"], r_, color=c, lw=2, label=LAB[s])
    ax.set_xlim(0, 1); ax.set_ylim(1e-4 * top_, 2 * top_)
    habiller(ax, "(c) RMS of the wall pressure fluctuation", "x/c", "rms C_p'")
    ax.legend(frameon=False, fontsize=8)

ax = axs[1, 1]
actif = Ep["Cp"]["tot"] > 1e-6 * Ep["Cp"]["tot"].max()      # faces ou il se passe quelque chose
for s, c in ((+1, CE), (-1, CI)):
    m = (cote_face == s) & actif
    ax.scatter(xf[m], 100 * Ep["Cp"]["R"][m], s=22, color=c, edgecolor="#fcfcfb", lw=1.5,
               label=LAB[s], zorder=3)
for r, v in cal:
    if abs(r - 2.0 / N) < 1e-9 or r == 1.0:
        ax.axhline(100 * v, color=ENCRE2, lw=1, ls=":")
        lab = "node-to-node alternation" if r < 1 else "wave one element long"
        ax.text(0.01, 100 * v, " " + lab, fontsize=8, color=ENCRE2, va="bottom")
ax.set_xlim(0, 1); ax.set_ylim(0, 100)
habiller(ax, f"(d) Share of C_p' energy in the highest mode (N = {N}) of each element", "x/c (element centre)", "%")
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.9))
fig.savefig(os.path.join(D, "paroi_film_cf.png"), dpi=130, facecolor=fig.get_facecolor())

if nt > 1:
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True,
                            gridspec_kw=dict(width_ratios=[1, 1, 1]))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, s, titre in ((axs[0], +1, "(a) Suction side"), (axs[1], -1, "(b) Pressure side")):
        st = S[s]; m = (st["x"] > 0.5) & (st["x"] <= 1.0)
        V = st["cp"][:, m] - st["cp"][:, m].mean(0)
        v = np.percentile(np.abs(V), 99)
        pc = ax.pcolormesh(st["x"][m], T - T[0], V, cmap=DIV, vmin=-v, vmax=v, shading="nearest")
        for xi_ in st["x"][m][st["interf"][m]]:
            ax.axvline(xi_, color=ENCRE2, lw=0.4, alpha=0.5)
        habiller(ax, f"{titre}: wall C_p'\n(grey lines = element boundaries)", "x/c", "(t − t₀) U∞/c")
        ax.grid(False)
        fig.colorbar(pc, ax=ax, shrink=0.85, label="C_p'")
    ax = axs[2]
    st = S[-1]; m = (st["x"] > 0.85) & (st["x"] <= 1.0)
    V = st["cp"][:, m] - st["cp"][:, m].mean(0)
    i0 = int(np.argmax(np.abs(V).max(1)))
    i0 = min(i0, nt - 7)
    for j, ton in zip((0, 3, 6), ("#184f95", "#3987e5", "#86b6ef")):
        ax.plot(st["x"][m], V[i0 + j], "-o", color=ton, lw=1.5, ms=4,
                label=f"t = {T[i0+j]:.4f}")
    for xi_ in st["x"][m][st["interf"][m]]:
        ax.axvline(xi_, color=ENCRE2, lw=0.6, ls=":")
    habiller(ax, f"(c) Pressure side near the TE: 3 snapshots\n{3*np.median(np.diff(T)):.3f} c/U∞ apart (dotted = element boundaries)", "x/c", "C_p'")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(D, "paroi_film_xt.png"), dpi=130, facecolor=fig.get_facecolor())

print("\n figures : paroi_film_cf.png" + (", paroi_film_xt.png" if nt > 1 else ""))

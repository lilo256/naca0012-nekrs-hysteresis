#!/usr/bin/env python3
"""
extract_montee.py  --  donnees de la branche montante (rapport, section 4.3)
==========================================================================
Pour chaque cas de la montee (et les references utiles) :
  - moyennes (ponderees en temps) et rms de CL, CD, CDp, CDv sur le regime
    permanent, pas de temps, derive residuelle ;
  - DECOLLEMENT A L'EXTRADOS par deux mesures independantes :
      (a) PAROI  : signe du frottement moyen Cf (resultats_csv/cf_cp_time_*),
                   c'est la definition standard (tau_w = 0) ;
      (b) PEIGNE : sondes A### a 0.008c de la paroi : ux moyen < 0 et
                   fraction du temps ou ux < 0 (gamma).
    + Cp,min moyen de l'extrados et profils Cf / Cp moyens (pour les figures).

USAGE (sur le cluster)
   cd ~/cdf_runs
   python3 ~/bin/extract_montee.py              # tout
   python3 ~/bin/extract_montee.py --sans-paroi # sans les fichiers Cf (rapide)
SORTIE
   deux tableaux a l'ecran  +  montee_data.json  (a rapatrier pour les figures)

REGLES (issues de la campagne)
 - lignes aberrantes des premiers pas apres un restart (|CL|~1e6) filtrees ;
 - drag_results.dat est ecrit en mode AJOUT : apres un restart, la portion
   recalculee apparait deux fois -> on garde la derniere version ;
 - un cas EN COURS a une derniere ligne incomplete : elle est ecartee, et le
   cas est marque "en_cours" (t atteint < endTime du .par) ;
 - une erreur sur un cas n'arrete plus l'extraction : le JSON est toujours ecrit ;
 - fenetre par defaut = seconde moitie du segment ; fenetres explicites pour
   les cas qui changent d'etat en cours de run (17,25 deg et froid 17 deg) ;
 - rms = ecart-type du signal dans la fenetre (amplitude d'oscillation du
   regime permanent), PAS l'incertitude sur la moyenne.
 - convention des fichiers de paroi (Tutoriel, 4.2) : colonnes x,y,z,cf_x,cf_y,
   cf_z,Cp ; cf_x = Cf scalaire DEJA signe (+ vers le bord de fuite) ;
   bord de fuite a l'origine du maillage.
"""
import numpy as np, json, os, re, sys, glob, time

SANS_PAROI = "--sans-paroi" in sys.argv
MAX_SNAP = 60          # instantanes de paroi lus au plus par fenetre (sous-echantillonnage regulier)

# (dossier, alpha, role, fenetres)   fenetres = None -> seconde moitie
#   une fenetre = (etiquette, t_debut, t_fin)  ; t_fin = None -> fin du run
CAS = [
    ("prod_a00",        0.00, "polaire",         None),
    ("prod_a5",         5.00, "polaire",         None),
    ("dom_a20_b30_v2", 10.00, "polaire",         None),
    ("prod_a15",       15.00, "chaine",          None),   # depart de la chaine (froid, 100 CTU)
    ("up_a15.25",      15.25, "chaine",          None),
    ("up_a15.50",      15.50, "chaine",          None),
    ("up_a15.75",      15.75, "chaine",          None),
    ("up_a16.00",      16.00, "chaine",          None),
    ("up_a16.25",      16.25, "chaine",          None),
    ("up_a16.50",      16.50, "chaine",          None),
    ("up_a16.75",      16.75, "chaine",          None),
    ("up_a17.00",      17.00, "chaine",          None),
    ("up_a17.25",      17.25, "chaine",          [("plateau_attache", 352.0, 366.0),
                                                  ("decroche",        425.0, None)]),
    ("up_a17.50",      17.50, "chaine_decroche", None),
    ("up_a17.75",      17.75, "chaine_decroche", None),
    ("up_a18.00",      18.00, "chaine_decroche", None),
    ("cold_a16.50",    16.50, "froid",           None),
    ("cold_a17.00",    17.00, "froid",           [("plateau_attache",  40.0,  62.0),
                                                  ("decroche",        110.0, None)]),
    ("cold_a18.00",    18.00, "froid",           None),
    ("cold_a19.00",    19.00, "froid",           None),
]

# Martin Ferrer (2025), Tables 4.1-4.2 et texte (CLmax = 1.477 a 15.5 deg)
MARINA = {0.0: (-0.0005, 0.0091), 5.0: (0.5279, 0.0148), 10.0: (1.028, 0.0205),
          15.0: (1.406, 0.0390), 15.5: (1.477, None), 16.0: (1.439, 0.0454),
          17.0: (1.401, 0.0569), 17.5: (1.071, 0.1907), 18.0: (1.075, 0.1967),
          18.5: (1.033, 0.2031), 19.0: (0.951, 0.3303), 20.0: (1.014, 0.381)}


# =============================================================================
#  LECTURE ROBUSTE
# =============================================================================
def charger(p, delim=None):
    """np.loadtxt, robuste aux fichiers d'un cas EN COURS (derniere ligne
    incomplete) et aux lignes parasites (en-tete repete apres un restart,
    nombre de colonnes different) : on ne garde alors que les lignes
    numeriques ayant le nombre de colonnes de l'en-tete."""
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
            if len(s) != ncol:
                continue
            try:
                bons.append([float(x) for x in s])
            except ValueError:
                pass
    return np.array(bons, dtype=float).reshape(-1, ncol)


def fin_prevue(d):
    """endTime du .par (pour reperer un cas encore en cours)."""
    for p in [os.path.join(d, "airfoil.par")] + glob.glob(os.path.join(d, "*.par")):
        try:
            for l in open(p):
                s = l.split("#")[0].strip()
                if s.lower().startswith("endtime") and "=" in s:
                    return float(s.split("=")[1])
        except Exception:
            pass
    return None


def monotone(t):
    """Masque qui ne garde, apres chaque restart, que la derniere version
    de la portion recalculee (le fichier est ecrit en ajout)."""
    keep = np.ones(len(t), bool)
    for r in np.where(np.diff(t) < 0)[0] + 1:          # 1re ligne de chaque nouveau segment
        keep[:r] &= t[:r] < t[r]
    return keep


def lire_forces(d):
    p = os.path.join(d, "drag_results.dat")
    if not os.path.isfile(p):
        return None
    a = charger(p)
    ok = (np.abs(a[:, 3]) < 2.0) & (a[:, 2] > 0) & (a[:, 2] < 2.0)
    nrej = int((~ok).sum())
    a = a[ok]
    a = a[monotone(a[:, 1])]
    return a, nrej


def lire_sondes(d):
    p = os.path.join(d, "probes_uvp.csv")
    if not os.path.isfile(p):
        return None
    h = [c.strip() for c in open(p).readline().strip().split(",")]
    s = charger(p, delim=",")
    if s.ndim != 2 or len(s) < 10:
        return None
    s = s[monotone(s[:, 0])]
    return h, s


# =============================================================================
#  EFFORTS
# =============================================================================
def stats(a, t0, t1):
    t = a[:, 1]
    m = (t >= t0) & (t <= t1)
    if m.sum() < 50:
        return None
    tt = t[m]
    w = np.r_[np.diff(tt), np.diff(tt)[-1]]
    moy = lambda y: float(np.sum(y[m] * w) / np.sum(w))
    rms = lambda y: float(np.std(y[m]))
    # derive : CL moyen sur 4 quarts successifs de la fenetre
    q = []
    for k in range(4):
        mk = (tt >= t0 + k * (t1 - t0) / 4) & (tt < t0 + (k + 1) * (t1 - t0) / 4)
        if mk.sum() > 10:
            q.append(a[m, 3][mk].mean())
    der_abs = float(max(q) - min(q)) if len(q) >= 3 else None
    der = der_abs / abs(np.mean(q)) * 100 if der_abs is not None and abs(np.mean(q)) > 0.05 else None
    dts = np.diff(tt)
    return dict(t0=float(tt[0]), t1=float(tt[-1]), duree=float(tt[-1] - tt[0]),
                CL=moy(a[:, 3]), CL_rms=rms(a[:, 3]),
                CD=moy(a[:, 2]), CD_rms=rms(a[:, 2]),
                CDp=moy(a[:, 5]), CDv=moy(a[:, 8]),
                derive_CL_pct=der, derive_CL_abs=der_abs,
                dt_median=float(np.median(dts)), dt_min=float(dts.min()))


# =============================================================================
#  DECOLLEMENT (b) : PEIGNE DE SONDES A### (0.008c au-dessus de l'extrados)
# =============================================================================
def peigne(sondes, t0, t1):
    if sondes is None:
        return None
    h, s = sondes
    m = (s[:, 0] >= t0) & (s[:, 0] <= t1)
    if m.sum() < 10:
        return dict(texte="(pas de sondes dans la fenetre)", x_sep=None, stations=[])
    A = sorted((int(n[4:]) / 100.0, float(s[m, j].mean()), float((s[m, j] < 0).mean()))
               for j, n in enumerate(h) if n.startswith("ux_A") and n[4:].isdigit())
    if not A:
        return None
    st = [dict(x=x, ux=u, gamma=g) for x, u, g in A]
    x_prev = None
    for x, u, g in A:
        if u < 0:
            txt = (f"retour entre x/c={x_prev:.2f} et {x:.2f}" if x_prev is not None
                   else f"retour des x/c={x:.2f}")
            return dict(texte=txt, x_sep=[x_prev, x], stations=st, n=int(m.sum()))
        x_prev = x
    return dict(texte=f"aucun retour moyen ({A[0][0]:.2f}-{A[-1][0]:.2f})", x_sep=None,
                stations=st, n=int(m.sum()))


# =============================================================================
#  DECOLLEMENT (a) : FROTTEMENT PARIETAL MOYEN (definition tau_w = 0)
# =============================================================================
RX_CF = re.compile(r"cf_cp_time_([0-9.eE+-]+?)(?:_rank_\d+)?\.csv$")


def index_paroi(d):
    """{t: [fichiers]} des dumps parietaux (un fichier par rang MPI)."""
    out = {}
    for f in glob.glob(os.path.join(d, "resultats_csv", "cf_cp_time_*.csv")):
        m = RX_CF.search(os.path.basename(f))
        if m:
            try:
                out.setdefault(round(float(m.group(1)), 6), []).append(f)
            except ValueError:
                pass
    return out


def lire_instantane(fichiers):
    blocs, noms = [], None
    for f in sorted(fichiers):
        with open(f, errors="ignore") as fh:
            h = [c.strip().lower() for c in fh.readline().strip().lstrip("#").split(",")]
        a = charger(f, delim=",")
        if a.ndim != 2 or len(a) == 0:
            continue
        noms = noms or h
        blocs.append(a)
    if not blocs:
        return None
    a = np.vstack(blocs)
    col = lambda *c: next((noms.index(n) for n in c if n in noms), None)
    ix, iy, icf, icp = col("x"), col("y"), col("cf_x", "cfx", "cf"), col("cp")
    if None in (ix, iy, icf):
        raise ValueError(f"colonnes inattendues dans resultats_csv : {noms}")
    return a[:, ix], a[:, iy], a[:, icf], (a[:, icp] if icp is not None else None)


def geometrie(x, y):
    """x/c le long de la corde (0 au BA) et cote (> 0 : extrados).
    Bord de fuite a l'origine du maillage ; BA = point le plus eloigne du BF."""
    i_bf = int(np.argmin(x ** 2 + y ** 2))
    xb, yb = x[i_bf], y[i_bf]
    i_ba = int(np.argmax((x - xb) ** 2 + (y - yb) ** 2))
    xa, ya = x[i_ba], y[i_ba]
    c = np.hypot(xb - xa, yb - ya)
    ex, ey = (xb - xa) / c, (yb - ya) / c
    xc = ((x - xa) * ex + (y - ya) * ey) / c
    yn = (-(x - xa) * ey + (y - ya) * ex) / c
    alpha = float(np.degrees(np.arctan2(ya - yb, xb - xa)))
    return xc, yn, alpha


def profil(xc, v, msk):
    """valeur moyenne par station x/c (fusionne les doublons des plans z), triee."""
    k = np.round(xc[msk], 5)
    u, inv = np.unique(k, return_inverse=True)
    return u, np.bincount(inv, weights=v[msk]) / np.bincount(inv)


def croisements(xs, cf, x_min=0.005, x_max=0.995, fusion=0.005):
    """Zones de Cf < 0 sur l'extrados, du BA vers le BF (bornes interpolees).
    decollement = debut de la 1re zone ; recollement = sa fin si elle est avant
    x_max. Plusieurs zones possibles (bulle de BA + decollement de BF, etc.) ;
    deux zones separees de moins de `fusion` sont fusionnees."""
    m = (xs >= x_min) & (xs <= x_max)
    xs, cf = xs[m], cf[m]
    if len(xs) < 5:
        return None, None, None
    zones, debut = [], (float(xs[0]) if cf[0] < 0 else None)
    for i in range(len(xs) - 1):
        xi = float(xs[i] + cf[i] / (cf[i] - cf[i + 1]) * (xs[i + 1] - xs[i])) \
            if cf[i] != cf[i + 1] else float(xs[i])
        if debut is None and cf[i] > 0 >= cf[i + 1]:
            debut = xi
        elif debut is not None and cf[i] <= 0 < cf[i + 1]:
            zones.append([debut, xi]); debut = None
    if debut is not None:
        zones.append([debut, float(xs[-1])])
    fz = []
    for z in zones:
        if fz and z[0] - fz[-1][1] < fusion:
            fz[-1][1] = z[1]
        else:
            fz.append(z)
    dx = np.gradient(xs)
    frac = float(dx[cf < 0].sum() / dx.sum())
    sep = fz[0][0] if fz else None
    rec = fz[0][1] if fz and fz[0][1] < float(xs[-1]) else None
    return sep, rec, dict(amont=bool(cf[0] < 0), frac_neg=frac, zones=fz)


def paroi(idx, t0, t1, alpha_nom):
    ts = sorted(t for t in idx if t0 <= t <= t1)
    if not ts:
        return dict(texte="(aucun instantane de paroi dans la fenetre)", n=0)
    if len(ts) > MAX_SNAP:
        ts = [ts[i] for i in np.linspace(0, len(ts) - 1, MAX_SNAP).round().astype(int)]
    CFh, CPh, CPb, seps, xh = [], [], [], [], None
    alpha_geo = None
    for t in ts:
        r = lire_instantane(idx[t])
        if r is None:
            continue
        x, y, cf, cp = r
        xc, yn, alpha_geo = geometrie(x, y)
        hx, hcf = profil(xc, cf, yn > 0)
        if xh is None:
            xh = hx
            if cp is not None:
                xb_, _ = profil(xc, cp, yn <= 0)
                xbas = xb_
        CFh.append(np.interp(xh, hx, hcf))
        if cp is not None:
            CPh.append(np.interp(xh, hx, profil(xc, cp, yn > 0)[1]))
            bx, bcp = profil(xc, cp, yn <= 0)
            CPb.append(np.interp(xbas, bx, bcp))
        s_, _, _ = croisements(hx, hcf)
        seps.append(np.nan if s_ is None else s_)
    if not CFh:
        return dict(texte="(fichiers de paroi illisibles)", n=0)
    cf_m = np.mean(CFh, axis=0)
    sep, rec, info = croisements(xh, cf_m)
    seps = np.array(seps, float)
    fin = np.isfinite(seps)
    res = dict(n=len(CFh), t=[float(ts[0]), float(ts[-1])], alpha_geo=alpha_geo,
               x_sep=sep, x_rec=rec, frac_neg=info["frac_neg"] if info else None,
               amont=info["amont"] if info else None,
               zones_cf_negatif=info["zones"] if info else None,
               x_sep_inst=[float(np.nanmin(seps)), float(np.nanmax(seps))] if fin.any() else None,
               frac_inst_decolle=float(fin.mean()),
               profil_cf_extrados=dict(x=[round(float(v), 5) for v in xh],
                                       cf=[float(f"{v:.5g}") for v in cf_m]))
    if CPh:
        cph = np.mean(CPh, axis=0)
        i = int(np.argmin(cph))
        res.update(cp_min=float(cph[i]), x_cp_min=float(xh[i]),
                   profil_cp=dict(x_haut=[round(float(v), 5) for v in xh],
                                  cp_haut=[float(f"{v:.5g}") for v in cph],
                                  x_bas=[round(float(v), 5) for v in xbas],
                                  cp_bas=[float(f"{v:.5g}") for v in np.mean(CPb, axis=0)]))
    if sep is None:
        res["texte"] = "attache (pas de Cf < 0 avant x/c=0.995)"
    elif len(info["zones"]) == 1 and rec is None:
        res["texte"] = (f"decollement x/c={sep:.3f}" if not info["amont"]
                        else "Cf < 0 des x/c=0.005 (decolle au BA)")
    else:
        res["texte"] = "Cf<0 sur " + ", ".join(f"{a:.3f}-{b:.3f}" for a, b in info["zones"][:3])
        if len(info["zones"]) > 3:
            res["texte"] += f" (+{len(info['zones']) - 3})"
    if alpha_geo is not None and abs(alpha_geo - alpha_nom) > 0.05:
        res["texte"] += f"  !! alpha du maillage {alpha_geo:.2f}"
    return res


# =============================================================================
#  PROGRAMME
# =============================================================================
out = {"version": 2, "genere": time.strftime("%Y-%m-%d %H:%M"), "cas": {},
       "marina": {str(k): v for k, v in MARINA.items()}}
PROBLEMES = []
W = 124
print("=" * W)
print(" BRANCHE MONTANTE — moyennes (ponderees en temps) et rms du regime permanent")
print("=" * W)
print(f" {'cas':15s} {'alpha':>5s} {'fenetre':16s} {'[t0 ; t1]':>15s} {'CL':>7s} {'rms':>6s}"
      f" {'CD':>7s} {'rms':>7s} {'CDp':>7s} {'CDv':>7s} {'deriv':>8s} {'dt med':>8s}  etat")
print("-" * W)
for d, alpha, role, fen in CAS:
    if not os.path.isdir(d):
        print(f" {d:15s} {alpha:5.2f}  ABSENT")
        continue
    try:
        r = lire_forces(d)
        if r is None or len(r[0]) < 50:
            print(f" {d:15s} {alpha:5.2f}  pas de drag_results.dat exploitable")
            continue
        a, nrej = r
        t = a[:, 1]
        tfp = fin_prevue(d)
        en_cours = bool(tfp is not None and t[-1] < tfp - 0.01)
        etat = f"EN COURS (t={t[-1]:.1f} / {tfp:g})" if en_cours else "fini"
        fenetres = fen or [("regime_permanent", t[0] + 0.5 * (t[-1] - t[0]), None)]
        try:
            sondes = lire_sondes(d)
        except Exception as e:
            sondes = None
            PROBLEMES.append(f"{d} : sondes illisibles ({e})")
        idx = {}
        if not SANS_PAROI:
            try:
                idx = index_paroi(d)
            except Exception as e:
                PROBLEMES.append(f"{d} : index paroi ({e})")
        ent = dict(alpha=alpha, role=role, segment=[float(t[0]), float(t[-1])],
                   t_fin_prevue=tfp, en_cours=en_cours, lignes_rejetees=nrej, fenetres={})
        for lab, t0, t1 in fenetres:
            t1 = t[-1] if t1 is None else t1
            st = stats(a, t0, t1)
            if st is None:
                print(f" {d:15s} {alpha:5.2f} {lab:16s}  fenetre vide ou trop courte")
                continue
            try:
                st["peigne"] = peigne(sondes, t0, t1)
            except Exception as e:
                st["peigne"] = None
                PROBLEMES.append(f"{d} [{lab}] : peigne ({e})")
            if not SANS_PAROI:
                try:
                    st["paroi"] = paroi(idx, t0, t1, alpha)
                except Exception as e:
                    st["paroi"] = None
                    PROBLEMES.append(f"{d} [{lab}] : paroi ({e})")
            ent["fenetres"][lab] = st
            if st["derive_CL_pct"] is not None:
                der = f"{st['derive_CL_pct']:6.2f}% "
            elif st["derive_CL_abs"] is not None:
                der = f"{st['derive_CL_abs']:.0e}a"      # absolu quand CL ~ 0
            else:
                der = "    -   "
            print(f" {d:15s} {alpha:5.2f} {lab:16s} [{st['t0']:6.1f};{st['t1']:6.1f}]"
                  f" {st['CL']:7.4f} {st['CL_rms']:6.4f} {st['CD']:7.5f} {st['CD_rms']:7.5f}"
                  f" {st['CDp']:7.5f} {st['CDv']:7.5f} {der:>8s} {st['dt_median']:8.2e}  {etat}")
        out["cas"][d] = ent
    except Exception as e:
        print(f" {d:15s} {alpha:5.2f}  ERREUR : {e}")
        PROBLEMES.append(f"{d} : {e}")
print("-" * W)
print(" rms   = ecart-type du signal dans la fenetre (amplitude d'oscillation) : barres d'erreur")
print(" deriv = variation du CL moyen entre les 4 quarts de la fenetre ; 'a' = valeur ABSOLUE")
print("         (quand CL ~ 0, un pourcentage n'a pas de sens)")

# --- tableau du decollement ---------------------------------------------------
print("\n" + "=" * W)
print(" DECOLLEMENT A L'EXTRADOS — paroi (Cf moyen) et peigne de sondes (0.008c)")
print("=" * W)
print(f" {'cas':15s} {'alpha':>5s} {'fenetre':16s} {'n':>3s}  {'paroi : Cf moyen':52s} "
      f"{'Cp,min (x/c)':>15s}  peigne 0.008c")
print("-" * W)
for d, ent in out["cas"].items():
    for lab, st in ent["fenetres"].items():
        p, g = st.get("paroi"), st.get("peigne")
        if p is None:
            ptxt, n, cpt = ("(non lu)" if SANS_PAROI else "(erreur)"), "", ""
        else:
            ptxt, n = p.get("texte", ""), str(p.get("n", ""))
            if (p.get("x_sep_inst") and p.get("x_sep") is not None and not p.get("amont")
                    and len(p.get("zones_cf_negatif") or []) == 1):
                ptxt += f" [inst. {p['x_sep_inst'][0]:.2f}-{p['x_sep_inst'][1]:.2f}]"
            cpt = f"{p['cp_min']:+.2f} ({p['x_cp_min']:.3f})" if "cp_min" in p else ""
        gtxt = g["texte"] if g else "(pas de sondes)"
        print(f" {d:15s} {ent['alpha']:5.2f} {lab:16s} {n:>3s}  {ptxt[:52]:52s} {cpt:>15s}  {gtxt}")
        if g and g.get("stations"):
            ux = " ".join(f"{s['ux']:+.2f}" for s in g["stations"])
            ga = " ".join(f"{s['gamma']:.2f}" for s in g["stations"])
            xs = " ".join(f"{s['x']:.2f}" for s in g["stations"])
            print(f" {'':39s} x/c   {xs}\n {'':39s} ux    {ux}\n {'':39s} gamma {ga}")
print("-" * W)
print(" paroi  : Cf moyen de l'extrados (cf_x signe, + vers le BF), x/c de 0.005 a 0.995 ;")
print("          [inst.] = etendue du point de decollement sur les instantanes (instationnarite)")
print(" peigne : ux moyen a 0.008c de la paroi ; gamma = fraction du temps ou ux < 0")
print("          (le decollement a la paroi est EN AMONT du retour vu a 0.008c)")

# --- comparaison avec Martin Ferrer (2025) aux angles communs -----------------
print("\n COMPARAISON AVEC MARTIN FERRER (2025)")
print(f" {'alpha':>5s} {'cas':15s} {'CL':>7s} {'CL MF':>7s} {'ecart':>7s}   {'CD':>7s} {'CD MF':>7s} {'ecart':>7s}")
for d, ent in out["cas"].items():
    a_ = ent["alpha"]
    if a_ not in MARINA or ent["role"] == "froid" or ent["en_cours"]:
        continue
    lab = "plateau_attache" if "plateau_attache" in ent["fenetres"] else None
    for lab_, st in ent["fenetres"].items():
        if lab and lab_ != lab:
            continue
        cl_m, cd_m = MARINA[a_]
        e_cl = f"{(st['CL'] - cl_m) / abs(cl_m) * 100:+6.1f}%" if abs(cl_m) > 0.01 else "   -  "
        e_cd = f"{(st['CD'] - cd_m) / cd_m * 100:+6.1f}%" if cd_m else "   -  "
        print(f" {a_:5.2f} {d:15s} {st['CL']:7.4f} {cl_m:7.4f} {e_cl}   {st['CD']:7.5f} "
              f"{(cd_m if cd_m else float('nan')):7.4f} {e_cd}")
print(" Note : le CLmax de Martin Ferrer est 1.477 a 15.5 deg (texte) ; 1.439 est sa valeur a 16 deg.")

if PROBLEMES:
    print("\n PROBLEMES RENCONTRES (l'extraction a continue) :")
    for p in PROBLEMES:
        print("   -", p)

with open("montee_data.json", "w") as f:
    json.dump(out, f, indent=1)
print("\n -> montee_data.json ecrit dans", os.getcwd())
print("    rapatrier :  rsync -avz USER@cdf:~/cdf_runs/montee_data.json .")

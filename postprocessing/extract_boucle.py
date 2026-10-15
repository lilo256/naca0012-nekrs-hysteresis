#!/usr/bin/env python3
"""
extract_boucle.py  --  donnees de la boucle d'hysteresis (rapport, section 4.5)
==============================================================================
Meme moteur que extract_montee.py (section 4.3), autre liste de cas :
  branche attachee de 0 a 17 deg (montee de 4.3 comprise : le JSON suffit pour tracer la boucle),
  montee apres decrochage, descente longue (depuis 20 deg),
  descente courte (depuis le depart a froid decroche a 17 deg), sauts de 1 deg jusqu'a 10 deg,
  departs a froid, test de perturbation a 13 deg (pert13_*), etats decroches relances avec un
  filtre 4 fois plus fort (stall*_hpf40*).
Pour chaque cas et chaque fenetre :
  - moyennes (ponderees en temps) et rms de CL, CD, CDp, CDv, derive (4 quarts) ;
  - decollement : paroi (zero du Cf moyen, resultats_csv/cf_cp_time_*) et peigne de sondes (0.008c) ;
  - profils moyens Cf (extrados) et Cp (deux faces) pour les figures ;
et pour chaque cas la SERIE TEMPORELLE de CL et CD moyennee par tranches de 0.1 CTU (figures
C_L(t), plans de phase (C_D, C_L)).

USAGE (sur le cluster)
   cd ~/cdf_runs
   python3 ~/bin/extract_boucle.py              # tout (lecture des fichiers de paroi : plusieurs minutes)
   python3 ~/bin/extract_boucle.py --sans-paroi # efforts, sondes et series seulement (rapide)
SORTIE
   trois tableaux a l'ecran  +  boucle_data.json  (a rapatrier pour les figures)

FENETRES (meme regle qu'en 4.3) : seconde moitie du cas par defaut ; fenetres explicites pour les
cas de 4.3 (memes valeurs que le tableau tab:upstroke_loads) et pour les cas qui changent d'etat
en cours de calcul (perturbation, filtre x4) : seulement la partie finale.
"""
import numpy as np, json, os, re, sys, glob, time

SANS_PAROI = "--sans-paroi" in sys.argv
MAX_SNAP = 60          # instantanes de paroi lus au plus par fenetre (sous-echantillonnage regulier)

# (dossier, alpha, role, fenetres)   fenetres = None -> seconde moitie
#   une fenetre = (etiquette, t_debut, t_fin)  ; t_fin = None -> fin du run
CAS = [
    # --- branche attachee (references ; montee de 4.3 : la 2e moitie de chaque pas = fenetres de 4.3) ---
    ("prod_a00",        0.00, "attache",          None),                                  # 30-60, comme 4.3
    ("prod_a5",         5.00, "attache",          None),                                  # 30-60, comme 4.3
    ("dom_a20_b30_v2", 10.00, "attache",          [("regime_permanent", 65.0, 85.0)]),   # comme 4.3
    ("att_a11.00",     11.00, "attache",          None),
    ("att_a12.00",     12.00, "attache",          None),
    ("att_a13.00",     13.00, "attache",          None),
    ("prod_a15",       15.00, "attache",          [("regime_permanent", 50.0, 100.0)]),  # comme 4.3
    ("up_a15.25",      15.25, "attache",          None),
    ("up_a15.50",      15.50, "attache",          None),
    ("up_a15.75",      15.75, "attache",          None),
    ("up_a16.00",      16.00, "attache",          None),
    ("up_a16.25",      16.25, "attache",          None),
    ("up_a16.50",      16.50, "attache",          None),
    ("up_a16.75",      16.75, "attache",          None),
    ("up_a17.00",      17.00, "attache",          [("regime_permanent", 275.0, 340.0)]), # comme 4.3
    ("cold_a16.50",    16.50, "froid_attache",    [("regime_permanent", 60.0, 120.0)]),  # comme 4.3
    # --- montee apres decrochage ---
    ("up_a17.25",      17.25, "montee_decroche",  [("plateau_attache", 352.0, 366.0),
                                                   ("decroche",        425.0, None)]),
    ("up_a17.50",      17.50, "montee_decroche",  [("decroche", 420.0, 450.0)]),
    ("up_a17.75",      17.75, "montee_decroche",  [("decroche", 470.0, 490.0)]),
    ("up_a18.00",      18.00, "montee_decroche",  [("decroche", 510.0, 530.0)]),
    ("up_a18.50",      18.50, "montee_decroche",  None),
    # --- descente longue (depart a froid a 20 deg, pas de 0.25 deg) ---
    ("cold_a20.00",    20.00, "descente_longue",  None),
    ("down_a19.75",    19.75, "descente_longue",  None),
    ("down_a19.50",    19.50, "descente_longue",  None),
    ("down_a19.25",    19.25, "descente_longue",  None),
    ("down_a19.00",    19.00, "descente_longue",  None),
    ("down_a18.75",    18.75, "descente_longue",  None),
    ("down_a18.50",    18.50, "descente_longue",  None),
    ("down_a18.25",    18.25, "descente_longue",  None),
    ("down_a18.00",    18.00, "descente_longue",  None),
    ("down_a17.75",    17.75, "descente_longue",  None),
    ("down_a17.50",    17.50, "descente_longue",  None),
    ("down_a17.25",    17.25, "descente_longue",  None),
    ("down_a17.00",    17.00, "descente_longue",  None),
    # --- descente courte (depart a froid decroche a 17 deg) ---
    ("cold_a17.00",    17.00, "descente_courte",  [("plateau_attache",  40.0,  62.0),
                                                   ("decroche",        110.0, None)]),
    ("down_a16.50",    16.50, "descente_courte",  None),
    ("down_a16.25",    16.25, "descente_courte",  None),
    ("down_a16.00",    16.00, "descente_courte",  None),
    ("down_a15.75",    15.75, "descente_courte",  None),
    ("down_a15.50",    15.50, "descente_courte",  None),
    ("down_a15.25",    15.25, "descente_courte",  None),
    ("down_a15.00",    15.00, "descente_courte",  None),
    ("down_a14.75",    14.75, "descente_courte",  None),
    ("down_a14.50",    14.50, "descente_courte",  None),
    ("down_a14.25",    14.25, "descente_courte",  None),
    ("down_a14.00",    14.00, "descente_courte",  None),
    # --- sauts de 1 deg (et 0.5 deg vers 13.5) ---
    ("jump_a14.00",    14.00, "saut",             None),
    ("jump_a13.50",    13.50, "saut",             None),
    ("jump_a13.00",    13.00, "saut",             None),
    ("jump_a12.00",    12.00, "saut",             None),
    ("jump_a11.00",    11.00, "saut",             None),
    ("jump_a10.00",    10.00, "saut",             None),
    # --- autres departs a froid (decroches) ---
    ("cold_a18.00",    18.00, "froid_decroche",   [("decroche", 60.0, 80.0)]),           # comme 4.3
    ("cold_a19.00",    19.00, "froid_decroche",   [("decroche", 60.0, 80.0)]),           # comme 4.3
    # --- test de perturbation a 13 deg : w x decroche + (1-w) x attache ---
    # fenetres finales alignees sur les tableaux B3 et B4 de MAJ_0710
    ("pert13_90",      13.00, "perturbation",     [("final", 600.0, None)]),
    ("pert13_70",      13.00, "perturbation",     [("final", 580.0, None)]),
    ("pert13_50",      13.00, "perturbation",     [("final", 590.0, None)]),
    ("pert13_60",      13.00, "perturbation",     [("plateau", 562.0, 575.0),   # palier intermediaire (pres de l'etat de bord)
                                                   ("final",   600.0, None)]),
    # --- etats decroches relances avec un filtre 4 fois plus fort ---
    ("stall10_hpf40",  10.00, "filtre_x4",        [("final", 690.0, None)]),
    ("stall11_hpf40",  11.00, "filtre_x4",        [("final", 665.0, None)]),   # convergence pas finie
    ("stall12_hpf40",  12.00, "filtre_x4",        [("final", 610.0, None)]),
    ("stall13_hpf40",  13.00, "filtre_x4",        [("final", 575.0, None)]),
    ("stall13_hpf40b", 13.00, "filtre_x4",        [("final", 600.0, None)]),
    # --- test de resolution : etat decroche de production a 13 deg relu a N = 4 (lance le 08/10) ---
    ("stall13_N4",     13.00, "resolution_N4",    [("final", 595.0, None)]),   # recolle vers t = 561-564 ; valeurs envoyees a Fer
    ("stall16_N4",     16.00, "resolution_N4",    [("cycle", 283.0, 295.0)]),  # reste decroche ; un cycle du mode lent (~12 CTU)
]
PERTURBATION_W = {"pert13_90": 0.90, "pert13_70": 0.70, "pert13_60": 0.60, "pert13_50": 0.50}   # part d'etat decroche
BIN_SERIE = 0.1        # largeur des tranches des series temporelles (CTU)

# Martin Ferrer (2025) : montee (Tables 4.1-4.2, CLmax = 1.477 a 15.5 deg dans le texte)
MARINA_MONTEE = {0.0: (-0.0005, 0.0091), 5.0: (0.5279, 0.0148), 10.0: (1.028, 0.0205),
                 15.0: (1.406, 0.0390), 15.5: (1.477, None), 16.0: (1.439, 0.0454),
                 17.0: (1.401, 0.0569), 17.5: (1.071, 0.1907), 18.0: (1.075, 0.1967),
                 18.5: (1.033, 0.2031), 19.0: (0.951, 0.3303), 20.0: (1.014, 0.381)}
# Martin Ferrer (2025) : descente (Table 4.3)
MARINA_DESCENTE = {20.0: (1.019, 0.3827), 19.0: (0.963, 0.3472), 18.5: (0.927, 0.3192),
                   18.0: (1.033, 0.1979), 17.5: (1.041, 0.1933), 17.0: (1.062, 0.1975),
                   16.0: (1.057, 0.1839)}
# Points de decollement de reference (x/c), d'apres les textes
REF_DECOLLEMENT = {
    "martin2025_montee": {"16": 0.76, "17": 0.58, "17.5": 0.50, "18": 0.37, "18.5": 0.19,
                          "19": "bord d'attaque (0.000965)", "20": "bord d'attaque (0.001091)"},
    "martin2025_descente": {"17": "0.20 puis recollement, 2e decollement 0.545",
                            "17.5": "0.17 puis 0.43", "18": "0.13 puis 0.32",
                            "18.5": "bord d'attaque", "19": "bord d'attaque"},
    "mittal2002_montee": {"5": 0.99, "10": 0.95, "15": 0.64, "16": 0.52, "17": 0.38, "18": 0.25,
                          "18.5": 0.09, "19": "bord d'attaque (decrochage)"},
    "mittal2002_descente": {"17": "attache (meme solution qu'en montee)", "18": "bord d'attaque",
                            "18.5": "bord d'attaque", "19": "bord d'attaque"},
}


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
#  SERIES TEMPORELLES
# =============================================================================
def serie(a, largeur=BIN_SERIE):
    """CL, CD moyens par tranches de 'largeur' CTU (moyenne simple des pas de la tranche)."""
    t = a[:, 1]
    if len(t) < 2:
        return None
    k = np.floor((t - t[0]) / largeur).astype(int)
    n = np.bincount(k)
    ok = n > 0
    tm = (np.bincount(k, weights=t) / np.maximum(n, 1))[ok]
    cl = (np.bincount(k, weights=a[:, 3]) / np.maximum(n, 1))[ok]
    cd = (np.bincount(k, weights=a[:, 2]) / np.maximum(n, 1))[ok]
    return dict(largeur=largeur, t=[round(float(v), 3) for v in tm],
                CL=[round(float(v), 5) for v in cl], CD=[round(float(v), 6) for v in cd])


# =============================================================================
#  PROGRAMME
# =============================================================================
out = {"version": 1, "genere": time.strftime("%Y-%m-%d %H:%M"), "cas": {}, "series": {},
       "marina_montee": {str(k): v for k, v in MARINA_MONTEE.items()},
       "marina_descente": {str(k): v for k, v in MARINA_DESCENTE.items()},
       "ref_decollement": REF_DECOLLEMENT, "perturbation_w": PERTURBATION_W}
PROBLEMES = []
W = 124
print("=" * W)
print(" BOUCLE D'HYSTERESIS — moyennes (ponderees en temps) et rms, fenetres de la section 4.3")
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
                   t_fin_prevue=tfp, en_cours=en_cours, lignes_rejetees=nrej,
                   n_instantanes_paroi=len(idx), fenetres={})
        if d in PERTURBATION_W:
            ent["w_decroche"] = PERTURBATION_W[d]
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
                der = f"{st['derive_CL_abs']:.0e}a"
            else:
                der = "    -   "
            print(f" {d:15s} {alpha:5.2f} {lab:16s} [{st['t0']:6.1f};{st['t1']:6.1f}]"
                  f" {st['CL']:7.4f} {st['CL_rms']:6.4f} {st['CD']:7.5f} {st['CD_rms']:7.5f}"
                  f" {st['CDp']:7.5f} {st['CDv']:7.5f} {der:>8s} {st['dt_median']:8.2e}  {etat}")
        out["cas"][d] = ent
        try:
            out["series"][d] = serie(a)
        except Exception as e:
            PROBLEMES.append(f"{d} : serie ({e})")
    except Exception as e:
        print(f" {d:15s} {alpha:5.2f}  ERREUR : {e}")
        PROBLEMES.append(f"{d} : {e}")
print("-" * W)
print(" rms   = ecart-type du signal dans la fenetre (amplitude d'oscillation, pas incertitude de la moyenne)")
print(" deriv = variation du CL moyen entre les 4 quarts de la fenetre ; 'a' = valeur ABSOLUE")

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
            cpt = f"{p['cp_min']:+.2f} ({p['x_cp_min']:.3f})" if "cp_min" in p else ""
        gtxt = g["texte"] if g else "(pas de sondes)"
        print(f" {d:15s} {ent['alpha']:5.2f} {lab:16s} {n:>3s}  {ptxt[:52]:52s} {cpt:>15s}  {gtxt}")
print("-" * W)
print(" paroi  : Cf moyen de l'extrados ; pour un etat decroche, le zero du Cf n'est PAS le point ou")
print("          la couche cisaillee quitte le profil (section 4.3) : lire les zones de Cf < 0")
print(" peigne : ux moyen a 0.008c de la paroi")

# --- comparaison avec Martin Ferrer (2025) ------------------------------------
print("\n COMPARAISON AVEC MARTIN FERRER (2025) — meme chemin : sa montee pour nos etats attaches et notre")
print(" montee apres decrochage, sa descente pour nos descentes et sauts ; les deux pour les departs a froid decroches")
print(f" {'alpha':>5s} {'cas':15s} {'fenetre':16s} {'CL':>7s} {'CL MF':>7s} {'ecart':>7s}   {'CD':>7s} {'CD MF':>7s} {'ecart':>7s}  ref")
for d, ent in out["cas"].items():
    a_ = ent["alpha"]
    if ent["role"] in ("perturbation", "filtre_x4", "resolution_N4"):
        continue
    for lab, st in ent["fenetres"].items():
        attache = ent["role"] in ("attache", "froid_attache") or lab == "plateau_attache"
        if attache or ent["role"] == "montee_decroche":
            refs = [("montee", MARINA_MONTEE)]
        elif ent["role"] == "froid_decroche" or (d == "cold_a17.00" and lab == "decroche"):
            # sa montee n'est decrochee qu'a partir de 17.5 deg
            refs = ([("montee", MARINA_MONTEE)] if a_ >= 17.5 else []) + [("descente", MARINA_DESCENTE)]
        else:
            refs = [("descente", MARINA_DESCENTE)]
        for nom, table in refs:
            if a_ not in table:
                continue
            cl_m, cd_m = table[a_]
            e_cl = f"{(st['CL'] - cl_m) / abs(cl_m) * 100:+6.1f}%" if abs(cl_m) > 0.01 else "   -  "
            e_cd = f"{(st['CD'] - cd_m) / cd_m * 100:+6.1f}%" if cd_m else "   -  "
            print(f" {a_:5.2f} {d:15s} {lab:16s} {st['CL']:7.4f} {cl_m:7.4f} {e_cl}   {st['CD']:7.5f} "
                  f"{(cd_m if cd_m else float('nan')):7.4f} {e_cd}  {nom}")

if PROBLEMES:
    print("\n PROBLEMES RENCONTRES (l'extraction a continue) :")
    for p in PROBLEMES:
        print("   -", p)

with open("boucle_data.json", "w") as f:
    json.dump(out, f, indent=1)
print("\n -> boucle_data.json ecrit dans", os.getcwd())
print("    rapatrier :  scp USER@cdf.upc.es:~/cdf_runs/boucle_data.json .")

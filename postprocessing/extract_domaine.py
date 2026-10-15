#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_domaine.py  --  extraction uniforme des cas de l'etude de domaine   [v2]
===============================================================================
Recalcule CL, CD, CDp, CDv sur une FENETRE COMMUNE pour tous les cas, avec de
quoi verifier que la moyenne est bien etablie.

CE QUI CHANGE PAR RAPPORT A LA v1
  1. Fenetre par defaut 20 CTU, et JAMAIS avant la mi-run : t0 = max(t_mi, t_fin - F).
     C'est ce qui corrige dom_a20_b30_dense (t_fin - 50 = 10 -> 30).
  2. DERIVE : variation de la moyenne entre sous-fenetres emboitees (fenetre
     entiere, sa derniere moitie, son dernier quart). C'est le critere
     d'etablissement : il mesure, au lieu de supposer un temps d'etablissement.
     Au-dela de DERIVE_MAX, le cas est signale [DERIVE].
  3. Recollement des restarts : si le temps recule (reprise depuis un checkpoint
     anterieur), les lignes anciennes recouvertes sont supprimees.
  4. Deux etalons automatiques, qui ne coutent aucun calcul :
       - dom_a7_b10 doit reproduire la baseline du chapitre maillage ;
       - les trois maillages (20c, 30c) doivent s'accorder sur CDp.
     Si l'un des deux echoue, la fenetre est trop longue : relancer avec 15, puis 12.
  5. Lecture correcte du nombre d'elements dans airfoil.re2 (la v1 renvoyait 3).
  6. t0/(a+b) conserve, mais comme information, jamais comme regle de rejet.

USAGE
   cd ~/cdf_runs
   python3 ~/bin/extract_domaine.py           # fenetre par defaut : 20 CTU
   python3 ~/bin/extract_domaine.py 15        # fenetre plus courte

SORTIE
   - tableau lisible a l'ecran + les deux etalons
   - domaine_data.json  (a rapatrier pour la redaction)
"""
import json
import os
import re
import sys

import numpy as np

FENETRE = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
DERIVE_MAX = 1.0          # % sur CDp entre sous-fenetres emboitees
TOL_ETALON = 0.3          # % admis sur l'etalon baseline
TOL_TRIO = 0.3            # % admis sur la dispersion des trois maillages (20c, 30c)

# baseline du chapitre maillage (moyenne ponderee par le temps sur t >= 20,
# impulsions de restart exclues)
BASELINE = dict(CL=1.018392, CD=0.020643, CDp=0.012456, CDv=0.008187)

CAS = [
    # (dossier,             a,   b,  commentaire)
    ("dom_a7_b10",           7,  10, "domaine de reference (Marina)"),
    ("dom_a7_b20",           7,  20, "OFAT sur b"),
    ("dom_a7_b30",           7,  30, "plan factoriel"),
    ("dom_a15_b10",         15,  10, "OFAT sur a"),
    ("dom_a20_b10",         20,  10, "plan factoriel"),
    ("dom_a20_b30_ORIGINE", 20,  30, "production, maillage 1"),
    ("dom_a20_b30_v2",      20,  30, "production, maillage 2 (.geo commun)"),
    ("dom_a20_b30_dense",   20,  30, "production, maillage 3 (lointain densifie)"),
    ("dom_a30_b45",         30,  45, "extension, verification du plateau"),
    ("prod_a00",            20,  30, "controle a portance nulle"),
    ("prod_a5",             20,  30, "polaire"),
    ("prod_a15",            20,  30, "polaire"),
]

TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def nelgt(d):
    """Nombre d'elements lu dans l'en-tete de airfoil.re2."""
    try:
        with open(os.path.join(d, "airfoil.re2"), "rb") as f:
            head = f.read(80).decode("ascii", "ignore")
        head = re.sub(r"^#v\d+", " ", head)          # marqueur de version
        n = re.findall(r"-?\d+", head)
        return int(n[0]) if n else None
    except Exception:
        return None


def charger(p):
    """drag_results.dat -> tableau, avec ou sans en-tete texte."""
    for kw in (dict(comments="#"), dict(skiprows=1)):
        try:
            a = np.loadtxt(p, **kw)
            if a.ndim == 2 and a.shape[0] > 100:
                return a
        except Exception:
            continue
    return None


def recoller(a):
    """Restarts : si le temps recule, on jette les lignes anciennes recouvertes."""
    t = a[:, 1]
    coupes = np.where(np.diff(t) <= 0)[0] + 1
    if coupes.size == 0:
        return a
    garde = []
    for idx in np.split(np.arange(len(a)), coupes):
        t0 = t[idx[0]]
        garde = [k for k in garde if t[k] < t0]
        garde.extend(idx.tolist())
    return a[np.array(garde)]


def moyennes(t, y, m):
    tt, yy = t[m], y[m]
    mu = float(TRAPZ(yy, tt) / (tt[-1] - tt[0]))
    rms = float(np.sqrt(TRAPZ((yy - mu) ** 2, tt) / (tt[-1] - tt[0])))
    return mu, rms


def derive(t, y, m):
    """Ecart relatif maximal entre la fenetre entiere, sa derniere moitie et son
    dernier quart : 0 % = moyenne etablie."""
    tt = t[m]
    ref = moyennes(t, y, m)[0]
    ecarts = []
    for frac in (0.5, 0.25):
        t0 = tt[-1] - frac * (tt[-1] - tt[0])
        mm = m & (t >= t0)
        if mm.sum() > 50:
            ecarts.append(abs(moyennes(t, y, mm)[0] / ref - 1) * 100)
    return max(ecarts) if ecarts else float("nan")


def lire(d, fen, a_dom, b_dom):
    p = os.path.join(d, "drag_results.dat")
    if not os.path.isfile(p):
        return None
    arr = charger(p)
    if arr is None:
        return None
    ok = (np.abs(arr[:, 3]) < 2.0) & (arr[:, 2] > 0) & (arr[:, 2] < 1.0)   # lignes aberrantes
    nrej = int((~ok).sum())
    arr = recoller(arr[ok])
    t = arr[:, 1]
    t_mi = t[0] + 0.5 * (t[-1] - t[0])
    t0 = max(t_mi, t[-1] - fen)                # <-- jamais avant la mi-run
    m = t >= t0
    if m.sum() < 50:
        return None
    r = dict(t_end=float(t[-1]), t_win=[float(t0), float(t[-1])],
             duree_win=float(t[-1] - t0), n_rejets=nrej, n_ech=int(m.sum()),
             t0_sur_ab=float(t0 / (a_dom + b_dom)))
    for nom, col in (("CL", 3), ("CD", 2), ("CDp", 5), ("CDv", 8)):
        mu, rms = moyennes(t, arr[:, col], m)
        r[nom], r[nom + "_rms"] = mu, rms
    r["derive_CDp"] = derive(t, arr[:, 5], m)
    r["derive_CL"] = derive(t, arr[:, 3], m)
    return r


# ---------------------------------------------------------------- extraction
out = {"fenetre_CTU": FENETRE, "derive_max": DERIVE_MAX, "cas": {}}
print("=" * 118)
print(f" ETUDE DE DOMAINE — fenetre commune {FENETRE:.0f} CTU, jamais avant la mi-run")
print("=" * 118)
print(f"{'cas':22s} {'a':>3s} {'b':>3s} {'elem':>6s} {'t_fin':>6s} {'win':>5s} {'t0/(a+b)':>8s} "
      f"{'CL':>10s} {'CD':>10s} {'CDp':>10s} {'CDv':>10s} {'deriv%':>7s}")
print("-" * 118)
for d, a_dom, b_dom, com in CAS:
    if not os.path.isdir(d):
        print(f"{d:22s} {'ABSENT':>40s}")
        continue
    r = lire(d, FENETRE, a_dom, b_dom)
    if r is None:
        print(f"{d:22s} {'(pas de donnees exploitables)':>40s}")
        continue
    r.update(a=a_dom, b=b_dom, commentaire=com, elements=nelgt(d))
    out["cas"][d] = r
    flag = "  [DERIVE]" if r["derive_CDp"] > DERIVE_MAX else ""
    print(f"{d:22s} {a_dom:3d} {b_dom:3d} {str(r['elements'] or '?'):>6s} {r['t_end']:6.1f} "
          f"{r['duree_win']:5.1f} {r['t0_sur_ab']:8.2f} {r['CL']:10.6f} {r['CD']:10.6f} "
          f"{r['CDp']:10.6f} {r['CDv']:10.6f} {r['derive_CDp']:7.2f}{flag}")
print("-" * 118)

# ---------------------------------------------------------------- etalons
print("\n ETALON 1 — dom_a7_b10 doit reproduire la baseline du chapitre maillage")
if "dom_a7_b10" in out["cas"]:
    r = out["cas"]["dom_a7_b10"]
    pires = []
    for q in ("CL", "CD", "CDp", "CDv"):
        e = 100 * (r[q] / BASELINE[q] - 1)
        pires.append(abs(e))
        print(f"   {q:4s} {r[q]:10.6f}  vs {BASELINE[q]:10.6f}   {e:+6.2f} %"
              f"   {'OK' if abs(e) < TOL_ETALON else '<<< FENETRE TROP LONGUE'}")
    out["etalon_baseline_max_pct"] = max(pires)
else:
    print("   (cas absent)")

trio = [k for k in ("dom_a20_b30_ORIGINE", "dom_a20_b30_v2", "dom_a20_b30_dense") if k in out["cas"]]
print("\n ETALON 2 — les trois maillages (20c, 30c) doivent s'accorder sur CDp")
if len(trio) == 3:
    v = np.array([out["cas"][k]["CDp"] for k in trio])
    disp = 100 * (v.max() - v.min()) / v.mean()
    out["etalon_trio_pct"] = float(disp)
    for k, x in zip(trio, v):
        print(f"   {k:22s} CDp = {x:.6f}")
    print(f"   dispersion : {disp:.2f} %   {'OK' if disp < TOL_TRIO else '<<< AU MOINS UN CAS NON ETABLI'}")
else:
    print(f"   (seulement {len(trio)}/3 cas presents)")

# ---------------------------------------------------------------- plan 2x2
need = ["dom_a7_b10", "dom_a7_b30", "dom_a20_b10", "dom_a20_b30_v2"]
if all(k in out["cas"] for k in need):
    A = out["cas"]["dom_a7_b10"]["CDp"]
    B = out["cas"]["dom_a7_b30"]["CDp"]
    C = out["cas"]["dom_a20_b10"]["CDp"]
    D = out["cas"]["dom_a20_b30_v2"]["CDp"]
    eff_a = 0.5 * ((C - A) + (D - B))
    eff_b = 0.5 * ((B - A) + (D - C))
    inter = 0.5 * ((D - C) - (B - A))
    out["factoriel"] = dict(effet_a=eff_a, effet_b=eff_b, interaction=inter,
                            ratio_inter_a=abs(inter / eff_a),
                            coins=dict(a7b10=A, a7b30=B, a20b10=C, a20b30=D))
    print("\n PLAN FACTORIEL 2x2  (grandeur pilote : CDp)")
    print(f"   b 10->30 a a=7c   : {(B - A) / A * 100:+7.2f} %")
    print(f"   a 7->20  a b=10c  : {(C - A) / A * 100:+7.2f} %")
    print(f"   b 10->30 a a=20c  : {(D - C) / C * 100:+7.2f} %")
    print(f"   a 7->20  a b=30c  : {(D - B) / B * 100:+7.2f} %")
    print(f"   effet principal a : {eff_a:+.3e}")
    print(f"   effet principal b : {eff_b:+.3e}")
    print(f"   INTERACTION a x b : {inter:+.3e}   ({abs(inter / eff_a):.2f} x l'effet de a)")

# ---------------------------------------------------------------- extrapolation
fam = [("dom_a7_b10", 7.0), ("dom_a20_b30_v2", 20.0), ("dom_a30_b45", 30.0)]
if all(k in out["cas"] for k, _ in fam):
    av = np.array([x for _, x in fam])
    cp = np.array([out["cas"][k]["CDp"] for k, _ in fam])
    cv = np.array([out["cas"][k]["CDv"] for k, _ in fam])
    print("\n FAMILLE AUTO-SIMILAIRE  (b/a ~ 1.5)  et EXTRAPOLATION")
    for i in range(len(fam) - 1):
        print(f"   {fam[i][0]:22s} -> {fam[i + 1][0]:22s} : CDp {(cp[i + 1] - cp[i]) / cp[i] * 100:+7.2f} %")
    out["extrapolation"] = {}
    for p in (1.0, 1.5, 2.0):
        M = np.column_stack([np.ones(3), av ** (-p)])
        Cinf, K = np.linalg.lstsq(M, cp, rcond=None)[0]
        err = [(c - Cinf) / Cinf * 100 for c in cp]
        out["extrapolation"][f"p={p}"] = dict(CDp_inf=float(Cinf), K=float(K),
                                              err_residuelle=[float(e) for e in err])
        print(f"   1/a^{p}: CDp_inf = {Cinf:.6f}   erreurs residuelles "
              f"a=7:{err[0]:+5.1f}%  a=20:{err[1]:+5.1f}%  a=30:{err[2]:+5.1f}%")
    Cmid = out["extrapolation"]["p=1.5"]["CDp_inf"]
    CDinf = Cmid + float(cv.mean())
    out["CD_infini_estime"] = CDinf
    print(f"\n   CD_infini estime (p=1.5, CDv={cv.mean():.6f}) : {CDinf:.6f}")
    for k, _ in fam:
        cd = out["cas"][k]["CD"]
        print(f"     {k:22s} CD={cd:.6f}  erreur de confinement {(cd - CDinf) / CDinf * 100:+6.2f} %")

with open("domaine_data.json", "w") as f:
    json.dump(out, f, indent=1)
print("\n -> domaine_data.json ecrit dans", os.getcwd())

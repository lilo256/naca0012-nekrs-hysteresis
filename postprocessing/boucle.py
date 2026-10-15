#!/usr/bin/env python3
"""
boucle.py  --  tableau complet de la boucle d'hysteresis (montee, descentes, sauts)
===================================================================================
Pour chaque cas de ~/cdf_runs (prod_a15*, up_a*, cold_a*, down_a*, jump_a*) :
  - son PARENT, retrouve en comparant son fichier startFrom aux fichiers des autres cas
    (next.sh copie le point de sauvegarde du parent dans le nouveau dossier) ;
  - la chaine a laquelle il appartient (on remonte les parents jusqu'a la racine) ;
  - le pas d'incidence depuis le parent (0,25 = quasi statique, plus = saut) ;
  - C_L, C_D, C_Dp, C_Dv moyens (ponderes en temps) sur les derniers W CTU, le rms de C_L,
    et la derive entre les deux moitiees de la fenetre (convergence).
Puis un tableau par angle : montee / descente longue / descente courte et sauts.

Lecture rapide : on n'analyse pas tout drag_results.dat, on y preleve ~1500 lignes reparties
dans la fenetre (acces direct dans le fichier). Suffisant pour des moyennes a 1e-3 pres.

USAGE
   python3 ~/bin/boucle.py          # fenetre = 20 derniers CTU (au plus la moitie du cas)
   python3 ~/bin/boucle.py 30       # 30 derniers CTU
"""
import glob, math, os, re, sys

RUNS = os.path.expanduser("~/cdf_runs")
W = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
NECH = 1500
COL = dict(t=1, CD=2, CL=3, CDp=5, CDv=8)
MOTIFS = ("prod_a15*", "up_a*", "cold_a*", "down_a*", "jump_a*")


def lire(b):
    s = b.split()
    if len(s) <= COL["CDv"]:
        return None
    try:
        v = [float(s[COL[k]]) for k in ("t", "CD", "CL", "CDp", "CDv")]
    except ValueError:
        return None
    return v if abs(v[2]) < 5 and all(math.isfinite(x) for x in v) else None


def ligne_apres(f, off):
    """premiere ligne complete qui commence apres l'octet off"""
    f.seek(off)
    c = f.read(2048)
    i = c.find(b"\n") if off > 0 else -1
    j = c.find(b"\n", i + 1)
    return lire(c[i + 1:j]) if j > 0 else None


def bornes(f, taille):
    f.seek(0)
    t0 = next((v[0] for v in map(lire, f.read(65536).splitlines()) if v), None)
    f.seek(max(0, taille - 8192))
    fin = [v for v in map(lire, f.read().splitlines()[1:-1] or []) if v]
    return t0, (fin[-1][0] if fin else None)


def octet_de(f, taille, t_cible):
    """position approximative de t_cible (t croissant ; les chevauchements de reprise ne genent pas)"""
    a, b = 0, taille
    for _ in range(40):
        if b - a < 4096:
            break
        m = (a + b) // 2
        v = ligne_apres(f, m)
        if v is None or v[0] < t_cible:
            a = m
        else:
            b = m
    return a


def moyenne_t(e, k):
    """moyenne ponderee en temps (trapezes) de la colonne k sur les echantillons tries par t"""
    if len(e) < 2 or e[-1][0] <= e[0][0]:
        return sum(x[k] for x in e) / len(e) if e else float("nan")
    s = sum(0.5 * (e[i][k] + e[i + 1][k]) * (e[i + 1][0] - e[i][0]) for i in range(len(e) - 1))
    return s / (e[-1][0] - e[0][0])


def analyse(d):
    p = os.path.join(d, "drag_results.dat")
    if not os.path.isfile(p) or os.path.getsize(p) < 1000:
        return None
    taille = os.path.getsize(p)
    with open(p, "rb") as f:
        t0, t1 = bornes(f, taille)
        if t0 is None or t1 is None:
            return None
        w = min(W, 0.5 * (t1 - t0))
        o0 = octet_de(f, taille, t1 - w)
        e = [ligne_apres(f, o0 + (taille - o0) * i // NECH) for i in range(NECH)]
    e = sorted((x for x in e if x and x[0] >= t1 - w - 1e-9), key=lambda x: x[0])
    if len(e) < 20:
        return None
    r = dict(t0=t0, t1=t1, w=e[-1][0] - e[0][0], n=len(e))
    for k, nom in ((1, "CD"), (2, "CL"), (3, "CDp"), (4, "CDv")):
        r[nom] = moyenne_t(e, k)
    m = r["CL"]
    r["CLr"] = (sum((x[2] - m) ** 2 for x in e) / len(e)) ** 0.5
    tm = 0.5 * (e[0][0] + e[-1][0])
    h1, h2 = [x for x in e if x[0] < tm], [x for x in e if x[0] >= tm]
    r["der"] = 100 * (moyenne_t(h2, 2) - moyenne_t(h1, 2)) / abs(m) if len(h1) > 2 and len(h2) > 2 else float("nan")
    return r


def par(d, cle):
    q = os.path.join(d, "airfoil.par")
    if not os.path.isfile(q):
        return None
    for l in open(q, errors="ignore"):
        m = re.match(rf"\s*{cle}\s*=\s*(\S+)", l.split("#")[0], re.I)
        if m:
            return m.group(1).strip('"')
    return None


def angle(nom):
    m = re.search(r"_a(\d+(?:\.\d+)?)", nom)
    return float(m.group(1)) if m else float("nan")


# ------------------------------------------------------------------ cas et analyses
dirs = sorted({x.rstrip("/") for m in MOTIFS for x in glob.glob(os.path.join(RUNS, m + "/"))})
C = {}
for d in dirs:
    nom = os.path.basename(d)
    r = analyse(d)
    if r is None:
        continue
    r.update(nom=nom, d=d, a=angle(nom), sf=par(d, "startFrom"), tf=par(d, "endTime"))
    C[nom] = r
    print(f"  lu {nom:14s}", end="\r", file=sys.stderr)
print(" " * 40, end="\r", file=sys.stderr)

# ------------------------------------------------------------------ parents
def meme_fichier(f, g):
    """meme taille, meme premier Mo et meme fin : l'en-tete porte le temps et le pas du point de
    sauvegarde, et les donnees different des les premiers octets entre deux champs distincts"""
    if os.path.getsize(f) != os.path.getsize(g):
        return False
    with open(f, "rb") as a, open(g, "rb") as b:
        if a.read(1 << 20) != b.read(1 << 20):
            return False
        n = os.path.getsize(f)
        a.seek(max(0, n - 65536)); b.seek(max(0, n - 65536))
        return a.read() == b.read()


for nom, r in C.items():
    r["par"] = None
    sf = r["sf"]
    if not sf or nom.startswith("cold_") or nom.startswith("prod_"):
        continue                                   # depart a froid (eventuellement prolonge)
    if "/" in sf:                                  # chemin absolu : le parent est dans le chemin
        pn = os.path.basename(os.path.dirname(sf))
        r["par"] = pn if pn != nom and pn in C else None
        continue
    f = os.path.join(r["d"], sf)
    if not os.path.isfile(f):
        continue
    for pn, q in C.items():
        g = os.path.join(q["d"], sf)
        if pn == nom or not os.path.isfile(g) or q["t0"] >= r["t0"] - 0.5:
            continue                               # un parent a commence avant l'enfant
        if meme_fichier(f, g):
            r["par"] = pn
            break


def racine(nom, vu=()):
    p = C[nom]["par"]
    return nom if p is None or p in vu else racine(p, vu + (nom,))


GROUPES = [("MONTEE (depuis prod_a15)", lambda n, rc: rc.startswith("prod_") or n.startswith("up_")),
           ("DESCENTE LONGUE (depuis le depart a froid a 20 deg)", lambda n, rc: rc == "cold_a20.00"),
           ("DESCENTE COURTE (depuis l'etat decroche du depart a froid a 17 deg)", lambda n, rc: rc == "cold_a17.00"),
           ("AUTRES DEPARTS A FROID / PARENT INCONNU", lambda n, rc: True)]


def ligne(r):
    p = r["par"]
    da = f"{r['a'] - C[p]['a']:+6.2f}" if p else "      "
    saut = "  SAUT" if p and abs(r["a"] - C[p]["a"]) > 0.51 else ""
    try:
        encours = "  t<endTime" if r["tf"] and r["t1"] < float(r["tf"]) - 0.05 else ""
    except ValueError:
        encours = ""
    return (f" {r['a']:6.2f}  {r['nom']:13s} {(p or '-'):13s} {da}  {r['t0']:6.1f}->{r['t1']:6.1f}"
            f"  {r['w']:5.1f}  {r['CL']:7.4f} {r['CLr']:.1e}  {r['CD']:.5f} {r['CDp']:.5f} {r['CDv']:.5f}"
            f"  {r['der']:+6.2f}%{saut}{encours}")


print("=" * 128)
print(f" BOUCLE D'HYSTERESIS  --  moyennes ponderees en temps sur les {W:g} derniers CTU (au plus la moitie du cas)")
print(" derive = ecart de C_L entre 2e et 1re moitie de la fenetre ; dalpha = pas depuis le parent (SAUT si > 0,5 deg)")
print("=" * 128)
tete = (f" {'alpha':>6s}  {'cas':13s} {'parent':13s} {'dalpha':>6s}  {'t debut->fin':>14s}  {'fen.':>5s}"
        f"  {'C_L':>7s} {'rms':>7s}  {'C_D':>7s} {'C_Dp':>7s} {'C_Dv':>7s}  {'derive':>7s}")
place = set()
for titre, test in GROUPES:
    noms = [n for n in C if n not in place and test(n, racine(n))]
    if not noms:
        continue
    place.update(noms)
    print(f"\n {titre}")
    print(tete)
    for n in sorted(noms, key=lambda n: (-C[n]["a"] if "DESCENTE" in titre else C[n]["a"], C[n]["t0"])):
        print(ligne(C[n]))

# ------------------------------------------------------------------ tableau par angle
col = {"montee": [], "longue": [], "courte": []}
for n, r in C.items():
    rc = racine(n)
    if rc.startswith("prod_") or n.startswith("up_"):
        col["montee"].append(r)
    elif rc == "cold_a20.00":
        col["longue"].append(r)
    elif rc == "cold_a17.00" or n.startswith("jump_"):
        col["courte"].append(r)


def val(lst, a):
    v = [r for r in lst if abs(r["a"] - a) < 1e-6]
    if not v:
        return f"{'-':>16s}"
    r = max(v, key=lambda r: r["t1"])
    mark = "s" if r["nom"].startswith("jump_") else ("*" if r["nom"].startswith("cold_") else " ")
    return f"{r['CL']:7.4f} {r['CD']:.4f}{mark}"


angles = sorted({r["a"] for lst in col.values() for r in lst}, reverse=True)
print("\n" + "=" * 128)
print(" PAR ANGLE : C_L  C_D     (s = saut, * = depart a froid)")
print(f" {'alpha':>6s}   {'montee':>16s}   {'descente longue':>16s}   {'desc. courte/sauts':>18s}")
for a in angles:
    print(f" {a:6.2f}   {val(col['montee'], a)}   {val(col['longue'], a)}   {val(col['courte'], a):>18s}")
print("=" * 128)

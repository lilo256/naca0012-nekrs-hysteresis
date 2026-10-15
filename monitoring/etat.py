#!/usr/bin/env python3
"""
etat.py  --  etat de tous les calculs de ~/cdf_runs, sur cdf ET nofx (le home est partage)
=========================================================================================
Pour chaque cas :
  - EN COURS  : drag_results.dat modifie il y a moins de ACTIF minutes ;
  - FINI      : arrive a son endTime dans les dernieres 48 h ;
  - ARRETE ?  : plus rien d'ecrit depuis ACTIF minutes alors que endTime n'est pas atteint
                (plantage, job tue, file d'attente...).
Donne : temps atteint / endTime, CTU restants, vitesse (CTU/h), heure de fin prevue, et les
moyennes de C_L, C_D, C_Dp, C_Dv sur les derniers FEN CTU (avec le rms de C_L).

Vitesse : lue dans le log du job (lignes "t=" et "elapsedStepSum="). A defaut, estimee entre
deux appels successifs du script (memoire dans ~/.etat_cache.json).

USAGE
   python3 ~/bin/etat.py            # ou, apres l'alias ci-dessous : etat
   python3 ~/bin/etat.py 10         # moyennes sur les 10 derniers CTU (defaut 5)
   echo "alias etat='python3 ~/bin/etat.py'" >> ~/.bashrc && source ~/.bashrc

Colonnes de drag_results.dat (indices a partir de 0) : a verifier une fois avec
"head -2 drag_results.dat". La colonne "somme" doit afficher ~0 (C_Dp + C_Dv - C_D) :
si ce n'est pas le cas, corriger COL ci-dessous.
"""
import glob, json, os, re, subprocess, sys, time
from datetime import datetime, timedelta

RUNS = os.path.expanduser("~/cdf_runs")
FEN = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0      # CTU pour les moyennes
ACTIF = 20.0                                                 # minutes sans ecriture -> plus actif
COL = dict(t=1, CD=2, CL=3, CDp=5, CDv=8)
CACHE = os.path.expanduser("~/.etat_cache.json")
JOURS = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]


# ------------------------------------------------------------------ lecture de la fin d'un fichier
def queue_lignes(p, octets):
    with open(p, "rb") as f:
        f.seek(0, 2)
        n = f.tell()
        f.seek(max(0, n - octets))
        brut = f.read().decode(errors="ignore").splitlines()
    return brut[1:] if n > octets else brut          # 1re ligne probablement coupee


def efforts(p, fen):
    """lignes numeriques des derniers `fen` CTU (lecture de la fin du fichier seulement)."""
    octets = 2_000_000
    while True:
        rows, ncol = [], None
        for l in queue_lignes(p, octets):
            s = l.split()
            try:
                v = [float(x) for x in s]
            except ValueError:
                continue
            if ncol is None:
                ncol = len(v)
            if len(v) == ncol:
                rows.append(v)
        if not rows:
            return None
        t = [r[COL["t"]] for r in rows]
        if t[-1] - t[0] >= fen or octets >= 256_000_000 or octets >= os.path.getsize(p):
            break
        octets *= 4
    tfin = rows[-1][COL["t"]]
    rows = [r for r in rows if r[COL["t"]] >= tfin - fen and abs(r[COL["CL"]]) < 5]
    return rows


def moy(v):
    return sum(v) / len(v) if v else float("nan")


def rms(v):
    m = moy(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5 if v else float("nan")


def end_time(d):
    p = os.path.join(d, "airfoil.par")
    if not os.path.isfile(p):
        return None
    for l in open(p, errors="ignore"):
        l = l.split("#")[0]
        m = re.match(r"\s*endTime\s*=\s*([0-9.eE+-]+)", l, re.I)
        if m:
            return float(m.group(1))
    return None


# ------------------------------------------------------------------ vitesse depuis le log du job
RX_T = re.compile(r"step=\s*(\d+)\s+t=\s*([0-9.eE+-]+)")
RX_S = re.compile(r"step=\s*(\d+)\s+elapsedStep=.*elapsedStepSum=\s*([0-9.eE+-]+)")


def paires(lignes):
    T, S = {}, {}
    for l in lignes:
        m = RX_T.search(l)
        if m:
            T[int(m.group(1))] = float(m.group(2))
            continue
        m = RX_S.search(l)
        if m:
            S[int(m.group(1))] = float(m.group(2))
    return sorted((k, T[k], S[k]) for k in T.keys() & S.keys())


def vitesse_log(d):
    cands = []
    for pat in ("NACA0012.*.out", "*.o[0-9]*", "logfile", "*.log", "nohup.out"):
        cands += glob.glob(os.path.join(d, pat))
    cands = [c for c in cands if os.path.isfile(c) and os.path.getsize(c) > 0]
    if not cands:
        return None
    log = max(cands, key=os.path.getmtime)
    try:
        with open(log, "rb") as f:
            debut = f.read(3_000_000).decode(errors="ignore").splitlines()
        fin = queue_lignes(log, 1_000_000)
    except OSError:
        return None
    a, b = paires(debut), paires(fin)
    if not a or not b or b[-1][2] - a[0][2] < 60:
        return None
    return (b[-1][1] - a[0][1]) / (b[-1][2] - a[0][2]) * 3600.0     # CTU par heure


# ------------------------------------------------------------------ jobs (file de la machine courante)
def jobs():
    out = {}
    try:
        q = subprocess.run(["qstat"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return out, ""
    for l in q.splitlines()[2:]:
        s = l.split()
        if not s or not s[0].isdigit():
            continue
        try:
            info = subprocess.run(["qstat", "-j", s[0]], capture_output=True, text=True, timeout=20).stdout
            m = re.search(r"^cwd:\s*(\S+)", info, re.M)
            if m:
                out[os.path.basename(m.group(1).rstrip("/"))] = (s[0], s[4], l[l.find("@") + 1:].split()[0] if "@" in l else "")
        except (OSError, subprocess.SubprocessError):
            pass
    return out, q


def horodate(h):
    fin = datetime.now() + timedelta(hours=h)
    return f"{JOURS[fin.weekday()]} {fin:%d/%m %H:%M}"


# ------------------------------------------------------------------ principal
# "maintenant" selon l'horloge du serveur de fichiers (QNAP), qui date aussi drag_results.dat.
# Il faut ECRIRE dans le fichier : rouvrir un fichier vide ne met pas sa date a jour sur NFS.
ref = os.path.join(os.path.expanduser("~"), ".etat_ref")
with open(ref, "w") as f:
    f.write(f"{time.time():.3f}\n")
try:
    os.utime(ref, None)
except OSError:
    pass
maintenant = os.path.getmtime(ref)
try:
    cache = json.load(open(CACHE))
except (OSError, ValueError):
    cache = {}
JOBS, QSTAT = jobs()

en_cours, finis, arretes = [], [], []
for d in sorted(glob.glob(os.path.join(RUNS, "*/"))):
    nom = os.path.basename(d.rstrip("/"))
    p = os.path.join(d, "drag_results.dat")
    if not os.path.isfile(p):
        continue
    age = (maintenant - os.path.getmtime(p)) / 60.0    # minutes
    if age > 48 * 60:
        continue
    rows = efforts(p, FEN)
    if not rows:
        continue
    t = rows[-1][COL["t"]]
    tf = end_time(d)
    fini = tf is not None and t >= tf - 1e-3
    r = dict(nom=nom, t=t, tf=tf, age=age, n=len(rows),
             w=rows[-1][COL["t"]] - rows[0][COL["t"]],
             CL=moy([x[COL["CL"]] for x in rows]), CLr=rms([x[COL["CL"]] for x in rows]),
             CD=moy([x[COL["CD"]] for x in rows]))
    ncol = len(rows[0])
    if ncol > max(COL["CDp"], COL["CDv"]):
        r["CDp"] = moy([x[COL["CDp"]] for x in rows])
        r["CDv"] = moy([x[COL["CDv"]] for x in rows])
    # Date de fichier dans le futur : le noeud qui ecrit a une horloge en avance (vu sur des cas
    # passes par nofx, ~ +45 h). La date ne dit alors plus rien : on juge l'activite sur la
    # progression de t entre deux appels d'etat (memoire dans CACHE).
    prev = cache.get(nom)
    r["futur"] = age < -5.0
    if r["futur"]:
        if prev and maintenant - prev[1] > 60:
            actif = t > prev[0] + 1e-9
        else:
            actif = None                      # premier passage : on ne sait pas encore
    else:
        actif = age <= ACTIF
    if fini:
        finis.append(r)
    elif actif or actif is None:
        v = vitesse_log(d)
        if v is None and prev and maintenant - prev[1] > 60:
            dt_ctu, dt_s = t - prev[0], maintenant - prev[1]
            v = dt_ctu / dt_s * 3600.0 if dt_ctu > 0 else None
        r["v"], r["incertain"] = v, actif is None
        en_cours.append(r)
    else:
        arretes.append(r)
    if not fini and not (r["futur"] and prev and maintenant - prev[1] <= 60):
        cache[nom] = [t, maintenant]
try:
    json.dump(cache, open(CACHE, "w"))
except OSError:
    pass


def valeurs(r):
    s = f"CL {r['CL']:7.4f} (rms {r['CLr']:.1e})  CD {r['CD']:.5f}"
    if "CDp" in r:
        s += f"  CDp {r['CDp']:.5f}  CDv {r['CDv']:.5f}  somme {r['CDp'] + r['CDv'] - r['CD']:+.0e}"
    return s + f"   [sur {r['w']:.1f} CTU]"


now = datetime.now()
print("=" * 118)
print(f" ETAT DES CALCULS  —  {JOURS[now.weekday()]} {now:%d/%m %H:%M}   (moyennes sur les {FEN:g} derniers CTU)")
print("=" * 118)
print(f"\n EN COURS ({len(en_cours)})")
print(f" {'cas':15s} {'t':>8s} {'fin':>7s} {'reste':>7s} {'vitesse':>11s}  {'fin prevue':22s} job")
for r in en_cours:
    reste = (r["tf"] - r["t"]) if r["tf"] is not None else float("nan")
    if r.get("v") and not r.get("incertain"):
        h = reste / r["v"]
        vit, fin = f"{r['v']:6.2f} CTU/h", f"{h:5.1f} h ({horodate(h)})"
    else:
        vit, fin = f"{'?':>11s}", "relancer etat dans 5 min"
    j = JOBS.get(r["nom"])
    js = f"{j[0]} {j[2]}" if j else "(autre cluster)"
    print(f" {r['nom']:15s} {r['t']:8.2f} {r['tf'] if r['tf'] else float('nan'):7.1f} {reste:7.2f} {vit}  {fin:22s} {js}")
    print(f" {'':15s} {valeurs(r)}")
print(f"\n FINIS DANS LES DERNIERES 48 h ({len(finis)})  -> leur noeud est libre")
for r in finis:
    quand = "fini (date du fichier en avance)" if r["futur"] else f"fini il y a {r['age']/60:5.1f} h"
    print(f" {r['nom']:15s} t={r['t']:8.2f}  {quand:32s} {valeurs(r)}")
print(f"\n ARRETES AVANT endTime ? ({len(arretes)})  -> verifier le log (plantage, job tue)")
for r in arretes:
    quand = "t n'avance plus depuis le dernier etat" if r["futur"] else f"plus d'ecriture depuis {r['age']/60:5.1f} h"
    print(f" {r['nom']:15s} t={r['t']:8.2f} / {r['tf'] if r['tf'] else float('nan'):7.1f}"
          f"   {quand}   {valeurs(r)}")
if QSTAT.strip():
    print("\n FILE DE CETTE MACHINE (qstat)")
    print(QSTAT.rstrip())
print("=" * 118)

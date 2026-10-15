#!/usr/bin/env python3
"""
spectre_sondes.py  --  bandes et pics spectraux des efforts et des sondes d'un cas
=================================================================================
1. rms par bande de Strouhal (base corde) : lacher [0.8-2.6], double [2.6-4.6],
   [6.5-9.5] (mode d'extrados ~7.8) et [9.5-12] (mode du sillage proche ~10.7)
   -> OU chaque mode est le plus fort ;
2. principaux pics (frequence, amplitude) des signaux clefs, et leur rapport a la
   frequence du lacher f1  -> un pic a f/f1 entier est un HARMONIQUE du lacher.

Pas variable : le signal est reechantillonne sur une grille reguliere (message).
Enregistrement long (> 10 CTU) : moyenne de Welch sur des segments de 10 CTU
(resolution ~0.1) ; court (film) : un seul segment (resolution ~1/duree).

USAGE
   cd ~/cdf_runs/film_a15 && python3 -u ~/bin/spectre_sondes.py      | tee spectre_pics.txt
   cd ~/cdf_runs/prod_a15 && python3 -u ~/bin/spectre_sondes.py 50   | tee spectre_pics.txt
   cd ~/cdf_runs/spec_a19.00 && python3 -u ~/bin/spectre_sondes.py 205 20 0.05   # etats decroches
   (arguments : temps de debut [defaut : 2e moitie si long, sinon debut + 0.05],
    longueur des segments de Welch en CTU [10], debut de la recherche de pics en St [0.5].
    Pour les etats decroches, dont l'energie est sous St 0.8 : segments de 20 CTU et pics des 0.05.)
"""
import numpy as np, os, sys

BANDES = [("lent", 0.05, 0.8), ("lacher", 0.8, 2.6), ("double", 2.6, 4.6), ("7-9", 6.5, 9.5), ("10-12", 9.5, 12.0)]
CLEFS = ["CL", "CD", "ux_A020", "ux_A050", "ux_A092", "uy_B050", "uy_B070", "uy_B090",
         "uy_W015p", "uy_W015m", "uy_REF"]
SEG = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0     # CTU par segment de Welch
FMIN = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5      # debut de la recherche de pics
FMAX = 15.0


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


def regulier(t, Y):
    """grille reguliere au pas median ; renvoie (t, Y, pas, reechantillonne?)"""
    d = np.diff(t)
    med = np.median(d)
    if np.all(np.abs(d[:-1] - med) <= 0.01 * med):     # seul le dernier pas (raccourci
        if abs(d[-1] - med) > 0.01 * med:               # pour tomber sur endTime) differe
            t, Y = t[:-1], Y[:-1]
        return t, Y, med, False
    tr = np.arange(t[0], t[-1], med)
    return tr, np.column_stack([np.interp(tr, t, Y[:, j]) for j in range(Y.shape[1])]), med, True


def spectre(y, dt):
    """spectre d'amplitude (une face, Hann, corrige) moyenne sur segments (Welch)."""
    y = y - y.mean()
    n = len(y)
    L = min(n, int(round(SEG / dt)))
    pas = max(1, L // 2)
    segs = [y[i:i + L] for i in range(0, n - L + 1, pas)] or [y]
    L = len(segs[0])
    w = np.hanning(L)
    nfft = L * max(1, min(8, 2 ** 18 // L))           # bourrage si segment court
    P = np.mean([np.abs(np.fft.rfft((s - s.mean()) * w, nfft)) ** 2 for s in segs], axis=0)
    f = np.fft.rfftfreq(nfft, dt)
    A = 2.0 * np.sqrt(P) / w.sum()                    # amplitude d'une sinusoide
    pw = 2.0 * P / (L ** 2 * np.mean(w ** 2)) * (L / nfft)   # rms^2 par raie
    return f, A, pw, L * dt


def bandes(f, pw, y):
    out = {"rms": float(np.std(y))}
    for nom, a, b in BANDES:
        m = (f >= a) & (f < b)
        out[nom] = float(np.sqrt(pw[m].sum()))
    return out


def pics(f, A, dur, n=6):
    """maxima locaux (separes d'au moins 2 lobes) dans [0.5, FMAX], tries par amplitude."""
    df = f[1] - f[0]
    k = max(1, int(round(2.0 / dur / df)))            # ~ largeur du lobe principal de Hann
    m = np.where((f >= FMIN) & (f <= FMAX))[0]
    cand = []
    for i in m:
        lo, hi = max(0, i - k), min(len(A), i + k + 1)
        if A[i] == A[lo:hi].max() and A[i] > 0:
            if 0 < i < len(A) - 1 and A[i - 1] > 0 and A[i + 1] > 0:
                a_, b_, c_ = np.log(A[i - 1]), np.log(A[i]), np.log(A[i + 1])
                den = a_ - 2 * b_ + c_
                dk = 0.5 * (a_ - c_) / den if den != 0 else 0.0
            else:
                dk = 0.0
            cand.append((A[i], f[i] + dk * df))
    cand.sort(reverse=True)
    cand = [c for c in cand if c[0] >= 0.02 * cand[0][0]] if cand else []   # >= 2 % du plus grand
    return [(fr, a) for a, fr in cand[:n]]


# ------------------------------------------------------------------------------------
t_arg = float(sys.argv[1]) if len(sys.argv) > 1 else None
SIG = {}          # nom -> (t, y)

if os.path.isfile("drag_results.dat"):
    a = charger("drag_results.dat")
    a = a[(np.abs(a[:, 3]) < 2) & (a[:, 2] > 0) & (a[:, 2] < 2)]
    a = a[monotone(a[:, 1])]
    SIG["_forces"] = (a[:, 1], a[:, [3, 2]], ["CL", "CD"])
if os.path.isfile("probes_uvp.csv"):
    h = [c.strip() for c in open("probes_uvp.csv").readline().strip().split(",")]
    s = charger("probes_uvp.csv", ",")
    s = s[monotone(s[:, 0])]
    noms = [n for n in h if (n.startswith("ux_A") or n.startswith("uy_B") or n.startswith("uy_W")
                             or n.startswith("uy_REF") or n.startswith("uy_SPAN"))]
    SIG["_sondes"] = (s[:, 0], s[:, [h.index(n) for n in noms]], noms)

res, spec, f1 = {}, {}, None
for cle, (t, Y, noms) in SIG.items():
    t_fin = t[-1]
    if t_arg is not None:
        t0 = t_arg
    else:
        t0 = t[0] + 0.5 * (t_fin - t[0]) if t_fin - t[0] > 2 * SEG else t[0] + 0.05
    m = t >= t0
    t, Y = t[m], Y[m]
    t, Y, dt, rs = regulier(t, Y)
    print(f" ({'efforts' if cle == '_forces' else 'sondes'} : [{t[0]:.2f} ; {t[-1]:.2f}] = "
          f"{t[-1]-t[0]:.2f} CTU, pas {dt:.2e}{', REECHANTILLONNE (pas variable)' if rs else ''})")
    for j, n in enumerate(noms):
        f, A, pw, dur = spectre(Y[:, j], dt)
        res[n] = bandes(f, pw, Y[:, j])
        spec[n] = (f, A, dur)

# frequence du lacher f1 : pic de CL dans [1.2 ; 2.4], interpole (pas la raie brute)
if "CL" in spec:
    f, A, dur = spec["CL"]
    m = (f >= 1.2) & (f <= 2.4)
    p1 = pics(f[m], A[m], dur, n=1)
    f1 = float(p1[0][0]) if p1 else None

print("=" * 100)
print(" 1. RMS PAR BANDE DE STROUHAL (base corde)")
print("=" * 100)
print(f" {'signal':12s} {'rms total':>10s}  {'lent':>10s} {'lacher':>10s} {'double':>10s} {'7-9':>10s} {'10-12':>10s}")
print(f" {'':12s} {'':10s}  {'[.05-0.8]':>10s} {'[0.8-2.6]':>10s} {'[2.6-4.6]':>10s} {'[6.5-9.5]':>10s} {'[9.5-12]':>10s}")
for n, r in res.items():
    print(f" {n:12s} {r['rms']:10.3e}  {r['lent']:10.3e} {r['lacher']:10.3e} {r['double']:10.3e} {r['7-9']:10.3e} {r['10-12']:10.3e}")

print("\n" + "=" * 100)
print(f" 2. PRINCIPAUX PICS  f (amplitude) [f/f1]    f1 = lacher = {f1:.3f}" if f1 else " 2. PRINCIPAUX PICS")
print("=" * 100)
for n in CLEFS:
    if n not in spec:
        continue
    f, A, dur = spec[n]
    ps = pics(f, A, dur)
    txt = "  ".join(f"{fr:6.2f} ({a:.1e}) [{fr/f1:4.2f}]" if f1 else f"{fr:6.2f} ({a:.1e})"
                    for fr, a in ps)
    print(f" {n:10s} {txt}")
print("-" * 100)
dur = next(iter(spec.values()))[2] if spec else 0
print(f" resolution ~ {2/dur:.2f} (lobe de Hann) ; segments de {dur:.1f} CTU")
print(" [f/f1] entier (1.00, 2.00, 3.00...) = harmonique du lacher ; non entier = mode independant")

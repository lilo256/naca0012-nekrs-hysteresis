#!/usr/bin/env python3
"""
strouhal.py -- estimation du nombre de Strouhal par FFT

Lit drag_results.dat, isole le regime etabli, interpole sur une grille
temporelle reguliere (car dt varie), puis calcule le spectre de CL(t)
et identifie la frequence dominante = Strouhal (car c=U=1).

USAGE : python3 strouhal.py <drag_results.dat> [t_debut]
"""
import numpy as np, sys

fichier = sys.argv[1] if len(sys.argv)>1 else "drag_results.dat"
t0 = float(sys.argv[2]) if len(sys.argv)>2 else 30.0

d = np.loadtxt(fichier, skiprows=1)
t, CL, CD = d[:,1], d[:,3], d[:,2]
m = t > t0
t, CL, CD = t[m], CL[m], CD[m]

if len(t) < 100:
    print(f"ATTENTION : seulement {len(t)} points apres t={t0}. Fenetre trop courte.")
    print("Utilise un run plus long ou baisse t_debut.")
    sys.exit()

duree = t[-1]-t[0]
print(f"Fenetre : {t[0]:.1f} -> {t[-1]:.1f} CTU  ({duree:.1f} CTU, {len(t)} points)")

# interpolation sur grille reguliere (dt varie a cause du targetCFL)
N = len(t)
t_reg = np.linspace(t[0], t[-1], N)
CL_reg = np.interp(t_reg, t, CL)
dt_reg = t_reg[1]-t_reg[0]

# retirer la moyenne + fenetre de Hann (reduit les fuites spectrales)
sig = CL_reg - CL_reg.mean()
sig = sig * np.hanning(N)

# FFT
fft = np.fft.rfft(sig)
freq = np.fft.rfftfreq(N, d=dt_reg)   # en CTU^-1 = Strouhal
psd = np.abs(fft)**2

# pic dominant (on ignore la composante 0 et les tres basses freq < 0.05)
valid = freq > 0.05
fpeak = freq[valid][np.argmax(psd[valid])]

print(f"dt moyen (regulier) : {dt_reg:.2e} CTU")
print(f"Frequence de Nyquist : {1/(2*dt_reg):.1f}  (freq max mesurable)")
print()
print(f">>> Strouhal dominant : St ~ {fpeak:.3f}")
print(f">>> Periode : {1/fpeak:.2f} CTU")
print()
# top 3 pics pour voir s'il y a plusieurs modes
idx = np.argsort(psd[valid])[::-1][:3]
print("Top 3 des pics spectraux :")
for i in idx:
    print(f"   St = {freq[valid][i]:.3f}   (periode {1/freq[valid][i]:.2f} CTU)")

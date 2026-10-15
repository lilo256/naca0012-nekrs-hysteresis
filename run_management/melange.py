#!/usr/bin/env python3
"""
melange.py  --  champ initial perturbe : w * etat A + (1 - w) * etat B   (test de F. Mellibovsky)
===============================================================================================
Les deux etats doivent etre au MEME angle, sur le MEME maillage et au MEME ordre N (verifier avec
`cmp` sur les .re2). Vitesse, pression et scalaires (k, tau) sont melanges element par element,
dans l'ordre global des elements (carte des elements lue dans chaque fichier), donc l'ordre
d'ecriture des deux fichiers peut differer.

Deux champs a divergence nulle sur le meme maillage donnent un melange a divergence nulle ; k et
tau restent positifs (combinaison a poids positifs). Le fichier ecrit garde l'en-tete et le temps
de l'etat A ; les coordonnees sont recopiees si l'un des deux fichiers les contient.

USAGE (sur le cluster, apres `pip install --user pymech`)
   python3 ~/bin/melange.py ETAT_A.f0XXXXX ETAT_B.f0YYYYY 0.90 SORTIE.f00000
   (A = decroche, B = attache, w = 0.90 -> 90 % decroche + 10 % attache)
"""
import sys
import numpy as np
from pymech.neksuite import readnek, writenek

if len(sys.argv) != 5:
    sys.exit(__doc__)
fa, fb, w, fout = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
if not 0.0 <= w <= 1.0:
    sys.exit("le poids w doit etre entre 0 et 1")

A = readnek(fa)
B = readnek(fb)
for nom, d in (("A", A), ("B", B)):
    if not hasattr(d, "elem"):
        sys.exit(f"lecture impossible : {fa if nom == 'A' else fb}")
print(f" A : {fa}   t = {float(A.time):.4f}   nel = {A.nel}   N+1 = {A.lr1}   var = {A.var}")
print(f" B : {fb}   t = {float(B.time):.4f}   nel = {B.nel}   N+1 = {B.lr1}   var = {B.var}")
if A.nel != B.nel or list(A.lr1) != list(B.lr1):
    sys.exit(" ARRET : nombre d'elements ou ordre polynomial differents (pas le meme maillage ?)")
if list(A.var[1:]) != list(B.var[1:]):
    sys.exit(" ARRET : les deux fichiers ne contiennent pas les memes champs")

# correspondance des elements par numero global
ib = {int(g): i for i, g in enumerate(B.elmap)}
try:
    paires = [(ea, B.elem[ib[int(g)]]) for ea, g in zip(A.elem, A.elmap)]
except KeyError as err:
    sys.exit(f" ARRET : element global {err} absent de B")

# controle du maillage si les deux fichiers ont les coordonnees
if A.var[0] and B.var[0]:
    ecart = max(float(np.abs(ea.pos - eb.pos).max()) for ea, eb in paires)
    print(f" ecart maximal des coordonnees A/B : {ecart:.2e}")
    if ecart > 1e-6:
        sys.exit(" ARRET : les coordonnees different, ce n'est pas le meme maillage")
elif B.var[0] and not A.var[0]:
    for ea, eb in paires:                  # recopier les coordonnees dans la sortie
        ea.pos = eb.pos.copy()
    A.var = (B.var[0],) + tuple(A.var[1:])

for ea, eb in paires:
    ea.vel = w * ea.vel + (1.0 - w) * eb.vel
    ea.pres = w * ea.pres + (1.0 - w) * eb.pres
    if A.var[3]:
        ea.temp = w * ea.temp + (1.0 - w) * eb.temp
    if A.var[4]:
        ea.scal = w * ea.scal + (1.0 - w) * eb.scal

writenek(fout, A)
C = readnek(fout)                           # relecture de controle
ec = max(float(np.abs(c.vel - a.vel).max()) for c, a in zip(C.elem, A.elem))
print(f" ecrit : {fout}   t = {float(C.time):.4f}   var = {C.var}   (w = {w:.2f} sur A)")
print(f" relecture : ecart maximal sur la vitesse {ec:.1e} (arrondi simple precision attendu)")

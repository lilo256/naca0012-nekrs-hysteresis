#!/bin/bash
# =============================================================================================
# prep_pert.sh  --  test de perturbation de F. Mellibovsky (30/09) : l'etat decroche recolle-t-il ?
# ---------------------------------------------------------------------------------------------
# Champ initial = W x etat decroche + (1 - W) x etat attache, au MEME angle et sur le MEME
# maillage (verifie par cmp des .re2). Le cas reprend les reglages de production du cas decroche
# (pas variable, CFL 1) et tourne DUREE CTU a partir du temps de l'etat decroche.
#
# USAGE  prep_pert.sh NOM DOSSIER_DECROCHE FICHIER_DECROCHE DOSSIER_ATTACHE FICHIER_ATTACHE W DUREE
#   ex.  prep_pert.sh pert13_90 jump_a13.00 airfoil0.f00011 att_a13.00 airfoil0.f00005 0.90 60
# SUIVI  etat ; le C_L dit tout : ~0,71 = reste decroche, ~1,28 = recolle (13 deg).
# =============================================================================================
set -e
[ $# -eq 7 ] || { sed -n '2,14p' "$0"; exit 1; }
NOM=$1; DS=$2; FS=$3; DA=$4; FA=$5; W=$6; DUREE=$7
RUNS=~/cdf_runs; D=$RUNS/$NOM
[ -e "$D" ] && { echo "$D existe deja : rien n'est modifie"; exit 1; }
for f in "$RUNS/$DS/$FS" "$RUNS/$DA/$FA" "$RUNS/$DS/airfoil.re2" "$RUNS/$DA/airfoil.re2"; do
    [ -f "$f" ] || { echo "absent : $f"; exit 1; }
done
cmp -s "$RUNS/$DS/airfoil.re2" "$RUNS/$DA/airfoil.re2" || { echo "ARRET : les deux cas n'ont pas le meme maillage"; exit 1; }

mkdir "$D"
for f in "$RUNS/$DS"/*.par "$RUNS/$DS"/*.udf "$RUNS/$DS"/*.re2 "$RUNS/$DS"/*.oudf "$RUNS/$DS"/*.okl \
         "$RUNS/$DS"/*.usr "$RUNS/$DS"/*.hpp "$RUNS/$DS"/job.sh; do
    [ -e "$f" ] && cp "$f" "$D"/
done
cd "$D"
python3 ~/bin/melange.py "$RUNS/$DS/$FS" "$RUNS/$DA/$FA" "$W" restart0.f00000 || { echo "melange en echec"; exit 1; }

T0=$(head -c 132 restart0.f00000 | awk '{print $8}')
TF=$(python3 -c "print(round(float('$T0') + $DUREE, 3))")
P=airfoil.par
sed -i -E '/^[[:space:]]*(startFrom|endTime)[[:space:]]*=/Id' $P
sed -i -E "s|^\[GENERAL\][[:space:]]*$|[GENERAL]\nstartFrom = \"restart0.f00000\"\nendTime = $TF|I" $P

echo "================ $NOM : W = $W decroche ($DS/$FS) + $(python3 -c "print(round(1-$W,3))") attache ($DA/$FA)"
echo " depart t = $T0, fin t = $TF"
grep -n -i -E '^[[:space:]]*\[|^[[:space:]]*(startFrom|endTime|dt|timeStepper|regularization|checkpointControl|checkpointInterval)[[:space:]]*=' $P
grep -n "ALPHA_DEG" airfoil.udf | head -2 || true
read -p "Soumettre $NOM ? (o/N) " r
if [ "$r" = "o" ]; then qsub job.sh; else echo "non soumis (qsub job.sh dans $D quand tu veux)"; fi

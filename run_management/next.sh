#!/bin/bash
# =============================================================================
#  next.sh  --  un pas de la campagne d'hysteresis
# =============================================================================
#  Prepare et soumet le cas suivant a partir du cas precedent converge.
#  Verifie TOUT avant de lancer ; refuse de soumettre si quelque chose cloche.
#
#  USAGE
#    ./next.sh <cas_precedent> <nouvel_angle> [duree_CTU]
#
#  EXEMPLES
#    ./next.sh prod_a15   15.25        # montee, duree par defaut
#    ./next.sh up_a15.25  15.50
#    ./next.sh up_a20.00  19.75 25     # DESCENTE (angle qui diminue)
#
#  La branche (montee/descente) est deduite du sens de variation de l'angle.
#  Les dossiers sont nommes up_aXX.XX ou down_aXX.XX.
# =============================================================================
set -e

BASE=~/cdf_runs
MESH=~/meshes_aoa
NELGT_ATTENDU=14124

PREV="$1"
A="$2"
DUREE="${3:-}"

if [ -z "$PREV" ] || [ -z "$A" ]; then
  echo "Usage : ./next.sh <cas_precedent> <nouvel_angle> [duree_CTU]"
  echo "Ex.   : ./next.sh prod_a15 15.25"
  exit 1
fi

PREVDIR="$BASE/$PREV"
[ -d "$PREVDIR" ] || { echo "ERREUR : $PREVDIR introuvable"; exit 1; }

# --- angle du cas precedent (lu dans son .udf) -------------------------------
APREV=$(grep -oP 'ALPHA_DEG\s*=\s*\K[0-9.]+' "$PREVDIR/airfoil.udf" | head -1)
[ -n "$APREV" ] || { echo "ERREUR : ALPHA_DEG illisible dans $PREV"; exit 1; }

# --- sens : montee ou descente ----------------------------------------------
SENS=$(python3 -c "print('up' if $A > $APREV else 'down')")
NEW="${SENS}_a${A}"
NEWDIR="$BASE/$NEW"

echo "=============================================================="
echo " PAS DE CHAINAGE"
echo "   precedent : $PREV        (alpha = $APREV)"
echo "   nouveau   : $NEW   (alpha = $A)"
echo "   branche   : $SENS"
echo "=============================================================="

# --- 0. le cas precedent est-il PHYSIQUE ? ----------------------------------
echo
echo ">>> 0. Plausibilite physique du cas precedent"
python3 -c "
import numpy as np, sys
d=np.loadtxt('$PREVDIR/drag_results.dat',skiprows=1)
ok=(np.abs(d[:,3])<2)&(d[:,2]>0)&(d[:,2]<0.2)
if (~ok).sum(): print('    lignes aberrantes filtrees :', int((~ok).sum()))
d=d[ok]; m=d[:,1]>=d[0,1]+0.5*(d[-1,1]-d[0,1])
CL,CD=d[m,3].mean(),d[m,2].mean()
print(f'    CL = {CL:.6f}   CD = {CD:.6f}')
sys.exit(0 if (abs(CL)<2 and 0<CD<0.2) else 1)
" || { echo "    !!! valeurs non physiques — le calcul a diverge."; exit 1; }

# --- 1. le cas precedent est-il converge ? ----------------------------------
echo
echo ">>> 1. Convergence du cas precedent"
if [ -f ~/bin/conv.py ]; then
  python3 ~/bin/conv.py "$PREVDIR" | sed -n '/dispersion CL/,/>>>/p'
  if ! python3 ~/bin/conv.py "$PREVDIR" | grep -q "CONVERGE"; then
    echo
    echo "    !!! $PREV n'est PAS converge."
    read -p "    Continuer quand meme ? (o/N) " r
    [ "$r" = "o" ] || exit 1
  fi
else
  echo "    (conv.py absent — controle saute)"
fi

# --- 2. maillage disponible ? ------------------------------------------------
echo
echo ">>> 2. Maillage"
MDIR="$MESH/aoa_$A"
[ -f "$MDIR/airfoil.re2" ] || { echo "ERREUR : $MDIR/airfoil.re2 introuvable"; exit 1; }
NELGT=$(head -c 40 "$MDIR/airfoil.re2" | grep -oE '[0-9]+' | sed -n 2p)
echo "    $MDIR  ->  $NELGT elements"
if [ "$NELGT" != "$NELGT_ATTENDU" ]; then
  echo "    !!! ATTENDU $NELGT_ATTENDU — le chainage va echouer."
  exit 1
fi

# --- 3. dernier checkpoint ---------------------------------------------------
echo
echo ">>> 3. Checkpoint de depart"
# tri par DATE : un checkpoint herite (ex. f00099 copie d'un cas amont)
# serait choisi a tort par un tri alphabetique alors que le cas courant
# n'est alle que jusqu'a f00011.
LAST=$(ls -t "$PREVDIR"/airfoil0.f0* 2>/dev/null | head -1)
[ -n "$LAST" ] || { echo "ERREUR : aucun checkpoint dans $PREV"; exit 1; }
CKPT=$(basename "$LAST")
T=$(tail -1 "$PREVDIR/drag_results.dat" | awk '{print $2}')
echo "    $CKPT   (t = $T CTU)"

# --- duree du pas ------------------------------------------------------------
if [ -z "$DUREE" ]; then
  DUREE=$(python3 -c "
a=$A
# moyennes convergees uniquement. Les SPECTRES (St ~ 0.5 -> periode 2 CTU,
# ~50 cycles necessaires) se font a part, sur 2-3 angles rejoues longs.
print(25 if 17.4<=a<=18.9 else (20 if a>=16.7 else 12))")
  echo "    duree auto : $DUREE CTU"
fi
TEND=$(python3 -c "print(round($T + $DUREE, 1))")
echo "    endTime = $TEND   ($T + $DUREE)"

# --- 4. creation du dossier --------------------------------------------------
echo
echo ">>> 4. Creation de $NEW"
[ -d "$NEWDIR" ] && { echo "ERREUR : $NEWDIR existe deja (rm -rf pour refaire)"; exit 1; }

rsync -a --exclude='airfoil0.f0*' --exclude='.cache' --exclude='resultats_csv/*' \
      --exclude='drag_results.dat' --exclude='probes_*.csv' \
      --exclude='*.out' --exclude='*.err' --exclude='*.log' --exclude='cmake.log' \
      --exclude='airfoil.re2' --exclude='airfoil.ma2' --exclude='airfoil.nek5000' \
      --exclude='wall_distance_rank_*' \
      "$PREVDIR/" "$NEWDIR/"

cd "$NEWDIR"
mkdir -p resultats_csv
cp "$LAST" .
cp "$MDIR/airfoil.re2" "$MDIR/airfoil.ma2" .

# --- 5. ALPHA_DEG ------------------------------------------------------------
sed -i -E "s|ALPHA_DEG   = [0-9.]+|ALPHA_DEG   = ${A}|" airfoil.udf
VERIF=$(grep -oP 'ALPHA_DEG\s*=\s*\K[0-9.]+' airfoil.udf | head -1)
echo
echo ">>> 5. ALPHA_DEG = $VERIF"
[ "$VERIF" = "$A" ] || { echo "    !!! echec du sed"; exit 1; }

# --- 6. .par -----------------------------------------------------------------
echo
echo ">>> 6. Fichier .par"
# Le .par peut contenir PLUSIEURS lignes startFrom/endTime (heritees de
# reglages manuels). On les supprime toutes puis on en reinsere une seule,
# sinon un ancien startFrom reste actif plus bas dans le fichier.
sed -i -E '/^[[:space:]]*#*[[:space:]]*startFrom/Id' airfoil.par
sed -i -E '/^[[:space:]]*#*[[:space:]]*endTime/Id'   airfoil.par
sed -i -E '/^[[:space:]]*#*[[:space:]]*checkpointInterval/Id' airfoil.par
sed -i -E "/^\[GENERAL\]/Ia checkpointInterval = 5" airfoil.par
sed -i -E "/^\[GENERAL\]/Ia endTime = ${TEND}"      airfoil.par
sed -i -E "/^\[GENERAL\]/Ia startFrom = ${CKPT}"    airfoil.par
NSF=$(grep -ciE '^[[:space:]]*startFrom' airfoil.par)
[ "$NSF" = "1" ] || { echo "    !!! $NSF lignes startFrom (attendu 1)"; exit 1; }
grep -inE "startFrom|endTime|checkpointInterval" airfoil.par | sed 's/^/    /'

# --- 7. controles finaux -----------------------------------------------------
echo
echo ">>> 7. Controles"
[ -d .cache ] && { echo "    !!! .cache present"; exit 1; } || echo "    .cache absent      OK"
[ -d resultats_csv ] && echo "    resultats_csv      OK"
for f in airfoil.udf airfoil.oudf my_drag.oudf cf.okl postProcessing.hpp airfoil.re2 airfoil.ma2 "$CKPT"; do
  [ -f "$f" ] || { echo "    !!! MANQUE : $f"; exit 1; }
done
echo "    tous les fichiers  OK"

# --- 8. soumission -----------------------------------------------------------
echo
read -p ">>> Soumettre ? (o/N) " r
if [ "$r" = "o" ]; then
  qsub -q menuts.q job.sh
  echo
  echo "    Verifier dans 3 min :"
  echo "      cd $NEWDIR"
  echo "      grep 't= ' NACA0012.*.out | head -2     # doit repartir a ~$T"
  echo "      grep -i 'not found' NACA0012.*.out      # doit etre vide"
else
  echo "    Non soumis. Dossier pret : $NEWDIR"
fi

#!/bin/bash
# =============================================================================================
# prep_t15.sh  --  test de stabilisation a 15 deg (demande de F. Mellibovsky, 30/09)
# ---------------------------------------------------------------------------------------------
# Cree ~/cdf_runs/<NOM> a partir de film_a15 : MEME champ initial que film_a15 et res_a15_N4
# (film_a15/airfoil0.f00000, t = 100.0062), MEME maillage, N = 3, dt FIXE 3.5e-5, fin a t = 106.
# Un seul reglage change par test :
#   hpf40  regularization = hpfrt + nModes=1 + scalingCoeff=40   (filtre 4 fois plus fort)
#   hpf2m  regularization = hpfrt + nModes=2 + scalingCoeff=10   (filtre sur les 2 modes du haut)
#   bdf3   timeStepper = tombo3                                   (ordre 3 en temps)
#   ptol   residualTol = 1e-08 dans [PRESSURE]                    (au lieu de 1e-06)
# Sorties espacees de 0.5 CTU (pas les 0.0063 CTU du film). Affiche le .par obtenu et demande
# confirmation avant qsub (a lancer depuis la frontale cdf).
#
# USAGE   prep_t15.sh t15_hpf40 hpf40
# ANALYSE (apres le run, t = 106) :
#   python3 -u ~/bin/compare_N3_N4.py ~/cdf_runs/film_a15 ~/cdf_runs/t15_hpf40 100.3   # meme fenetre, par paires
#   cd ~/cdf_runs/t15_hpf40 && python3 -u ~/bin/spectre_sondes.py 102 2 0.5          # 102-106, a comparer a prod_a15
# =============================================================================================
set -e
NOM=$1; MOD=$2
RUNS=~/cdf_runs; SRC=$RUNS/film_a15; D=$RUNS/$NOM
if [ -z "$NOM" ] || [ -z "$MOD" ]; then
    echo "usage : prep_t15.sh NOM {hpf40|hpf2m|bdf3|ptol}"; exit 1
fi
case $MOD in hpf40|hpf2m|bdf3|ptol) ;; *) echo "reglage inconnu : $MOD"; exit 1;; esac
[ -e "$D" ] && { echo "$D existe deja : rien n'est modifie"; exit 1; }
[ -f "$SRC/airfoil0.f00000" ] || { echo "champ initial absent : $SRC/airfoil0.f00000"; exit 1; }

mkdir "$D"
for f in "$SRC"/*.par "$SRC"/*.udf "$SRC"/*.re2 "$SRC"/*.oudf "$SRC"/*.okl "$SRC"/*.usr "$SRC"/*.hpp "$SRC"/job.sh; do
    [ -e "$f" ] && cp "$f" "$D"/
done
cp "$SRC/airfoil0.f00000" "$D/restart0.f00000"
cd "$D"
P=airfoil.par
[ -f "$P" ] || { echo "airfoil.par absent de $SRC"; exit 1; }

# depart, fin, pas de temps fixe
sed -i -E '/^[[:space:]]*(startFrom|endTime)[[:space:]]*=/Id' $P
sed -i -E 's|^\[GENERAL\][[:space:]]*$|[GENERAL]\nstartFrom = "restart0.f00000"\nendTime = 106|I' $P
sed -i -E 's|^([[:space:]]*dt[[:space:]]*=).*|\1 3.5e-05|I' $P

# sorties tous les 0.5 CTU
if grep -q -i -E '^[[:space:]]*(checkpointControl|writeControl)[[:space:]]*=[[:space:]]*steps' $P; then
    IV=14286          # 0.5 CTU / 3.5e-5
else
    IV=0.5
fi
sed -i -E "s#^([[:space:]]*(checkpointInterval|writeInterval)[[:space:]]*=).*#\1 $IV#I" $P

# le reglage teste
case $MOD in
  hpf40) sed -i -E 's|^([[:space:]]*regularization[[:space:]]*=).*|\1 hpfrt + nModes=1 + scalingCoeff=40|I' $P ;;
  hpf2m) sed -i -E 's|^([[:space:]]*regularization[[:space:]]*=).*|\1 hpfrt + nModes=2 + scalingCoeff=10|I' $P ;;
  bdf3)  sed -i -E 's|^([[:space:]]*timeStepper[[:space:]]*=).*|\1 tombo3|I' $P ;;
  ptol)  awk '/^[[:space:]]*\[/ {s = toupper($0)}
              { if (s ~ /\[PRESSURE\]/ && $0 ~ /^[[:space:]]*residualTol[[:space:]]*=/) print "residualTol = 1e-08"; else print }' \
             $P > $P.tmp && mv $P.tmp $P ;;
esac

echo "================ $NOM ($MOD) : lignes actives du .par ================"
grep -n -i -E '^[[:space:]]*\[|^[[:space:]]*(startFrom|endTime|dt|timeStepper|regularization|polynomialOrder|advectionSubCyclingSteps|checkpointControl|checkpointInterval|writeControl|writeInterval|residualTol|preconditioner)[[:space:]]*=' $P
echo "---------------------------------------------------------------------"
grep -n "ALPHA_DEG" airfoil.udf | head -2 || true
ls -la "$D"
read -p "Soumettre $NOM ? (o/N) " r
if [ "$r" = "o" ]; then qsub job.sh; else echo "non soumis (qsub job.sh dans $D quand tu veux)"; fi

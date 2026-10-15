#!/bin/bash
# =============================================================================
#  archiver.sh  --  sauvegarde un cas AVANT un restart
# =============================================================================
#  POURQUOI : au restart, NekRS remet le compteur de champs a f00000 et
#  ECRASE les .f0* existants. Sans archivage prealable, les champs de la
#  phase precedente sont perdus definitivement.
#
#  USAGE
#    cd ~/cdf_runs/<cas>
#    ~/bin/archiver.sh <etiquette>
#
#  EXEMPLE
#    cd ~/cdf_runs/cold_a17.00
#    ~/bin/archiver.sh transition_decrochage
#
#  RESULTAT : ~/archives/<cas>__<etiquette>/
#    - tous les airfoil0.f0*        (champs)
#    - drag_results.dat             (efforts)
#    - probes_uvp.csv , probes_positions.csv
#    - resultats_csv/               (Cf, Cp parietaux)
#    - airfoil.par , airfoil.udf , Airfoil_mesh.geo  (config, tracabilite)
#    - INFO.txt                     (date, temps atteint, CL/CD, nb fichiers)
#
#  Les .out/.err (souvent > 1 Go) ne sont PAS archives.
# =============================================================================
set -e

ARCHIVES=~/archives
CAS=$(basename "$PWD")
ETIQ="${1:-$(date +%Y%m%d_%H%M)}"
DEST="$ARCHIVES/${CAS}__${ETIQ}"

[ -f drag_results.dat ] || { echo "ERREUR : pas de drag_results.dat ici."; \
                             echo "         Lance depuis un dossier de cas."; exit 1; }
[ -d "$DEST" ] && { echo "ERREUR : $DEST existe deja. Change d'etiquette."; exit 1; }

echo "=============================================================="
echo " ARCHIVAGE"
echo "   cas       : $CAS"
echo "   etiquette : $ETIQ"
echo "   vers      : $DEST"
echo "=============================================================="

# --- etat du cas -------------------------------------------------------------
NF=$(ls airfoil0.f0* 2>/dev/null | wc -l)
TFIN=$(tail -1 drag_results.dat | awk '{print $2}')
TAILLE=$(du -shc airfoil0.f0* drag_results.dat resultats_csv probes_*.csv 2>/dev/null | tail -1 | cut -f1)
echo " champs .f0*     : $NF"
echo " temps atteint   : $TFIN CTU"
echo " volume a copier : $TAILLE"
echo

mkdir -p "$DEST"

# --- champs ------------------------------------------------------------------
if [ "$NF" -gt 0 ]; then
  echo " -> champs ..."
  cp airfoil0.f0* "$DEST"/
  [ -f airfoil.nek5000 ] && cp airfoil.nek5000 "$DEST"/
fi

# --- resultats ---------------------------------------------------------------
echo " -> resultats ..."
cp drag_results.dat "$DEST"/
for f in probes_uvp.csv probes_positions.csv wall_distance_rank_0000.csv; do
  [ -f "$f" ] && cp "$f" "$DEST"/
done
[ -d resultats_csv ] && cp -r resultats_csv "$DEST"/

# --- configuration (tracabilite) ---------------------------------------------
echo " -> configuration ..."
for f in airfoil.par airfoil.udf airfoil.oudf my_drag.oudf cf.okl \
         postProcessing.hpp Airfoil_mesh.geo job.sh; do
  [ -f "$f" ] && cp "$f" "$DEST"/
done

# --- fiche d'information -----------------------------------------------------
{
  echo "ARCHIVE : $CAS / $ETIQ"
  echo "date            : $(date)"
  echo "source          : $PWD"
  echo "temps atteint   : $TFIN CTU"
  echo "champs archives : $NF"
  ALPHA=$(grep -oP 'ALPHA_DEG\s*=\s*\K[0-9.]+' airfoil.udf 2>/dev/null | head -1)
  echo "ALPHA_DEG       : ${ALPHA:-inconnu}"
  NELGT=$(head -c 40 airfoil.re2 2>/dev/null | grep -oE '[0-9]+' | sed -n 2p)
  echo "elements        : ${NELGT:-inconnu}"
  echo
  python3 -c "
import numpy as np
d=np.loadtxt('drag_results.dat',skiprows=1)
ok=(np.abs(d[:,3])<2)&(d[:,2]>0)&(d[:,2]<1.0); d=d[ok]
t=d[:,1]; m=t>=t[0]+0.5*(t[-1]-t[0])
print(f'segment    : {t[0]:.2f} -> {t[-1]:.2f} CTU')
print(f'CL  moyen  : {d[m,3].mean():.6f}   (RMS {d[m,3].std():.3e})')
print(f'CD  moyen  : {d[m,2].mean():.6f}   (RMS {d[m,2].std():.3e})')
if d.shape[1]>=11:
    print(f'CDp moyen  : {d[m,5].mean():.6f}')
    print(f'CDv moyen  : {d[m,8].mean():.6f}')
" 2>/dev/null || echo "(statistiques indisponibles)"
} > "$DEST/INFO.txt"

echo
echo "=============================================================="
cat "$DEST/INFO.txt"
echo "=============================================================="
echo " OK -> $DEST"
echo " Total archives : $(du -sh $ARCHIVES 2>/dev/null | cut -f1)"
echo
echo " Tu peux maintenant relancer : les .f0* seront ecrases sans risque."

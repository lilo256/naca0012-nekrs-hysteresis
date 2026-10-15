#!/bin/bash
# =============================================================================
#  make_meshes.sh  --  generation en lot des maillages d'incidence
# =============================================================================
#  A LANCER SUR LE PC, dans le dossier qui contient Airfoil_mesh.geo
#  (~/Bureau/stage_docs/code/mesh_modif/meshes_aoa).
#
#  USAGE
#    ./make_meshes.sh 14.75 14.50 14.25 14.00
#
#  Pour chaque angle : aoa_<A>/airfoil.re2 + airfoil.ma2, verifies, puis
#  proposition de depot sur le cluster (un seul rsync, arborescence gardee).
#
#  CE QUE LES TESTS ONT MONTRE (et que le script controle)
#   - Le .geo maille lui-meme, mais seule la commande
#       gmsh Airfoil_mesh.geo -3 -order 2 -format msh2
#     produit des HEX20 (aretes courbes). Sans -order 2 : HEX8, avec le
#     MEME nombre d'elements (14124) -> le controle du nombre ne suffit pas.
#   - gmsh2nek refuse les HEX8 mais sort quand meme avec le code 0 :
#     on verifie donc la presence du .re2, jamais le code de retour.
#   - La taille du .re2 varie de quelques centaines d'octets selon l'angle
#     (arete courbe ecrite seulement si elle l'est vraiment) ; le .ma2 est
#     constant. -> comparaison a une reference : .ma2 exact, .re2 a 0,1 %.
#   - L'angle est en radians dans le .geo : aoa=<A>*Pi/180;
#   - Message "Unknown number option 'Mesh.SecondOrderExperimental'" :
#     option retiree des gmsh recents, sans consequence.
# =============================================================================
set -u

NELGT=14124
GEO=Airfoil_mesh.geo
CLUSTER=${CLUSTER:-USER@cdf}      # a adapter : export CLUSTER=login@cdf
TOL_RE2=0.1                    # % d'ecart toleré sur la taille du .re2

# valeurs du .geo de production (etude de domaine close)
ATT_A="20*c"; ATT_B="30*c"; ATT_XFIRSTU="1.2e-5"; ATT_NORD="2"

# gmsh : le portable 4.15 (celui des 21 maillages de production), sinon PATH
GMSH=${GMSH:-$HOME/gmsh-4.15.0-Linux64/bin/gmsh}
[ -x "$GMSH" ] || GMSH=$(command -v gmsh || true)

BASE=$(pwd)
T0=$SECONDS

# --- 0. controles prealables -------------------------------------------------
[ $# -ge 1 ] || { echo "Usage : ./make_meshes.sh 14.75 14.50 ..."; exit 1; }
[ -n "$GMSH" ] && [ -x "$GMSH" ] || { echo "ERREUR : gmsh introuvable (GMSH=...)"; exit 1; }
for t in gmsh2nek genmap; do
  command -v $t >/dev/null || { echo "ERREUR : $t introuvable dans le PATH"; exit 1; }
done
[ -f "$GEO" ] || { echo "ERREUR : $GEO introuvable dans $BASE"; exit 1; }

param() {   # valeur d'une affectation du .geo, espaces retires
  grep -oE "^[[:space:]]*$1[[:space:]]*=[^;]*" "$GEO" | head -1 | sed 's/.*=//; s/[[:space:]]//g'
}
echo "=============================================================="
echo " Parametres de $GEO"
ok_geo=1
for p in "a:$ATT_A" "b:$ATT_B" "xfirstu:$ATT_XFIRSTU" "nord:$ATT_NORD"; do
  nom=${p%%:*}; att=${p#*:}; val=$(param "$nom")
  if [ "$val" = "$att" ]; then etat="OK"; else etat="!!! attendu $att"; ok_geo=0; fi
  printf "   %-8s = %-10s %s\n" "$nom" "$val" "$etat"
done
echo "   gmsh     : $GMSH"
[ $ok_geo = 1 ] || { echo " Ce n'est pas le .geo de production — arret."; exit 1; }

# reference de taille : un maillage de production deja present
REF=""
for d in aoa_15.00 aoa_*; do
  [ -f "$d/airfoil.re2" ] && [ -f "$d/airfoil.ma2" ] && { REF=$d; break; }
done
[ -n "$REF" ] && echo "   reference de taille : $REF" \
              || echo "   (aucun maillage de reference : comparaison de taille sautee)"

# --- 1. generation -----------------------------------------------------------
OK=(); DEJA=(); ECHECS=()
echec() { echo "    !!! $1"; ECHECS+=("$A"); cd "$BASE"; }

for arg in "$@"; do
  arg=${arg/,/.}                                   # 14,75 -> 14.75
  A=$(LC_ALL=C printf "%.2f" "$arg" 2>/dev/null) || { echo "angle invalide : $arg"; continue; }
  D="aoa_${A}"
  echo "=============================================================="
  echo " $A deg  ->  $D"

  if [ -f "$D/airfoil.re2" ] && [ -f "$D/airfoil.ma2" ]; then
    echo "    existe deja — ignore (supprimer $D pour le refaire)"
    DEJA+=("$A"); continue
  fi
  rm -rf "$D"; mkdir -p "$D"; cp "$GEO" "$D/"; cd "$BASE/$D"

  # incidence (radians dans le .geo)
  sed -i -E "s|^([[:space:]]*aoa[[:space:]]*=)[^;]*;|\1${A}*Pi/180;|" "$GEO"
  n=$(grep -cE "^[[:space:]]*aoa[[:space:]]*=${A}\*Pi/180;" "$GEO")
  [ "$n" = "1" ] || { echec "substitution de aoa ratee ($n ligne(s))"; continue; }

  # gmsh : -order 2 indispensable pour des HEX20
  "$GMSH" "$GEO" -3 -order 2 -format msh2 -o airfoil.msh > gmsh.log 2>&1
  [ -s airfoil.msh ] || { echec "gmsh n'a rien produit — voir $D/gmsh.log"; continue; }
  read -r h20 h8 <<< "$(awk '/^\$Elements/{f=1;getline;next} /^\$EndElements/{f=0}
                          f&&$2==17{a++} f&&$2==5{b++} END{print a+0, b+0}' airfoil.msh)"
  echo "    gmsh     : $h20 HEX20, $h8 HEX8"
  [ "$h20" = "$NELGT" ] && [ "$h8" = "0" ] || { echec "attendu $NELGT HEX20 et 0 HEX8"; continue; }

  # gmsh2nek (code 0 meme en erreur -> on teste le fichier)
  printf '3\nairfoil\n0\n0\nairfoil\n' | gmsh2nek > gmsh2nek.log 2>&1
  [ -s airfoil.re2 ] || { echec "gmsh2nek n'a pas produit airfoil.re2 — voir $D/gmsh2nek.log"; continue; }
  N=$(head -c 40 airfoil.re2 | grep -oE '[0-9]+' | sed -n 2p)
  echo "    gmsh2nek : $N elements"
  [ "$N" = "$NELGT" ] || { echec "attendu $NELGT elements dans le .re2"; continue; }

  # genmap
  printf 'airfoil\n0.2\n' | genmap > genmap.log 2>&1
  [ -s airfoil.ma2 ] || { echec "genmap n'a pas produit airfoil.ma2 — voir $D/genmap.log"; continue; }

  # coherence avec la reference
  s2=$(stat -c %s airfoil.re2); m2=$(stat -c %s airfoil.ma2)
  if [ -n "$REF" ]; then
    r2=$(stat -c %s "$BASE/$REF/airfoil.re2"); rm2=$(stat -c %s "$BASE/$REF/airfoil.ma2")
    ecart=$(awk -v a="$s2" -v b="$r2" 'BEGIN{d=(a-b)/b*100; printf "%.4f", (d<0?-d:d)}')
    echo "    tailles  : re2 $s2 (ref $r2, ecart ${ecart} %)   ma2 $m2 (ref $rm2)"
    if [ "$m2" != "$rm2" ] || awk -v e="$ecart" -v t="$TOL_RE2" 'BEGIN{exit !(e>t)}'; then
      echec "tailles incoherentes avec $REF — maillage different de la production"; continue
    fi
  else
    echo "    tailles  : re2 $s2   ma2 $m2"
  fi

  rm -f airfoil.msh* gmsh.log gmsh2nek.log genmap.log    # on garde le .geo : trace de l'angle
  echo "    OK"
  OK+=("$A"); cd "$BASE"
done

# --- 2. bilan ----------------------------------------------------------------
echo "=============================================================="
echo " BILAN  ($((SECONDS - T0)) s)"
for A in "${OK[@]}";     do echo "   aoa_$A   OK   ($NELGT HEX20, verifie)"; done
for A in "${DEJA[@]}";   do echo "   aoa_$A   deja present (non regenere)"; done
for A in "${ECHECS[@]}"; do echo "   aoa_$A   ECHEC — journaux dans aoa_$A/"; done
[ $((${#OK[@]} + ${#DEJA[@]})) -gt 0 ] || exit 1

# --- 3. depot sur le cluster (un rsync, arborescence preservee par -R) -------
#  les "deja presents" sont inclus : rsync ne renvoie rien s'ils sont a jour
FILES=(); DIRS=()
for A in "${OK[@]}" "${DEJA[@]}"; do
  FILES+=("aoa_$A/airfoil.re2" "aoa_$A/airfoil.ma2"); DIRS+=("aoa_$A")
done
echo "=============================================================="
echo " Depot :"
echo "   rsync -avzR ${FILES[*]} $CLUSTER:~/meshes_aoa/"
read -r -p " Deposer maintenant sur le cluster ? (o/N) " rep
[ "$rep" = "o" ] || { echo " Non depose."; exit 0; }

rsync -avzR "${FILES[@]}" "$CLUSTER:~/meshes_aoa/" || { echo "!!! rsync a echoue"; exit 1; }
echo " Verification sur le cluster :"
ssh "$CLUSTER" "bash -s" <<EOF
for d in ${DIRS[*]}; do
  f=~/meshes_aoa/\$d/airfoil.re2
  if [ -f "\$f" ]; then
    printf "   %-12s %s elements\n" "\$d" "\$(head -c 40 "\$f" | grep -oE '[0-9]+' | sed -n 2p)"
  else
    printf "   %-12s ABSENT\n" "\$d"
  fi
done
EOF

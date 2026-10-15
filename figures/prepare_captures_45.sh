#!/bin/bash
# =============================================================================
# prepare_captures_45.sh -- a lancer sur cdf (apres ssh, tape d'abord « hostname » seul) :
#     cd ~/cdf_runs && bash prepare_captures_45.sh
# Pour chaque capture de la liste en bas : cherche dans le dossier du cas le fichier de champ dont le
# temps (en-tete) est le plus proche de l'instant voulu, le copie dans ~/captures_45/<nom>/ avec le
# fichier qui porte le maillage (code de champ commencant par X), ecrit vis.nek5000, puis fait
# l'archive ~/captures_45.tar a rapatrier par scp. Ne modifie rien dans ~/cdf_runs.
# =============================================================================
set -u
RUNS=~/cdf_runs
OUT=~/captures_45
mkdir -p "$OUT"
printf "%-17s %-15s %8s %10s  %-18s %s\n" "capture" "cas" "voulu" "trouve" "fichier" "remarque"
while read -r nom cas t; do
  [ -z "${nom:-}" ] && continue
  case "$nom" in \#*) continue ;; esac
  d="$RUNS/$cas"
  if [ ! -d "$d" ]; then printf "%-17s %-15s %8s  ABSENT : dossier introuvable\n" "$nom" "$cas" "$t"; continue; fi
  best=""; bestdt=1e9; besttf=""; mesh=""
  for f in "$d"/airfoil0.f0*; do
    [ -e "$f" ] || continue
    read -r tf code <<< "$(head -c 132 "$f" | awk '{print $8, $12}')"
    if [ -z "$mesh" ] && [[ "$code" == X* ]]; then mesh="$f"; fi
    dt=$(awk -v a="$tf" -v b="$t" 'BEGIN{d=a-b; if(d<0)d=-d; printf "%.6f", d}')
    if awk -v a="$dt" -v b="$bestdt" 'BEGIN{exit !(a<b)}'; then best="$f"; bestdt="$dt"; besttf="$tf"; fi
  done
  if [ -z "$best" ]; then printf "%-17s %-15s %8s  AUCUN fichier airfoil0.f0*\n" "$nom" "$cas" "$t"; continue; fi
  rem=""
  if awk -v a="$bestdt" 'BEGIN{exit !(a>0.5)}'; then rem="ATTENTION : instant voulu absent, pris le plus proche"; fi
  if [ -z "$mesh" ]; then rem="$rem ; ATTENTION : pas de fichier avec maillage (X)"; fi
  mkdir -p "$OUT/$nom"
  if [ "$best" = "$mesh" ] || [ -z "$mesh" ]; then
    cp "$best" "$OUT/$nom/airfoil0.f00001"; n=1
  else
    cp "$mesh" "$OUT/$nom/airfoil0.f00001"; cp "$best" "$OUT/$nom/airfoil0.f00002"; n=2
  fi
  printf "filetemplate: airfoil%%01d.f%%05d\nfirsttimestep: 1\nnumtimesteps: %d\n" $n > "$OUT/$nom/vis.nek5000"
  printf "%-17s %-15s %8s %10.3f  %-18s %s\n" "$nom" "$cas" "$t" "$besttf" "$(basename "$best")" "$rem"
done <<'LISTE'
# --- figure 1 : la descente seule (fig_down_fields) ---
d20              cold_a20.00     65
d18p5            down_a18.50     250
d17p5            down_a17.50     405
d17              down_a17.00     465
d16              down_a16.00     275
d15p5            down_a15.50     345
d14              down_a14.00     570
d10              jump_a10.00     659.9
# facultatif (12 deg) :
d12              jump_a12.00     590
# --- figure 2 : montee contre descente (fig_updown_fields), en plus de d16, d17, d18p5, d10 ---
u16              up_a16.00       148
u17              up_a17.00       340
u18p5            up_a18.50       540
u10              dom_a20_b30_v2  80
# --- figure 3 : les etats a 13 deg (fig_hyst_fields, section 4.5.4) ---
a13_attached     att_a13.00      130
a13_stalled      jump_a13.00     560
a13_filter4      stall13_hpf40b  625
a13_blend70_565  pert13_70       565
a13_blend70_570  pert13_70       570
# quand stall13_N4 sera fini (dernier instantane) : enlever le # et mettre l'instant
# a13_N4         stall13_N4      600
LISTE
tar cf ~/captures_45.tar -C ~ captures_45 && echo && echo "archive : ~/captures_45.tar ($(du -h ~/captures_45.tar | cut -f1))"

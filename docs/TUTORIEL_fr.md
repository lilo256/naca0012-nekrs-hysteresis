# Tutoriel — Simulation du NACA 0012 sous NekRS (k–τ RANS)

> **Note (octobre 2026).** Version 1 du tutoriel, écrite fin juillet 2026, à la fin de l'étude de
> maillage. Deux choses ont changé ensuite :
> - la **production** a été faite à **N = 3** (maillage de 14 124 éléments HEX20, domaine 20c × 30c),
>   pas à N = 2 comme le recommande le §3 ci-dessous : voir le rapport de stage, chapitre 3 ;
> - la campagne a tourné sur la grappe **`cdf`** (GPU L4), avec les scripts de ce dépôt.
>
> Le reste (pièges du maillage, des UDF, des restarts, du post-traitement) est toujours valable.
> Le `README.md` à la racine donne l'enchaînement à jour.

*De la géométrie à l'analyse statistique et fréquentielle des efforts aérodynamiques.*
*Basé sur la référence de M. Martin Ferrer (2025, Mesh 7, N = 2) et l'étude de reprise.*

> **Comment lire ce document.** Chaque section donne la marche à suivre, puis un encadré **⚠ Piège**
> listant les erreurs qui coûtent des jours. Les pièges viennent tous d'erreurs réellement commises.

---

## 0. Objectif et principe

On simule l'écoulement autour d'un profil NACA 0012 à **Re = 10⁶**, incidence variable. Fermeture
**RANS k–τ** (variante du k–ω robuste près de la paroi).

Principe NekRS à retenir dès le départ :

- **NekRS est un solveur 3D**, éléments spectraux, **(N+1)³ points GLL par élément**, ordre `N`
  **isotrope**.
- **Pas de mode 2D natif**. Un cas « 2D » se fait par **une seule couche d'éléments en z** avec
  conditions périodiques.
- Conséquence : à ordre élevé, on calcule `N+1` plans en z **redondants** pour un écoulement 2D.

---

## 1. Environnement

### 1.1 Cluster
Grappes GPU de l'UPC avec stockage partagé (`cdf` : GPU NVIDIA L4 ; `nofx` : Tesla T4).
Un job = un nœud. On répartit les cas sur plusieurs nœuds en parallèle. Voir `MEMO_cluster_fr.md`.

### 1.2 Poste local (Linux)
Outils : NekRS v26.0.1, Gmsh **4.15** (binaire portable ; la 4.12 des dépôts apt pose des problèmes de
tolérance), outils Nek5000 (`gmsh2nek`, `genmap`), ParaView.

```bash
git clone https://github.com/Nek5000/Nek5000.git
cd Nek5000/tools && ./maketools gmsh2nek genmap
echo 'export PATH=$PATH:$HOME/Nek5000/bin' >> ~/.bashrc
ssh-keygen -t ed25519      # puis envoyer la clé publique à l'administrateur du cluster
```

> **⚠ Piège — terminal VS Code.** Le terminal du **Flatpak VS Code est sandboxé** : pas de `sudo`,
> `dpkg`, `apt`, `rsync`. Toute administration système passe par le vrai terminal.

---

## 2. Génération du maillage (Gmsh → gmsh2nek → genmap)

Chaîne : `.geo` → `.msh` (hexa, ordre 2) → `gmsh2nek` → `.re2` → `genmap` → `.ma2`.
En lot, avec contrôles : `mesh/make_meshes.sh`.

### 2.1 Le `.geo`
- Extrusion **mono-élément** en z, périodicité z.
- **Maillage courbe** : `nord = 2`, `SecondOrderLinear = 0`, et surtout la commande
  `gmsh Airfoil_mesh.geo -3 -order 2 -format msh2` (sans `-order 2` : HEX8, avec le **même** nombre
  d'éléments — le compte ne suffit pas comme contrôle).
- L'incidence est **gravée dans le maillage** (`aoa = <A>*Pi/180;`, en radians) : un maillage par angle.

> **⚠ Piège — `Geometry.Tolerance`.** `1e-14` provoque *« Could not find extruded vertex »* sur
> certaines machines. **`Geometry.Tolerance = 1e-8`** résout complètement.

### 2.2 Conversion
```bash
gmsh2nek        # dimension 3, nom du .msh
genmap          # tolérance 0.01
```

> **⚠ Piège — gmsh2nek sort avec le code 0 même en cas d'échec** (HEX8 refusés) : vérifier la
> présence du `.re2`, jamais le code de retour.

> **⚠ Piège — courbure au bord d'attaque.** Contrôle physique : le pic de succion `Cp,min` doit être
> piqué et stable (≈ −5,2 à x/c ≈ 0,0025 à 10°). Contrôle visuel dans ParaView :
> `Nonlinear Subdivision Level` ≥ 2.

---

## 3. Configuration NekRS (`.par`)

Voir `case_template/airfoil.par` (réglages de production : N = 3, pas variable, CFL cible 1,
régularisation `hpfrt` nModes = 1, scalingCoeff = 10).

> **⚠ Choix de l'ordre polynomial.** Le cas étant quasi-2D, les `N+1` plans GLL en z sont
> redondants. À coût égal, l'ordre bas donne plus de résolution utile dans le plan. L'étude de
> maillage a pourtant retenu N = 3 (mêmes efforts que N = 2 à 0,3 % avec moitié moins d'éléments).
> Les tests de fin de stage (N = 4 sur la branche décrochée, rapport §4.4–4.5) montrent que la
> couche cisaillée décollée reste sous-résolue à N = 3 : à réévaluer en priorité.

> **⚠ Piège — dt et FFT.** En `variableDt`, le pas varie et le spectre exige une ré-interpolation sur
> grille régulière (les scripts de `postprocessing/` le font). Pour un spectre propre, relancer à
> `dt` constant. Après un restart, NekRS fait souvent un pas initial minuscule (`dt ~ 1e-7`) : normal
> une fois.

> **⚠ Piège — CFL et sous-itérations.** Augmenter le CFL réduit le nombre de pas mais augmente le
> nombre d'itérations pression par pas : au-delà d'un certain point, le run devient *plus lent*.

---

## 4. UDF (`.udf` / `.oudf`)

### 4.1 Efforts aérodynamiques
Sortie brute dans `drag_results.dat` :
```
tstep  time  dragX dragY dragZ  dragPX dragPY dragPZ  dragVX dragVY dragVZ
```

> **⚠ Piège — rotation d'incidence (CD ×10 !).** La géométrie est **déjà inclinée dans le maillage** ;
> l'écoulement libre est selon **+X**. Donc `dragX = traînée`, `dragY = portance`,
> `AOA_ROT_DEG = 0`. **Ne PAS re-projeter** les efforts d'un angle d'incidence.

### 4.2 Dumps de paroi `cf_cp`
`x,y,z,cf_x,cf_y,cf_z,Cp` par point de paroi dans `resultats_csv/`.

> **⚠ Piège — écrasement au restart.** `tstep` repart de 0 au restart : **nommer les dumps par le temps
> physique**, pas par `tstep`.

> **⚠ Convention Cf.** **`cf_x` est déjà le Cf scalaire signé** (+ vers le bord de fuite) ;
> `cf_y = cf_z = 0`. L'utiliser directement, ne pas re-projeter.

### 4.3 Sondes
Évaluation ponctuelle de (u, v, p) → séries temporelles (`probes_uvp.csv`, positions dans
`probes_positions.csv`). **Ajout = recompilation de l'UDF** : à prévoir avant le run.
Placement : couche cisaillée de l'extrados, proche sillage **décalé en y** de part et d'autre de
l'axe (sur l'axe, on voit la fréquence double), sillage lointain, et une sonde **amont témoin** dont
le spectre doit être plat.

> **Double usage des sondes.** Si sonde et effort partagent fréquence et phase, l'oscillation des
> efforts est physique (ou du moins dans l'écoulement) ; si la sonde est lisse pendant que l'effort
> oscille, c'est un artefact de calcul ou de réduction.

---

## 5. Lancer et redémarrer sur le cluster

```bash
qsub job.sh            # depuis la frontale, un nœud par cas
```

> **⚠ Piège — compteur de champs remis à zéro.** Au restart, NekRS **repart de `f00000`** et écrase
> les `.f0*` du run précédent. **Archiver avant** (`run_management/archiver.sh`). Les chaînages de
> ce dépôt créent un nouveau dossier par pas (`next.sh`), ce qui évite le problème.

> **⚠ Piège — maillage seulement dans `f00000`.** Dans chaque run, seul `airfoil0.f00000` contient
> les coordonnées (code `X` dans l'en-tête : `head -c 132 airfoil0.f00000`). Pour ParaView ou pour
> une relance avec interpolation (`+int`), partir d'un fichier qui porte le maillage.

---

## 6. Post-traitement — discipline statistique

- **Purger le transitoire** avant tout calcul de statistiques.
- **Échantillons indépendants ≠ nombre de points** : `N_indep = T_fenêtre / τ` (τ = temps
  d'autocorrélation). Barre d'erreur = `σ / √N_indep`, recoupée par une moyenne par blocs.
- **Sensibilité à la fenêtre** : moyenner sur plusieurs fenêtres emboîtées. Dispersion < barre
  d'erreur → stable ; tendance monotone → dérive, ne pas conclure.
- **`drag_results.dat` est écrit en mode ajout** : après un restart, la portion recalculée apparaît
  deux fois (garder la dernière) et les premiers pas peuvent être aberrants (|CL| ~ 1e6, à filtrer).
  Les scripts `extract_*.py` le font.

> **⚠ Piège — y⁺ qui explose.** Ne jamais interpoler la distance-paroi en fonction de x/c sur tous
> les nœuds du volume : prendre, pour chaque point de paroi, le nœud interne le plus proche
> (`scipy.spatial.cKDTree`).

> **⚠ Piège — faux décollement.** Ne pas trier les deux faces mélangées pour calculer une tangente :
> traiter extrados et intrados séparément. Pour un état décroché, le zéro du Cf moyen n'est pas le
> point où la couche cisaillée quitte la paroi (rapport §4.3).

---

## 7. Validation (checklist, 10°)

| Grandeur | Attendu / référence |
|---|---|
| `y⁺` moyen | ~0,2 (Martin Ferrer : 0,66), max < 1 |
| `y1` (1er pt GLL) | `0,276 × xfirstu` à N = 3 (fraction GLL `(1 − 1/√5)/2`) |
| `Cp,min` | ≈ −5,2 à x/c ≈ 0,0025 |
| `Cp,max` | ≈ +1,0 (point d'arrêt) |
| décollement | aucun à 10° |

---

## 8. Études de sensibilité (résumé ; détails dans le rapport, chapitre 3)

- **Maillage (OFAT)** : combiner les optima de séries séparées peut donner **pire** que la référence
  (+1,26 % sur CD contre +0,68 %) → toujours tester le maillage combiné.
- **GCI / Richardson** : seulement si l'indicateur de plage asymptotique vaut ~1.
- **Domaine** : le confinement surestime CD, effet en **1/a²** porté par CDp (CDv invariant).
  Extrapoler CDp en 1/a² ; cible = littérature, pas la référence (elle-même confinée).

---

## 9. Analyse fréquentielle

- **Fenêtre de Hann**, moyenne de Welch, résolution `ΔSt ≈ 2 / longueur du segment`.
- **Détrender** avant FFT.
- **Contrôle croisé** : le spectre de CD doit piquer à 2× la fréquence de CL pour un lâcher alterné.
- **Modes parasites** (rapport §4.4) : un pic dont la fréquence suit l'ordre polynomial N
  (par exemple 10,66 à N = 3 → 13,9–14,5 à N = 4) est numérique. Toujours vérifier un pic rapide en
  changeant N avant de l'interpréter.

---

## 10. Récapitulatif des pièges

1. `AOA_ROT_DEG = 0` : ne pas re-projeter les efforts (CD ×10 sinon).
2. `Geometry.Tolerance = 1e-8` ; `gmsh ... -order 2 -format msh2` pour des HEX20.
3. Nommer les dumps par le temps physique.
4. Archiver les `.f0*` avant tout restart ; le maillage n'est que dans `f00000`.
5. `cf_x` = Cf scalaire signé → l'utiliser directement.
6. y⁺ : nœud interne le plus proche, pas d'interpolation sur tout le volume.
7. Décollement : traiter chaque face séparément.
8. Purger le transitoire ; convergence = sensibilité à la fenêtre.
9. Échantillons indépendants = T/τ.
10. GCI seulement dans la plage asymptotique.
11. Domaine : extrapoler en 1/a² sur CDp.
12. Un pic rapide qui change avec N est numérique.
13. Terminal VS Code sandboxé : administration système via le vrai terminal.
14. CFL plus haut ⇒ plus de sous-itérations ⇒ possible ralentissement net.

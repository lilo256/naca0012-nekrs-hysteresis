# MÉMO — NACA0012 / NekRS sur le cluster UPC

> **Note (octobre 2026).** Mémo écrit en juillet 2026 sur la grappe `nofx` (frontale `ibfluids01`,
> nœud `ibfluids02`, Tesla T4). La campagne de production a ensuite tourné sur la grappe **`cdf`**
> (nœuds L4, file `menuts.q`, même module `nekRS-26`, même logique Grid Engine). Les commandes
> restent valables ; remplacer `nofx` par `cdf`, le script de job par `job.sh` du dossier
> `case_template/`, et `USER` par ton identifiant. Voir aussi le `README.md` à la racine.

> Aide-mémoire complet : accès, fichiers, compilation, lancement, post-traitement.
> Cas : NACA0012, RANS k-τ, Re = 1e6, AoA = 10°, NekRS v26.0.1.

---

## 0. LA CARTE — 3 machines, 3 rôles

| Machine | Rôle | GPU ? | `qsub` ? |
|---|---|---|---|
| **ton PC** | maillage, post-traitement, graphiques | non | non |
| **`ibfluids01`** (frontale) | on **soumet** les jobs d'ici | non | ✅ **OUI** |
| **`ibfluids02`** (nœud de calcul) | le calcul **tourne** ici | ✅ Tesla T4 | ❌ NON |

**Le disque est partagé** entre la frontale et les nœuds (home sur le stockage QNAP).
Un fichier écrit d'un côté est visible de l'autre. Rien à recopier entre les deux.

**Règle d'or** : avec Grid Engine, tu ne te connectes **jamais** au nœud de calcul.
Tu restes sur la frontale et c'est `qsub` qui envoie le travail sur le nœud GPU
(grâce à la ligne `#$ -q ...` du script de job).

---

## 1. SE CONNECTER

```bash
ssh USER@nofx        # ou USER@cdf.upc.es
```

**Toujours vérifier où on est :**

```bash
hostname
```

⚠️ **LE PIÈGE N°1** : ne colle **jamais** un bloc de commandes juste après un `ssh` ou un `exit`.
Elles partent trop vite et s'exécutent sur la mauvaise machine. Après un `ssh` ou un `exit` :
tape `hostname` **tout seul**, attends la réponse, et seulement ensuite continue.

---

## 2. FICHIERS — naviguer, voir, éditer, supprimer

```bash
cd ~/cdf_runs/<cas>       # aller dans le dossier du cas
ls -lt                    # lister par date (le plus récent en haut)
ls -lh                    # avec la taille lisible (Ko, Mo)
nano airfoil.par          # éditer (Ctrl+O Entrée = sauver, Ctrl+X = quitter, Ctrl+W = chercher)
head -20 fichier.log ; tail -20 fichier.log ; grep endTime airfoil.par
cp airfoil.par airfoil.par.bak     # sauvegarde avant modification
```

**Écraser un fichier entièrement** (pratique pour les kernels) :
```bash
cat > cf.okl << 'EOF'
...contenu...
EOF
```
Les guillemets autour de `EOF` sont **importants** (sinon le shell interprète les `$`).

---

## 3. TRANSFÉRER PC ↔ SERVEUR

Toujours lancé **depuis ton PC**, jamais depuis le serveur.

```bash
rsync -avz airfoil.re2 airfoil.ma2 USER@cdf.upc.es:~/cdf_runs/<cas>/      # PC -> serveur
rsync -avz USER@cdf.upc.es:~/cdf_runs/<cas>/drag_results.dat .            # serveur -> PC
rsync -avz USER@cdf.upc.es:~/cdf_runs/<cas>/resultats_csv/ resultats_csv/ # un dossier
rsync -avz USER@cdf.upc.es:~/cdf_runs/<cas>/wall_distance_rank_\*.csv .   # motif * à échapper
rsync -avz --exclude='.cache' --exclude='*.f0*' cas/ USER@cdf.upc.es:~/cdf_runs/cas/
```

⚠️ **Le `/` final compte** : `dossier/` envoie le *contenu*, `dossier` envoie le *dossier lui-même*.
Vérifier un transfert : `md5sum airfoil.re2` des deux côtés.

---

## 4. MAILLAGE (sur le PC)

Voir `mesh/make_meshes.sh` (génération en lot, avec contrôles). À la main :

```bash
gmsh Airfoil_mesh.geo -3 -order 2 -format msh2 -o airfoil.msh 2>&1 | tee gmsh.log
grep -i "division by zero" gmsh.log && echo ">>> ECHEC" || echo ">>> OK"
gmsh2nek        # dimension : 3     |  nom : airfoil
genmap          # nom : airfoil     |  tolérance : 0.01
```
→ produit `airfoil.re2` et `airfoil.ma2`. **Jacobien minimum > 0** (gmsh, Tools > Statistics).

### Les corrections apportées au `.geo`

| Ligne | Correction | Pourquoi |
|---|---|---|
| `nord = 2` | (au lieu de 3) | gmsh2nek n'accepte que HEX20/HEX27 ; `nord=3` sort des HEX32 |
| `ikeepl1k = 0` | (au lieu de 1) | le solveur Newton `Compk` tombe sur la racine triviale k=1 → 0/0 |
| `Mesh.HighOrderOptimize = 2` | ajouté | désentortille les éléments courbes (Jacobiens négatifs sinon) |
| `xfirstu = 1.2e-5` | (au lieu de 4e-6) | 1ʳᵉ maille relâchée ×3 → dt plus grand, éléments moins repliés |

---

## 5. QUOI RECOMPILER ?

| Fichier | Recompiler ? |
|---|---|
| `airfoil.par` | ❌ **non** — lu au démarrage |
| `airfoil.re2` / `.ma2` | ❌ non |
| `airfoil.udf`, `postProcessing.hpp` | ✅ oui (C++) |
| `airfoil.oudf`, `my_drag.oudf`, `cf.okl` | ✅ oui (kernels GPU) |
| `airfoil.usr` | ✅ oui (Fortran) |

nekRS détecte les changements tout seul : un simple `qsub` recompile ce qu'il faut.
Pour voir les erreurs tout de suite (sur un nœud GPU) :
```bash
module load nekRS-26
nekrs --setup airfoil.par --build-only 1 --backend CUDA 2>&1 | tee build.log
grep -i "error" build.log || echo ">>> BUILD OK"
```
Vider le cache (rare, plusieurs minutes) : `rm -rf .cache`.

---

## 6. LANCER UN CALCUL (depuis la frontale uniquement)

```bash
grep endTime airfoil.par        # vérifier ce qu'on lance
qsub job.sh
qstat                           # qw = en attente | r = en cours | (vide) = terminé
qdel <ID>                       # tuer un job
qstat -j <ID> | grep -i reason  # pourquoi ça n'avance pas ?
```

Points obligatoires du script de job (voir `case_template/job.sh`) :
- ⚠️ **`--mca osc pt2pt`** dans la ligne `mpirun` : sans lui, `MPI_Win_create` → `MPI_ERR_WIN`.
- ⚠️ **`module load nekRS-26` DANS le script** : l'environnement du shell n'est pas transmis au job.

---

## 7. SUIVRE UN CALCUL

`python3 ~/bin/etat.py` (dossier `monitoring/`) donne l'état de tous les cas. À la main :

```bash
tail -f NACA0012.<ID>.out       # Ctrl+C arrête l'affichage, PAS le job
grep "FLUID p"   NACA0012.<ID>.out | tail -5     # itérations pression
grep "divUErr"   NACA0012.<ID>.out | tail -5     # incompressibilité
grep "dt="       NACA0012.<ID>.out | tail -5     # pas de temps + CFL
grep -i Jacobian NACA0012.<ID>.out               # doit être VIDE
```

| Indicateur | Sain | Malade |
|---|---|---|
| `FLUID p : iter` | **4 – 50** | 500 = plafond, il a abandonné |
| `divUErr` (2ᵉ nombre) | **< 0.05** | > 1 → la masse n'est pas conservée |
| `CFL` | ~1 | saut brutal → divergence imminente |
| Jacobien | rien | `Invalid element Jacobian < 0` |

---

## 8. REDÉMARRER DEPUIS UN CHECKPOINT

Voir `run_management/next.sh` (chaînage automatique). À la main, dans `[GENERAL]` du `.par` :
```ini
startFrom = airfoil0.f00007
endTime = 50
```
Vérifier : `grep "t= " NACA0012.<ID>.out | head -2` → le premier `t=` doit valoir le temps du
checkpoint, **pas `1e-09`**. Archiver les `.f0*` avant (`run_management/archiver.sh`) : le compteur
repart de `f00000`.

---

## 9. POST-TRAITEMENT

| Fichier | Contenu |
|---|---|
| `drag_results.dat` | `tstep time dragX dragY dragZ dragPX… dragVX…` → **CD = dragX, CL = dragY** |
| `resultats_csv/cf_cp_time_<t>_rank_<r>.csv` | `x,y,z,cf_x,cf_y,cf_z,Cp` → **`cf_x` = le Cf scalaire signé** |
| `wall_distance_rank_<r>.csv` | `x,y,z,wall_distance` → tout le domaine |
| `probes_uvp.csv` | séries temporelles des sondes (positions : `probes_positions.csv`) |
| `airfoil0.f0000X` | champs 3D (ParaView via un fichier `.nek5000`) |

y⁺ : `Cf → τ_w = ½ρU∞²Cf → u_τ = √(τ_w/ρ) → y⁺ = y₁ u_τ/ν` (ρ = 1, U∞ = 1, ν = 1e-6).

---

## 10. NETTOYER

**Ne jamais transférer** : `.cache/` (compilé pour un GPU précis), les `.f0000*` (lourds).
Pas de `zip` sur le serveur : `tar czf archive.tar.gz dossier/`.

---

## 11. LES PIÈGES — appris à la dure

| Symptôme | Cause | Solution |
|---|---|---|
| `is not a submit host` | `qsub` lancé depuis un nœud de calcul | revenir sur la frontale |
| `[CUDA] mode is not enabled` | lancé sur une machine sans GPU | passer par `qsub` |
| `MPI_ERR_WIN: invalid window` | RMA MPI cassé | ajouter `--mca osc pt2pt` |
| `FLUID p : iter 500` | SEMFEM indisponible | `preconditioner = semfem` |
| CFL saute à 72 → divergence | pas de temps trop agressif | `targetCFL = 1` |
| `Invalid element Jacobian < 0` | éléments courbes repliés | `Mesh.HighOrderOptimize = 2` |
| `Syntax error in COMMON statement` | Fortran fixe : `!` mal placé | le `!` doit être en **colonne 1** |
| `identifier "denom" is undefined` | OKL : variable hors de `@inner` | déclarer **dans** la boucle `@inner` |
| Le `.par` modifié ne change rien | lu au démarrage seulement | `qdel` → modifier → `qsub` |
| Les commandes partent au mauvais endroit | bloc collé après `ssh`/`exit` | une commande à la fois, `hostname` |

---

## 12. LES CIBLES (Martin Ferrer 2025, Table 4.1 — Re = 1e6, AoA = 10°)

| Grandeur | Cible |
|---|---|
| **CL** | **1.028** |
| **CD** | **0.0205** |
| Pic de succion Cp | ≈ −5 |
| y+ moyen | ≈ 0.66 (< 1 obligatoire) |

---

## 13. QUI DEMANDER QUOI

- **L'administrateur du cluster** → installation, modules, compilation de nekRS, MPI
- **Le responsable du stage (F. Mellibovsky)** → physique, validation, stratégie, maillage

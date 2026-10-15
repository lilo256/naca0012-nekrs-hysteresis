# NACA 0012 at Re = 10⁶ with NekRS (k–τ URANS): case setup, run chaining and post-processing

Code and procedures of the internship *"Static stall and hysteresis of the NACA 0012 at
Re = 10⁶ with NekRS"* (K. Rejeb, UPC, 2026, supervised by F. Mellibovsky).
The work extends the k–τ study of Martin Ferrer (2025) with a higher-order, larger-domain setup,
a full upstroke/downstroke campaign, perturbation tests and resolution tests.

The tag **`v1.0-report`** is the state of the code at the submission of the internship report
(16 October 2026); every figure and table of the report was produced with it.

> Comments and console messages in the scripts are in French. Each script starts with a docstring
> or header that gives its purpose and its exact usage: read it first.

---

## 1. Repository layout

| Folder | Content |
|---|---|
| `case_template/` | A complete production case (α = 15°, cold start): `.par`, `.udf`, `.oudf`, kernels, `.usr`, SGE job script, probe positions, the `.geo` and the mesh (`.re2`, `.ma2`). |
| `case_N4_restart/` | `.par` and `.udf` of a restart at polynomial order N = 4 from an N = 3 state (resolution test). |
| `mesh/` | `make_meshes.sh`: batch generation and checking of one mesh per angle. |
| `run_management/` | Chaining and preparation of runs: `next.sh` (one angle step), `prep_pert.sh` + `melange.py` (blended initial conditions), `prep_t15.sh` (stabilisation tests), `archiver.sh`. |
| `monitoring/` | `etat.py` (status of every run), `conv.py` (convergence of the mean loads). |
| `postprocessing/` | Extraction of loads, separation and probe statistics, spectra, wall analysis. |
| `figures/` | Scripts that produce the figures of the report from the extracted data. |
| `legacy/` | Scripts of the mesh and domain studies (July–August 2026), run on the PC: `moyenne_forces_v2.py` (time-weighted means, autocorrelation and block error bars, window sensitivity), `check_yplus.py` (y⁺ from the nearest interior node: reliable), `surface_cp_cf_yplus.py` (mean C_p, C_f and separation; **its y⁺ is wrong**, use `check_yplus.py`), `compare_cases.py` (significance between cases), `strouhal.py`, `serieC_figures.py`. |
| `docs/` | `TUTORIEL_fr.md` (full pipeline with its pitfalls) and `MEMO_cluster_fr.md` (cluster cheat sheet), in French. |

Field files, time series and wall dumps are **not** in the repository (several hundred GB). They
stay on the cluster, one folder per run in `~/cdf_runs/` (naming in §5).

---

## 2. Workflow

### 2.1 Mesh (local PC)
One mesh per angle: the incidence is engraved in the mesh (`aoa = <A>*Pi/180;` in the `.geo`),
the free stream is along +x and the trailing edge is at the origin.
```bash
cd mesh && ./make_meshes.sh 15.00 15.25      # Gmsh 4.15 -> gmsh2nek -> genmap, with checks
```
Production mesh: 14 124 curved HEX20 elements, domain 20c upstream/laterally and 30c downstream,
first element at the wall `xfirstu = 1.2e-5`.

### 2.2 Case and submission (cluster `cdf`, from the front end)
```bash
cp -r case_template ~/cdf_runs/my_case && cd ~/cdf_runs/my_case
# put the mesh of the right angle (airfoil.re2, airfoil.ma2) and set ALPHA_DEG in airfoil.udf
grep -E "endTime|startFrom|polynomialOrder" airfoil.par
qsub job.sh
```
Production settings: N = 3, adaptive time step with target CFL 1, `hpfrt` regularisation with
nModes = 1 and scalingCoeff = 10. Cost on one NVIDIA L4: about 1 h per convective time unit (CTU)
at N = 3 and 3.5 h per CTU at N = 4.

### 2.3 Chaining in incidence
`run_management/next.sh` prepares one quasi-static step (e.g. 15.00° → 15.25°) from the last
checkpoint of the previous run: plausibility and convergence checks of the parent run, mesh check,
absolute `endTime`, `ALPHA_DEG` in the UDF, then asks for confirmation before `qsub`.
Each step is a new folder, so no checkpoint is ever overwritten.

### 2.4 Monitoring
```bash
python3 monitoring/etat.py        # all runs: time reached, speed, expected end, mean C_L, C_D
```

### 2.5 Post-processing (on the cluster, from `~/cdf_runs`)
| Script | Purpose | Report |
|---|---|---|
| `extract_domaine.py` | loads of the domain study | ch. 3 |
| `extract_montee.py` | upstroke: mean loads, separation (wall C_f and probe comb), C_p | §4.3 |
| `extract_boucle.py` | whole hysteresis loop (upstroke, downstrokes, jumps, perturbation, filter and N = 4 tests) → `boucle_data.json` | §4.5 |
| `boucle.py` | quick table of the loop, with the parent of each run | §4.5 |
| `analyse_sondes.py` | probe statistics, Strouhal numbers, correlations | §4.4 |
| `spectre_sondes.py` | band r.m.s. and peaks of the loads and probes (Welch, Hann) | §4.4 |
| `spectre_film.py`, `paroi_film.py` | fixed-step "film" run at 15°: spectra, wall C_f/C_p, grid-locking test | §4.4 |
| `compare_N3_N4.py` | N = 3 against N = 4 from the same state | §4.4 |
| `extract_series_montee.py` | time series of the upstroke runs for the figures | §4.3 |
| `extract_spectra_44.py` | spectra of the frequency study (input of `fig_frequency_44.py`) | §4.4 |

### 2.6 Figures
`figures/` reads the JSON files written by the extraction scripts (or the raw data on the cluster)
and writes PDF + PNG: `fig_polaire_montee.py` and `fig_upstroke_*.py` (§4.3), `figures_44.py` and
`fig_spectres_CL.py` and `fig_frequency_44.py` (§4.4), `fig_N4.py` (§4.4–4.5). Vorticity snapshots
were made in the ParaView GUI (reader `.nek5000`, Gradient with vorticity, ω_z from −5 to 5) and
assembled with `montage_vorticity_44.py` and `montage_fields_45.py` (`prepare_captures_45.sh`
prepares the folders of field files for the captures).

---

## 3. Conventions

- Free stream along **+x**, U∞ = 1, chord c = 1, ν = 10⁻⁶: times are in CTU and
  frequencies are Strouhal numbers based on the chord.
- `drag_results.dat`: `tstep time dragX dragY dragZ dragPX dragPY dragPZ dragVX dragVY dragVZ`,
  with **C_D = dragX and C_L = dragY** (no projection: the incidence is in the mesh).
- `resultats_csv/cf_cp_time_<t>_rank_<r>.csv`: `x,y,z,cf_x,cf_y,cf_z,Cp`; **`cf_x` is the signed
  skin friction** (positive towards the trailing edge).
- Probes (`probes_positions.csv`, series in `probes_uvp.csv`): `A###` comb at 0.008c above the
  suction side (x/c = 0.###), `B###` separated shear layer, `W###p/m` wake above/below the axis,
  `REF` upstream witness (flat spectrum expected), `SPAN#` spanwise checks.

---

## 4. Pitfalls that cost days

1. **Mesh only in `f00000`.** In every run only `airfoil0.f00000` contains the coordinates (code
   starting with `X` in `head -c 132 airfoil0.f00000`). Use it for ParaView and for restarts with
   interpolation.
2. **Restart at a different order** (`case_N4_restart/`): `startFrom = "file+int"` with a file that
   carries the mesh, and the UDF with enlarged host buffers (`mesh->Nlocal + 4096`); the stock UDF
   crashes at N = 4.
3. **`drag_results.dat` is appended**: after a restart, the recomputed part appears twice and the
   first steps can be aberrant (|C_L| ~ 10⁶). The extraction scripts filter both.
4. **Gmsh**: `-order 2 -format msh2` is required for HEX20 (without it: HEX8 with the same element
   count); `Geometry.Tolerance = 1e-8`; gmsh2nek returns 0 even on failure, check that the `.re2`
   exists.
5. **Job script**: `module load nekRS-26` inside the script and `mpirun --mca osc pt2pt`.
6. **ParaView**: the VisIt Nek5000 reader loads no field by default (tick them before Apply); the
   old ParaView installed on the cluster (`pvbatch`) calls the image gradient `Gradient`; use
   `GradientOfUnstructuredDataSet` there, or work in the GUI on a PC.
7. **Spurious fast modes**: a fast peak whose frequency changes with N (10.66 at N = 3 → 13.9–14.5
   at N = 4 at the trailing edge) is numerical (report §4.4). Check any fast peak with N before
   interpreting it.

More pitfalls, with their symptoms: `docs/TUTORIEL_fr.md` §10 and `docs/MEMO_cluster_fr.md` §11.

---

## 5. Run naming on the cluster (`~/cdf_runs/`)

| Prefix | Meaning |
|---|---|
| `dom_a<a>_b<b>` | domain study at 10° (a = upstream/lateral extent, b = wake length, in chords) |
| `prod_a<α>` | production cold starts (0°, 5°, 15°) |
| `up_a<α>` | upstroke, quasi-static steps of 0.25° |
| `cold_a<α>` | cold starts at a given angle |
| `down_a<α>` | downstroke, quasi-static steps |
| `jump_a<α>` | downstroke by jumps of 0.5–1° |
| `att_a<α>` | attached branch below 15° |
| `pert13_<w>` | blend of stalled (w) and attached (1 − w) states at 13° |
| `stall<α>_hpf40` | stalled states restarted with a fourfold filter |
| `stall<α>_N4`, `res_a15_N4` | restarts at N = 4 |
| `film_a15`, `t15_*`, `spec_a*` | fixed-step runs for spectra and stabilisation tests |

---

## 6. Where to start (next intern)

1. Read `docs/TUTORIEL_fr.md` and the report (chapters 3 and 4).
2. Rerun `case_template/` for a few CTU to check the toolchain.
3. Priorities left open by the report (chapter 5): N = 4 with the fourfold filter at 15° and 16°;
   N = 5 or a trailing-edge refinement; an N = 4 downstroke from 16° to bound the loop at N = 4.

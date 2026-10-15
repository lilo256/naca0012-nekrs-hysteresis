# case_template — production case (α = 15°, cold start, N = 3)

Copied from `~/cdf_runs/prod_a15` (August 2026). All production runs use these files; only the
mesh (`airfoil.re2`, `airfoil.ma2`, one per angle, see `../mesh/`), `ALPHA_DEG` in `airfoil.udf`
and the `startFrom` / `endTime` lines of `airfoil.par` change from one run to the next.

| File | Role |
|---|---|
| `airfoil.par` | solver settings (N = 3, adaptive dt with target CFL 1, k–τ, `hpfrt` regularisation) |
| `airfoil.udf`, `postProcessing.hpp` | user functions: loads (`drag_results.dat`), wall dumps (`resultats_csv/`), probes (`probes_uvp.csv`) |
| `airfoil.oudf`, `my_drag.oudf`, `cf.okl` | GPU kernels (boundary conditions, loads, skin friction) |
| `airfoil.usr` | Fortran user file |
| `probes_positions.csv` | probe coordinates (mesh frame: trailing edge at the origin) |
| `job.sh` | SGE job script (submit with `qsub job.sh` from the front end) |
| `Airfoil_mesh.geo`, `airfoil.re2`, `airfoil.ma2`, `airfoil.nek5000` | geometry, mesh at 15°, ParaView entry file |

Changing `.udf`, `.oudf`, `.okl`, `.usr` or `.hpp` triggers a recompilation at the next start;
changing `.par` does not.

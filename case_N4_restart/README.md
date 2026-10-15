# case_N4_restart — restart of an N = 3 state at N = 4

Copied from `~/cdf_runs/stall13_N4` (resolution test of the stalled state at 13°, report §4.5.4).
Only two files differ from `case_template/`:

- `airfoil.par`: `polynomialOrder = 4` and `startFrom = "restart0.f00000+int"`. The `+int`
  interpolates the N = 3 field onto the N = 4 points; the start file **must contain the mesh**
  (in a run, only `airfoil0.f00000` does; check the `X` in `head -c 132 <file>`).
- `airfoil.udf`: host buffers enlarged to `mesh->Nlocal + 4096` (the stock UDF crashes at N = 4).
  `ALPHA_DEG` must match the mesh.

Same mesh, filter and CFL as the parent run. Cost: about 3.5 h per CTU on one L4 GPU.

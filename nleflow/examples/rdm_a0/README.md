# The RDM A = 0 example

The single-accumulator racing-diffusion flow whose training box reaches
A = 0 exactly.
`spec.yaml` is the recipe of the committed card `rdm_ss_base_s801_e271`
(`nleCards/`): context `v, B, t0, s, A`, A as
`asinh(A / 0.05)`, the three-part A mixture, [256, 128], 64 bins, range ±5,
batch 1024, 300 epochs, 30 kept checkpoints. `spec_smoke.yaml` is the same
with 1 % of the data (240 000 training rows, 24 000 validation rows) and 2
epochs: end to end in about a minute on 4 cores.

Needs R with EMC2 (DMC2 main) in a dedicated library (for `rWald` and
`dRDM`) and `jsonlite`; `arrow` for parquet, else the simulator writes CSV.

```sh
cd nleflow                                   # the package directory, after uv sync
export NLEFLOW_RLIB=~/R/nle-lib              # the library holding EMC2 (DMC2 main)
uv run nleflow simulate examples/rdm_a0/spec_smoke.yaml      # 5 s: data_smoke/{train,val}
uv run nleflow train    examples/rdm_a0/spec_smoke.yaml      # 15 s: runs/rdm_a0_smoke/
uv run nleflow steps    examples/rdm_a0/runs/rdm_a0_smoke
uv run nleflow export   examples/rdm_a0/runs/rdm_a0_smoke    # 15 s: runs/rdm_a0_smoke/export/
Rscript -e 'source(system.file("scripts", "nle_verify.R", package = "EMC2"))' \
    examples/rdm_a0/runs/rdm_a0_smoke/export/rdm_a0_smoke_e1_weights.json
```

(`R_PROFILE_USER=/dev/null` and `R_LIBS` as in [install and check](../../docs/install-and-check.md)
when another EMC2 is on the path.) The full recipe is the same four commands
with `spec.yaml` (24 million rows, 300 epochs: hours on a workstation, as the
project ran it on tux26). Checkpoint selection (score bias over the kept window, then
one SBC cell) is EMC2's validation kit, outside this package.

`simulate.R` is the project's `simulation/sim_rdm_a0.R` driven by the spec:
the box, `zerobox_c` and the A strata are read from `NLEFLOW_SPEC`, the row
counts from `NLEFLOW_ROWS` / `NLEFLOW_VAL_ROWS`, the data seed from
`NLEFLOW_SIM_SEED` (default 4876, the project's; validation uses 4877).
`training.seed` is the network's seed. One smoke-run fact worth knowing: a card
exported after 2 epochs on 1 % of the data registers and passes its golden
check because the golden check tests the port, not the fit; whether a flow
is fit to infer with is the validation kit's question.

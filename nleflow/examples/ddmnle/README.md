# The DDMnle example (the scaled DDM)

The flow over the unit-scale decision time u = (rt − t0)(s/a)² of a diffusion
with a = s = 1, conditioned on κ = v a/s², η = sv a/s², Z and SZ, with a
choice classifier for P(R = 2); t0, st0, a and s are applied exactly by the
card's evaluator (EMC2's `flow_scaled` kind; NLEFLOW_PLAN.md section 2,
`handover/SCALED_DDM_PLAN.md`). `spec.yaml` is the recipe of the committed
card `ddm_sc_s1106_e245` (`handover/nle-methods/port/export/`): [256, 128],
64 bins on log u ∈ [−12, 2], batch 1024, 300 epochs, 30 kept checkpoints,
600 000 draws × 40 trials, the box and strata of `simulation/scaled.R`, and
the [256, 256, 256] classifier trained on exact P(R = 2) targets
(`retrain_classifier.py`'s tuned configuration). `spec_smoke.yaml` is 1 % of
the data (6000 draws × 40) and 2 epochs; the classifier keeps its 194 epochs
on 2000 targets. DDMnle itself is not retrained here: the committed card and
its golden file are the references the tests use.

Needs R with EMC2 `dev-flow` in a dedicated library (`rDDM`, `dDDM`),
WienR (the exact choice probabilities) and jsonlite; arrow for parquet, else
CSV.

```sh
cd nleflow
export NLEFLOW_RLIB=~/R/nle-m0-lib
uv run nleflow simulate examples/ddmnle/spec_smoke.yaml      # 10 s: data_smoke/{train,val,choice}
uv run nleflow train    examples/ddmnle/spec_smoke.yaml      # 30 s: flow (2 epochs) + classifier
uv run nleflow export   examples/ddmnle/runs/ddmnle_smoke    # 30 s: a flow_scaled card + golden vectors
Rscript -e 'source(system.file("scripts", "nle_verify.R", package = "EMC2"))' \
    examples/ddmnle/runs/ddmnle_smoke/export/ddmnle_smoke_e1_weights.json
uv run nleflow onnx examples/ddmnle/runs/ddmnle_smoke/export/ddmnle_smoke_e1_weights.json
```

As run on the cloud instance (4 cores): the smoke card registers with
`twin = DDM` as `flow_scaled`, card_version 1; `nn_golden_check` 3.6e-13;
`nle_verify.R` PASS on the golden check, the compiled-vs-R path (5.7e-14)
and the oracle (8.3e-14; set `EMC2_NLE_PORT=handover/nle-methods/port/R`
for that line); float64 gate 3.8e-13 (log pdf) and 1.1e-14 (z), float32 gap
7.5e-5. The flow's validation loss after 2 epochs on 1 % is −1.107 against
the project's −1.436 after 300 epochs on all of it: a smoke run, not a fit.

`simulate.R` is `simulation/sim_ddm_scaled.R` and `scaled.R` driven by the
spec: it runs EMC2's DDM at a = s = 1, st0 = 0 and t0 = 0.05 and stores
u = rt − 0.05; the box, zero-box constants, strata (`sampling`) and the four
regimes of exact zeros (`regimes`) come from the spec; the choice sets are
fresh draws with `WienR::pWDM` probabilities. One difference from the
project: it draws pseudo-random uniforms where the project drew shifted
Sobol points (`randtoolbox`), to keep its dependencies to EMC2, WienR and
jsonlite. Data seeds: 5101 train, 5102 validation, 5103/5104 the choice sets
(`NLEFLOW_SIM_SEED` moves them together); `training.seed` is the network's.

The ONNX graph takes the natural DDM parameters (v, a, t0, s, Z, SZ as the
start-point proportion, sv, st0), rt and the choice, and does the scaling,
the shift and the st0 branch inside; see the package README for the two
signatures and the agreement measured against EMC2.

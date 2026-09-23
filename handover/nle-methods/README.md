# ABC — calibration-grade neural likelihoods for evidence accumulation models

A neural likelihood is usually validated on held-out log-likelihood or pointwise
error. **For inference those metrics are close to irrelevant.** This repository
contains the diagnostics that do predict whether an approximation distorts a
posterior, the recipe for training one that does not, and validated artefacts
for the DDM and RDM that load directly into EMC2.

The evidence for that claim is blunt: EMC2's default-precision analytic is
*pointwise much worse* than our flow (median worst-case error 0.33 vs ~0.12)
and *calibration-wise much better* (|g| ~1e-4 vs +0.013 on `a`). The orderings
are opposite, because what matters is **where** the error sits, not how big it is.

## Layout

| directory | what it holds |
|---|---|
| `simulation/` | regenerates every training input from a fixed seed |
| `flow/` | JAX/Flax training and export (Python) |
| `port/` | float64 R + C++ reference evaluators, and their golden-vector tests |
| `evaluation/` | **the diagnostics** — score bias, its analytic control, the posterior-shift screen, SBC harness |
| `artefacts/` | the selected flows, as weights JSON and EMC2-loadable bundles |
| `results/` | the SBC and score-bias objects backing every number we report |
| `docs/` | the methods document, the recipe, EMC2 integration, and the traps checklist |

## Data is regenerated, not deposited

`simulation/` reproduces the training inputs **bit-for-bit** from the recorded
seeds — verified: 16,000,000 rows, all 13 columns identical, shards
byte-identical, in 11.9 minutes. So no multi-gigabyte download is needed and
there is no risk of a deposit drifting from the code that made it.

```r
Rscript simulation/sim_ddm.R          # 16e6 rows, DDM
Rscript simulation/sim_rdm.R          # 16e6 rows, RDM single accumulator
Rscript simulation/sim_ddm_st0zero.R  # 16e6 rows, st0 = 0 variant
```

Check against `MANIFEST.sha256` to confirm your regeneration matches ours.

Two directories are **created by running the pipeline** and are deliberately
absent from a fresh clone: `data/` (by the commands above) and
`results/model_states/` (by training). Everything else a script reads is
included — the SBC prior draws the estimators condition on, the Fisher matrix,
the exported flow weights under `port/export/`, and the SBC objects under
`results/sbc/`. Verified by rehearsing this package in a clean directory.

## Quickstart: use an artefact

```r
# install the dev-nn EMC2 into a DEDICATED library -- it reports the same
# version as the release build and will otherwise overwrite it silently
.libPaths(c("~/Rlib-devnn", .libPaths()))
library(EMC2)
file.copy("artefacts/ddm_cap256w_c4.rds",
          system.file("extdata", "flownn", "ddm_ens5.rds", package = "EMC2"),
          overwrite = TRUE)
```
Bundle filenames are **hardcoded** in `ddmnn_get()`, so swapping a model means
replacing the file, not adding one.

## Quickstart: check a likelihood you have built

```r
Rscript evaluation/score_bias_draws.R 200 <export_tag>   # g, six parameters
Rscript evaluation/score_bias_t0.R    200 <export_tag>   # g_t0, restricted region
Rscript evaluation/test_posterior_shift.R <export_tag> 60 2000   # predicted SBC bias
```
Both print an **analytic control** first. If the control is not ~0, the
estimator is wrong and its numbers mean nothing — this is not optional, and ours
was silently 25–40% wrong until the control was added.

## The recipe

1. Simulate at scale (16e6 rows, one trial per draw).
2. Train **one** run at large batch (1024), retaining many checkpoints.
3. **Select** the best checkpoint on score bias. Never average — averaging
   converges to the member mean; selection finds the best member (22x, DDM).
4. Confirm with SBC.
5. **Replicate across >=3 seeds before believing any training intervention.**

Step 5 is not boilerplate. Run-to-run variance at fixed hyperparameters is
comparable to any effect worth reporting. Every claim in this project that
rested on one run per condition was later retracted; none replicated across
>=3 seeds was. See `docs/TRAPS.md`.

## Status of the artefacts

| artefact | SBC | note |
|---|---|---|
| `rdm_small.rds` | passes N=100 and N=400 | worst \|std bias\| 0.090 |
| `ddm_cap256w_c4.rds` | passes N=100–800 | `t0` carries a +0.28% offset; see `docs/RECIPE.md` |
| `ddm_st0zero.rds` | passes N=400 | 7-parameter st0=0 build, once `st0` is named a constant (see `docs/TRAPS.md`): worst std bias +0.282, coverage 0.936-0.960, ranks non-uniform on a, t0, Z |

## Provenance

`SBC_PLAN.md` in the parent project is the full audit trail — 36 phases,
including the claims we retracted and why. It travels with this repository as
provenance rather than as reading.

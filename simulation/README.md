# `simulation/`

Simulates choice–RT data and computes **ground-truth likelihoods**
for the three evidence accumulation models (LNR, RDM, DDM), in R.

## Working directory

> **Run from the repository root.** Paths are root-relative (`data/...`) and
> sources are loaded as `simulation/...`.

## Run

```sh
Rscript simulation/sim_code.R
```

Key packages: **EMC2**
(model definitions, design objects, analytic likelihoods), `randtoolbox`
(Sobol' draws for training parameters), `arrow`/`dplyr`.

## Files

| File | Role |
|---|---|
| [`sim_functions.R`](sim_functions.R) | simulation + analytic-likelihood kernels and the `simulate_data` / `param_matrix` drivers |
| [`designs.R`](designs.R) | EMC2 `design` objects, parameter bounds, sampled-scale transforms (incl. single-accumulator variants) |
| [`sim_code.R`](sim_code.R) | top-level driver: calls `simulate_data` for every set/budget |
| [`prior_predictives.R`](prior_predictives.R) | prior-predictive draws |
| [`tests/test_designs.R`](tests/test_designs.R) | checks on the design objects |

## What `sim_code.R` produces (under `data/`)

| Subtree | Content | Seed |
|---|---|---|
| `prior/<MODEL>/` | prior-predictive samples (1,000 param sets) | `1` |
| `train/<budget>/{DDM, RDM_single, LNR_single}/` | **single-accumulator** training sets | `TRAIN_SEED=3876` |
| `val/{RDM_single, LNR_single, DDM}/` | validation sets (10,000 param sets) | `VAL_SEED=1656` |
| `test/{RDM, LNR, DDM}/` | **two-accumulator** test sets (1,000 param sets × 2,000 trials) | `TEST_SEED=2251` |
| `kde_fit/<budget>/{LNR, RDM, DDM}/` | KDE fitting data at test parameters | `SEED_KDE=2231` |
| `kde_fit/val/{LNR, RDM, DDM}/` | KDE **bandwidth-selection** validation data | `2026` |

Each output folder holds `parameters.parquet`, `trials/trials_*.parquet`
(columns include `param_id`, `R`, `rt`, `ll`, plus model parameters), and a
`sim_time.RDS` (elapsed seconds, used by the cost figures).

### Budgets

- **Training** (`small`/`medium`/`large`) = **50,000 / 200,000 / 1,000,000**
  total trials for the DDM and RDM. The LNR budgets are halved (**25,000 / 100,000 / 500,000**).
- **KDE fitting** (`small`/`medium`/`large`) = **1,000 / 10,000 / 100,000**
  trials per parameter set (split across the two stimuli).
- The training set uses **Sobol' sampling**; test/validation sets use
  independent uniform proposals (to avoid Sobol-like sequences leaking into
  evaluation).

### Single- vs two-accumulator variants

The flow is trained on **single-accumulator** data (`*_single`, one racing
accumulator). Evaluation/test sets are **two-accumulator** (the full racing
likelihood), so analytic `ll` reflects the complete model. `designs.R` defines
both variants.

## Determinism

Every call passes an explicit `seed`, so outputs are **bit-exact for a fixed R +
EMC2 version**. 

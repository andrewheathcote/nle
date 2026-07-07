# `kde/`

Kernel density estimation (KDE) of choice–RT likelihoods, in the style of Lin,
Heathcote & Holmes (2019): fit per-`(param_id, response)` densities, select a
bandwidth, and evaluate against the analytic test likelihood.

## Working directory

> **Run from the repository root.** Inputs are read from `data/...` and outputs
> written to `results/...`.

## Run

```sh
Rscript kde/KDE.R          # fit/evaluate + write predictions
Rscript kde/KDE_timing.R   # per-call timing of KDE fit + eval
```

## Files

| File | Role |
|---|---|
| [`kde_functions.R`](kde_functions.R) | `fit_kde` (per-response `density()` → `approxfun`, scaled by `p(R)`) and `eval_kde` (log-density with a `log(1e-10)` floor) |
| [`KDE.R`](KDE.R) | driver: `evaluate_bws`, the fit/eval loop, writes `bw_eval.rds` + `kde_ll_pred.csv` |
| [`KDE_timing.R`](KDE_timing.R) | timing driver (reuses `fit_kde`/`eval_kde`) |

## Inputs / outputs

- **Reads:** `data/kde_fit/<budget>/<MODEL>/trials/` (fitting data),
  `data/kde_fit/val/<MODEL>/trials/` (bandwidth-selection set),
  `data/test/<MODEL>/trials/` (evaluation set). Columns used:
  `param_id`, `R`, `rt`, `ll`.
- **Writes:** `results/<MODEL>/<budget>/bw_eval.rds` (a list:
  `[[1]]` the bandwidth grid metrics, `[[2]]` the selected `best_kde` fit) and
  `results/<MODEL>/<budget>/kde_ll_pred.csv`.

## Bandwidth selection

`evaluate_bws()` fits the KDE at each bandwidth in `c(0.1, 0.01, 0.001)` and
scores it on the validation set by correlation, MSE, and a Monte-Carlo KL
estimate (`mean(ll_true − est)`). The returned `best_kde` is the **KL-optimal**
bandwidth (closest to 0).

## The `read_existing_KDE` switch (`KDE.R` line ~72)

- `read_existing_KDE <- TRUE` (current): **re-evaluates** the stored
  `bw_eval.rds[[2]]` fits over `data/test`, rather than refitting from scratch.
  Use this to reproduce the saved predictions without the (expensive) refit.
- `read_existing_KDE <- FALSE`: refits across the bandwidth grid, selects the
  KL-optimal bandwidth, and overwrites `bw_eval.rds` + `kde_ll_pred.csv`.

## `kde_ll_pred.csv` must be keyed (row order is not otherwise recoverable)

`arrow::open_dataset("data/test/<MODEL>/trials")` is a multi-file read whose
**row order is non-deterministic** across calls, so an `est` column written by
row position cannot be paired back to its truth (naive pairing gives
`cor ≈ 0.01` against a ceiling of ≈ 0.98). The current `KDE.R` therefore writes
the identifying keys with each prediction, so `kde_ll_pred.csv` has columns
`param_id, R, rt, ll, est`. The analysis
([`../results/tables_paper/tables.R`](../results/tables_paper/tables.R)) joins
KDE `est` onto the flow/truth table **on those keys** (plus
`rt_key = round(rt, 12)`), making the pairing order-independent. If you have
older files that stored only `est`, regenerate them with the current `KDE.R`
(`read_existing_KDE = TRUE`).

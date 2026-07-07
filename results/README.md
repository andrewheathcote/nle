# `results/`

Analysis, paper figures, and summary tables comparing the **normalizing flow
(NLE)** and **KDE** likelihood approximators against ground truth. This is where
the paper's numbers come from.

## Working directory

> **Run from the repository root.** The analysis scripts resolve paths with
> `here::here("results", ...)` / `here::here("data", ...)`, so the project root
> must be the working directory (or run inside the `.Rproj`).

## What is tracked here

Tracked: analysis **code** (`*.R`), the figure writeup
([`figures_paper/OVERVIEW.md`](figures_paper/OVERVIEW.md)), every paper-figure
panel (`figures_paper/*.pdf`), and the summary tables
(`tables_paper/*.csv`). **Not** tracked (regenerable / in the deposit):
per-model/budget predictions (`<MODEL>/<budget>/*.{csv,rds}`), timing
(`*/timing/**`), checkpoints (`model_states/**`), the superseded
`prelim_results/`, rendered reports and caches, and scratch files. See the root
[`.gitignore`](../.gitignore).

Code for anything that feeds a paper figure/table lives here; the **bulk
outputs** it consumes are regenerable from `data/` + the pipeline (or pulled from
the deposit).

## Reproduce figures & tables

```sh
Rscript results/figures_paper/plotting.R    # paper-figure PDFs → figures_paper/
Rscript results/tables_paper/tables.R       # summary table     → tables_paper/approx_stats.csv
```

## Files

| File | Produces |
|---|---|
| [`figures_paper/plotting.R`](figures_paper/plotting.R) | the paper-figure PDFs in `figures_paper/` (sources `tables_paper/tables.R`) |
| [`tables_paper/tables.R`](tables_paper/tables.R) | `tables_paper/approx_stats.csv` |
| [`ll_analysis.R`](ll_analysis.R) | standalone log-likelihood analysis |
| [`truncation_correction.R`](truncation_correction.R) / [`truncation_plots.R`](truncation_plots.R) | DDM truncation analysis + figures |

## Inputs the analysis reads

- `results/<MODEL>/<budget>/flow_ll_pred.csv` — flow predictions (self-aligned:
  its `ll` column is the analytic truth for the same row).
- `results/<MODEL>/<budget>/kde_ll_pred.csv` — KDE predictions, keyed
  (`param_id, R, rt, ll, est`); see the alignment note below.
- `results/<MODEL>/timing/N=<N>/{flow_timing.json,times.csv}` — evaluation timing.
- `results/model_states/<budget>_budget/<head>/training_log.json` — loss curves +
  per-epoch times (heads: `LNR`, `RDM`, `DDM_flow`, `DDM_classifier`).
- `data/test/<MODEL>/trials/` and `data/train|kde_fit/.../sim_time.RDS`.

## KDE alignment (why the numbers are trustworthy)

`KDE.R` writes `kde_ll_pred.csv` with the identifying keys `param_id, R, rt`
next to `ll` (analytic truth) and `est` (KDE estimate). `tables.R` joins the KDE
`est` onto the flow/truth table **on those keys** (`param_id, R, model, budget`,
plus `rt_key = round(rt, 12)`), not on row position — so the `est ↔ truth`
pairing never depends on the non-deterministic row order
`arrow::open_dataset` returns. If you have older, unkeyed `kde_ll_pred.csv`
files, regenerate them with the current `KDE.R` (`read_existing_KDE = TRUE`).
See also [`../kde/README.md`](../kde/README.md).

## See also

[`figures_paper/OVERVIEW.md`](figures_paper/OVERVIEW.md) — a consolidated
walkthrough of every paper figure and the summary statistics.

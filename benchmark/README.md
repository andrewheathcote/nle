# `benchmark/`

Computational-efficiency benchmarks: per-call **evaluation timing** of the flow
vs. KDE vs. the analytical/numerical likelihood, plus the supporting data
generation.

## Working directory

> **Run everything from the repository root.** The R scripts use
> `simulation/...` and `benchmark/scripts_julian/...`; the Python flow-timing
> scripts read `./results/model_states/...` and write `results/<MODEL>/timing/...`.

## Run order

| # | Script | Does |
|---|---|---|
| 1 | [`scripts_julian/get_data_params.R`](scripts_julian/get_data_params.R) | simulates the eval datasets (`N=100/1000/10000`) and writes the **`eval_params.csv`** parameter grids (`param_matrix(..., seed = 2)`) per model |
| 2 | [`scripts_julian/get_kde_train_at_eval_params.R`](scripts_julian/get_kde_train_at_eval_params.R) | simulates KDE *training* data **at the eval params** (`kde_train/`) so KDE can be fit and timed at evaluation parameters |
| 3 | `scripts_julian/{LNR,RDM}_timing.R`, [`scripts_julian/dWDM_C_eval.R`](scripts_julian/dWDM_C_eval.R) | analytic / numerical likelihood timing (DDM uses the WienR `dWDM` C path) |
| 3' | `scripts_julian/{RDM,LNR,DDM}_flow_timing.py` | flow likelihood timing |

Outputs go to `results/<MODEL>/timing/N=<N>/` (`times.csv` for analytic,
`flow_timing.json` for the flow), consumed by the results analysis (see
[`../results/`](../results/)).

## Inputs you must provide

- **`eval_params.csv`** (`scripts_julian/{DDM,LNR,RDM}/eval_params.csv`):
  **tracked** (small, deterministic, `seed = 2`). The eval datasets and
  `kde_train/` under those dirs are regenerable bulk and gitignored — regenerate
  with steps 1–2 above, or get them from the deposit.
- **Trained flow checkpoints** for the flow-timing scripts, at the canonical

  ```
  results/model_states/large_budget/<head>/        head ∈ {RDM, LNR, DDM_flow, DDM_classifier}
  results/model_states/large_budget/<head>/train_loader.pkl
  results/model_states/large_budget/<head>/schedule_config.json
  ```

  The training notebook writes checkpoints **directly** to this canonical
  location (see [`../flow/README.md`](../flow/README.md#where-checkpoints-go)),
  with `train_loader.pkl` (the fitted feature scaler) and `schedule_config.json`
  (the LR schedule) as siblings. Checkpoints + these siblings otherwise come from
  the [deposit](../README.md#data--checkpoints).

## ⚠️ `reevaluate_DDM_ll.R` — one-off, already applied, do **not** re-run

[`scripts_julian/one_off/reevaluate_DDM_ll.R`](scripts_julian/one_off/reevaluate_DDM_ll.R)
was a **single-use repair** that recomputed the DDM analytic `ll` and overwrote
`data/test/DDM/trials/trials_000{1,2}.parquet` **in place**. `data/test/DDM`
already carries the corrected `ll`, so re-running it would re-write live test
data for no benefit. It has been moved into `one_off/` and guarded
(`if (!interactive()) stop(...)`) so it cannot run non-interactively. Do not
port it for portability — it is archived, not maintained.

# Likelihood Approximation in Evidence Accumulation Models: KDE vs. Normalizing Flow

This project compares **kernel density estimation** (KDE; Lin et al., 2019) and
**neural likelihood estimation** (NLE; Boelts et al., 2022; Fengler et al., 2021; Papamakarios et al., 2019) via a **normalizing flow**
at approximating choice–RT likelihoods in three evidence accumulation models:
the **log-normal race** (LNR), the **racing diffusion model** (RDM), and the
full **diffusion decision model** (DDM). It evaluates both *approximation
accuracy* and *computational efficiency*.

## Pipeline

```
Simulate (R) ──► Train flow (Python, JAX/Flax) ──┐
            └─► Fit KDE (R)                       ├──► Evaluate (accuracy + timing)
                                                  ┘
```

Each stage owns its language; cross-stage I/O is **parquet on disk** under
`data/` and CSV/RDS/JSON under `results/`.

## Working-directory contract

> **Run R scripts and the benchmark scripts from the
> repository root.** They use paths relative to the root (`data/...`,
> `results/...`, `simulation/...`). **Run the flow training notebook from
> `flow/notebooks/`** — it uses `../../data/...`. Every per-directory README
> restates the working directory it assumes.

## Directory layout

| Path | Role | Language | Tracked? |
|---|---|---|---|
| [`simulation/`](simulation/) | Data simulation + analytical likelihoods | R | yes |
| [`kde/`](kde/) | KDE fitting, bandwidth selection, evaluation | R | yes |
| [`flow/`](flow/) | Normalizing-flow package + training notebook | Python (JAX/Flax) | yes |
| [`benchmark/`](benchmark/) | Timing: flow vs KDE vs analytical | R + Python | yes |
| [`results/`](results/) | Analysis code + paper figures/tables | R | **partly** (see below) |
| `data/` | Pipeline inputs (prior, train, val, test, kde_fit) | parquet | no (gitignored) |

### `results/` — what is tracked

`results/` mixes small, valuable deliverables with multi-GB regenerable bulk, so
it is tracked **selectively** (see [`.gitignore`](.gitignore)):

- **Tracked:** analysis code (`*.R`), the figure/table writeups
  (`*.md`), every paper-figure panel under `results/figures_paper/*.pdf`, and the
  summary tables `results/tables_paper/*.csv`.
- **Ignored (regenerable, obtained from the deposit):** per-model/budget
  predictions (`results/<MODEL>/<budget>/*.{csv,rds}`), timing JSON/CSV under
  `results/*/timing/`, trained checkpoints under `results/model_states/`, rendered
  reports and caches, and scratch files.

> If a file appears in a paper figure or table, its **code** lives in `results/`
> (tracked) and its **bulk output** is regenerable / in the deposit.

## Installation

### R (simulation, KDE, benchmark, results)

R dependencies are pinned with [`renv`](https://rstudio.github.io/renv/).
Captured R version: **4.5.3**.

```r
renv::restore()
# renv::status()   # should report "No issues found" (no drift)
```

### Python (flow training & flow timing)

The [`flow/`](flow/) package is installable; dependencies are pinned in
`flow/uv.lock` / `flow/pyproject.toml` (Python 3.11–3.12).

```sh
cd flow
uv sync                 # preferred; or: pip install -e .
# GPU (flow training): pip install -e ".[gpu]"   (installs jax[cuda12])
```

`flow/requirements.txt` is a pinned mirror of the runtime deps for environments
that consume only a requirements file.

## Reproduce from a fresh clone

All commands run from the **repo root** unless noted. Get the inputs and trained
checkpoints from the [deposit](#data--checkpoints) (fast path) **or** regenerate
them by running the pipeline below.

| # | Step | Command | Working dir | Notes / approx. cost |
|---|---|---|---|---|
| 1 | Restore R env | `Rscript -e 'renv::restore()'` | root | one-off |
| 2 | Simulate data | `Rscript simulation/sim_code.R` | root | writes `data/`; DDM is the expensive model (large budget ≈ hours). See [simulation/README.md](simulation/README.md) |
| 3 | Train flows | run [`flow/notebooks/flow_training.py`](flow/notebooks/flow_training.py) cell-by-cell | `flow/notebooks/` | per-model × per-budget cells (RDM/LNR/DDM). Writes checkpoints **directly** to the canonical `results/model_states/<budget>_budget/<head>/`. See [flow/README.md](flow/README.md) |
| 4 | Fit / evaluate KDE | `Rscript kde/KDE.R` | root | reads `data/kde_fit/`, `data/test/`; writes `results/<MODEL>/<budget>/`. See [kde/README.md](kde/README.md) |
| 5 | Timing benchmarks | scripts in `benchmark/scripts_julian/` | root | see [benchmark/README.md](benchmark/README.md) for run order & required checkpoints |
| 6 | Regenerate figures/tables | `Rscript results/figures_paper/plotting.R`; `Rscript results/tables_paper/tables.R` | root | rebuilds the paper PDFs + `approx_stats.csv`. See [results/README.md](results/README.md) |

### Data & checkpoints

`data/`, the bulk of `results/`, and the trained checkpoints
are gitignored. They are deposited (with a thesis copy) at:

**Deposit (OSF):** https://osf.io/sqr2z

Simulated data can be found on figshare: https://doi.org/10.6084/m9.figshare.32873795

Everything is **regenerable** by running the pipeline from scratch; the deposit
is the fast path.

### Trained-checkpoint location

The training notebook writes checkpoints **directly** to the canonical layout the
timing and analysis code read — no manual move/rename:

```
results/model_states/<budget>_budget/<head>/
        heads: RDM | LNR | DDM_flow | DDM_classifier
        siblings: train_loader.pkl (fitted scaler), schedule_config.json, training_log.json
```

with `<budget>` ∈ {small, medium, large}. The flow timing scripts read
`results/model_states/large_budget/<head>/`. See
[flow/README.md](flow/README.md) and [benchmark/README.md](benchmark/README.md).

## Reproducibility — seeds & determinism

- **R simulation / KDE.** Fixed seeds make outputs **bit-exact** for a given R +
  package version: simulation uses `TRAIN_SEED=3876`, `VAL_SEED=1656`,
  `TEST_SEED=2251`, `SEED_KDE=2231` (and `2026` for the KDE-bandwidth validation
  set, `1` for prior predictives); the timing eval parameters use `seed=2`.
- **Python flow training.** The flow is **JAX/Flax** (not PyTorch), seeded with
  `jax.random.key(100)` and `nnx.Rngs(...)`. Reported equivalence is loss-curve / scatter shape, not bit
  identity (bit-equality is only claimable on CPU).

## Roadmap (not yet implemented)

- **MCMC** posterior sampling consuming the flow or KDE likelihood.
- **Simulation-based calibration (SBC).**

These have no code in the repo yet.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).

## References

Boelts, J., Lueckmann, J.-M., Gao, R., & Macke, J. H. (2022). Flexible and
efficient simulation-based inference for models of decision-making. *eLife*, 11,
e77220. <https://doi.org/10.7554/eLife.77220>

Fengler, A., Govindarajan, L. N., Chen, T., & Frank, M. J. (2021). Likelihood
approximation networks (LANs) for fast inference of simulation models in
cognitive neuroscience. *eLife*, 10, e65074.
<https://doi.org/10.7554/eLife.65074>

Lin, Y.-S., Heathcote, A., & Holmes, W. R. (2019). Parallel probability density
approximation. *Behavior Research Methods*, 51(6), 2777–2799.
<https://doi.org/10.3758/s13428-018-1153-1>

Papamakarios, G., Sterratt, D., & Murray, I. (2019). Sequential neural
likelihood: Fast likelihood-free inference with autoregressive flows.
*Proceedings of the Twenty-Second International Conference on Artificial
Intelligence and Statistics*, 837–848.
<https://proceedings.mlr.press/v89/papamakarios19a.html>

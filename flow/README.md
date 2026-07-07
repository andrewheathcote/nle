# `flow/`

Python package (JAX/Flax) implementing the **normalizing-flow likelihood** for
the LNR, RDM, and DDM evidence accumulation models, plus the notebook that
trains the flows.

## Working directory

> **Run the training notebook from `flow/notebooks/`.** Its paths are relative to
> that folder: data at `../../data/...` and outputs at `../../results/...` (both
> resolve to the repo root). The flow **timing** scripts in
> [`../benchmark/`](../benchmark/) are instead run from the **repo root**.

## Install

```sh
cd flow
uv sync                 # or: pip install -e .
pip install -e ".[gpu]" # GPU training (jax[cuda12]); optional
pip install -e ".[notebooks]"  # jupytext + ipykernel, to run the notebook
```

`requirements.txt` is a pinned mirror of the runtime deps; `pyproject.toml` /
`uv.lock` are authoritative.

## Module layout

- `src/flow/config.py` — flow architecture: `spline_flow`, `make_mlp_conditioner`,
  `make_choice_classifier`, and `joint_log_prob(conditioner, classifier, rt,
  params, response)` (DDM flow + choice classifier, 5-arg). `spline_flow_ddm` is
  **not currently used** (kept for reference; see its docstring).
- `src/flow/data.py` — `batches(...)` parquet batch generator, `PreloadedBatches`
  (preloads + standardizes; carries the fitted scaler), `sign_rts`.
- `src/flow/checkpoints.py` — Orbax-backed `make_checkpoint_manager` /
  `save_checkpoint` / `restore_checkpoint`, keeping the **best validation-loss**
  step (`best_fn`); `best_step()` returns it.
- `src/flow/training.py` — `loss_fn`, `train_step`, `train_model`,
  `train_model_classifier` (DDM choice head), timed evaluators used by the
  benchmark, and the `save_obj`/`load_obj`/`plot_*` helpers the notebook uses.

## Training — the notebook is the driver

The canonical entry point is the jupytext-paired notebook:

- **Tracked:** [`notebooks/flow_training.py`](notebooks/flow_training.py) (py:percent mirror).
- **Generated / gitignored:** `notebooks/flow_training.ipynb`. 

> The notebook is the real and
> only documented training path. **Run it cell by cell from `flow/notebooks/`**
> (it is a research notebook — do not "Run All"; everything below the
> `# OLD CODE !!!` marker near the end is legacy/non-functional and is **not**
> part of the pipeline).

The notebook has a section per model (**RDM**, **LNR**, **DDM**), each repeated
for the **small / medium / large** budgets. Every section:

1. points `TRAIN_PATH` / `VAL_PATH` at `../../data/train/<budget>/<MODEL>/trials`
   and `../../data/val/<MODEL>/trials`,
2. seeds JAX with `jax.random.key(100)` → `nnx.Rngs(default=..., sampling=...)`,
3. builds an MLP conditioner (`make_mlp_conditioner`), a `warmup_cosine_decay`
   AdamW optimizer, and `PreloadedBatches` train/val loaders (`standardize=True`;
   the val loader reuses the train loader's fitted `scaler`),
4. calls `train_model(...)` (or, for the DDM choice head,
   `train_model_classifier(...)`), which checkpoints the **best-val-loss** step
   and writes a `training_log.json` (per-epoch train/val losses + times),
5. restores the best step and writes flow predictions to
   `../../results/<MODEL>/<budget>/flow_ll_pred.csv`.

The **DDM** combines a spline flow over `rt` (conditioned on the DDM params + the
response `R`) with a separate **choice classifier** via
`joint_log_prob(flow, classifier, rt, ctx, response)`.

## Where checkpoints go

The notebook writes checkpoints **directly to the canonical location** that the
timing and analysis code read — no manual move/rename is needed:

```
results/model_states/<budget>_budget/<head>/
        heads: RDM | LNR | DDM_flow | DDM_classifier
        siblings: train_loader.pkl  (fitted feature scaler, via save_obj)
                  schedule_config.json  (LR schedule)
                  training_log.json  (per-epoch losses + times)
```

with `<budget>` ∈ {`small`, `medium`, `large`}. The flow **timing** scripts
([`../benchmark/`](../benchmark/)) read `results/model_states/large_budget/<head>/`
plus the `train_loader.pkl` / `schedule_config.json` siblings; the analysis
([`../results/`](../results/)) reads `training_log.json`. Do **not** relocate
checkpoints out of `results/model_states/` — that would break the timing scripts.

## Tests

```sh
pytest flow/tests        # smoke test for the package API
```

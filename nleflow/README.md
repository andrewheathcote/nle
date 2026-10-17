# nleflow

Train a normalizing-flow neural likelihood for an evidence-accumulation model
from a model specification, and export it as a card that EMC2 (branch
`dev-nle-m0`) loads with `register_nn_model()`. This is milestone M1 of
`handover/NLEFLOW_PLAN.md`: the package skeleton with the race-model path and
the RDM A = 0 worked example. The DDM path (flow + classifier, the scaled DDM
card, ONNX) is milestone M2.

```
nleflow spec     <spec>            validate a spec and print it resolved
nleflow simulate <spec>            run the spec's simulator command
nleflow train    <spec>            train; checkpoints under runs/<name>/
nleflow steps    <run>             the kept checkpoints and their validation loss
nleflow export   <run> --step N    write a card_version 1 card + golden vectors
nleflow golden   <card>            rewrite a card's golden vectors from its JSON
```

What a flow is, how the cards are evaluated and validated, and the card
format are EMC2's: `vignette("neural-likelihood-cards")` on `dev-nle-m0` is
the contract this package writes to.

## Installation (the Python side of NLEFLOW_PLAN.md section 9.2)

Each step has its one-line check. R and EMC2 `dev-nle-m0` are needed only
for the simulator of the RDM example (`EMC2:::rWald`) and for registering
the exported card; their steps are in the plan's section 9.2.

1. **uv** (https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
   (macOS also: `brew install uv`). Check: `uv --version` prints 0.8 or later.
2. **A Python 3.11 or 3.12 interpreter**; uv fetches one when none is present:
   `uv python install 3.11`. Check: `uv python find 3.11` prints a path.
3. **The environment from the lock**, in this directory:
   `uv sync --frozen --group dev`. This installs the pinned versions every
   shipped flow was trained and exported with (jax 0.5.3, flax 0.10.6,
   distrax 0.1.5, optax 0.2.8, orbax-checkpoint 0.11.1, numpy 2.2.6,
   scipy 1.15.3) plus pyarrow, pyyaml, onnx, onnxruntime and pytest, into
   `.venv/`. 11 s with a warm cache on the cloud instance, a few minutes cold.
   Check: `uv run python -c "import jax, flax, distrax, onnx, onnxruntime; print(jax.__version__)"`
   prints `0.5.3`.
4. **The command.** Check: `uv run nleflow --help` lists the six commands
   (or `source .venv/bin/activate` once and use `nleflow` directly).
5. **The tests.** Check: `uv run pytest` passes (12 tests; a 13th runs when
   `NLEFLOW_EMC2_LIB=<library holding EMC2 dev-nle-m0>` is set and registers a
   card in R). About 30 s on 4 cores.
6. **GPU (optional, untested):** `uv sync --frozen --group dev --extra gpu`
   installs `jax[cuda12]`. Every shipped flow was trained on CPU.

Threads: `nleflow train` and `export` limit themselves to `min(8, cores)`
threads (`--cores` to change), the plan's "at most 8 cores" rule.

## The spec

A YAML or JSON file (`nleflow/spec.py` documents every key and validates it;
`nleflow spec <file>` prints the resolved form). The RDM A = 0 example,
`examples/rdm_a0/spec.yaml`:

```yaml
name: rdm_a0                      # label of runs and cards made from it
model: RDM                        # the analytic twin: an EMC2 model name
response: race                    # race | choice (choice: M2)
context: [v, B, t0, s, A]         # the network's input order; asserted everywhere
transforms: {v: log, B: log, t0: log, s: log, A: zerobox}
zerobox_c: {A: 0.05}              # the network sees asinh(A / 0.05); exact at A = 0
stored: {A: log}                  # the data store log A, -Inf at A = 0 (probit for SZ)
box_natural: {v: [1.0e-3, 5], B: [0.1, 3], t0: [0.05, 1], s: [0.1, 2], A: [0, 2]}
sampling: {A: {uniform: 0.70, asinh_uniform: 0.15, zero: 0.15}}   # the rest uniform
pre: null                         # t0: a decision-time flow (optional, unproven)
spline: {bins: 64, range: [-5, 5]}
conditioner: {widths: [256, 128], norm: false, affine: false}    # affine: optional, unproven
training: {rows: 24000000, val_rows: 1000000, batch: 1024, epochs: 300, keep: 30, peak_lr: 1.0e-4, seed: 801}
data: {train: data/train, val: data/val}    # relative to the spec file
simulator: Rscript simulate.R               # writes the data (below)
control_limits: {s: [0.1, 2], vb_over_s2_max: 354}   # the Wald control's limits
mass_tolerance: 1.0e-3
```

YAML note: write exponents with a sign (`1.0e-4`, not `1e-4`) or as plain
integers; PyYAML reads `24e6` as a string and the spec refuses it.

**The two unproven options** (plan section 13.3.3): `conditioner.affine`
(two more conditioner outputs, a per-context location and scale on log
time) and `pre: t0` (the flow models rt − t0 and t0 leaves the context).
Both are off by default, both are exported as the card fields `affine` /
`affine_scale` and `pre` / `pre_bounds_natural` that EMC2 and the R reference
evaluate, and neither is part of the recipe until the seed study's reading
says so. The pipeline test exercises both.

## The data contract (`nleflow/data.py`)

One row per trial; columns `param_id`, each context name on the stored
(EMC2 sampled) scale with `-Inf` at an exact zero of a zero-box input, `rt`,
and optionally `ll` (the analytic log density, for diagnostics) and `R`. A
dataset is a directory of parquet files, or of CSV files (`write.csv`'s
format, `-Inf` spelled so) when R has no `arrow`; the two load identically.
`nleflow simulate` runs the spec's `simulator` command with
`NLEFLOW_SPEC` (the resolved spec as JSON), `NLEFLOW_TRAIN`, `NLEFLOW_VAL`,
`NLEFLOW_ROWS`, `NLEFLOW_VAL_ROWS`, `NLEFLOW_SEED` and `NLEFLOW_CORES` set
(`nleflow/simulate.py`); any program that writes the contract will do.

Training (`nleflow/train.py`) is the NLE project's `train_race.py`: the
zero-box encoding computed in float64 from the stored value, the context
standardised by the training moments (`scaler.json` in the run directory),
the spline flow's mean negative log density as the loss, AdamW under a
warmup-cosine schedule, one Orbax checkpoint per epoch with the `keep`
latest kept and the best validation loss marked, no early stopping. The flow
itself (`nleflow/flow/`) is the project's `flow/src/flow/config.py` and
`checkpoints.py`, vendored unchanged.

## The card and its gate (`nleflow/export.py`)

`nleflow export <run> [--step N]` (default: the best validation loss) writes
`<run>/export/<name>_e<N>_weights.json`, `_golden.csv` and
`_intermediates.json`. The card is version 1 of EMC2's vignette: `kind
flow_race`, `model` (the twin), `context_names`, `context_transforms`,
`zerobox_c`, `bounds_sampled` / `bounds_natural`, `scaler`, `mlp`, `spline`,
`affine` (+ `affine_scale`), `pre` (+ `pre_bounds_natural`), `ll_floor_log`,
`support` (`{"time": [0, "Inf"]}`: JSON has no infinity, and jsonlite reads
the string `"Inf"` as R's `Inf`), `mass_tolerance`, `beyond_support`,
`control_limits`, plus provenance (`label`, `budget`, `checkpoint_step`,
`exported`, `exporter_version`, `training`, `export_gate`). The `validation`
block is written only once the spec records something in it, so EMC2's
message that the card records no validation fires for every fresh export:
loading a network is not the same as being able to infer with it.

Before anything is written, the float64 NumPy port that produces the golden
vectors (`nleflow/golden.py`) is gated against the flow itself evaluated by
JAX in float64 (`jax_enable_x64`): log density and spline inversion must
agree to 1e-8 over the 575 golden points, or the export refuses. The gap to
the float32 flow (the one trained; the number the project's `export_flow.py`
used to gate on, at 5e-3) is printed for information. On the smoke card:
float64 gate 2.0e-13 (log pdf) and 7.4e-15 (z); float32 gap 3.6e-5.

`nleflow golden <card>` rewrites the golden file from the card alone (the
project's `golden.py`, moved here; `handover/nle-methods/flow/scripts/golden.py`
now calls this module). The golden grid, row order and storage scales are
the exporters' and are documented in `nleflow/golden.py`.

## Checking a card in EMC2

```r
.libPaths(c("~/R/nle-m0-lib", .libPaths())); library(EMC2)
m <- register_nn_model("rdm_a0_smoke_e1_weights.json", twin = RDM)
nn_golden_check(m)    # PASS golden check ... max |diff| 2.70e-13 (log_pdf) against 1e-09
```
and the verification command,
`Rscript -e 'source(system.file("scripts", "nle_verify.R", package = "EMC2"))' <card>`,
which prints one PASS/FAIL line per shipped artefact and per card and exits 1 on any FAIL.

## Layout

```
pyproject.toml, uv.lock       the pinned environment
src/nleflow/spec.py           the spec schema and validation
src/nleflow/data.py           the data contract, encodings, scaler, batches
src/nleflow/train.py          nleflow train
src/nleflow/export.py         nleflow export: card v1 + float64 gate
src/nleflow/golden.py         golden vectors (race and joint cards)
src/nleflow/simulate.py       nleflow simulate
src/nleflow/flow/             the flow, vendored from the NLE project
examples/rdm_a0/              the RDM A = 0 example: spec.yaml, spec_smoke.yaml, simulate.R
tests/                        pytest: golden regression, spec, pipeline (+ R registration)
```

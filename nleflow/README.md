# nleflow

Train a normalizing-flow neural likelihood for an evidence-accumulation model
from a model specification, and export it as a card that EMC2 (branch
`dev-nle-m0`) loads with `register_nn_model()`. Milestones M1 and M2 of
`handover/NLEFLOW_PLAN.md`: the race-model path with the RDM A = 0 example,
and the choice-model path (flow + choice classifier; the scaled DDM card,
DDMnle) with its ONNX export.

```
nleflow spec       <spec>            validate a spec and print it resolved
nleflow simulate   <spec>            run the spec's simulator command
nleflow train      <spec>            train; checkpoints under runs/<name>/ (+ the classifier)
nleflow steps      <run>             the kept checkpoints and their validation loss
nleflow export     <run> --step N    write a card_version 1 card + golden vectors
nleflow golden     <card>            rewrite a card's golden vectors from its JSON
nleflow onnx       <card>            a DDM flow card as an ONNX graph (two signatures)
nleflow onnx-check <model> <card>    the graph against the golden rows and EMC2 cases
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
4. **The command.** Check: `uv run nleflow --help` lists the eight commands
   (or `source .venv/bin/activate` once and use `nleflow` directly).
5. **The tests.** Check: `uv run pytest` passes (25 tests; three of them run
   only when `NLEFLOW_EMC2_LIB=<library holding EMC2 dev-nle-m0>` is set, to
   register cards in R and compare the ONNX graph with EMC2, and one needs
   the EMC2 checkout beside this repository for its golden file; the rest
   skip). About 80 s on 4 cores.
   The ONNX tests need `onnx` and `onnxruntime` from the lock; check:
   `uv run python -c "import onnxruntime as ort; print(ort.__version__)"` prints 1.20 or later.
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

**A choice model** (the scaled DDM, `examples/ddmnle/spec.yaml`) sets
`response: choice`, a `classifier`, `scaling: {kind: ddm_unit}` and, for the
simulator, per-input strata in the long form (`kind`, `share`, `range`) plus
`regimes`, the shares of draws with each set of inputs exactly 0:

```yaml
response: choice
context: [kappa, eta, Z, SZ]
transforms: {kappa: identity, eta: zerobox, Z: probit, SZ: zerobox}
zerobox_c: {eta: 0.01, SZ: 0.05}
stored: {eta: log, SZ: probit}
box_natural: {kappa: [-10, 10], eta: [0, 8], Z: [0.1, 0.9], SZ: [0, 0.99]}
sampling:
  kappa: [{kind: uniform, share: 0.5}, {kind: uniform, share: 0.5, range: [-3, 3]}]
  eta: [{kind: log_uniform, share: 0.75, range: [0.01, 8]}, {kind: asinh_uniform, share: 0.25}]
  SZ: [{kind: asinh_uniform, share: 1, range: [0.05, 0.99]}]
regimes: [{share: 0.55}, {share: 0.15, zero: [eta]}, {share: 0.15, zero: [SZ]}, {share: 0.15, zero: [eta, SZ]}]
scaling: {kind: ddm_unit, sz: proportion}
spline: {bins: 64, range: [-12, 2]}
classifier: {widths: [256, 256, 256], lr: 1.49e-3, batch: 256, epochs: 194, loss: mse}
training: {rows: 24000000, val_rows: 1000000, trials_per_draw: 40, choice_rows: 200000, choice_val_rows: 20000, ...}
data: {train: data/train, val: data/val, choice_train: data/choice/train.csv, choice_val: data/choice/val.csv}
```

The flow is conditioned on (standardised context, R) with R appended raw;
the classifier maps the standardised context to the logit of P(R = 2) and is
trained after the flow on the exact probabilities of the two choice sets
(`retrain_classifier.py`'s tuned recipe: sigmoid output, MSE on the
probability, AdamW under warmup-cosine), with the flow's standardiser. It is
written to the run as `classifier.json`; `nleflow train --classifier-from
<run>` reuses another run's classifier when the two standardisers agree to
1e-3 (the exporters' rule). The DDM's own eight-parameter flows (`flow_joint`
cards, no `scaling`) are the same path without the scaling block.

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

A choice model exports a `flow_joint` card, or a `flow_scaled` card when the
spec has `scaling` (`model: DDMsc`, the structured `scaling` block,
`context_encoding: zerobox`, `classifier_mlp`, `classifier_scaler`; EMC2
registers it with `twin = DDM`). Its gate compares the joint log density of
both responses over the 1150 golden points (JAX `joint_log_prob` in float64);
on the smoke card 3.8e-13.

## The ONNX export (`nleflow/onnx_export.py`)

`nleflow onnx <card>` writes the LAN-shaped graph of a `flow_joint` or
`flow_scaled` card in two signatures (plan section 7.3), float64, opset 18,
IR version 10 (onnxruntime ≤ 1.30 loads it; 1.31 here):

| signature | inputs | output |
|---|---|---|
| `trialwise` | one `[N, k + 2]` tensor: the k parameters, rt, choice (−1 lower, +1 upper) | `[N]` log p(rt, choice \| θ) |
| `shared` | `theta [k]`, `rt [N]`, `choice [N]`; the conditioner runs twice (once per response) and the classifier once | `[N]` |

For a `flow_joint` card the k inputs are the context on the stored scale
(`export_onnx_ddm.py`'s contract: log, probit, −Inf at an exact zero; the
graph applies the zero-box encoding and the st0 = 0 support rule). For a
`flow_scaled` card they are the eight natural DDM parameters v, a, t0, s, Z,
SZ (the start-point proportion), sv, st0: the graph computes κ, η, the
encodings, the probit of Z, c = (s/a)², u = (rt − t0) c and the st0 branch,
exactly as EMC2's `nn_eval_scaled()` does (c g(u) when st0 = 0 or the window
is narrower than 1e-7 d, otherwise [G(u₁) − G(max(u₀, 0))] / st0 with G the
network's defective CDF p_R Φ(z)), −Inf for rt ≤ t0 and the card's floor
outside the box. onnxruntime has no double-precision `Erf` or `Asinh`, so the
graph carries R's own pnorm (Cody) and qnorm (AS 241) as graph operations
(`nleflow/rmath.py` is the NumPy twin, tested against scipy to 1e-14) and
asinh(x) = log(x + √(x² + 1)).

`nleflow onnx-check <model> <card> [--cases cases.csv]` compares the graph
with the card's golden rows and with EMC2's evaluation over prior draws
(`scripts/onnx_cases_scaled.R`: 200 draws × 100 trials in each of the six
regimes of `sbc/scaled_cell.R`, half the draws with s ≠ 1). Measured on the
committed DDMnle card, both signatures:

| comparison | rows above the floor | max \|Δ log p\| | median |
|---|---|---|---|
| golden rows | 1010 | 5.3e-11 | 7e-15 |
| EMC2, st0 = 0 (a density) | 38 400 | 4.4e-11 | 1e-14 |
| EMC2, st0 > 0 (a CDF difference) | 78 400 | 5.6e-9 | 4e-15 |

The 1e-11 target of the plan holds where the likelihood is a density up to
the spline's conditioning at its range boundary (the 5e-11 rows sit at
z = 2.0, the identity-tail handover). On the st0 > 0 branch the difference
G(u₁) − G(u₀) of two CDFs amplifies round-off: the NumPy float64 port itself
differs from EMC2 by the same 3e-9 on those rows, and the graph differs from
the port by no more. The defaults of `onnx-check` are therefore 1e-10 for the
density rows and 1e-8 for the window rows, and the README records both.

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
src/nleflow/onnx_export.py    nleflow onnx / onnx-check
src/nleflow/rmath.py          R's pnorm and qnorm in NumPy (the graph's algorithms)
src/nleflow/simulate.py       nleflow simulate
scripts/onnx_cases_scaled.R   EMC2 reference cases for onnx-check
src/nleflow/flow/             the flow, vendored from the NLE project
examples/rdm_a0/              the RDM A = 0 example: spec.yaml, spec_smoke.yaml, simulate.R
examples/ddmnle/              the scaled DDM example (DDMnle): spec.yaml, spec_smoke.yaml, simulate.R
tests/                        pytest: golden regression, spec, rmath, pipeline, onnx (+ R checks)
```

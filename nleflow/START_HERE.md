# Start here

For Julian and Russ: what nleflow is, how to install and check it, where the cards
are, and which page to read for each task. Thirty minutes from install to your own
card in EMC2 and in ONNX: [TUTORIAL.md](TUTORIAL.md).

## What nleflow is

A Python package (JAX) that trains a **neural likelihood** for an evidence-accumulation
model from a short model specification (`spec.yaml`) and a simulator you write (R or
Python), and exports it as a **card**: one JSON file with the weights, the input
transforms, the training box and golden vectors. EMC2 loads a card with
`register_nn_model()` and uses it like any other model (design, prior, fit, SBC); the
same card exports to an **ONNX** graph for use outside EMC2 (onnxruntime, HSSM-style
code). It also converts a published **LAN** (an ONNX multilayer perceptron, e.g.
HSSM's) into an EMC2 card.

Two kinds of network are trained:

| path | what it learns | card kind | worked example |
|---|---|---|---|
| race | the density of each accumulator's finishing time; EMC2 combines them | `flow_race` | `examples/rdm_a0`, `examples/lba` |
| choice | rt given the response (a spline flow) and P(response) (a classifier) | `flow_joint`, `flow_scaled` | `examples/ddm_svst0`, `examples/ddmnle` |

**What it is not.** It does not fit models (EMC2 does), it does not train LANs (it
converts them), and a card that passes every check here is still not shown to give
calibrated posteriors: that is EMC2's job (`nn_cell()`, `nn_score_bias()`, `run_sbc()`;
EMC2's vignette "Neural likelihoods").

## Install and check

Full steps, one check each: [docs/install-and-check.md](docs/install-and-check.md). In short:

1. **R >= 4.5 and EMC2** with the neural-likelihood code, in a library of its own.
   Today that is EMC2 branch **`dev-flow`** (the nle work on top of `dev` after the
   `joint_move` sampler merge, origin/dev ff374317; it reads a card's `fixed` field;
   stage H3). Until it is pushed to `ampl-psych/EMC2` it is local to AH's machine;
   `dev-nle-ondev` (the same nle work on `dev` before `joint_move`) also works, and
   `dev-nle-m0` on GitHub works for everything except registering a card with fixed
   parameters from the card alone (see "Known limits").
   ```sh
   MAKEFLAGS=-j4 R CMD INSTALL --preclean --library=$HOME/R/nle-lib <EMC2 source>
   export R_LIBS=$HOME/R/nle-lib R_PROFILE_USER=/dev/null     # always both (a second EMC2 can shadow yours)
   ```
2. **Python**: `uv sync --frozen --group dev` in `nleflow/` (uv fetches Python 3.11).
3. **The one verification command** (R; installs nothing, checks every shipped card
   against its golden vectors, and any card you name):
   ```sh
   Rscript -e 'source(system.file("scripts", "nle_verify.R", package = "EMC2"))' [card.json ...]
   ```
   Check: one `PASS` line per card and `all checks passed`.
4. **The tests**: `R_PROFILE_USER=/dev/null uv run pytest -rs` in `nleflow/`; set
   `NLEFLOW_EMC2_LIB` (the EMC2 library; R_LIBS for its dependencies), `NLEFLOW_RLIB`
   and `NLEFLOW_EMC2_SRC` (an EMC2 source tree) to turn on the R cases.
5. **Or Docker**: `docker/Dockerfile` (R, EMC2, the lock); `docker run <image> nle-verify`.

## Where the cards are

Cards live in this repository, `handover/nle-methods/port/export/`, each as
`<tag>_weights.json` with `<tag>_golden.csv` (and `_intermediates.json`) beside it:

| card | model | kind | notes |
|---|---|---|---|
| `ddm_svst0_e<N>` | DDM with sv and st0 free, s = 1, SZ = 0 fixed | `flow_joint` | `examples/ddm_svst0`; its README has the gates |
| `ddm_sc_s1106_e245` | the full DDM, scaled (DDMnle) | `flow_scaled` | the adopted DDM network; `examples/ddmnle` |
| `lba_e299` | LBA | `flow_race` | `examples/lba` |
| `rdm_ss_base_s801_e271`, `rdm_rdm_small_e195` | RDM | `flow_race` | `examples/rdm_a0` |
| `lan_ddm_uniform_st` | HSSM's DDM with uniform st (a converted LAN) | `mlp_joint` | `examples/lan_in` converts the current file |

EMC2 itself ships `ddm_cap256w_c4`, `ddm_st0zero` and `rdm_small` (`DDMnn`, `RDMnn`).

## A map

| to do this | read | run |
|---|---|---|
| the whole path once, on small data | [TUTORIAL.md](TUTORIAL.md) | `examples/ddm_svst0/run_smoke.sh` |
| train a choice flow (DDM variants) | [docs/train-a-choice-flow.md](docs/train-a-choice-flow.md) | `examples/ddm_svst0`, `examples/ddmnle` |
| train a race flow (RDM, LBA) | [docs/train-a-race-flow.md](docs/train-a-race-flow.md) | `examples/lba`, `examples/rdm_a0` |
| fit data with a card in EMC2 (register, check, prior in the box, fit vs the analytic model) | EMC2's `vignette("using-a-trained-flow")` (dev-flow, or dev-nle-ondev >= cb40f173) | `inst/scripts/using_a_trained_flow_data.R` |
| export a card and check it in EMC2 | [docs/export-and-check-a-card.md](docs/export-and-check-a-card.md) | each example's `check_in_emc2.R` |
| a card as ONNX, and check it | [docs/onnx-out.md](docs/onnx-out.md) | `nleflow onnx`, `nleflow onnx-check` |
| use the ONNX graph from Python | [examples/onnx_use](examples/onnx_use/README.md) | `examples/onnx_use/run.sh` |
| bring a published LAN into EMC2 | [docs/lan-in.md](docs/lan-in.md) | `examples/lan_in/run.sh` |
| the spec, the data contract, the card, the converter | [README.md](README.md) (reference) | |
| the card format (the contract with EMC2) | EMC2's `vignette("neural-likelihood-cards")` | |
| why the package looks as it does; its progress log | `handover/NLEFLOW_PLAN.md` | |

## Known limits

- **ONNX is float32 when published, float64 when ours.** A LAN in float32 matches its
  float64 card to about 1e-5 in log density (the golden check's tolerance is 1e-4 and
  is recorded in the card); our graphs are float64 and match EMC2 to about 1e-13 (1e-8
  where st0 > 0 makes the density a difference of two CDFs).
- **Cards with fixed parameters need EMC2 >= 1d9fb41d** (`dev-flow` or `dev-nle-ondev`). A twin
  parameter held at a value in training (`fixed` in the spec, e.g. s = 1 and SZ = 0 for
  `ddm_svst0`) is recorded in the card; older EMC2 ignores it, so registration needs
  `exceptions = unlist(jsonlite::fromJSON(card)$fixed)` and `nle_verify.R` fails on
  the card.
- **Priors inside the training box.** A network returns its floor outside the box it
  was trained on, which truncates a posterior there; `nn_cell()` and `run_sbc()` refuse
  a prior whose mean ± 4 sd leaves it (`nn_in_box()`).
- **rt resolution.** When t0 is sampled, set `rt_resolution = 1/1000` (or finer) in
  `make_emc()`: EMC2's default 1/60 floors the response times and biases t0 by about
  −1.9 posterior sd at N = 100, for the analytic likelihood too; `make_emc()` warns
  when a neural-likelihood design does this.
- **A trained flow varies by seed and by checkpoint** as much as by architecture.
  Screen several checkpoints of a run (`nleflow steps`, then EMC2's score-bias check)
  before adopting one, as DDMnle and the RDM card were (the plan's sections 2 and 3).
- **Laptop runs at <= 8 cores.** `nleflow train` uses `min(8, cores)` threads; a full
  recipe (24 M rows, 300 epochs) is about 7 h on an Apple M2 Max.

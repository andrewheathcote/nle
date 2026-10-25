# Start here

What nleflow is, how to install and check it, where the cards
are, and which page to read for each task. Thirty minutes from install to your own
card in EMC2 and in ONNX: [TUTORIAL.md](TUTORIAL.md).

## What nleflow is

A Python package (JAX) that trains a **neural likelihood** for an evidence-accumulation
model from a short model specification (a `spec.yaml`; the worked ones are `nleflow/examples/<name>/spec.yaml`) and a simulator you write (R or
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
   That is the EMC2 on the `main` branch of https://github.com/andrewheathcote/DMC2 (private; ask for access):
   `git clone https://github.com/andrewheathcote/DMC2`. It reads a card's `fixed` field,
   ships the cards listed below and has the vignette "Using a trained flow".
   ```sh
   MAKEFLAGS=-j4 R CMD INSTALL --preclean --library=$HOME/R/nle-lib <EMC2 source>
   export R_LIBS=$HOME/R/nle-lib R_PROFILE_USER=/dev/null     # always both (a second EMC2 can shadow yours)
   ```
2. **Python**: `uv sync --frozen --group dev` in `nleflow/` (uv fetches Python 3.11).
3. **The one verification command** (R; installs nothing, checks every shipped card
   against its golden vectors, and any card you name):
   ```sh
   Rscript -e 'source(system.file("scripts", "nle_verify.R", package = "EMC2"))' [card.json ...]
   # nle_verify.R is part of the installed EMC2 (inst/scripts/); system.file finds it
   ```
   Check: one `PASS` line per card and `all checks passed`.
4. **The tests**: `R_PROFILE_USER=/dev/null uv run pytest -rs` in `nleflow/`; set
   `NLEFLOW_EMC2_LIB` (the EMC2 library; R_LIBS for its dependencies), `NLEFLOW_RLIB`
   and `NLEFLOW_EMC2_SRC` (an EMC2 source tree) to turn on the R cases.
5. **Or Docker**: `docker/Dockerfile` (R, EMC2 from the repository above or from a source tarball, the lock); `docker run <image> nle-verify`, `pytest`, `emc2-tests` ([install and check](docs/install-and-check.md), section 4).

## Where the cards are

The cards are in `nleCards/` at the root of this folder, each as
`<tag>_weights.json` with `<tag>_golden.csv` (and `_intermediates.json`) beside it;
`nleCards/SHA256SUMS` lists their checksums.

| card | model | kind | in EMC2 | notes |
|---|---|---|---|---|
| `ddm_svst0_e<N>` | DDM with sv and st0 free, s = 1, SZ = 0 fixed | `flow_joint` | no | `examples/ddm_svst0`; its README has the gates |
| `ddm_sc_s1106_e245` | the full DDM, scaled (DDMnle) | `flow_scaled` | yes, identical | the adopted DDM network; `examples/ddmnle` |
| `lba_e299` | LBA | `flow_race` | no | `examples/lba` |
| `rdm_ss_base_s801_e271` | RDM, A = 0 | `flow_race` | yes, identical | `examples/rdm_a0` |
| `rdm_rdm_small_e195` | RDM, small | `flow_race` | no | used by the golden-vector tests |
| `ddm_z0_s503_e282` | DDM, eight-parameter flow (earlier network) | `flow_joint` | no | used by the ONNX tests only |
| `lan_ddm_uniform_st` | HSSM's DDM with uniform st (a converted LAN) | `mlp_joint` | no | `examples/lan_in` converts the current file |
| `rtcon2_r3_s2_e006` (RTCONnn) | RTCON2 rating model, K response categories | `flow_cat` | yes (only there) | not an nleflow example |
| `crdm_aff_log_h2_k12_clip_100k_dt0.0005` (CRDMnn) | conflict racing diffusion model (CRDM), A = 0 | `flow_race` | yes (only there) | Lüken's flow; not an nleflow example |

**Why a card can be in both places.** EMC2 ships four cards in `inst/extdata/flownn/`:
`ddm_sc_s1106_e245_weights.json` (DDMnle, the single DDM flow covering all variability
cases), `rdm_ss_base_s801_e271_weights.json` (the RDM flow), `rtcon2_r3_s2_e006_weights.json`
(RTCONnn) and `crdm_aff_log_h2_k12_clip_100k_dt0.0005_weights.json` (CRDMnn, the conflict
racing diffusion model as a hybrid race: the Wald where the pulse is exactly 0, the flow
elsewhere). They load through `register_nn_model()` or the `DDMnn()`, `RDMnn()`, `RTCONnn()`
and `CRDMnn()` model functions (`DDMnn()` defaults to the DDMnle card). `nleCards/` holds
byte-identical copies of the first two so that nleflow's tests and examples run without
EMC2 (training, export and ONNX need only Python), plus the cards EMC2 does not ship.
`nleCards/SHA256SUMS` lets you check that a copy still matches EMC2's.

## A map

| to do this | read | run |
|---|---|---|
| the whole path once, on small data | [TUTORIAL.md](TUTORIAL.md) | `nleflow/examples/ddm_svst0/run_smoke.sh` |
| train a choice flow (DDM variants) | [docs/train-a-choice-flow.md](docs/train-a-choice-flow.md) | `examples/ddm_svst0`, `examples/ddmnle` |
| train a race flow (RDM, LBA) | [docs/train-a-race-flow.md](docs/train-a-race-flow.md) | `examples/lba`, `examples/rdm_a0` |
| use the shipped cards in EMC2: specify the RDM and RTCON2 models, and fit one participant with DDMnle against the standard DDM and a 1e-8 DDM (prior in the box, rt resolution, checks) | EMC2's `vignette("using-a-trained-flow")` (in the EMC2 repository above) | `inst/scripts/using_a_trained_flow_data.R` (reruns the fits) |
| export a card and check it in EMC2 | [docs/export-and-check-a-card.md](docs/export-and-check-a-card.md) | `nleflow/examples/<name>/check_in_emc2.R` |
| a card as ONNX, and check it | [docs/onnx-out.md](docs/onnx-out.md) | `nleflow onnx`, `nleflow onnx-check` |
| use the ONNX graph from Python | [examples/onnx_use](examples/onnx_use/README.md) | `examples/onnx_use/run.sh` |
| bring a published LAN into EMC2 | [docs/lan-in.md](docs/lan-in.md) | `examples/lan_in/run.sh` |
| the spec, the data contract, the card, the converter | [README.md](README.md) (reference) | |
| the card format (the contract with EMC2) | EMC2's `vignette("neural-likelihood-cards")` | |

## Known limits

- **ONNX is float32 when published, float64 when ours.** A LAN in float32 matches its
  float64 card to about 1e-5 in log density (the golden check's tolerance is 1e-4 and
  is recorded in the card); our graphs are float64 and match EMC2 to about 1e-13 (1e-8
  where st0 > 0 makes the density a difference of two CDFs).
- **Cards with fixed parameters need the EMC2 above.** A twin
  parameter held at a value in training (`fixed` in the spec, e.g. s = 1 and SZ = 0 for
  `ddm_svst0`) is recorded in the card; older EMC2 ignores it, so registration needs
  `exceptions = unlist(jsonlite::fromJSON(card)$fixed)` and `nle_verify.R` fails on
  the card.
- **Priors inside the training box.** A network returns its floor outside the box it
  was trained on, which truncates a posterior there; `nn_cell()` and `run_sbc()` refuse
  a prior whose mean ± 4 sd leaves it (`nn_in_box()`).
- **rt resolution: use 1/1000 or finer, and train at full precision.** Our recommendation
  is `rt_resolution = 1/1000` in `make_emc()`, which matches the resolution the shipped
  cards were trained and validated at. When you train a new network, keep the training
  response times at full precision: do not floor or round them (the example simulators
  write them unrounded), and validate the card at 1 ms or finer. At sampling time, the
  default `rt_resolution = 1/60` floors each RT to a 1/60 s grid. That removes the
  average delay a 60 Hz display adds to real data (Stevenson et al., "Throw your
  response times in the bin"), but on data without such a delay, such as simulations or
  recordings accurate to 1 ms, it shifts t0 down by about 8 ms (about −1.3 posterior sd
  at N = 100). The bias is in the data, not in the likelihood, so it holds for analytic
  and neural likelihoods alike; `make_emc()` warns when a neural-likelihood design
  samples t0 at a coarser resolution. Use a coarser resolution only if you know your RTs
  carry the noise it corrects, and match it to that noise.
- **The analytic DDM's precision is a speed setting.** WienR's precision defaults to 5e-3, which is up to 0.04
  off in the log density with sv, SZ and st0 all free; `design(model = DDM(precision = 1e-8), ...)` is the
  model to compare a card with (`design()` takes a model list, so models with options go in directly).
- **`RTCONnn()` is the way to use the RTCON2 card.** `register_nn_model()` on that card alone errors, as it has
  no analytic twin to take the parameterisation from; `RTCONnn()` supplies it.
- **A trained flow varies by seed and by checkpoint** as much as by architecture.
  Screen several checkpoints of a run (`nleflow steps`, then EMC2's score-bias check)
  before adopting one, as DDMnle and the RDM card were.
- **Laptop runs at <= 8 cores.** `nleflow train` uses `min(8, cores)` threads; a full
  recipe (24 M rows, 300 epochs) is about 7 h on an Apple M2 Max.

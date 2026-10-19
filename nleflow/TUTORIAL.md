# Tutorial: your own card, in EMC2 and in ONNX

About 30 minutes, of which the machine needs about @@MACHINE@@ (Apple M2 Max, 8 cores;
timed by following this page in a fresh clone). You train a small choice flow for
EMC2's DDM with drift and non-decision-time variability, check it, use it in EMC2,
export it to ONNX and use it from Python; then bring a published LAN into EMC2. The
network is trained on 10 % of the full recipe's data for 10 epochs: enough to see
every step work, not a card to fit data with (the full recipe is
`examples/ddm_svst0/run_full.sh`, about 7 h).

Every step prints a check line; this page shows what to expect. Before you start,
do [install and check](docs/install-and-check.md) (EMC2 in a library of its own, `uv
sync`), or read [START_HERE.md](START_HERE.md).

## 0. Where you are (1 minute)

From the repository root:

```sh
export NLE=$PWD/nleflow
export NLEFLOW_RLIB=$HOME/R/nle-lib      # your EMC2 library (dev-nle-ondev); a list a:b also works
export R_LIBS=$NLEFLOW_RLIB R_PROFILE_USER=/dev/null
NF="uv run --project $NLE nleflow"
mkdir -p ~/nle-tutorial && cd ~/nle-tutorial
cp $NLE/examples/ddm_svst0/{spec_smoke.yaml,simulate.R,check_simulator.R,check_in_emc2.R} .
```

`R_PROFILE_USER=/dev/null` and `R_LIBS` matter: a second EMC2 elsewhere on your R
path can be picked up instead of yours.

## 1. The spec (5 minutes reading)

Open `spec_smoke.yaml`. A spec is everything nleflow needs to know about a model:

- `model: DDM` names EMC2's analytic twin; `response: choice` makes this a choice
  flow (rt given the response, plus a classifier for P(response)).
- `context` lists the network's inputs, EMC2's parameter names on EMC2's sampled
  scale (`transforms`); `sv` and `st0` are **zero-box** inputs, exact at 0.
- `fixed: {s: 1, SZ: 0}` are parameters held at a value in training: not inputs, and
  recorded in the card so EMC2 uses those values.
- `box_natural` is the training box. Outside it the card returns its floor.
- `sampling` and `regimes` say how draws fill the box (here 30 % of draws have
  `sv = 0` exactly, 30 % `st0 = 0`).
- `spline`, `conditioner`, `classifier` and `training` are the network and its
  schedule; `simulator` is the command that writes the data (`simulate.R`, which
  uses EMC2's DDM and WienR).

Make it yours: a new name, a narrower drift box, 10 % of the data, 10 epochs.

```sh
sed -e 's/^name: .*/name: my_ddm/' \
    -e 's/^  v: \[-5, 5\]/  v: [-4, 4]/' \
    -e 's/^  rows: .*/  rows: 2400000/' -e 's/^  val_rows: .*/  val_rows: 100000/' \
    -e 's/^  choice_rows: .*/  choice_rows: 20000/' -e 's/^  choice_val_rows: .*/  choice_val_rows: 2000/' \
    -e 's/^  epochs: 2/  epochs: 10/' -e 's/^  keep: .*/  keep: 3/' \
    -e 's/data_smoke/data/g' spec_smoke.yaml > my_ddm.yaml
$NF spec my_ddm.yaml | grep -E '^name|rows|epochs|  v:'
```

Check: `name: my_ddm`, `v: [-4, 4]`, the new rows and `epochs: 10`. `nleflow spec`
validates the spec and prints it resolved; a mistake stops here with a message
naming the field.

## 2. Data (@@SIM@@)

```sh
$NF simulate --cores 8 my_ddm.yaml
Rscript check_simulator.R data/val
```

Check: `=== 2,400,000 rows ...; sv = 0 in 0.300 and st0 = 0 in 0.300 of draws` (and
the same for 100 000 validation rows), two `choice set` lines; then the simulator's
PIT, `pWDM(rt, R) / P(R)`, uniform (KS p not small, deciles near 0.100) overall and
per regime and response, and the `ll` column against `dWDM` to about 1e-5. This
tests your simulator, not the network: if the PIT is not uniform, the data are not
the model.

## 3. Train (@@TRAIN@@)

```sh
$NF train --cores 8 my_ddm.yaml
$NF steps runs/my_ddm
```

Check: ten `epoch` lines with the validation loss falling, `flow done`, then the
classifier's lines ending `classifier done: val mean |err| ...`; `steps` lists the
three kept checkpoints and marks the best.

## 4. Export the card (@@EXPORT@@)

```sh
$NF export runs/my_ddm
export CARD=$(ls $PWD/runs/my_ddm/export/my_ddm_e*_weights.json); echo $CARD
```

Check: `float64 gate: port vs JAX float64 ... (tolerance 1e-08)` — the card, evaluated
by the reference port of EMC2's code, reproduces the trained network — and three
`wrote` lines: the card (`_weights.json`), its golden rows (`_golden.csv`, the
network's own log densities at fixed points, which every consumer must reproduce)
and intermediates. The card is the one file EMC2 needs.

## 5. In EMC2 (@@EMC2@@)

The one verification command, on your card alone:

```sh
Rscript -e 'source(system.file("scripts", "nle_verify.R", package = "EMC2"))' $CARD
```

Check: a `PASS golden check` line for each card EMC2 ships and for yours, then `all
checks passed`. (With EMC2 older than 1d9fb41d it fails on this card: older EMC2 does
not read the card's `fixed`; see START_HERE's known limits.)

In R (started in this shell, so `CARD` is set), the card is a model like any other:

```r
library(EMC2)
card <- Sys.getenv("CARD")
m <- register_nn_model(card, twin = DDM)   # s = 1, SZ = 0 come from the card
nn_golden_check(m)
design_my <- design(factors = list(subjects = 1, S = c("left", "right")), Rlevels = c("left", "right"),
                    formula = list(v ~ S, a ~ 1, t0 ~ 1, Z ~ 1, sv ~ 1, st0 ~ 1), model = m)
```

(`twin = DDM` lets EMC2 compare the card with the analytic DDM.) Then the example's
check, which does that through a design and per trial against WienR:

```sh
Rscript check_in_emc2.R $CARD data/val cases.csv
```

Check: in R, `PASS golden check` and a design; from the script, `compiled vs R path ... max |diff|` below 1e-11 (EMC2's
compiled likelihood and its R path agree); the network against the analytic DDM
through the design and per trial on 1e5 held-out rows (`mean log p - log q` is the
KL estimate; after 10 epochs on 10 % of the data it is @@KL@@, against 0.0002 for the
full card); `wrote cases.csv` (EMC2's evaluation of the card, the ONNX reference).

When you sample with this design, set `rt_resolution = 1/1000` in `make_emc()`
(EMC2's default 1/60 biases t0), keep the prior inside the box (`nn_in_box()`), and
check calibration before trusting a posterior: `nn_cell()`, `nn_score_bias()`,
`run_sbc()` in EMC2's vignette "Neural likelihoods".

## 6. In ONNX (@@ONNX@@)

```sh
$NF onnx $CARD
for s in trialwise shared; do $NF onnx-check ${CARD%_weights.json}_$s.onnx $CARD --cases cases.csv; done
uv run --project $NLE python $NLE/examples/onnx_use/use_onnx.py $CARD
```

Check: two `wrote` lines (`_trialwise.onnx`, a parameter vector per trial;
`_shared.onnx`, one vector for a block of trials); four `PASS against 1e-10` lines
(each graph on the golden rows and on EMC2's cases, including which rows are −Inf
and which at the floor); then `use_onnx.py`, which uses the graphs with numpy and
onnxruntime only: the input layout, the density's mass (@@MASS@@ here; 1 for a
trained card), the golden rows `PASS` with both graphs, and the time per trial.
[examples/onnx_use](examples/onnx_use/README.md) explains the stored scale the graph
takes and shows the full card's numbers.

## 7. A published LAN into EMC2 (@@LAN@@)

```sh
$NLE/examples/lan_in/run.sh
```

Check: the download's sha256, the preset's two known versions, `PASS` for the
conversion (`card vs float64 graph` about 1e-14; vs the published float32 graph
about 5e-6), `PASS golden check`, `all checks passed`, and the LAN against EMC2's DDM:
per trial a median 0.037 log units, each wrong parameter mapping 10 to 61 times
worse. [examples/lan_in](examples/lan_in/README.md) reads the output.

## Next

- The full recipe for this model and its gates: [examples/ddm_svst0](examples/ddm_svst0/README.md)
  (`run_full.sh`, `gate_full.sh`).
- Your own model: [train a choice flow](docs/train-a-choice-flow.md) or [a race
  flow](docs/train-a-race-flow.md); the simulator contract is in [README.md](README.md).
- Several checkpoints, not one: a flow varies by seed and by checkpoint as much as by
  architecture; screen them (`nleflow steps`, EMC2's score-bias check) before adopting
  one.

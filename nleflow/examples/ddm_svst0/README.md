# The DDM (sv, st0) example: an eight-parameter choice flow

EMC2's DDM with drift variability `sv` and non-decision-time variability
`st0` free and start-point variability off, as a `flow_joint` card (no
scaling): a flow over rt given (v, a, t0, Z, sv, st0) and the response, and a
classifier for P(R = 2). `s = 1` and `SZ = 0` are held fixed in training
(`fixed` in the spec; the card records them). `sv` and `st0` are zero-box
inputs, exact at 0, with four regimes of exact zeros. The card registers with
`twin = DDM` and the fixed values as exceptions:

```r
card <- "ddm_svst0_e<N>_weights.json"
m <- register_nn_model(card, twin = DDM, exceptions = unlist(jsonlite::fromJSON(card)$fixed))  # s = 1, SZ = 0
```

| input | transform | box (natural) |
|---|---|---|
| v | identity | [-5, 5] (half the draws in [-2, 2]) |
| a | log | [0.5, 3] |
| t0 | log | [0.05, 0.6] |
| Z | probit | [0.2, 0.8] |
| sv | zero-box, c = 0.1 | [0, 3]; exactly 0 in 30 % of draws |
| st0 | zero-box, c = 0.01 | [0, 0.4]; exactly 0 in 30 % of draws |
| s, SZ | fixed | 1, 0 |

The nonzero `sv` and `st0` are half uniform, half asinh-uniform (dense near
0); the regimes are 55 % both free, 15 % `sv = 0`, 15 % `st0 = 0`, 15 % both
0. The spline runs over log rt in [-3.5, 3.5] (0.03 s to 33 s); the simulated
log rt lies in [-2.9, 2.5], so the identity tails carry nothing in the data.
When `st0 = 0` the card has density 0 for rt ≤ t0 (the zero-box DDM card's
support rule, applied by EMC2 and by the ONNX graph); for `st0 > 0` the flow
models the smear itself.

```sh
cd nleflow/examples/ddm_svst0
export NLEFLOW_RLIB=<library holding EMC2 and WienR> R_PROFILE_USER=/dev/null   # e.g. ~/R/nle-ondev-lib:~/R/nle-m4-lib
./run_smoke.sh          # 1 min: the whole chain below on 1 % of the data, 2 epochs
./run_full.sh           # the full recipe: simulate + 300 epochs, about 7.2 h at 8 cores
./gate_full.sh          # the full card's gates, 2 min (EMC2_FIXED_LIB=<a current EMC2> adds nle_verify.R on the card alone)
```

`run_smoke.sh` runs, in order:

| step | check |
|---|---|
| `nleflow spec spec_smoke.yaml` | validates |
| `nleflow simulate` | `=== 240,000 rows ...; sv = 0 in 0.30 and st0 = 0 in 0.30 of draws ...` and two `choice set` lines |
| `check_simulator.R` | the PIT `pWDM(rt, R) / P(R)` of the rows is uniform overall and per regime and response (KS p not small, deciles near 0.100); the `ll` column matches `dWDM` at precision 1e-10 to ~3e-6 |
| `nleflow train` | the flow's epoch lines, `classifier done` |
| `nleflow export` | `float64 gate: ... (tolerance 1e-08)`, three `wrote` lines |
| `nleflow golden` + `cmp` | the golden file rewritten from the card alone is identical |
| `nn_golden_check()` | `PASS golden check ... (flow_joint, 1150 rows)` |
| `check_in_emc2.R` | compiled vs R path below 1e-11; the network vs the analytic DDM through a design and per trial; writes `cases.csv` |
| `nle_verify.R` | passes with the current EMC2; fails with older EMC2 (below) |
| `nleflow onnx` + `onnx-check --cases` | both signatures `PASS` on the golden rows and on the EMC2 cases |

`check_simulator.R` tests the simulator, not the network: given the response,
`pWDM(rt | θ) / P(R | θ)` of every simulated row is uniform when the data
follow WienR's DDM. `check_in_emc2.R` registers the card, builds a
two-stimulus design through `nn_cell()` and compares the compiled likelihood
with the R path and with the analytic DDM over 50 prior draws; then, per
trial, the card's density with the analytic `ll` of up to 1e5 held-out rows
(the mean of `log p - log q` estimates the KL divergence), overall and by
regime, near the onset (rt − t0 < 0.05 s) and in the tail (rt > 3 s); with a
third argument it writes EMC2's evaluation over 200 prior draws × 100 trials
in each of the four regimes as the reference cases of `nleflow onnx-check
--cases`. The cases avoid rt = t0 exactly: the graph takes log t0, and
onnxruntime's batched Exp may differ from R's `exp` by an ulp, which decides
the tie at the discontinuous onset of an `st0 = 0` density either way (42 of
1600 such rows did so in the trialwise graph).

**`nle_verify.R` needs the current EMC2 for this card.** It registers a card with
the card alone. The current EMC2 takes the card's `fixed` values as
the default `exceptions` (and refuses an exception that contradicts them); older
EMC2 ignores them, so registration stops with "the network
was trained without 's' ... declare the value it was fixed at with exceptions =
c(s = <value>)". With older EMC2, register with the exceptions as above;
`run_smoke.sh` and `gate_full.sh` do.

On the laptop (Apple M2 Max, 8 cores), the smoke chain: data in 5 s, flow 2
epochs in 3 s, classifier in 2 s; float64 gate 5.1e-13 (log pdf), 8.0e-15 (z),
float32 gap 5.9e-5; `nn_golden_check` 3.1e-13; compiled vs R path 9.9e-14;
ONNX against the golden rows 2.9e-13 and against 79 200 EMC2 rows 2.8e-13
(both signatures); the simulator's PIT KS p 0.51 to 0.95. After 2 epochs on
1 % the card is far from the DDM (per trial |Δ log p| median 0.33): a pipeline
check, not a fit.

The full recipe is `spec.yaml`: 24 000 000 rows (2 400 000 draws × 10
trials), 1 000 000 validation rows, 300 epochs, 30 checkpoints kept, the
[256, 128] conditioner with 64 bins, the [256, 256, 256] classifier on 200 000
exact P(R = 2) targets (WienR's `pWDM`). On the laptop (8 cores) it took 7.2 h:
the simulation 4.1 min, the flow 424.8 min (about 81 s an epoch), the classifier
3.7 min.

## The full card: `ddm_svst0_e299`

In `nleCards/`: `ddm_svst0_e299_weights.json` (sha256
`67739e2dca9da692e0b241ea16389eadc5933d68e6873f535f47bef314a04374`), its golden file
and intermediates. The best checkpoint is the last (validation loss −0.21778; flat
to five digits from epoch 290). `gate_full.sh` (2 min), with an older EMC2 for the
EMC2 checks and the current one for the second `nle_verify.R`:

| gate | result |
|---|---|
| float64 gate (port vs JAX float64, 1116 points) | log pdf 9.08e-12, z 1.07e-14 (tolerance 1e-08); float32 gap 4.3e-3 (information) |
| golden from the card alone | identical (`cmp`) |
| `nn_golden_check()` | PASS, 1150 rows, 7.58e-12 against 1e-09 |
| compiled vs R path (50 draws, N = 200) | 9.95e-14, all finite |
| network vs analytic DDM through the design (50 draws × 200 trials) | mean \|Δ\| 2.79 (0.0139 a trial); max 118.5 is the one draw with t0 (0.403 s) above the fastest rt (0.380 s), where the DDM's density is 0 (floored) and the card's is not; the other 49 draws: mean 0.43, max 3.43 |
| per trial vs WienR, 1e5 held-out rows | mean log p − log q 0.0002 (KL); \|Δ\| median 0.0124, 90 % 0.0390, 99 % 0.0998 |
| — by regime | sv = 0: 0.0118 / 0.0915 (median / 99 %); st0 = 0: 0.0120 / 0.0976; both 0: 0.0110 / 0.0869; \|v\| > 3: 0.0150 / 0.1165 |
| — onset and tail | rt − t0 < 0.05 s: 0.0168 / 0.1755; rt > 3 s: 0.0363 / 0.3498 |
| `nle_verify.R`, EMC2 252e155f | FAIL: "the network was trained without 's'" (expected, above) |
| `nle_verify.R`, current EMC2 | `all checks passed` (the card: 7.58e-12) |
| ONNX, both signatures, golden rows | 1106 rows above the floor: 3.31e-12; −Inf and floor rows agree; PASS against 1e-10 |
| ONNX, both signatures, EMC2 cases | 79 200 rows: 8.19e-11; −Inf and floor rows agree; PASS against 1e-10 |
| simulator PIT (1e5 rows of data/val) | KS p 0.096 to 0.934 overall, per regime and response; `ll` vs `dWDM` 6.4e-06 |

The card is close to the DDM per trial (99 % of held-out rows within 0.1 log
units, the KL estimate 2e-4) and worst in the far tail (rt > 3 s, 1.7 % of rows).
These gates check the network and its transport into EMC2 and ONNX, not inference:
the card records no validation, and a posterior check (`nn_cell()`,
`nn_score_bias()`, `run_sbc()`) is the next step before it is used to fit data.

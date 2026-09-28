# Prompt for the EMC2 (dev-nle) session: Malte Lüken's race flows and the hybrid CRDM race

Written 2026-09-28 by the NLE session (stage M3 of
`~/Downloads/julian/nle/handover/MALTE_STAGES.md`). Hand the block below to the
EMC2 session as-is. Compute: a laptop. Every acceptance test here runs in
seconds to a few minutes at 8 cores; no server, no SBC measurement.

---

dev-nle takes the shipped networks and the zero-box DDM flow
(`ZEROBOX_REGISTRATION_DONE.md`). Two more kinds of network are coming, both
from Malte Lüken (`github.com/maltelueken/eamax`,
`.../neural-race-model-estimation`): **RDM flows** in his configuration, and a
**CRDM flow**, a racing diffusion in which one accumulator's mean path carries
a conflict pulse, `M(t) = v t + amp (e t / tau) exp(-t / tau)`. The CRDM has no
analytic likelihood and no EMC2 model. `register_nn_model()` cannot take either
card today: it stops at "the card has no context_transforms". Add the five
things below, in dev-nle's own idiom, with the gate at the end as the
acceptance test.

Trained checkpoints have been requested from Malte and have not arrived. The
work does not wait for them: three **fixture cards** with randomly initialised
weights exist now, converted by the converter the trained ones will go
through. Their densities model nothing. The point is that dev-nle must equal
the NLE project's R port on them, exactly.

## The specification is code, not this text

All under `~/Downloads/julian/nle` (branch `dev-sbc-calibration`). Where this
prompt and the code disagree, the code is right.

| what | file | validated |
|---|---|---|
| flow evaluator, R | `port/R/flow_race.R` (`flow_load`, `flow_knots`, `flow_eval`) | vs his `evaluate_pdf_sf`, float64: log pdf ≤ 3.6e-11, log sf ≤ 7.4e-13 (`port/tests/validate_malte.R`) |
| fixture cards and golden vectors | `malte/fixtures/{rdm_final,rdm_plain,crdm_final}/{card.json,golden.csv}` | — |
| card writer | `flow/scripts/malte_to_card.py` | refuses what it cannot export exactly |
| CRDM simulator and Volterra density, C++ | `port/src/crdm.cpp` (`crdm_mean`, `rcrdm_acc`, `crdm_volterra_grid`, `dcrdm_volterra`) | vs his solver ≤ 3.7e-13; simulator vs his draws KS p ≥ 0.33 (`port/tests/validate_crdm.R`, data in `malte/crdm_ref/`) |
| hybrid accumulator, race assembly, CRDM twin, in plain R | `sbc/malte_reference.R` (`malte_ref_acc`, `race_ref_ll`, `CRDM_ref`) | simulator vs quadrature race at 2e5 trials: choice within 0.7 se, KS p ≥ 0.22, total mass 1.00000 |
| the gate | `sbc/gate_malte_dev_nle.R` | self-test passes on today's dev-nle; mock run 7/7 |

## What the cards are

`flow_race` cards (one spline flow for one accumulator's finishing time), with
fields the shipped `rdm_small` does not have:

| field | value | meaning |
|---|---|---|
| `context_names` | `v, s, b` (RDM), `v, amp, tau, s, b` (CRDM) | network inputs, **natural scale** |
| `context_scale`, `parameter_scale` | `"natural"` | there is no `context_transforms` and no `bounds_sampled` |
| `bounds_natural` | RDM `v [0, 8]`, `s, b [0.25, 3.5]`; CRDM `v [0, 8]`, `amp [0, 1]`, `tau [0.01, 0.5]`, `s, b [0.25, 3]` | training box, in `context_names` order |
| `input_scaling` | `eps`, `loc`, `scale`, one per input | the network sees `(log(x + eps) - loc) / scale`; there is no `scaler`. Absent in `rdm_plain`, which takes the natural values as they are |
| `affine`, `affine_scale` | `true`; `min_scale`, `offset` | two more conditioner outputs after the `3K + 1` spline outputs, `loc` and `raw_scale`; see below |
| `pre` | `"t0"` | the flow models decision time `rt - t0`; `t0` is not an input |
| `mlp` | one hidden layer (`rdm_plain`) or two (`*_final`), width 128, `gelu_tanh`, no LayerNorm | output width `3K + 1`, or `3K + 3` with `affine` |
| `spline` | 12 bins, range [-5, 5], `boundary_slopes = "identity"` | as `rdm_small` otherwise |
| `model` | `rdm_final`, `rdm_plain`, `crdm_final` | not the name of an EMC2 model; the twin is passed |

His flows have **no start-point variability**: `A = 0` exactly, so the
network's threshold input `b` is EMC2's `B + A = B`.

## 1. `flow_race` cards with `input_scaling`, `affine`, natural-scale context, `pre` from the card

- **Natural-scale context.** A card with `context_scale = "natural"` and no
  `context_transforms` has transform `"identity"` for every input, and
  `bounds_sampled` equal to `bounds_natural`. The parameter keeps the twin's
  transform on EMC2's sampled scale (log for `v`, `B`, `s`), and the network
  is fed the natural value, as for zero-box. `nn_tf(x, "identity")` already
  does this; what is missing is reading it from the card.
- **Input scaling**, inside the evaluator, in place of the `scaler`:
  `h_j = (log(x_j + eps_j) - loc_j) / scale_j`. Refuse a card that has both
  `input_scaling` and `scaler`. With neither (`rdm_plain`) the network sees
  the natural values. `flow_build()` currently requires `scaler` and
  `bounds_sampled`.
- **Affine stage.** With `K` bins the last two outputs are
  `loc = raw[3K + 2]` and `scale = softplus(raw[3K + 3] + offset) + min_scale`
  (both constants from the card's `affine_scale`). With `y = (log t - loc) /
  scale` and `z = g^{-1}(y)` (identity tails in `y`):
  `log pdf = dnorm(z, log) + log|dg^{-1}/dy| - log(scale) - log t`,
  `cdf = pnorm(z)`. `flow_build()` currently requires output width `3K + 1`.
- **`pre` from the card.** `register_nn_model()`'s `pre` defaults to the
  card's `pre`; an argument that contradicts the card is refused. The compiled
  path already handles a named `pre`.
- **The box is closed at `amp = 0`**, where the CRDM flow was trained and
  where the model is the plain RDM. The other edges stay as they are.
- **`b` and `B`.** The gate registers with `pars = c("v", "s", "B")`, that is,
  the network's `b` input is EMC2's `B`, and `exceptions = c(A = 0)`. That is
  exact only because `A = 0`. Either keep the rename and refuse any other
  value of `A` for these cards, or feed the network the twin's `Ttransform`
  column `b`; the gate's call must work either way. A registration that leaves
  `A` unfixed must be refused (it is today: "the network would ignore them").

## 2. The hybrid race

`register_nn_model(..., hybrid = "amp")`: a row whose natural-scale `amp` is
**exactly 0** is an unpulsed accumulator and is scored by EMC2's own Wald
(`v`, `b = B`, `s`, `A = 0`, decision time `rt - t0`); every other row goes to
the network. The network never sees the unpulsed rows, so they carry no
approximation error and cost no conditioner pass. Without `hybrid` the same
card is the **all-flow** route (every row to the network, `amp = 0` included),
which stays registrable: it is the fallback and the nesting check.

- Every path: `dfun`, `pfun`, single and trial-wise parameters, and the
  **compiled likelihood** (`c_name = "NN"`). The SBC cells need the compiled
  path; `wald_functions.h` is already in `src/`.
- `hybrid` naming a parameter that is not a network input is refused.
- Record it in the registration as `nn$hybrid` (`"amp"`, or `NULL`).
- Not tested by the gate, recommended: apply the box on `v`, `s`, `b` to the
  Wald rows too, so that hybrid and all-flow reject the same parameter vectors.

## 3. The design

Which accumulator is pulsed is a property of the trial, as which accumulator
matches is. No new design machinery is needed on the R side. This already
works on today's dev-nle with the analytic `RDM` standing in:

```r
design(factors = list(subjects = 1, S = c("left", "right"), D = c("left", "right")),
       Rlevels = c("left", "right"), matchfun = function(d) d$S == d$lR,
       functions = list(lD = function(d) factor(d$D == d$lR, levels = c(FALSE, TRUE))),
       formula = list(v ~ lM, B ~ 1, t0 ~ 1, tau ~ 1, amp ~ 0 + lD),
       constants = c(amp_lDFALSE = -Inf), model = <the hybrid model>)
```

`mapped_pars()` gives `amp` exactly 0 on the `lDFALSE` rows and the sampled
value on the others. What is needed:

- **`amp`** has transform `exp`, default `-Inf` (exactly zero: no pulse, the
  model is the RDM), bounds `[0, 1]` from the card with a bound exception at
  0, as zero-box has for `sv`, `SZ`, `st0`.
- **The compiled mapping must give the same exact zeros.** Under dummy coding
  the `lDTRUE` rows multiply the `-Inf` constant by 0. The R mapping handles
  it; the gate's check 4 compares the compiled likelihood with the reference
  on exactly this design, so a `NaN` there shows up as a floored likelihood.
- **`nn_cell()` takes `functions =`** and passes it to both designs (network
  and control). Today it has no such argument.
- `nn_cell()`'s box check must treat the exact zero of `amp` as inside.

## 4. `CRDM()`: the hand-written twin, exported

An EMC2 model function, type `"RACE"`, parameters `v, B, A, t0, s, amp, tau`.
`sbc/malte_reference.R::CRDM_ref` is a working R implementation with the
defaults, transforms and bounds to copy; the package version should be the
same model with the loops in C++ (`port/src/crdm.cpp` is standalone Rcpp and
can be taken as it is).

- `CRDM(dt = 5e-4, t_max = 3, sim_dt = 5e-4)`, callable with no arguments so
  that it can be a twin and a control. A cheaper control is
  `function() CRDM(dt = 0.002)`.
- **`rfun`**: unpulsed accumulators are exact Wald draws. A pulsed accumulator
  is simulated (`rcrdm_acc`) only as far as the fastest unpulsed accumulator
  of its trial, because it matters only if it finishes first. No horizon is
  assumed and nothing is redrawn. Do not replace this by "simulate to `t_max`
  and redraw the trials without a finisher": a redraw that is not conditioned
  on the first attempt having failed puts too much mass early. A trial in
  which every accumulator is pulsed is simulated to `t_max` and must produce a
  finisher, else stop. `set.seed()` must reproduce the data (`rcrdm_acc` uses
  R's generator).
- **`dfun`, `pfun`**: the Volterra density and CDF, **one solve per distinct
  parameter row per call**, interpolated at that row's decision times. The
  M2 functions take scalar parameters and solve once per call, so the twin
  groups trials by `(v, amp, tau, s, b)` and calls once per group; rows of a
  group are not adjacent (winners and losers interleave). Rows with `amp`
  exactly 0 are the closed-form Wald. Decision time ≤ 0 gives 0.
  The grid must cover each group's largest decision time
  (`dcrdm_volterra` holds its end values outside the grid); the recursion is
  causal, so extending the grid leaves earlier values unchanged.
- `A` admits only 0. `amp = 0` and `v = 0` are bound exceptions.
- Cost and accuracy of one solve at `t_max = 3`, for the documentation:

  | dt | time | max error of the density, relative to its peak |
  |---|---|---|
  | 0.002 | 5 ms | 5.9e-3 |
  | 0.001 | 22 ms | 2.1e-3 |
  | 0.0005 | 86 ms | 5.6e-4 |

## 5. Documentation and tests

A card-contract paragraph in `?register_nn_model` (natural-scale context,
`input_scaling`, `affine`, `pre` from the card, `hybrid`), a help page for
`CRDM`, and package tests on the three fixtures with golden values from the R
port, so that a later evaluator change cannot silently break them.

## Traps the NLE session has already paid for

- **`amp = 0` is not evidence about the Volterra solver.** There the recursion
  sum vanishes and the density is the Wald term itself, at any `dt`. Test the
  solver at `amp > 0`.
- **Do not compare against jax's `logsf` beyond z = 20.** It is wrong by up to
  3.7e-9 there; `golden.csv` carries `log_sf_exact` for those points.
- **All three fixture cards are called `card.json`.** `basename()` makes their
  labels identical in messages; the registry keys by path, so nothing
  collides, but a label from the card's `model` field would read better.
- **The shipped `rdm_small` is checkpoint 195.** `port/export/` also holds
  e207 and others; they are different networks.
- **Two EMC2 installs.** The gate stops unless the EMC2 it loaded is the one
  under the library it was given.

## Acceptance

```
cd ~/Downloads/julian/nle
Rscript sbc/gate_malte_dev_nle.R ~/R/nle-sbc-lib        # 9 lines; all must pass
Rscript sbc/gate_zerobox_dev_nle.R ~/R/nle-sbc-lib      # must stay 3/3
Rscript sbc/gate_dev_nle.R ~/R/nle-sbc-lib              # must stay 5/5
```

On today's dev-nle the first prints 1 passed, 8 failed: check 0 passes and
check 1a stops at the missing `context_transforms`.
`GATE_MALTE_MOCK=1 Rscript sbc/gate_malte_dev_nle.R` runs the same checks with
the R references standing in for dev-nle, which shows what each check expects.

The calls the gate makes, which fix the interface:

```r
register_nn_model(card, pars = c("v", "s", "B"), twin = RDM, exceptions = c(A = 0))          # rdm_final, rdm_plain
register_nn_model(card, pars = c("v", "amp", "tau", "s", "B"), twin = CRDM,
                  exceptions = c(A = 0), hybrid = "amp")                                     # hybrid
register_nn_model(card, pars = c("v", "amp", "tau", "s", "B"), twin = CRDM, exceptions = c(A = 0))   # all-flow
nn_cell(model, mean =, sd =, formula =, constants = c(amp_lDFALSE = -Inf),
        factors =, Rlevels =, matchfun =, functions = list(lD = ...))
```

| check | passes when |
|---|---|
| 0 | the gate's reference race equals dev-nle's compiled likelihood on the shipped `rdm_small` (passes today, 1.0e-13) |
| 1a | the four registrations succeed |
| 1b | `nn$pre == "t0"`; all transforms `identity`; `c_name == "NN"`; `A` bounded to 0; `amp` default `-Inf`, exception 0, upper bound 1; `nn$hybrid`; three refusals (A unfixed, `hybrid` on a card without `amp`, `pre` contradicting the card) |
| 2 | `dfun`, `pfun` equal the R port on `rdm_final` and `rdm_plain`: log pdf < 1e-7, CDF < 1e-8; exactly 0 for `rt <= t0` and outside the box |
| 3 | the same for `crdm_final`, hybrid (Wald where `amp == 0`) and all-flow, with pulsed and unpulsed rows interleaved |
| 4 | `mapped_pars()` shows exact zeros and `A = 0`; the likelihood the sampler uses equals the reference race to 1e-7 per trial, hybrid CRDM and `rdm_final` |
| 5a | `CRDM()`'s `dfun`, `pfun` equal `dcrdm_volterra` at dt 5e-4 to 1e-10, in under 5 s for 6 parameter rows × 40 trials (one solve per trial would take 28 s) |
| 5b | `make_data()`: choice proportion within 4 se of the quadrature race and KS p > 0.001, congruent and incongruent |
| 6 | `nn_cell(functions = )` builds network and control designs; a 2-replicate `nn_sbc_cell()` completes for the hybrid CRDM and for `rdm_final` |

Checks 0 and 4 call `EMC2:::nn_dadm`, `EMC2:::nn_trialwise` and
`EMC2:::get_pars_matrix_oo`. If one is renamed, say so in the completion note.

Reinstall to `~/R/nle-sbc-lib` when done, write
`SBC-staging/2026-09-24_dev-nle_phase6/MALTE_FLOWS_DONE.md` in the form of
`ZEROBOX_REGISTRATION_DONE.md` (commit, what changed, the gate's numbers), and
say so.

---

## What the NLE session will then do

Re-run the gate, then stages M4 to M6 of `MALTE_STAGES.md`: register the
trained cards when Malte sends them, score bias and posterior shift for both
RDM flows, the RDM SBC cells on tux26, and the CRDM tests against the Volterra
control. That compute is the NLE project's, not the EMC2 session's.

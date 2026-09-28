# Malte RDM / CRDM work: stages, models, prompts

Companion to `handover/MALTE_CRDM_PLAN.md` (the what and why). This file is the
**how, in stages that each start from a cleared context.** Every stage reads
its inputs from files, writes its result to files, and ends by appending one
line to the log at the bottom. Nothing depends on a conversation.

Decisions taken (AH, 2026-09-28): hybrid race in dev-nle; real-data study
deferred (so the fitting parameterisation is moot for now); checkpoints
requested from Malte by email.

## Rules that save more tokens than the model choice does

- **Start each stage with `/clear`, set the model, paste the stage's prompt.**
  The prompt names the files to read; do not re-read the repositories.
- **Every script prints its acceptance test as one line** (`PASS max|diff|
  3e-13`). Read that line, not the log.
- **Long jobs run in the background with a watcher that prints a table.**
- **Absolute paths.** The shell's working directory resets between calls, and
  the port exists in two copies (`port/` at the repo root and
  `handover/nle-methods/port/`); patch both, check `git status` on both.
- **Limits:** 8 cores on the laptop; 72 on tux26. Commit after each stage.
- Memory (`MEMORY.md`) carries the standing traps — dual EMC2 install, EMC2's
  own port copy, the st0-default trap — and is loaded automatically.

## Stages

| # | stage | model | needs | blocked on |
|---|---|---|---|---|
| M1 | Converter, golden vectors, R port additions | **Sonnet** | Malte's code only (random-initialised conditioners are enough for an exactness check) | nothing |
| M2 | CRDM simulator and Volterra reference in C++ (standalone Rcpp), validated against `eamax` | **Sonnet** | `eamax` | nothing |
| M3 | Prompt for the EMC2 session + the NLE-side acceptance gate for it | **Fable / Opus** | M1, M2 | nothing else |
| — | *EMC2 session implements M3's prompt in dev-nle* | (its own session) | M3 | — |
| M4 | Register trained cards; score bias and posterior shift for both RDM flows | **Sonnet** to adapt scripts, **Haiku** to run | trained checkpoints; dev-nle with M3 | Malte; EMC2 session |
| M5 | RDM SBC cells on tux26 (A = 0 cell, N = 100 and 400, both flows + control) | **Haiku** | M4 | tux26 free |
| M6 | CRDM tests: nesting check, score bias and posterior shift against Volterra, network SBC cell, step-size sensitivity, one Volterra control cell at 250 replicates | **Sonnet** to adapt, **Haiku** to run | M3 done in dev-nle; CRDM checkpoints | compute after 1 Oct |
| M7 | Seed study: our pipeline with the affine stage and decision-time option, four seeds per arm, screened | **Fable / Opus** for the training-code change, **Haiku** to run | M1 | compute after 1 Oct |
| M8 | Read M4–M7 against the registered expectations; write the comparison section | **Fable / Opus** | M4–M7 | — |
| Z | Zero-box v2: export → screen → select → four SBC cells → read | **Haiku** to run, **Fable / Opus** to read the cells | tux26 pipeline marker | training (~09:00 CEST 29 Sep) |

M1 and M2 are independent and can run in either order or in parallel
sessions. M3 should not start before both have passed, because their outputs
are its specification.

---

## Stage M1 — converter, golden vectors, R port  (Sonnet)

**Deliverables**

1. `malte/` at the repo root: a `uv` virtual environment (Python 3.11) with
   `eamax` v0.1.1 and `confrdm` installed from clones of
   `github.com/maltelueken/eamax` and
   `github.com/maltelueken/neural-race-model-estimation` (branch
   `development`). Our `flow/.venv` has jax 0.5.3 and `eamax` needs jax ≥ 0.9,
   so it must be a separate environment. CPU only.
2. `malte/make_fixtures.py`: builds **randomly initialised** conditioners in
   three configurations, saves each with its sidecar through his
   `save_conditioner`, and writes float64 golden vectors (`log_pdf`, `log_sf`
   at ≥ 200 (θ, t) points spanning the box interior, the edges, both spline
   tails and the identity region) from **his** `evaluate_pdf_sf` under
   `jax_enable_x64`:
   - RDM final: context `v, s, b`; affine; log inputs
     (eps `v 0.05, s 0.05, b 0.05`; box `v [0, 8]`, `s, b [0.25, 3.5]`);
     two hidden layers; 12 bins; width 128;
   - RDM plain: context `v, s, b`; one hidden layer; 12 bins; width 128;
     natural-scale inputs;
   - CRDM final: context `v, amp, tau, s, b`; affine; log inputs (eps
     `v 0.05, amp 0.01, tau 0.005, s 0.05, b 0.05`; box `v [0, 8]`,
     `amp [0, 1]`, `tau [0.01, 0.5]`, `s, b [0.25, 3]`); two hidden layers;
     12 bins; width 128.
   Random weights make the densities meaningless and the check exact, which is
   the point: trained checkpoints will go through the same converter.
3. `flow/scripts/malte_to_card.py`: Orbax checkpoint + sidecar → our JSON card
   (the format `port/R/flow_race.R` loads), with these additions recorded in
   the card: `affine` (true/false; when true the last two conditioner outputs
   are `loc` and `raw_scale`, `scale = softplus(raw_scale + c) + 1e-3` with
   `c = log(expm1(1 − 1e-3))`, and `log t = loc + scale · y` after the
   spline); `input_scaling` (`eps`, `loc`, `scale` per input; the network
   sees `(log(x + eps) − loc) / scale`; absent for natural-scale inputs);
   `pre = "t0"` (the flow models decision time `rt − t0`); `context_scale =
   "natural"`; `bounds_natural` from the training box; spline `range`,
   `boundary_slopes`, `min_bin_size`; hidden layers in order (`linear1`,
   `linear_mid1`, …, `linear2`); activation `gelu_tanh` (flax `nnx.gelu`
   defaults to the tanh approximation — verified).
4. `port/R/flow_race.R` **and** `handover/nle-methods/port/R/flow_race.R`:
   evaluate such a card — input scaling, any number of hidden layers, the
   affine stage, identity tails in the standardised coordinate. Existing cards
   (`rdm_small`) must evaluate bit-identically to before.
5. `port/tests/validate_malte.R`: R port against the golden vectors for all
   three fixtures, and `rdm_small` against its existing golden file.

**Acceptance (one line each):** R port vs golden max |diff| < 1e-10 on
`log_pdf` and `log_sf`, all three fixtures; `rdm_small` unchanged to 0.

**Out of scope:** the compiled evaluator, dev-nle, anything CRDM-specific
beyond the five-input flow, `boundary_slopes = "unconstrained"` (his final
configurations use `identity`; the converter must **refuse** anything else
rather than export it wrongly).

### Prompt for M1

```
Read handover/MALTE_STAGES.md, section "Stage M1", and do exactly that stage.
Background, only if you need it: handover/MALTE_CRDM_PLAN.md section 1. His
flow code is confrdm_jax/flows_affine.py in the paper repository and
eamax/flows/model.py; our reference evaluator is port/R/flow_race.R. Do not
read beyond those three files and the ones the stage names.

Work in /Users/andrew/Downloads/julian/nle with absolute paths; at most 8
cores. When the acceptance lines pass, commit, append one line to the log at
the bottom of handover/MALTE_STAGES.md (date, what passed, the numbers), and
stop. If an acceptance test fails and the cause is not a port bug you can
fix, stop and report what differs and where, with the smallest reproducing
case.
```

---

## Stage M2 — CRDM simulator and Volterra reference  (Sonnet)

**Deliverables**

1. `port/src/crdm.cpp` (Rcpp, standalone; mirrored in
   `handover/nle-methods/port/src/`), exporting:
   - `crdm_mean(t, v, amp, tau)`: `v t + amp (e t / tau) exp(−t / tau)`
     (gamma shape 2), and its derivative;
   - `rcrdm_acc(n, v, amp, tau, s, b, dt, t_max)`: first-passage times of
     one pulsed accumulator — exact mean path plus Brownian increments on the
     grid, Brownian-bridge crossing correction, within-step dequantisation,
     `Inf` when there is no crossing by `t_max` (his
     `first_passage_from_mean`);
   - `dcrdm_volterra(t, v, amp, tau, s, b, dt, t_max)`: density and CDF of one
     pulsed accumulator by the Fortet recursion (his `solve_volterra_fpt`),
     **solved once per call** on the grid and interpolated at `t` — the caller
     passes all trials that share a parameter row in one call.
2. `malte/make_crdm_reference.py` (uses the M1 environment): golden values of
   his `solve_volterra_fpt` density and CDF at dt = 0.002 and 0.0005 for ~20
   parameter sets across the CRDM box, and 1e5 draws per set from his
   simulator for five of them.
3. `port/tests/validate_crdm.R`.

**Acceptance (one line each):** `dcrdm_volterra` vs his solver at the same dt,
max relative diff < 1e-8 where density > 1e-12; at `amp = 0` and dt = 0.0005,
vs the closed-form Wald density (`statmod` or our own), max relative diff
< 2e-3 in the central 99% of mass (this is the solver's discretisation, and
the number is the result); convergence in dt reported as a three-row table;
simulator vs his draws, two-sample KS p > 0.01 on all five sets, and vs
`dcrdm_volterra`'s CDF, KS p > 0.01; timing of one `dcrdm_volterra` call at
dt = 0.002, 0.001, 0.0005.

### Prompt for M2

```
Read handover/MALTE_STAGES.md, section "Stage M2", and do exactly that stage.
His reference implementations are eamax/accumulators/pulse.py, volterra.py
and diffusion.py (clone under malte/ from stage M1; if malte/ does not exist,
create the environment as stage M1 item 1 describes). Read only those.

Work in /Users/andrew/Downloads/julian/nle with absolute paths; at most 8
cores. When the acceptance lines pass, commit, append one line with the
numbers to the log in handover/MALTE_STAGES.md, and stop. If the Volterra
solver does not converge to the Wald at amp = 0 within the stated tolerance,
do not loosen the tolerance: report the table and stop.
```

---

## Stage M3 — the EMC2 session's prompt and its acceptance gate  (Fable / Opus)

**Deliverables**

1. `~/Documents/Projects/EMCpackage/SBC-staging/2026-09-24_dev-nle_phase6/MALTE_FLOWS_PROMPT.md`:
   what dev-nle must add, in its own idiom, with M1's R port and M2's C++ as
   the specification —
   (i) `flow_race` cards with `affine`, `input_scaling`, natural-scale
   context and `pre = "t0"`;
   (ii) the **hybrid race**: a per-accumulator switch so that the network
   scores the pulsed accumulator and EMC2's own Wald the other;
   (iii) the design: a distractor-match factor `lD` beside `lM`, `amp ~ 0 +
   lD` with the unpulsed level a named constant at exact zero (sampled scale
   −Inf; `p_types`, bound exception and transform as for zero-box);
   (iv) a hand-written twin for CRDM with `rfun` = the M2 simulator racing a
   Wald, and the M2 Volterra density as the reference `dfun` for controls,
   hoisted to one solve per distinct parameter row.
2. `sbc/gate_malte_dev_nle.R`, the NLE-side gate, runnable on the three M1
   fixtures before any trained card exists: registration; `dfun` equals the R
   port (< 1e-7) for RDM-final, RDM-plain and the hybrid CRDM race, with `amp`
   exactly zero on the unpulsed accumulator; `mapped_pars()` shows exact zeros
   and `A = 0`; the twin's `rfun` and `make_data()` run; a 2-replicate
   `nn_sbc_cell()` completes.

### Prompt for M3

```
Read handover/MALTE_STAGES.md section "Stage M3", handover/MALTE_CRDM_PLAN.md
section 4.1, and the log at the bottom of MALTE_STAGES.md for what M1 and M2
delivered. Read the zero-box prompt
(~/Documents/Projects/EMCpackage/SBC-staging/2026-09-24_dev-nle_phase6/ZEROBOX_REGISTRATION_PROMPT.md)
and sbc/gate_zerobox_dev_nle.R as the pattern for both deliverables, and
dev-nle's R/nn_register.R and src/flow_race.cpp
(~/Documents/Projects/EMCpackage/EMCrelease/EMC2-nle) for what exists. Write
the two deliverables. The gate will fail until the EMC2 session has done its
part: run it only far enough to confirm that it fails for that reason and no
other. Commit, log one line, and give me the one-sentence instruction to hand
the EMC2 session.
```

---

## Stages M4 – M8

Prompts are written when their inputs exist (trained checkpoints; dev-nle
passing the M3 gate), because they depend on what M1–M3 actually delivered.
Each will follow the same form: read this file's log, do one stage, print
the acceptance lines, log, stop. Registered expectations are in
`MALTE_CRDM_PLAN.md` §3.3 and must not be edited after M4 begins.

---

## Stage Z — zero-box v2, from wherever it is  (Haiku to run; Fable / Opus to read)

State lives on tux26 and in `handover/ZERO_BOX_RETRAIN_PLAN.md` §11 and its
progress log. The tux26 pipeline (`flow/scripts/zerobox_v2_pipeline_tux.sh`)
runs training then classifier and export by itself and writes
`~/emc2nn/nle/handover/nle-methods/zb2_pipeline.marker` when the cards are
out.

### Prompt for Z (run part; Haiku)

```
Read handover/ZERO_BOX_RETRAIN_PLAN.md section 11 and the last five lines of
its progress log. Then:
1. ssh andrew@tux26psy.fmg.uva.nl and check
   ~/emc2nn/nle/handover/nle-methods/zb2_pipeline.marker and zb2_pipeline.out.
   If the marker is absent and training is still running, report the epoch
   and the expected finish and stop.
2. If present: from /Users/andrew/Downloads/julian/nle run
   PREFIX=zb2 nohup bash sbc/zerobox_screen_all.sh > sbc/zerobox_screen_all.out 2>&1 &
   and wait for sbc/zerobox_screen/done.marker. Print
   sbc/zerobox_screen/selection.out and the posterior-shift lines from
   sbc/zerobox_screen_all.out.
3. The cards are already on tux26 (no copying). There, from ~/emc2nn/nle, run
   python3 sbc/zerobox_card_normalise.py handover/nle-methods/port/export/ddm_zb2_*_weights.json
   (idempotent), then
   sbc/gate_zerobox_dev_nle.R ~/emc2nn/nle/.Rlibs-nle <selected card>, and
   only if it prints 3 passed launch, from ~/emc2nn/nle,
   OUT=sbc/nci/results_zerobox_v2 CARD=<selected card> setsid nohup bash sbc/zerobox_s4_launch.sh > sbc/zerobox_s4_v2.out 2>&1 &
   (two cells at a time at 36 cores = 72).
4. When sbc/nci/results_zerobox_v2/done.marker exists, print each cell's
   results/summary.csv as a table, copy the four summary.csv files into
   sbc/nci/results_zerobox_v2/<regime>/ locally, append the numbers to the
   plan's progress log, commit, and stop. Do not interpret the results.
Use absolute paths; at most 8 cores locally.
```

The reading of those four cells against v1 and against the acceptance rule in
§11 is a separate, short Fable / Opus session: "Read
handover/ZERO_BOX_RETRAIN_PLAN.md §11 and its progress log and tell me whether
v2 replaces the shipped flow, ships beside it, or needs the gradient-clipping
lever (MALTE_CRDM_PLAN.md §2 item 4)."

---

## Log

- 2026-09-28 plan and stages written; email to Malte drafted
  (`~/Downloads/julian/EMAIL_TO_MALTE.md`). Zero-box v2 training at epoch 25
  of 300 on tux26, 6.2 min per epoch.

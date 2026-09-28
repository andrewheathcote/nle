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
- **Every stage ends by handing over the next one.** After the commit and the
  log line, the final report names the next stage (from the table below, given
  what has passed), the **model** to set, and the **full prompt** to paste,
  ready to copy, adjusted for anything the stage just learned (paths, changed
  tolerances, traps found). If the next stage is blocked, say on what. Do not
  wait to be asked. (AH, 2026-09-28.)
- Memory (`MEMORY.md`) carries the standing traps — dual EMC2 install, EMC2's
  own port copy, the st0-default trap — and is loaded automatically.

## Stages

| # | stage | model | needs | blocked on |
|---|---|---|---|---|
| M1 | Converter, golden vectors, R port additions | **Sonnet** | Malte's code only (random-initialised conditioners are enough for an exactness check) | nothing |
| M2 | CRDM simulator and Volterra reference in C++ (standalone Rcpp), validated against `eamax` | **Sonnet** | `eamax` | nothing |
| M3 | Prompt for the EMC2 session + the NLE-side acceptance gate for it | **Fable / Opus** | M1, M2 | nothing else |
| — | *EMC2 session implements M3's prompt in dev-nle* | **Fable** (its own session; AH, 2026-09-28) | M3 | **done 2026-09-28**, dev-nle 14c12c79 (pushed to origin/dev-nle) |
| M4 | Register trained cards; score bias and posterior shift for both RDM flows | **Sonnet** to adapt scripts, **Haiku** to run | trained checkpoints; dev-nle with M3 | Malte; EMC2 session |
| T | **tux26 library**: install dev-nle 14c12c79 into a new library, run the three gates there | **Haiku** | dev-nle pushed (done) | nothing; must precede any Malte run on tux26 (M5, M6, M7b screens) |
| M5 | RDM SBC cells on tux26 (A = 0 cell, N = 100 and 400, both flows + control) | **Haiku** | M4, **T** | tux26 free |
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
and diffusion.py (clones and the uv environment are already in malte/ from
stage M1: activate malte/.venv; malte/README.md has the rebuild steps if it is
missing). Read only those.

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

## Stage M7a — seed study: the training-code change  (Fable / Opus)

Written 2026-09-28, when its inputs existed (M1; dev-nle 14c12c79 evaluates affine and
decision-time race cards). M7a is the code and a laptop smoke test. The 20 training runs
(M7b, Haiku) wait for compute after 1 Oct.

**Deliverables**

1. `flow/src/flow/config.py`, `flow/scripts/train_race.py`: two options for the RDM flow,
   off by default so that existing tags train bit-identically:
   - `--affine`: two more conditioner outputs after the `3K + 1` spline outputs, `loc` and
     `raw_scale`, `scale = softplus(raw_scale + c) + 1e-3`, `c = log(expm1(1 - 1e-3))`,
     `log t = loc + scale * spline(z)`. Exactly the card semantics of
     `port/R/flow_race.R` (`affine`, `affine_scale`), so the export needs no new evaluator.
   - `--decision-time`: `t0` leaves the context and the flow is trained on `rt - t0`
     (card field `pre = "t0"`), as `train_ddm.py --decision-time` does for the DDM.
2. `flow/scripts/export_flow.py`: writes `affine`, `affine_scale`, `pre`; its golden
   vectors and port check cover both options. Context stays ours (`v, B, [t0,] s, A`, log
   scale, `scaler`), so an arm differs from the baseline in one thing only.
3. `sbc/seed_study_launch_tux.sh` and `sbc/seed_study_screen.sh`: five arms x four seeds
   (baseline; + affine; + decision-time; + both; his configuration as published, trained
   with his code in `malte/.venv` and exported by `flow/scripts/malte_to_card.py`), 30
   retained checkpoints per run, every retained checkpoint screened with
   `sbc/score_bias_race.R`. Output: one table of `|g|` per arm, seed, checkpoint, parameter.
4. The criterion, registered in the launch script's header before any number exists: an
   arm helps if its median `|g|` is below the baseline's best quartile
   (`MALTE_CRDM_PLAN.md` section 3.2, B8).

**Acceptance (one line each), laptop, 8 cores:** default options reproduce an existing
tag's first-epoch loss to the last digit; for each of the four of our arms a 2-epoch smoke
run on a 1% subset exports, passes the port golden check (< 1e-10 in float64), and
registers in dev-nle with `dfun` equal to the R port (< 1e-7); the screen script runs on
the four smoke cards and prints its table.

### Prompt for M7a

```
Read handover/MALTE_STAGES.md section "Stage M7a" and its log, and
handover/MALTE_CRDM_PLAN.md section 2 rows 1 and 2 and section 3.2 row B8. Do
exactly that stage. Our code: flow/src/flow/config.py, flow/src/flow/training.py,
flow/scripts/train_race.py, flow/scripts/export_flow.py; the decision-time
precedent is flow/scripts/train_ddm.py and export_flow_ddm.py (search
"decision_time"); the evaluator that defines the card is port/R/flow_race.R; his
affine stage is confrdm_jax/flows_affine.py under
malte/neural-race-model-estimation. Read only those.

Traps already paid for: train_race.py's CONTEXT order must equal
export_flow.py's MODEL_REGISTRY order (a transposed A and s gave a score bias
500x too large with a healthy loss); EMC2 has its own copy of the port, so
check dev-nle against port/R/flow_race.R and never against itself; prepend
~/R/nle-sbc-lib (dev-nle 14c12c79) to .libPaths and assert find.package("EMC2")
is under it; the shipped rdm_small is checkpoint 195, not e207; the port exists
in two copies (port/ and handover/nle-methods/port/), patch both.

Do not start the 20 training runs: compute is tux26-only until 1 Oct and tux26
is committed to zero-box v2. Work in /Users/andrew/Downloads/julian/nle with
absolute paths; at most 8 cores. When the acceptance lines pass, commit, append
one line with the numbers to the log, and end by giving the model and full
prompt for M7b (Haiku: launch, watch, screen, print the table, do not
interpret).
```

---

## Stage T — dev-nle for the Malte work on tux26  (Haiku)

tux26's `~/emc2nn/nle/.Rlibs-nle` holds dev-nle d49d190b, the library stage Z's cells were
gated on. It cannot register Malte's cards. **Do not update it.** The Malte work gets its own
library, `~/emc2nn/nle/.Rlibs-malte`, built from origin/dev-nle at 14c12c79 or later.
Every M5, M6 and M7b prompt must name `.Rlibs-malte` and must start by checking
`~/emc2nn/nle/handover/nle-methods/rlibs_malte.marker`; if the marker is absent, stage T
runs first. Whoever writes those prompts copies this rule into them.

### Prompt for T

```
On tux26 (ssh andrew@tux26psy.fmg.uva.nl), from ~/emc2nn/nle:
1. git pull so that sbc/gate_malte_dev_nle.R, sbc/malte_reference.R,
   port/src/crdm.cpp and malte/fixtures are present. If the pull would
   overwrite local changes, stop and report them.
2. Install EMC2 from github ampl-psych/EMC2, branch dev-nle, into the NEW
   library ~/emc2nn/nle/.Rlibs-malte (create it; remotes::install_github with
   lib = that path, or clone and R CMD INSTALL --library=that path). Do NOT
   install into ~/emc2nn/nle/.Rlibs-nle and do not touch it: zero-box cells
   run on it. Report the installed commit; it must be 14c12c79 or a
   descendant.
3. Run, each with the new library as its argument, and print every PASS/FAIL
   line and the final counts:
     Rscript sbc/gate_malte_dev_nle.R   ~/emc2nn/nle/.Rlibs-malte
     Rscript sbc/gate_zerobox_dev_nle.R ~/emc2nn/nle/.Rlibs-malte <the zero-box card used on tux26>
     Rscript sbc/gate_dev_nle.R         ~/emc2nn/nle/.Rlibs-malte
   Expected: 9/9, 3/3, 5/5. Use at most 8 cores if zero-box cells are
   running (check with top first).
4. Only if all three pass, write the date, the commit and the three counts
   to ~/emc2nn/nle/handover/nle-methods/rlibs_malte.marker, append one line
   to the log in handover/MALTE_STAGES.md, commit, and stop. If anything
   fails, print the error text and stop; do not edit a gate or the package.
```

---

## Stages M4 – M8

Prompts are written when their inputs exist (trained checkpoints; dev-nle
passing the M3 gate), because they depend on what M1–M3 actually delivered.
Each will follow the same form: read this file's log, do one stage, print
the acceptance lines, log, stop. Registered expectations are in
`MALTE_CRDM_PLAN.md` §3.3 and must not be edited after M4 begins.
Any of them that runs on tux26 uses `~/emc2nn/nle/.Rlibs-malte` and requires stage T's marker.

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
- 2026-09-28 M1 PASS. `malte/` uv env (py3.11, jax 0.10.2, flax 0.12.8; eamax 6d7f2ff, confrdm fae8add),
  `malte/make_fixtures.py`, `flow/scripts/malte_to_card.py` (refuses non-identity boundary slopes and a box
  that does not reproduce the recorded input scaling), affine / input-scaling / n-hidden support in both
  copies of `port/R/flow_race.R`, `port/tests/validate_malte.R`. R vs his `evaluate_pdf_sf`, float64, max|diff|:
  log_pdf 7.3e-12 (rdm_final), 3.4e-11 (rdm_plain), 3.6e-11 (crdm_final); log_sf <= 7.4e-13 for z < 20.
  rdm_small (e207) vs golden 6.8e-13 and vs the pre-extension evaluator 0 (bit-identical). Note: jax's
  `norm.logsf` is wrong by up to 3.7e-9 for z >= 20 (survival < e-87; R and scipy agree with each other);
  there the port is held to scipy's exact value (golden.csv `log_sf_exact`), matches to <= 7.7e-12.
  Irrelevant to any likelihood, but do not compare trained flows to jax logsf beyond z = 20.

- 2026-09-28 M2 PASS. `port/src/crdm.cpp` (+ mirror in handover/nle-methods/port/src; the two `flow_ddm.cpp` copies already differed before M2, untouched),
  `malte/make_crdm_reference.py` -> `malte/crdm_ref/` (20 Sobol sets in the crdm_final box, 3 with amp = 0; dt 0.002 and 0.0005, t_max 3; 1e5 draws x 5 sets at dt 0.001),
  `port/tests/validate_crdm.R`. `dcrdm_volterra` vs his `solve_volterra_fpt` (float64), max rel diff pdf 9.3e-14 (dt 0.002), 3.7e-13 (dt 0.0005); cdf <= 1.3e-14.
  amp = 0 vs closed-form Wald: 4.0e-14 -- exact by construction, not a discretisation test (the kernel factor -v + dM/dt vanishes, so the recursion sum is 0 and the
  density is the Wald term itself; same in his solver). Real convergence, amp > 0, vs dt 0.00025, max |pdf err|/peak: 5.9e-3 (dt 0.002), 2.1e-3 (0.001), 5.6e-4 (0.0005),
  order ~1.5-1.9; max |cdf err| 1.8e-2, 2.6e-3, 6.1e-4. Simulator vs his draws KS p >= 0.33; vs Volterra CDF (dt 0.0005) KS p >= 0.084 (all five sets).
  Timing per call, t_max 3: 5 ms (dt 0.002), 22 ms (0.001), 86 ms (0.0005). dcrdm_volterra holds end values outside [dt, t_max] like his jnp.interp.

- 2026-09-28 M3 DONE (deliverables written; the gate fails on today's dev-nle, as it must). Prompt:
  `~/Documents/Projects/EMCpackage/SBC-staging/2026-09-24_dev-nle_phase6/MALTE_FLOWS_PROMPT.md` (copy in `handover/MALTE_FLOWS_PROMPT.md`).
  Gate: `sbc/gate_malte_dev_nle.R`, references in `sbc/malte_reference.R` (hybrid accumulator, race assembly, `CRDM_ref` twin in plain R).
  On dev-nle d49d190b (`~/R/nle-sbc-lib`): 1 passed, 8 failed; check 0 (gate's reference race vs compiled likelihood on shipped rdm_small) 1.0e-13;
  check 1a stops at "the card has no context_transforms", 5a at "dev-nle exports no CRDM()"; no other error (`sbc/gate_malte_dev_nle_pre.out`).
  Mock run (`GATE_MALTE_MOCK=1`, R references stand in for dev-nle): 7 passed, 2 skipped (`sbc/gate_malte_mock.out`). Reference simulator vs
  quadrature race at 2e5 trials: P(pulsed wins) z = +0.69 / -0.26, KS p >= 0.22, mass 1.00000. Zero-box gate unchanged, 3/3.
  Found: shipped rdm_small is checkpoint 195, not e207; design(functions = lD) with amp_lDFALSE = -Inf already maps to exact zeros in R
  (the compiled mapping is untested until check 4 runs); nn_cell() has no `functions` argument. Interface fixed by the gate:
  `pars = c("v","s","B")` with `exceptions = c(A = 0)`, `hybrid = "amp"`, exported `CRDM(dt, t_max, sim_dt)`.
- 2026-09-28 Model for the EMC2 session that implements `MALTE_FLOWS_PROMPT.md`: **Fable** (AH). Reason: compiled-likelihood changes
  (hybrid switch in the batched native path, exact zeros in the compiled mapping) where an error is a small bias, not a crash, and
  requirements in the prompt that the gate does not test (box on Wald rows, refusal of A != 0, package tests).
- 2026-09-28 EMC2 session DONE, verified from the NLE side. dev-nle 14c12c79 (local, not pushed) installed in `~/R/nle-sbc-lib`
  (built 04:57 UTC). `gate_malte_dev_nle.R` 9/9 (`sbc/gate_malte_dev_nle_post.out`): check 0 1.0e-13; 2: log pdf 6.6e-12 / 8.7e-12,
  cdf <= 1.6e-13; 3: hybrid 6.4e-13 (Wald rows vs closed form 1.1e-13), all-flow 6.0e-12; 4: compiled vs reference race 1.2e-13 and
  1.5e-13, exact zeros; 5a: 5.9e-16 of peak, 0.43 s; 5b: .829 vs .817 and .510 vs .497, KS p .417 / .846; 6: both SBC cells complete.
  Zero-box gate 3/3, usability gate 5/5. Full package suite against the installed library, run here because the EMC2 session had not
  re-run it: 1143 expectations, 0 failures, 94 skipped (tests marked CRAN-only and unconfigured cards).
  One change outside the prompt: the training-region check holds a value built from several sampled parameters to +-k sd of the
  combination, sqrt(sum sd^2), not the linear sum; caused by this gate's check-6 prior (v sd .15 + v_lMTRUE sd .1 reached 8.96 > 8).
  Read and judged sound: the mapped value's prior is normal with that sd, so an effect now gets the same +-4 sd as an intercept;
  intercept-only cells are unchanged. Awaiting AH's decision to keep or revert.
  Traps for M5: EMC2's analytic `pWald` overflows for v b / s^2 > 354, which is inside his RDM box (v 8, b 3.5, s .25 gives 448), and
  the RDM control uses it, so keep s fixed at 1 in the comparison cell (v b <= 28); hybrid Wald rows use a log-form Wald
  (`src/nle_wald.h`) and differ from `pWald` by up to 2.5e-10. tux26: install 14c12c79 into a NEW library directory, not
  `.Rlibs-nle`, so stage Z's cells run on the library they were gated on. Next: M7a (prompt above); M4-M6 wait for Malte's checkpoints.
- 2026-09-28 Decisions (AH): **keep** the new training-region rule (sd of the combination); dev-nle 14c12c79 **pushed** to
  origin/dev-nle (ampl-psych/EMC2). M7a cleared to start. Note: running the package suite with `load_package = "installed"` deletes
  `tests/testthat/_snaps/RACE.md` in the worktree (testthat removes snapshots of skipped tests); it was restored before the push.
- 2026-09-28 Stage T added (AH asked that the tux26 update be part of the plan, not a note): new library `.Rlibs-malte`,
  three gates, marker file; M5 depends on it in the table and every tux26 prompt must check the marker.


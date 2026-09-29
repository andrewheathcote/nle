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
| M4 | **Re-scoped 2026-09-29: the deciding stage for the RDM.** Density error, score bias and posterior shift against the Wald, A = 0 cell, s = 1: his final flow, his plain flow, our flow selected from the seed study's baseline arm. Criteria in "Stages M4 – M6, re-scoped" below | **Fable / Opus** | trained checkpoints (arrived 2026-09-29); baseline arm trained | nothing; laptop, 8 cores |
| G | **Gadi library**: copy the bundle `SBC-pending/2026-10-01_gadi_dev-nle`, install dev-nle 14c12c79 into `~/R/nle-lib`, run `check_nle_lib.pbs`; required before any cell: `TESTS expectations 783 failed 0 errors 0` and mean log-likelihoods 131.286 / 131.325 / 131.359 | **Haiku** | SSH key (works since 2026-09-29) | **not before 1 Oct** (allocation) |
| T | **tux26 library**: install dev-nle 14c12c79 into a new library, run the three gates there | **Haiku** | dev-nle pushed (done) | **done 2026-09-28**; redo for any later dev-nle commit |
| M5 | RDM SBC cells **on Gadi** (not tux26): our selected flow with its analytic control, A = 0 cell and standard cell, N = 100 and 400. **His RDM flows are in M5 only if M4's rule says NOT SEPARATED** | **Haiku** to run, **Fable / Opus** to read | M4, **G** | Gadi allocation, from 1 Oct |
| M6 | CRDM tests: nesting check, score bias and posterior shift against Volterra, step-size sensitivity (laptop); network SBC cell, Volterra control **250 replicates at dt 2e-3 plus 50 at dt 5e-4** (Gadi) | **Sonnet** to adapt, **Haiku** to run | CRDM cards (exported 2026-09-29), **G** | Gadi allocation, from 1 Oct |
| M7 | Seed study: our pipeline with the affine stage and decision-time option, four seeds per arm, screened; **since 2026-09-28 on the A = 0 box** (`RDM_A0_BOX_PLAN.md`), so its baseline arm is the new RDM flow | **Fable / Opus** for the training-code change (M7a, **done 2026-09-28**), **Haiku** to run (M7b) | M1 | M7b: **wave 1 of 5 runs launched 2026-09-28 08:46 CEST**; the other 11 + Malte's 4 need cores (stage Z overlap) |
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

## Stage M7b — seed study: launch, watch, screen, print  (Haiku)

Written 2026-09-28 at the end of M7a; **revised the same day** after AH's decisions to train
on the A = 0 box and to start on tux26's idle cores (`handover/RDM_A0_BOX_PLAN.md`, section 9
has the state as run). **The prompt below supersedes the first version.** What differs from
the text under "State of tux26" and "Timing": the code and the A0 data are on tux26 in
`~/emc2nn/nle-malte`, the tux26 smoke has passed, a run takes about 14 to 18 h at 7 cores,
and wave 1 (`ss_base_s801..804`, `ss_aff_s801`) is running. Still absent on tux26: Malte's
python environment.

Everything it runs was smoke-tested end to end on the laptop (five arms, one seed, 2 epochs
on 1% of the data, his arm 300 steps): `sbc/seed_study/m7asmoke/` (old box) and
`sbc/seed_study/a0smoke/` (A = 0 box).

**What M7a delivered, as M7b uses it**

- `sbc/seed_study_launch_tux.sh`: 20 runs, 8 at a time at 9 cores (72). Its header holds the
  registered criterion; do not edit it. It has a preflight that names what is missing and
  starts nothing, never restarts a complete run, and writes
  `sbc/seed_study/ss/train_done.marker` only when 20 of 20 are complete.
- `sbc/seed_study_screen.sh`: export, card check, score-bias screen in two cells, table.
  `sbc/seed_study_table.R` applies the registered rule and prints the verdict lines.
- Tags: `ss_<arm>_s<seed>`; arms `base aff dt both malte`; our seeds 801-804 (also the
  batch-order seed), his 2026-2029. Cards: `port/export/rdm_ss_<arm>_s<seed>_e<checkpoint>`.
- His arm is `malte/train_conditioner_ckpt.py` in Malte's environment, which now needs
  `hydra-core` (`malte/README.md`).

**State of tux26 on 2026-09-28 (read-only look):** `~/emc2nn/nle/flow/.venv` exists (python
3.12, jax 0.5.3) but its `flow` package is `~/emc2nn/nle/flow/src`, which predates M7a, so
the scripts put their own root's `flow/src` first on `PYTHONPATH` and the preflight asserts
it. Absent on tux26: the RDM training data (`data/train/rdm16m`, 1.8 GB), the RDM validation
data, `sbc/nci/results/RDMnn_N100_R500.rds`, and any Malte python environment. `uv` is at
`~/.local/bin/uv`; `taskset` exists; 96 cores.

**Timing, measured on the laptop at 8 threads:** our epoch on the full data 203 s, so 300
epochs are about 17 h per run there; his 100k steps about 45 min. The screen costs about
6 s per export and about 1 min per run and cell at 8 cores.

**Acceptance (one line each):** preflight passed; the tux26 smoke prints `0 failing line(s)`
and both controls below 1e-3; `20 of 20 runs complete`; the screen prints the summary and the
four `PRIMARY VERDICTS` lines; `table.csv` and `summary.csv` are on the laptop and committed.

### Prompt for M7b

Revised 2026-09-28 12:00 CEST: **every run of the study is running or queued on tux26**, so
this stage launches no training. **Amended 2026-09-29 (AH): it starts at step 3**, because the
baseline arm was screened on the laptop in M4. It waits, screens once, and prints.

```
Read handover/MALTE_STAGES.md section "Stage M7b" and the last six entries of
its log. Do exactly that stage: wait, screen, print the table. Do not
interpret the numbers, do not edit any script, do not change the criterion,
seeds, arms or defaults. Work from /Users/andrew/Downloads/julian/nle on the
laptop (absolute paths) and over ssh andrew@tux26psy.fmg.uva.nl, where
everything runs from ~/emc2nn/nle-malte with
  export PY=~/emc2nn/nle/flow/.venv/bin/python LIB=~/emc2nn/nle/.Rlibs-malte

Rules. Start no training run, delete nothing under results/, kill nothing.
The only way to ask how training stands is
  STATUS=1 bash sbc/seed_study_launch_tux.sh
which starts nothing and ends with a line
  STATUS c complete, r running, n not started of 20
Ignore the "x of y runs complete" lines in sbc/seed_study_launch_wave*.out:
the runs were launched in five overlapping batches and those counts are per
batch. Malte's four runs were complete on 29 Sep. Do not ask questions; if
something fails, print the evidence named below, write it to the log, and
stop. Judge how busy tux26 is from the idle figure of `top -bn1 | grep Cpu`,
never from the load average.

Steps 1 and 2 are DONE and must not be repeated (amended 2026-09-29): the
baseline arm finished training at 02:40 CEST on 29 Sep and was screened on
the laptop in stage M4 (log entry "M4 part E"). Its results are on the laptop
and committed: sbc/seed_study/ss/table_base.csv and summary_base.csv. Do not
overwrite or delete those two files. Start at step 3.

3. Wait for the whole study. Poll in the background, not by hand, e.g.
     until ssh andrew@tux26psy.fmg.uva.nl 'cd ~/emc2nn/nle-malte &&
       export PY=~/emc2nn/nle/flow/.venv/bin/python &&
       STATUS=1 bash sbc/seed_study_launch_tux.sh |
         grep -q "^STATUS 20 complete, 0 running, 0 not started of 20"';
       do sleep 600; done
   On 29 Sep at 04:19 CEST six runs were still training (epochs 214 to 292
   of 300), so expect completion that morning. If a run shows "not started
   or stopped" with more than 0 and fewer than 300 epochs, it died: print the
   last 30 lines of sbc/seed_study/ss/logs/<run>.log and stop. Then, and only
   then, run
     bash sbc/seed_study_launch_tux.sh > sbc/seed_study_launch_final.out 2>&1
   It finds every run complete, starts nothing, prints the table of 20 runs
   and writes sbc/seed_study/ss/train_done.marker. Required: "20 of 20 runs
   complete". If it prints "start", something was incomplete: say so in the
   log and wait for it to finish.

4. Full screen (about 2 h):
     CORES=32 nohup bash sbc/seed_study_screen.sh > sbc/seed_study_screen.out 2>&1 &
   Use CORES=8 if `pgrep -fa "zerobox_s4|nn_sbc"` shows stage Z's cells
   running. Done when sbc/seed_study/ss/screen_done.marker exists.

5. Print, verbatim and without comment: sbc/seed_study_launch_final.out from
   the run table to the end, and sbc/seed_study_screen.out from the line
   "=== 4. table ===" to the end. Copy sbc/seed_study/ss/table.csv,
   summary.csv, card_check.out, failed_exports.txt and the .out files to the
   laptop's sbc/seed_study/ss/ and commit them. Append one line to the log in
   handover/MALTE_STAGES.md: runs complete, failed exports per arm, the two
   controls, seconds per epoch per arm (last epoch in each run's log), and
   the four PRIMARY VERDICTS lines as printed.

6. The baseline arm is screened again here, on tux26, as part of the full
   screen. Report in the log line whether it agrees with the laptop screen:
   the number of failed exports in arm base (laptop: 1,
   rdm_ss_base_s804_e279) and the base rows of the summary against
   sbc/seed_study/ss/summary_base.csv. A difference is reported, not fixed.
7. End by naming the next stage, its model and its full prompt. Our flow was
   already selected in M4 (rdm_ss_base_s801_e271); what follows this stage is
   the reading of the five arms against the registered rule, and whether the
   shipped flow should be selected from another arm (Fable / Opus). Say which
   of G, M5 and M6 are still open, from the log.
```

---

## Stage T — dev-nle for the Malte work on tux26  (Haiku)

tux26's `~/emc2nn/nle/.Rlibs-nle` holds dev-nle d49d190b, the library stage Z's cells were
gated on. It cannot register Malte's cards. **Do not update it.** The Malte work gets its own
library, `~/emc2nn/nle/.Rlibs-malte`, built from origin/dev-nle at 14c12c79 or later.
Every M5, M6 and M7b prompt must name `.Rlibs-malte` and must start by checking
`~/emc2nn/nle/handover/nle-methods/rlibs_malte.marker`; if the marker is absent, stage T
runs first. Whoever writes those prompts copies this rule into them.

**As run on 2026-09-28, which corrects the prompt below:** `~/emc2nn/nle` and the EMC2 source on
tux26 are plain copies, not git checkouts, so there is nothing to pull. dev-nle went over as
`git archive 14c12c79` to `~/emc2nn/src/EMC2-nle-14c12c79` and was installed with
`R CMD INSTALL --library=~/emc2nn/nle/.Rlibs-malte`. tux26's `port/R/flow_race.R` predates M1, and
stage Z's gate reads it, so it was **not** overwritten: the Malte files (M1 port, `crdm.cpp`,
fixtures, reference data, the gate) live in a separate root, **`~/emc2nn/nle-malte`**, laid out
like the repo. Malte runs on tux26 start from that root with `.Rlibs-malte`; new scripts and cards
are copied into it, never into `~/emc2nn/nle`.

### Prompt for T (superseded in steps 1 and 2 by the note above)

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

## Stages M4 – M6, re-scoped 2026-09-29 (AH's decisions; registered before any M4 number)

Source: `~/Documents/Projects/EMCpackage/SBC-staging/2026-09-29_malte_trained/PLAN.md` section 5.
This section replaces the M4 – M6 rows of `MALTE_CRDM_PLAN.md` section 3.2 (B4 – B6) where they
differ. Section 3.3 of that file (the registered expectations) stands unedited.

**What was known when this was written** (so the registration can be read for what it is).
Known: the density error of *his* flows over their whole boxes and on his own grid (part B of
this stage, log below), including a quick look in the region of the cell (s = 1, b 0.6 to 1.6,
v 1 to 4: KS median .0010 for his final flow, .0035 for his plain flow). Not known: anything
about *our* flow (no checkpoint of the baseline arm had been exported or screened: M7b's
screen had not run), and no score bias or posterior shift of any of the three flows in this cell.

### M4: is our RDM flow as good as or better than his?

**Cell.** The A = 0 comparison cell: `v ~ lM`, `B ~ 1`, `t0 ~ 1` sampled (v, v_lMTRUE, B, t0),
s constant at 1, A = 0 exactly (truth, control and every flow; his flows take b = B). Prior
means log 1.5, 0.4, log 1, log 0.2, sd 0.1 on the sampled scale. Parameter draws: the 500
stored draws of `sbc/nci/results/RDMnn_N100_R500.rds` with A dropped, as in the seed-study screen.
Reference: the closed-form Wald, EMC2's from `~/R/nle-sbc-lib` (dev-nle 14c12c79, asserted).

**Flows.** HF: his final flow (`malte_rdm_aff_log_h2_k12_clip_100k`). HP: his plain flow
(`malte_rdm_plain_h1_k12_noclip_500k`). O: ours, selected as follows.

**Selection of O, fixed before it is compared with anything.** The 120 retained checkpoints
of `ss_base_s801..804` go through `sbc/seed_study_screen.sh` (`ARMS=base`), which measures
score bias on **draws 1 to 200** in both cells (std and A0). O is the checkpoint with the
smallest W = max over both cells and all parameters (t0 included) of |g_p|. The tests of
`RDM_A0_BOX_PLAN.md` section 4 (density error flat in A, nesting at A = 0, the standard cell
no worse than `rdm_b1k`) are reported for O; a failure is reported and does not change the
selection. Carried beside it, outside the verdict: the checkpoint of median W (rank 60 of 120),
to show what an unselected run gives. His flows are single runs, ours is one of 120: that is
the procedure we would ship, and it is said wherever the comparison is reported.

**Measures. All on draws 201 to 400, which the selection never saw.**

| | measure | how | scalar used in the rule | floor |
|---|---|---|---|---|
| D | density error | per accumulator (two per draw, 400 parameter sets), decision-time grid of 4000 log-spaced points between the Wald's 1e-10 and 1 - 1e-10 quantiles, trapezoid, censored ends: KL(Wald \|\| flow), KS, and E_true \|log f_flow - log f\| | median KL over the 400 sets (90% point, maximum and KS reported) | 1e-4 nats per trial |
| G | score bias | `sbc/score_bias_race.R`, `SBR_CELL=A0`, H = 0.01, t0 by the restricted-region estimator relative to the analytic value; analytic control must be below 1e-3 | max over v, v_lMTRUE, B, t0 of \|g_p\| | 0.005 |
| P | posterior shift | 200 datasets of N = 400 trials simulated from the Wald race at prior draws (seeds 7001 to 7200), the **same datasets for every flow**; defensive mixture of the Laplace approximations of the analytic and of each flow's posterior (covariances times 1.5^2), 4000 draws, importance weights to each posterior. Per parameter: mean over datasets of (flow posterior mean - analytic posterior mean) / analytic posterior sd. Control: the analytic posterior's predicted standardised bias within +-0.15 for every parameter (2 se at 200 datasets; `MALTE_CRDM_PLAN.md` B5 registered +-0.15 at 48 to 60 datasets, where it is 1.2 se and fails by chance two times in three, hence 200) | max over the four parameters of \|mean shift\| | 0.10 posterior sd |

Floors: below them a difference is immaterial. 1e-4 nats per trial is 0.04 nats on 400 trials;
a shift of 0.10 sd is below what the screen resolves against SBC (+-0.15, memory
`regression-baseline-sbc-result`) and far below the 0.5 gate; |g| 0.005 is the level of our
shipped RDM flow in its own cell (0.004).

**Reading one measure** (X is D, G or P; O against one of his flows H; se of X_O - X_H from
2000 bootstrap resamples of draws or datasets, paired, because the scalars are medians and maxima):

- **tie** if both X_O and X_H are below the floor, or |X_O - X_H| <= 2 se;
- **ours better** if X_O < X_H by more than 2 se and X_H is above the floor;
- **ours worse** if X_O > X_H by more than 2 se and X_O is above the floor.

**Verdict, O against HF (primary) and O against HP (secondary):**

| the three readings | verdict | SBC? |
|---|---|---|
| no "ours worse" | **OURS AS GOOD OR BETTER** | no |
| no "ours better", and P reads "ours worse" | **HIS BETTER** | no; M8 reports it, and the remedy is sought in the seed study's other arms |
| anything else (a "worse" beside a "better"; or "worse" on D or G only, P a tie) | **NOT SEPARATED** | yes |
| a screen is invalid: G's control >= 1e-3, P's control outside +-0.15, or median importance ESS below 200 of 4000 for O or H | **NOT SEPARATED** | yes |

P carries the second row because it is the only one of the three that measures what inference
returns (prior included, all parameters moving together); D and G are its causes, not its
substitutes (`PAPER_REVIEW.md` items 3 and 6).

**The SBC cell, if the rule calls for one:** the A = 0 cell, N = 400, 250 replicates, for O
and for the flow of his it was not separated from, with one analytic control (the cell is the
same, so one control serves both). On Gadi, from 1 October. Read by standardised bias against
the 0.5 gate and KS on the ranks; ours is as good or better if its worst |standardised bias|
is not larger than his by more than 0.13 (2 se of a difference at 250 replicates).

**Acceptance of M4** (what it must print, whatever the verdict): cards' SHA256; the three
controls; the table of D, G, P with se for HF, HP, O and the median checkpoint; the two
verdict lines, `VERDICT O vs HF: ...` and `VERDICT O vs HP: ...`.

### M5: RDM SBC cells

His RDM flows leave M5 unless M4 prints NOT SEPARATED. What stays is the validation of our
own new flow (`RDM_A0_BOX_PLAN.md` section 4, test 5): O with its analytic control in the
A = 0 cell and in the standard cell, N = 100 and 400. On Gadi, as self-contained bundles
(`nn_sbc_cell(archive_dir = , run = FALSE)`), after stage G.

### M6: CRDM

Laptop (8 cores): the nesting check (flow at amp = 0 against the Wald; trap: at amp = 0 the
Volterra recursion *is* the Wald at any dt, so this tests the flow, never the solver); score
bias and posterior shift against Volterra; the step sensitivity (the dt 0.0005, 0.005 and 0.05
cards against one reference). Reference for the screens: dt 1.25e-4 where a solve is
affordable, because at dt 5e-4 the reference's own KS error (up to 1.5e-3) is the size of the
dt 0.0005 flow's (part B). Never evaluate Volterra on grids out to 20 s: the cost is the
square of the longest decision time.

Gadi: the network SBC cell (hybrid race, N = 400, 500 replicates) and the Volterra control,
**250 replicates at dt 2e-3 and 50 replicates at dt 5e-4** (AH). Items 2 to 5 of
`PAPER_REVIEW.md` are what these cells test: the missing posterior reference (2), a constant
shift that correlation cannot see (3), recovery against a coarser simulator (4), and
references at the flow's own dt (5). The cost per cell is in the table below.

**Cost per cell** (added 2026-09-29 after the registration above; costs, not results).
AH's measured costs per likelihood evaluation on 400 trials: hybrid network 0.10 ms, `CRDM()`
at dt 2e-3 1.4 ms, at dt 5e-4 18 ms. Measured here (`sbc/malte_m6/cost_per_replicate.R`, laptop,
three chains on three cores, N = 400): one replicate of the network cell takes 33.5 core-s,
of the dt 2e-3 control 392 core-s, of the dt 5e-4 control 4993 core-s (one replicate). So a
replicate is about 280 000 likelihood evaluations (392 s / 1.4 ms), and the dt 5e-4 figure
predicted from that, 280 000 x 18 ms = 5040 core-s, agrees with the measurement.

| cell | replicates | laptop core-s per replicate | laptop core-hours | Gadi core-hours at 2x (to be replaced by the check job's timing lines) |
|---|---|---|---|---|
| network (hybrid race) | 500 | 33.5 | 4.7 | 9 |
| Volterra control, dt 2e-3 | 250 | 392 | 27 | 54 |
| Volterra control, dt 5e-4 | 50 | 4993 | 69 | 139 |
| all three | | | 101 | about 200 (400 SU in the normal queue) |

**Walltime is the constraint, not core-hours.** `run_sbc` runs replicates one after another
with three chains, so a 48-cpu job used as it stands keeps 3 cores busy: 250 x 131 s = 9 h
(dt 2e-3) and 50 x 1664 s = 23 h (dt 5e-4) on the laptop's cores, twice that on Gadi's, the
second close to the 48 h limit. M6's bundles must therefore split each control into 16
chunks of replicates with their own seeds, 16 x 3 chains = 48 cores, and join the chunks
afterwards: about 1.2 h (dt 2e-3) and 3 h (dt 5e-4) of walltime at 2x. `cores_per_chain`
is the alternative for the dt 5e-4 cell only (18 ms per evaluation is long enough for
particle-level forks to pay); the chunked form is the default because its efficiency is known.

### Where things run

Laptop: M4 and the M6 screens, **never more than 8 cores in total**, counting every job
running at once; each job's core count is stated when it is started. tux26: the seed study's
training and its screen, nothing else. Gadi: every SBC cell, from 1 October (project tg53,
`/scratch/tg53/ajh575/`, PBS only, 48 cpus and 128 GB per cell job, single-threaded BLAS,
library `~/R/nle-lib`, asserted by every cell; results come down promptly because scratch
expires). **Everything uploaded to or run on Gadi is under `/scratch/tg53/ajh575`, because the
home directory is small (AH, 2026-09-29): bundles, cells, logs, `TMPDIR`, and the library
itself, which is installed in `/scratch/tg53/ajh575/R/nle-lib` with `~/R/nle-lib` a symlink to
it** (stage G below). Nothing is submitted to Gadi before 1 October or before stage G's two lines.

---

## Stage G — dev-nle on Gadi  (Haiku)

Written 2026-09-29. The SSH key works since that day
(`ssh -o BatchMode=yes ajh575@gadi.nci.org.au hostname` printed `gadi-login-06.gadi.nci.org.au`),
so the NLE session manages Gadi directly. **Nothing is copied, installed or submitted before
1 October 2026**; `sbc/gadi/gadi_stage_G.sh` reads Gadi's date and refuses (tested 29 Sep:
`REFUSED: Gadi's date is 20260929`).

**Everything on Gadi lives under `/scratch/tg53/ajh575`** (AH, 2026-09-29: the home directory is
small; 3.4 GB used on 29 Sep). The bundle goes to `/scratch/tg53/ajh575/2026-10-01_gadi_dev-nle`,
temporary files to `/scratch/tg53/ajh575/tmp`, and the library to
`/scratch/tg53/ajh575/R/nle-lib`, with `~/R/nle-lib` a symlink to it, so the agreed name and the
assertion `find.package("EMC2")` under `~/R/nle-lib` (both sides through `normalizePath`) hold
and home holds one symlink. Scratch expires: a purged library is rebuilt by `setup` from the
tarball. `~/R/ss-lib`, `~/R/ssd-lib` and the default library are never touched.

Read before the first run, 2026-09-29 (none of it has run on Gadi): `install_nle_lib.sh` takes
`NLE_LIB` from the environment and builds with `-fp-model=precise` through a private
`Makevars`; it needs the `module` function, so over ssh it runs under `bash -lc`. It installs
WienR, jsonlite and testthat from CRAN on the login node (compute nodes have no network, so
the check job must find them installed). `check_nle_lib.pbs` asks for 4 cpus, 16 GB, 40 min,
and names no `storage`, like AH's working template `sbc/nci/SBCens_N400.pbs`. The bundle's
README still says Claude has no login and that the library is in home; the script here
supersedes both.

**Acceptance:** `GADI CHECK PASS`, which requires in `check_nle_lib.log`
`TESTS expectations 783 failed 0 errors 0`, the three mean log-likelihoods 131.286 / 131.325 /
131.359, and EMC2 loaded from `nle-lib`. Then, and only then, cells.

### Prompt for G

```
Read handover/MALTE_STAGES.md: section "Stage G", section "Stages M4 - M6,
re-scoped" and the last six log entries. Do exactly stage G. Work from
/Users/andrew/Downloads/julian/nle on the laptop, absolute paths. Do not edit
any script. Nothing runs on tux26. The laptop never runs more than 8 cores in
total.

Everything uploaded to or run on Gadi is under /scratch/tg53/ajh575; nothing
is written to the home directory except the symlink ~/R/nle-lib, which the
script makes. PBS only. Never type or ask for a password: if
  ssh -o BatchMode=yes ajh575@gadi.nci.org.au hostname
does not print a gadi-login node, stop, print the commands with
  DRY=1 bash sbc/gadi/gadi_stage_G.sh setup
  DRY=1 bash sbc/gadi/gadi_stage_G.sh submit
for AH to run, and say so.

1. bash sbc/gadi/gadi_stage_G.sh setup   > sbc/gadi/stage_G_setup.out 2>&1
   It refuses before 1 October 2026 (Gadi's date). Required at the end of the
   output: "* DONE (EMC2)", the line "EMC2 3.4.1.9001 from ...nle-lib", "CRDM
   exported: TRUE | hybrid argument: TRUE", and ~/R/nle-lib shown as a symlink
   into /scratch/tg53/ajh575/R/nle-lib. If the BLAS printed is R's reference
   BLAS, say so in the log line.
2. bash sbc/gadi/gadi_stage_G.sh submit  (prints the job id). Poll with
   bash sbc/gadi/gadi_stage_G.sh status   every 5 minutes, in the background,
   until the job has left the queue. At most 40 minutes of walltime.
3. bash sbc/gadi/gadi_stage_G.sh verify  > sbc/gadi/stage_G_verify.out 2>&1
   Print that file verbatim. "GADI CHECK PASS" is the acceptance. On FAIL
   print the failing tests from sbc/gadi/stage_G/check_nle_lib.log and stop:
   a failure at exact zeros or -Inf points to fast-math (Makevars.icpx not
   used). Submit no cell.
4. Commit sbc/gadi/stage_G*. Append one log line: node, library path, BLAS,
   the TESTS line, the three timing lines as printed, and the ratio of Gadi's
   ms per evaluation to the laptop's (0.10, 18.1, 1.40), which replaces the
   "2x" in the cost table of the re-scoped section.
5. End with the next stage, its model and its full prompt: the cells built by
   stage M6a (sbc/gadi/cells/), submitted in the order its log line gives.
```

---

## Stage M6a — the cells for Gadi, and the CRDM screens on the laptop  (Sonnet to build, Haiku to run; Fable / Opus reads)

Written 2026-09-29 at the end of M4. Can start at once: it needs nothing from Gadi.

**Registered before any CRDM score bias or posterior shift exists** (known on 2026-09-29: the
density numbers of part B only). Screens as in M4, with the Volterra reference at dt 1.25e-4
and t_max 6 in place of the Wald, cell of `sbc/gate_malte_dev_nle.R` check 6 (v, v_lMTRUE, B,
t0, tau, amp_lDTRUE; means log 2, .5, log 1, log .3, log .1, log .4; sd .15, .1, .15, .1, .2,
.15), N = 400, 200 prior draws for G, 200 datasets for P. Floors as in M4: G 0.005, P 0.10
posterior sd. A card **passes the screen** if G and P are below their floors with both
controls valid (G's control below 1e-3; P's control within +-0.15; median ESS above 200).
Expected: the dt 0.0005 card passes; the dt 0.05 card fails by a wide margin (its KS is .03
at the median); the dt 0.005 card is the open one, and it is the test of `PAPER_REVIEW.md`
item 5. Nesting check: the dt 0.0005 card at amp = 0 against the Wald, KS and KL as in part
B; it tests the flow, not the solver.

**Cells to build** as self-contained bundles under `sbc/gadi/cells/<name>/`, each with its
PBS file (project tg53, 48 cpus, 128 GB, `place=pack`, `-l wd`, single-threaded BLAS exports,
the three modules), each asserting `find.package("EMC2")` under `~/R/nle-lib`, each writing
only under its own directory in `/scratch/tg53/ajh575`:

| name | what | replicates | why |
|---|---|---|---|
| `rdm_a0_N400` | A = 0 cell, N = 400: O (`rdm_ss_base_s801_e271`), HF (`malte_rdm_aff_log_h2_k12_clip_100k`), one analytic control | 250 each | M4's verdict O vs HF was NOT SEPARATED |
| `rdm_o_a0_N100`, `rdm_o_std_N100`, `rdm_o_std_N400` | O with its control | 500 network, 250 control | M5: validation of our new flow |
| `crdm_net_N400` | hybrid race, dt 0.0005 card | 500 | M6 |
| `crdm_ctl_dt2e-3` | Volterra control, 16 chunks | 250 | M6 (AH) |
| `crdm_ctl_dt5e-4` | Volterra control, 16 chunks | 50 | M6 (AH) |

### Prompt for M6a

```
Read handover/MALTE_STAGES.md: sections "Stages M4 - M6, re-scoped", "Stage
G" and "Stage M6a", and the last eight log entries. Do stage M6a. Work in
/Users/andrew/Downloads/julian/nle, absolute paths. HARD LIMIT: the laptop
never runs more than 8 cores in total, counting everything running at once;
check what is running before each job and state the core count of each job
you start. Nothing runs on tux26. Nothing is copied to or run on Gadi in this
stage. Do not edit the registered criteria; do not edit sbc/malte_m4*.R.

Traps already paid for: prepend ~/R/nle-sbc-lib (dev-nle 14c12c79) to
.libPaths and assert find.package("EMC2") is under it; keep s = 1 in every
cell with an RDM control (pWald overflows for v b / s^2 > 354); CRDM()'s
default t_max is 3; the Volterra solve costs the square of the longest
decision time, so t_max 6 at most and never a grid to 20 s; at amp = 0 the
recursion is the Wald at any dt and is no evidence about the solver;
port/export is not tracked by git, so record the SHA256 of every card a
bundle carries; our cards are port/export/<tag>_weights.json and condition on
A through asinh(A / 0.05), evaluated at A = 0 exactly.

A. CRDM screens (laptop). Adapt sbc/malte_m4_lib.R and sbc/malte_m4.R into
   sbc/malte_m6_lib.R and sbc/malte_m6.R: the accumulator with the pulse
   (amp, tau) uses the card, the other the Wald (the hybrid race); the
   reference is dcrdm_volterra (port/src/crdm.cpp) at dt 1.25e-4, t_max 6,
   one solve per distinct parameter row. Simulate data with EMC2's make_data
   on the CRDM design. First a smoke run on 4 datasets with the numbers
   masked, then the full run. Print per card (dt 0.0005, 0.005, 0.05): the
   controls, G per parameter, P per parameter with se, and one line
   "SCREEN <card>: PASS" or "FAIL (<which>)". Print the nesting check line.
B. Bundles. For each row of the table in "Stage M6a", build the bundle with
   nn_sbc_cell(archive_dir = , run = FALSE) where it applies; the two
   Volterra controls are 16 chunks of replicates with seeds 9001 to 9016, run
   side by side at 3 cores each inside one 48-cpu job, and a join script.
   Walltimes from the cost table in the re-scoped section, doubled again for
   safety, never above 48 h. Smoke every bundle on the laptop with 2
   replicates (dt 5e-4: 1 replicate, 3 cores, about 30 min) and confirm the
   result file is written and read back by the join script.
C. Write sbc/gadi/cells/SUBMIT_ORDER.md: rdm_a0_N400 first, then the CRDM
   cells, then the M5 cells, with core-hours per cell.
D. Commit. Append one log line per part with its numbers. End with the next
   stage, its model and its full prompt: stage G if it has not run (its
   prompt is in the file), otherwise the submission of the cells.
```

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
- 2026-09-28 Stage T DONE. tux26: dev-nle 14c12c79 in `~/emc2nn/nle/.Rlibs-malte` (built 05:24 UTC); `.Rlibs-nle` untouched (still
  built 2026-09-27 14:35 UTC), as is `~/emc2nn/nle/port/R/flow_race.R`. Gates against the new library: Malte 9/9 in 1.0 min
  (`sbc/gate_malte_dev_nle_tux.out`; check 0 1.3e-13, 2: 1.8e-12 / 8.7e-12, 3: 1.5e-12 / 6.0e-12, 4: 1.2e-13 / 4.1e-13, 5a 5.9e-16,
  5b identical to the laptop draw for draw), zero-box 3/3 on ddm_zb_s903_e271 (<= 1.2e-12, 450 -Inf points), usability 5/5
  (reg_s402 card absent on tux26, so that part of check 2 did not run). Marker `handover/nle-methods/rlibs_malte.marker` written.
  Malte root on tux26: `~/emc2nn/nle-malte`. Done while zero-box v2 trained (3 jobs, about 20 of 96 cores).
- 2026-09-28 M7a PASS (code and laptop smoke; no training run started). Options `--affine`, `--decision-time`, `--shuffle-seed` in
  `flow/scripts/train_race.py` (+ `affine=` in `flow/src/flow/config.py`; `training.py` unchanged), exporter writes `affine`, `affine_scale`,
  `pre`, mirrored in `handover/nle-methods/flow`; `port/tests/validate_m7a.R` in both port copies (the two `flow_race.R` are identical and
  needed no patch); `malte/train_conditioner_ckpt.py`; `sbc/seed_study_launch_tux.sh` (criterion registered in its header),
  `sbc/seed_study_screen.sh`, `sbc/seed_study_table.R`; `sbc/score_bias_race.R` takes decision-time and natural-scale cards and an A = 0 cell.
  Default path, rdm_b1k's configuration on the full 16M rows, first epoch before vs after the change, batch order seeded: train
  1.1992952823638916 / val 0.7444201707839966 both times (`sbc/seed_study/m7a_first_epoch.out`). **The acceptance line as written cannot be
  met by any code:** an existing tag's first epoch is not reproducible, because `PreloadedBatches` shuffles with an unseeded generator
  (rdm_b1k recorded 1.2014626); the test was therefore HEAD against working tree under one injected shuffle seed. Smoke, 2 epochs on 1%
  (160k rows), through the launch and screen scripts: golden max|diff| 2.0e-13 (base), 9.9e-14 (aff), 3.1e-13 (dt), 2.0e-13 (both), all
  < 1e-10; dev-nle 14c12c79 `dfun`/`pfun` vs this repo's R port <= 8.3e-14, < 1e-7, all four register; float64 port vs distrax float32
  <= 1.5e-4; screen controls 0.00021 (std) and 0.00025 (A0); table printed, 80 rows (`sbc/seed_study/m7asmoke/`; its |g| are of 2-epoch
  flows and mean nothing). Found: (1) **our flow is weak at A -> 0**: rdm_b1k_e242 has |g| <= 0.004 in the standard cell and 0.09-0.33 in
  the A = 0 cell, density error 0.066 at A <= 0.001 against 0.014 at A = 0.3, because A is uniform on [1e-4, 2] and conditioned on its log
  (0.5% of draws below 0.01); this handicaps our side of every A = 0 comparison (M4, M5, arm 5 of M7) and is registered as a caveat in the
  launch header; (2) R's log(1e-4) is 2e-15 below the card's numpy bound, so A = 1e-4 is out of the box unless clamped to the card's own
  bound (done in the screen; M4/M5 must do the same); (3) tux26 has no RDM data, no Malte environment, and an installed `flow` package
  that predates M7a, so the scripts put their root's `flow/src` first and assert it; (4) his trainer needs `hydra-core`, added to
  `malte/.venv` and the lock file. Laptop timing at 8 threads: 203 s per epoch on the full data, his 100k steps about 45 min. Next: M7b
  (Haiku, prompt above), blocked until 1 Oct and tux26 free.
- 2026-09-28 Decisions (AH): the RDM flow gets an A = 0 box (`handover/RDM_A0_BOX_PLAN.md`), **folded into the seed study** (arms 1-4
  train on it, so the baseline arm is the new flow), 24e6 rows, first wave now. Done the same day (Fable): code stage A0a passes on the
  laptop (sampler test 13/13; default path bit-identical, 1.1992952823638916 / 0.7444201707839966; five-arm smoke on the A0 data: golden
  <= 7.1e-13, dev-nle vs R port <= 1.4e-13 including at A = 0 exactly; controls 0.00021 / 0.00025); launch header amended before any
  run or number (A0 cell at A = 0 exactly; handicap caveat withdrawn, text kept). tux26: data simulated in `~/emc2nn/nle-malte`
  (24e6 + 1e5 rows, sampler test 13/13, Wald KS p 0.41 / 0.08 / 0.35 on 6e5-row blocks); smoke of two arms, one epoch: golden <= 7.3e-13,
  dev-nle <= 3.4e-13, **epoch 156 s (base) and 209 s (both) at 7 cores, so 14 to 18 h per run**, not 30. **Wave 1 launched 08:46 CEST:**
  `ss_base_s801..804`, `ss_aff_s801`, cores 40-74, expected about 23:00 CEST. Traps: tux26's `arrow` is in `~/emc2nn/Rlib-devnn` only;
  the launch script now refuses to start a run that is in progress (a relaunch would have wiped it). M7b prompt rewritten above.
- 2026-09-28 11:13 CEST, **wave 2 launched** (AH: the remaining runs, 24 cores): 11 of ours (`ss_aff_s802..804`, `ss_dt_s801..804`,
  `ss_both_s801..804`) and Malte's 4 (`ss_malte_s2026..2029`), 6 at a time at 4 cores, pinned to cores 0-23, output
  `sbc/seed_study_launch_wave2.out`. Malte's environment built on tux26 in `~/emc2nn/nle-malte/malte` (eamax 6d7f2ff, confrdm fae8add,
  jax 0.10.2, flax 0.12.8; `uv pip freeze` equals `requirements.lock` after pinning tfp-nightly to 20260927). **tux26 is oversubscribed:**
  another user (mdonzal) runs about 36 R jobs, load average 103 on 96 cores after the launch. Epoch times: wave 1 went from 165 s to
  about 250 s; wave 2 runs 205-266 s. Expected, if the load stays: wave 1 done about 04:30 on 29 Sep; wave 2's first six about 07:00-10:30
  on 29 Sep, its last runs on the morning of 30 Sep. Every run of the study is now launched or queued; M7b launches nothing.
- 2026-09-28 11:51 CEST, **correction to the entry above: tux26 is not oversubscribed.** Measured with top: 25.7% idle; our jobs use
  47.5 cores (zero-box included), mdonzal's 23. The load average (83-103) counts runnable threads, and each of our runs carries 38-61
  threads, so it overstates use. Pinning confirmed with taskset: wave 1 on cores 40-74 (7 each, using 3.1-3.3), wave 2 on cores 0-23
  (4 each, using 2.2-2.6). The slow epochs at 11:20 were a transient; at 11:51 wave 1 is at 177 s per epoch and wave 2 at 190 (aff)
  to 244 s (dt). Expected: wave 1 about 01:00 on 29 Sep; wave 2's first six 04:00-09:00 on 29 Sep; wave 2 complete between the late
  evening of 29 Sep and the early morning of 30 Sep.
- 2026-09-28 11:56 CEST, **the last nine runs launched** (AH: 24 more cores, idle 31% before): `ss_aff_s804`, `ss_dt_s803`,
  `ss_both_s803`, `ss_dt_s804` on cores 24-39; `ss_both_s804` on 76-79; Malte's four one after another on 80-83 (outputs
  `sbc/seed_study_launch_wave3a/b/c.out`). All 16 of our runs are training; wave 2's queue will find its remaining entries running or
  complete and skip them, so **its final "x of 15" count will be short and means nothing**. New `STATUS=1` mode of the launch script
  prints the state of all 20 runs and starts nothing; the M7b prompt is rewritten around it and launches no training. Expected: Malte's
  runs by the evening of 28 Sep, wave 1 about 01:00 on 29 Sep, everything between 04:00 and 09:00 on 29 Sep.
- 2026-09-29 **State of M7b, found by reading tux26 at 04:19 CEST: it has not finished.** Training complete for the baseline arm
  (4 of 4, last at 02:40), `aff` 3 of 4, `both` 3 of 4, `dt` 1 of 4, Malte's 4 of 4; six runs still training (epochs 214 to 292 of
  300). No export, no screen, no table and no marker existed on tux26. M7b's steps 3 to 5 (wait, final status, full screen on tux26)
  are still to do; its step 2 is superseded by the laptop screen of part E below.
- 2026-09-29 M4 part A PASS (converter). Reproduced on Malte's RDM final checkpoint: `StandardRestore()` stops with "Topology
  mismatch" (`_sharding` records `cuda:0`; the fixtures record `cpu:0`). Restoring every leaf as a numpy array is the right fix:
  three independent routes to the weights of that checkpoint (numpy leaves; `StandardRestore` with a numpy target tree; his own
  `load_conditioner` into his own conditioner) agree to the bit, max|diff| 0, float32, 6 leaves. Applied to
  `flow/scripts/malte_to_card.py`, which now also refuses a non-finite or non-float leaf; **the copy under `handover/nle-methods`
  did not exist and was created** (`handover/nle-methods/flow/scripts/malte_to_card.py`). `validate_malte.R`: ACCEPTANCE PASS,
  log_pdf 7.3e-12 / 3.4e-11 / 3.6e-11, unchanged from M1; fixture cards re-converted with the patched converter differ from the
  stored ones in the export date only, stored files' SHA256 unchanged. Five cards exported to `port/export/malte_<name>.json`:
  **SHA256 identical to `cards/SHA256` in the staging folder, all five** (`sbc/malte_m4/cards.sha256`). The hash includes the export
  date, so it reproduces on 2026-09-29 only.
- 2026-09-29 M4 part B (first look checked; `sbc/malte_density_check.R`, outputs in `sbc/malte_m4/density_part*.out`, 6 cores).
  **We reproduce Malte's `density_comparison.nc` set by set**: max |ours - his| over his 45 + 3 x 540 sets, KL 2.1e-7, W1 7.1e-8,
  KS 2.8e-7 (his evaluator is float32; my preset 1e-7 for the RDM KL was missed at 1.5e-7). His medians: RDM final KL 1.4e-5,
  KS 9.4e-4; CRDM dt 0.0005 KL 9.7e-5, KS 2.5e-3; dt 0.005 KL 2.4e-4, KS 3.3e-3; dt 0.05 KL 2.5e-2, KS 3.1e-2 (90% .19, max .43).
  **`first_look.R`: its CDF numbers can be relied on, its KL numbers only as orders of magnitude, as it said.** On its own
  parameter draws (reproduced exactly from the seed) the proper KS equals its KS to the digits printed; its KL has the right median
  but single values off by factors of -22 to 20, because the rectangle rule on its grid puts the reference's mass between 0.74 and
  1.003 (truncation at 20 s and below 1 ms) where the proper grid gives 1.0000000 to 1.0000032. Its worst RDM final KL, 4.5e-4, is
  5.9e-5 done properly; its worst plain KL, 7.6, is 14.3. It evaluated `CRDM()` beyond that function's default t_max of 3.
  **Verdict 1, plain RDM flow: confirmed, and the cause is not "s or b near 0" but the ratio b / s.** For b / s < 0.2 its KS equals
  the reference's mass below exp(-5) = 6.7 ms to three decimals (median .49 for b / s < 0.1): without an affine stage its spline
  lives on log t in [-5, 5], so it cannot put mass below 6.7 ms. It fails on his own grid too, at s = 3, b = 0.5 (KL .48 to .75),
  which is inside the box of the final flow. For b / s > 3 it is poor as well (KS median .013). In the M4 cell (b / s about 1) it
  is merely three to four times worse than the final flow. **Verdict 2, CRDM at dt 0.05: confirmed, and it is his own result**:
  proper KS median .018, 90% .064, max .16 on the first look's draws; .031 / .19 / .43 on his grid, identical against a reference
  at dt 1.25e-4. New: the dt 5e-4 Volterra reference has its own error, KS up to 1.5e-3 (90% 2.9e-4), KL up to 3.4e-4; his worst
  KL for the dt 0.0005 flow, .079, is .011 against the finer reference, so that maximum is mostly the reference's error.
- 2026-09-29 M4 part C DONE. M4 to M6 re-scoped in this file **before any M4 number existed** and committed as 4370337 before our
  flow was exported (section "Stages M4 - M6, re-scoped"; what was already known is stated there). Cost per cell, measured:
  network 33.5 core-s per replicate, control dt 2e-3 392, dt 5e-4 4993 (predicted from AH's per-evaluation costs: 5040); laptop
  core-hours 4.7 (500 replicates), 27 (250), 69 (50); about 200 core-hours on Gadi at 2x. `run_sbc` uses 3 cores, so the controls
  must be chunked (16 x 3 chains) to fit a 48-cpu job and the 48 h limit.
- 2026-09-29 M4 part D: **the Gadi key works** (`gadi-login-06.gadi.nci.org.au`, no password). Nothing was copied, installed or
  submitted: it is 29 September. Read-only look: home 3.4 GB used, no `~/R/nle-lib`, tg53 scratch 149 GiB of 1 TiB. AH, during the
  stage: everything uploaded to or run on Gadi goes under `/scratch/tg53/ajh575` because home is small; incorporated in stage G
  (library in scratch, `~/R/nle-lib` a symlink). `sbc/gadi/gadi_stage_G.sh` (setup, submit, status, verify) written and its date
  guard tested; its setup and submit have not run. Bundle read: no fault found that blocks a first run; notes in stage G.
- 2026-09-29 **M4 part E DONE. Registered verdicts: `O vs HF: NOT SEPARATED: SBC cell needed`; `O vs HP: OURS AS GOOD OR BETTER`.**
  Baseline arm screened on the laptop (4 cores; checkpoints copied from tux26, scaler identical in the four runs, sha 24636e63):
  119 of 120 exported (`rdm_ss_base_s804_e279` fails the float32 tolerance, 1.8e-2), card check 0 failing lines, controls .00021
  (std) and .00025 (A0); `sbc/seed_study/ss/table_base.csv`, `summary_base.csv`. W over the 120: min .0022, quartiles .0103 /
  .0139 / .0208; seed 801 holds the 5 best, seed 802's best is rank 85. **O = `rdm_ss_base_s801_e271`** (W .0022), median
  checkpoint `rdm_ss_base_s803_e295` (W .0139). On draws 201 to 400 (5 cores; controls: G .00025, P .145, Wald mass 1.0000003 to
  1.0000006, median ESS above 1900 of 4000):

  | | D: median KL (KS median) | G: max \|g\| | P: max \|shift\|, posterior sd | worst parameter of P |
  |---|---|---|---|---|
  | HF, his final | 2.7e-5 (.0012) | .0043 | .021 (se .003) | B +.021 |
  | HP, his plain | 4.4e-4 (.0039) | .0371 | .431 (se .010) | t0 +.431, B -.269 |
  | O, ours selected | 1.5e-4 (.0025) | .0010 | .041 (se .003) | B +.041, v +.040 |
  | ours, median checkpoint | 1.6e-4 (.0017) | .0076 | .113 (se .007) | t0 +.113 |

  O against HF: D ours worse (KL 5.6 times his, above the floor), G tie (both below .005; ours is four times smaller), P tie
  (both below 0.10 sd). A "worse" on D alone with P a tie is the third row of the registered table, hence NOT SEPARATED. The
  unselected checkpoint against HF reads HIS BETTER on all three. P's control passed at .145 against .15, which is 2 se: it is
  marginal, on v and v_lMTRUE. `RDM_A0_BOX_PLAN.md` section 4 for O: density error over A = 0 to 1 between .0068 and .0118
  (target .014: PASS); standard cell max |g| .0022 (rdm_b1k: .004: PASS). Files: `sbc/malte_m4/{G,D,P,verdict,flatA}.out`,
  `selection.csv`. Read for AH: the predicted difference between O and HF is 0.02 posterior sd, and the SBC cell the rule calls
  for resolves 0.13, so it will confirm that both pass rather than rank them. Next: M6a (prompt above), then G on 1 October.
- 2026-09-29 M7b prompt amended at AH's request: steps 1 and 2 removed (baseline arm screened on the laptop in M4), it starts
  at step 3 with its own polling command, protects `table_base.csv` and `summary_base.csv`, and compares the tux26 screen of
  the baseline arm with the laptop's.
- 2026-09-29 M6a part A (CRDM screens; `sbc/malte_m6_lib.R`, `sbc/malte_m6.R`, adapted from the M4 pair): nesting check (dt 0.0005
  card at amp = 0 vs Wald) max KL 5.4e-5, max KS 3.3e-3 (`sbc/malte_m6/nest.out`). **G, full run, 200 draws, 5 cores, 96 min**
  (`sbc/malte_m6/G_full.out`, `G.rds`): v, v_lMTRUE, B, tau, t0 controls all clean (<= 0.0019); **amp_lDTRUE control is 0.02077,
  21x the 1e-3 floor — invalid**, which reads `SCREEN <card>: FAIL (G control)` for all three cards under the registered rule.
  dt 0.0005 and dt 0.005's own amp_lDTRUE values (-0.017, +0.016) are the same order as the control's artifact, so this screen
  cannot currently separate real flow bias from solver/estimator noise on that one parameter (H = 0.01, dt 1.25e-4); not
  root-caused here (candidates: smaller H, finer or adaptive dt). dt 0.05's amp_lDTRUE (-0.382) stands independently of the
  control problem and confirms M4 part B's finding that dt 0.05 is poor on its own terms. **P (posterior shift) not completed:**
  a 2-dataset smoke (N = 50, 2 cores, one card) ran 4.5 h without finishing one dataset, genuinely computing throughout (99%
  CPU) — each dataset's Laplace fit against the Volterra reference costs a fresh Volterra solve (~5 s, benchmarked directly)
  per `optim()` evaluation, which compounds past laptop-affordable; not priced in the registered cost table (which covered only
  D/G-style per-draw cost, not a full posterior optimisation against the reference). Abandoned rather than left running for days
  or reported with fabricated numbers. Full readout: `sbc/malte_m6/screen_final.out`.
- 2026-09-29 M6a part B (bundles; `sbc/gadi/cells/`): all 8 bundles built — `nn_sbc_cell(run = FALSE)` for the 6 network cells
  (`rdm_a0_N400/{O,HF}`, `rdm_o_a0_N100`, `rdm_o_std_N100`, `rdm_o_std_N400`, `crdm_net_N400`), `run_sbc() + prior(type =
  "single")` for the two chunked Volterra controls (`crdm_ctl_dt2e-3`, `crdm_ctl_dt5e-4`, 16 chunks each, seeds 9001-9016,
  exact replicate counts 250 and 50 split across chunks). Every bundle smoke-run for real at the required settings (2
  replicates/3 cores; 1 replicate/3 cores for dt 5e-4, ~76 min measured — the prompt's own "~30 min" estimate was short, real
  chains run serially at cores_for_chains = 1 for a single replicate): result files confirmed written and read back
  (`nn_sbc_summary()`, `summary.csv`, `sbc_nn.RData`/`sbc_control.RData`). Two real bugs found and fixed by these smoke runs:
  an `Lbar()` indexing bug in M6a's own G mode (silently produced all-NA slopes on the first smoke); `nn_sbc_cell`'s
  auto-generated `run_cell.R` has no `.libPaths()` before `library(EMC2)`, so it silently loads the wrong EMC2 unless `R_LIBS`
  is exported first (the dual-install trap) — fixed in `pbs_template.sh`. Also fixed: `run_cell.R` runs from `bundle/`, not the
  archive root, so each bundle's card copy now lives at `bundle/port/export/<tag>*.json`; cards' SHA256 recorded in each
  build's log, files themselves not committed (`sbc/gadi/cells/**/port/export/*` added to `.gitignore`, matching the existing
  rule for `port/export/*`).
- 2026-09-29 M6a part C: `sbc/gadi/cells/SUBMIT_ORDER.md` written — `rdm_a0_N400` first (resolves M4's own verdict, cheapest),
  then the CRDM cells in increasing cost, then the M5 `rdm_o_*` validation cells last. Core-hours per cell from the re-scoped
  section's cost table (RDM cells' core-hours are a rough, unregistered estimate: no measured RDM per-replicate cost exists).
  Measured (not just the cost table) one-replicate wall times from this session's smoke runs: dt 2e-3 ~15 min, dt 5e-4 ~76 min;
  extrapolated per-chunk time on Gadi (replicates DO parallelise there at cores_per_chain = 3) is comfortably inside the
  requested walltimes (4 h / 8 h). Noted explicitly: P's infeasibility (part A) does not gate any cell here — the cells' own
  SBC machinery (`nn_sbc_cell` / plain `run_sbc`, ordinary MCMC) is unrelated code, already smoke-validated for every cell.
- 2026-09-29 M6a part D: committed (`a371bfa`). Next: stage G (prompt below, unchanged from when it was written; Gadi's date
  guard still refuses before 1 October). M6a's own finding for whoever reads this before G: the amp_lDTRUE score-bias control
  needs attention before this screen can be trusted for a real per-card verdict, and P's Laplace-against-Volterra route needs a
  cheaper method (or should be skipped in favour of going straight to the Gadi SBC cell, which is what the registered rule
  calls for on an invalid screen in any case).
- 2026-09-29 AH had been reading an older draft; re-reviewed the PeerJ submission (`~/Downloads/Neural_Density_Estimation_
  for_Race_Models__PeerJ_.pdf`) against `PAPER_REVIEW.md`'s ten items (full table:
  `~/Documents/Projects/EMCpackage/SBC-staging/2026-09-29_malte_trained/PAPER_REVIEW.md`, "Update 2026-09-29" section, not
  tracked by this repo's git). Fixed on his side: 1 (methods now match the results), 5 (Fig. 3's accuracy reference is now the
  fixed fine grid, not matched-dt), 6 (MAD dropped for Wasserstein/KS), 10 (simulator description now matches the
  Brownian-bridge port, and states the t_max-censoring rule). **Still open, unchanged from the version item 2 was written
  against: item 2 (no CRDM posterior reference; Discussion still calls numerical inference "too slow"), item 4 (test data
  simulated coarser, dt 0.001, than the network is trained on, dt 0.0005), item 8 (100 datasets, now up to 9 CRDM ones per
  condition discarded for non-convergence, was 6), item 9 (one seed per configuration -- AH confirmed with Malte directly).**
  Item 3 (C2ST rise) is now explained (posterior-width difference) rather than waved away, but still no standardised-shift
  statistic and hierarchical C2ST (0.625-0.664) is still well above chance. Commit confirmed by Malte, full hash
  `fae8addc52448c36f16b04068cf42f7acc0bc527` -- matches this repo's `confrdm fae8add` (M1, M7a): no change to our checkout.
  Nothing here weakens the case for M6/M8; if anything items 2, 4, 8, 9 are exactly what our results speak to and the paper's
  central CRDM inference claim still rests on parameter recovery alone. Point 2 is feasible and already built, not yet run:
  the M6a bundles above (`crdm_net_N400`, `crdm_ctl_dt2e-3`, `crdm_ctl_dt5e-4`) are precisely the reference posterior and SBC
  control this item calls for, smoke-tested, a few hours of wall-clock once Gadi is available -- the numeric rebuttal to
  "too slow." Still blocked only on stage G.

### Prompt for the stage after M6a

Stage G has not run (`sbc/gadi/gadi_stage_G.sh setup`/`submit` have not been called; Gadi's date guard refuses before 1
October 2026). Its full prompt is unchanged from where it was written, "Stage G" above:

```
Read handover/MALTE_STAGES.md: section "Stage G", section "Stages M4 - M6,
re-scoped" and the last six log entries. Do exactly stage G. Work from
/Users/andrew/Downloads/julian/nle on the laptop, absolute paths. Do not edit
any script. Nothing runs on tux26. The laptop never runs more than 8 cores in
total.

Everything uploaded to or run on Gadi is under /scratch/tg53/ajh575; nothing
is written to the home directory except the symlink ~/R/nle-lib, which the
script makes. PBS only. Never type or ask for a password: if
  ssh -o BatchMode=yes ajh575@gadi.nci.org.au hostname
does not print a gadi-login node, stop, print the commands with
  DRY=1 bash sbc/gadi/gadi_stage_G.sh setup
  DRY=1 bash sbc/gadi/gadi_stage_G.sh submit
for AH to run, and say so.

1. bash sbc/gadi/gadi_stage_G.sh setup   > sbc/gadi/stage_G_setup.out 2>&1
   It refuses before 1 October 2026 (Gadi's date). Required at the end of the
   output: "* DONE (EMC2)", the line "EMC2 3.4.1.9001 from ...nle-lib", "CRDM
   exported: TRUE | hybrid argument: TRUE", and ~/R/nle-lib shown as a symlink
   into /scratch/tg53/ajh575/R/nle-lib. If the BLAS printed is R's reference
   BLAS, say so in the log line.
2. bash sbc/gadi/gadi_stage_G.sh submit  (prints the job id). Poll with
   bash sbc/gadi/gadi_stage_G.sh status   every 5 minutes, in the background,
   until the job has left the queue. At most 40 minutes of walltime.
3. bash sbc/gadi/gadi_stage_G.sh verify  > sbc/gadi/stage_G_verify.out 2>&1
   Print that file verbatim. "GADI CHECK PASS" is the acceptance. On FAIL
   print the failing tests from sbc/gadi/stage_G/check_nle_lib.log and stop:
   a failure at exact zeros or -Inf points to fast-math (Makevars.icpx not
   used). Submit no cell.
4. Commit sbc/gadi/stage_G*. Append one log line: node, library path, BLAS,
   the TESTS line, the three timing lines as printed, and the ratio of Gadi's
   ms per evaluation to the laptop's (0.10, 18.1, 1.40), which replaces the
   "2x" in the cost table of the re-scoped section.
5. End with the next stage, its model and its full prompt: the cells built by
   stage M6a (sbc/gadi/cells/), submitted in the order its log line gives
   (sbc/gadi/cells/SUBMIT_ORDER.md). Before submitting, read
   handover/MALTE_STAGES.md's M6a log entries: the amp_lDTRUE score-bias
   control failed on the laptop screen and P (posterior shift) could not be
   completed there: neither blocks a cell submission (the cells run ordinary
   MCMC-based SBC, unrelated code, already smoke-tested), but both are open
   questions this file should carry forward, not silently drop.
```

Model: Haiku (as originally assigned to stage G).

# An RDM flow whose box reaches A = 0

Written 2026-09-28, after stage M7a (`handover/MALTE_STAGES.md`, log). Status: **decided
and running.** Decisions (AH, 2026-09-28): fold into the seed study; 24e6 rows; start the
first wave now on tux26's idle cores. Code stage done, data simulated on tux26, **wave 1
launched 2026-09-28 08:46 CEST** (section 9). Sections 1 to 8 are the plan as written
before the decisions; where a measurement has replaced an estimate, section 9 says so.

## 1. Why

The comparison with Malte Lüken's flow is at A = 0 (`MALTE_CRDM_PLAN.md` section 3.1), and
our flow is weakest exactly there. Measured in M7a on `rdm_b1k_e242`:

| | standard cell (A about 0.3) | A = 0 cell (flow at A = 1e-4) |
|---|---|---|
| score bias, max over parameters | 0.004 | 0.33 |
| density error, E_true \|log f_flow - log f\| | 0.014 | 0.066 |

The box's lower bound is A = 1e-4, not the problem in itself. The problem is the
coordinate: A is drawn uniformly on [1e-4, 2] and conditioned on log A, so the 0.5% of
draws with A < 0.01 are stretched over 46% of the input's range. The zero-box DDM had the
same disease in sv, SZ and st0 and the same cure applies.

## 2. Specification

**Encoding.** `u = asinh(A / c)`, `c_A = 0.05`: exactly 0 at A = 0, linear below about
0.05, log-like above. The density depends on A to first order (the mean distance to
threshold is B + A/2), so a linear coordinate near zero is the right one, and 0.05 is half
the smallest threshold in the box (B >= 0.1). `v, B, t0, s` keep the log. Card fields as
for the zero-box DDM: `context_transforms$A = "zerobox"`, `zerobox_c = {A: 0.05}`.

**Box.** A in [0, 2], continuous to zero. Everything else unchanged
(v 1e-3..5, B 0.1..3, t0 0.05..1, s 0.1..2).

**Sampling of A: a three-part mixture.** The zero-box v1 failure was training density
moved away from where the SBC cell measures, so the standard cell must not lose rows.

| stratum | share | purpose |
|---|---|---|
| uniform on [0, 2], natural scale | 70% | today's density, where the standard cell lives |
| uniform in asinh(A / 0.05) on [0, 2] | 15% | dense coverage of the approach to zero |
| A = 0 exactly | 15% | the Wald: the comparison cell and the nesting check |

**Budget: 24e6 rows** (16e6 today). Rows in the standard cell's prior region
(A in [0.2, 0.45]): 2.0e6 today, 2.75e6 planned. Rows with 0 < A < 0.01: 80e3 today,
250e3 planned, plus 3.6e6 at exactly zero. The sampler test asserts both shares.

**Storage.** Parameters on EMC2's sampled scale, `log A = -Inf` at zero, as the zero-box
data does; the asinh context is computed in one place (loader and exporter).

**Architecture and recipe.** Unchanged: [256, 128], 64 bins, batch 1024, 300 epochs,
30 retained checkpoints.

Checked already (2026-09-28, laptop): `EMC2:::rWald` and `dRDM` accept A = 0 (draws
against the closed-form Wald CDF, KS p = 0.36; every log density finite); simulation runs
at 0.7 s per 1e6 rows; `RDM()$p_types[["A"]]` is `-Inf`, so a design that omits A gives
exactly A = 0; dev-nle 14c12c79 has the `zerobox` transform code and reads `zerobox_c`.
**Not yet checked:** that dev-nle's compiled *race* path applies the zerobox transform (it
was gated for the DDM only). That is the first acceptance line of the code stage.

## 3. Code changes (stage A0a)

1. `simulation/sim_rdm_a0.R` + `simulation/test_rdm_a0.R`: the mixture sampler, training
   and validation sets (`data/train/rdm24m_a0`, `data/val/RDM_a0`), and a test that asserts
   the shares, the exact zeros and finite `ll`.
2. `flow/scripts/train_race.py --zerobox-A`: context column A becomes asinh(exp(log A) / c)
   before the scaler. Off by default, composes with `--affine` and `--decision-time`.
3. `flow/scripts/export_flow.py`: `context_transforms`, `zerobox_c`, bounds with a lower
   bound of 0 in the asinh coordinate; golden vectors include A = 0 rows.
4. `port/R/flow_race.R` (both copies) needs nothing: it takes the network's input
   coordinate. `port/tests/validate_m7a.R` and `sbc/score_bias_race.R` learn to build that
   coordinate for a zerobox card, and the screen's A = 0 cell evaluates at exactly zero
   (the 1e-4 clamp stays for old cards only).
5. `sbc/seed_study_launch_tux.sh`: data, the `--zerobox-A` flag, and the header, amended
   **before any number exists**: the first caveat (A0 handicaps our flows) is withdrawn and
   the reason recorded.

**Acceptance, laptop:** sampler test passes; default options still train bit-identically
(`sbc/seed_study/m7a_first_epoch.py`); a 2-epoch smoke of each arm on 1% exports, passes
golden < 1e-10 including A = 0 rows, registers in dev-nle with `dfun` equal to the R port
< 1e-7 at A = 0 and A > 0; the screen runs in both cells with controls < 1e-3.

## 4. Tests of the trained flow

1. Score bias in both cells, every retained checkpoint (the seed-study screen).
2. **The M7a measurement repeated:** density error at A = 0, 0.001, 0.01, 0.05, 0.1, 0.3, 1.
   Target: flat in A, no worse than 0.014 anywhere.
3. Nesting: at A = 0 the flow against the closed-form Wald.
4. Standard cell must not regress: score bias no worse than `rdm_b1k`'s checkpoints.
5. SBC is M5's: the A = 0 cell at N = 100 and 400, with the control.

## 5. Sequencing: fold it into the seed study (recommended)

The seed study (M7b) has not started and would train 16 of our runs on the old data,
with a registered caveat that its comparison with Malte's flow is handicapped. Training
the new flow separately would add four more runs and leave that caveat standing. Instead:
**the seed study's four arms train on the new data with `--zerobox-A`.** Its baseline arm,
four seeds and 120 screened checkpoints, *is* the new RDM flow's candidate set. Cost over
the seed study as planned: 1.5x the rows, no extra runs.

The baseline arm runs first, so the new flow exists before the other arms finish.

## 6. Timing

Inputs. Per epoch on tux26: zero-box v2 measures 6.2 min for 24e6 rows at 12 threads, of
which it uses 6.6 cores; v1 measured about 4 min. Same network, so **one RDM run on 24e6
rows is 20 to 31 h**; I plan on 30 h and 7 cores per run. The tux26 smoke measures it.
tux26 has 96 cores; your ceiling is 72; zero-box v2 holds 36 of them until about 09:00 CEST
on 29 September; stage Z's cells then want 72 for about 6 h.

| step | who | duration | earliest finish (CEST) |
|---|---|---|---|
| A0a: code, tests, laptop smoke | Fable / Opus, laptop | 1 to 2 h | 28 Sep, evening |
| A0b setup: copy, simulate 24e6 rows, Malte environment, tux26 smoke | Haiku | about 1 h | 28 Sep, late evening |
| Wave 1: baseline arm, 4 seeds (+ 1 run of the next arm), 35 cores | tux26 | about 30 h | 30 Sep, about 04:00 |
| Screen of the baseline arm: **the new flow's candidates** | Haiku | about 1 h | 30 Sep, morning |
| Remaining 11 of our runs + Malte's 4 (under 1 h each), 10 at a time | tux26 | about 35 h after wave 1 | 1 Oct, afternoon |
| Full screen and table | Haiku | about 2 h | 1 Oct, evening |

So: **new flow's candidates screened on the morning of 30 September; whole seed study on
the evening of 1 October**, a day or two earlier than waiting for 1 October to start it.
If only the new flow is wanted (four runs, no seed study), the first three rows and the
screen stand as they are and the rest is dropped.

The estimate slips by about a day if a run really takes 31 h and the waves cannot overlap
stage Z; it improves by about 10 h if the RDM epoch is nearer v1's 4 min.

## 7. Decisions needed

1. **Fold into the seed study (recommended) or train the new flow alone.**
2. **Cores during stage Z.** Wave 1 (35 cores) and stage Z's cells (72) overlap on
   29 September. Either the total may reach 96 for those 6 h, or stage Z runs at 60
   (`CPC=10`, about 7 h), or wave 1 waits until Z's cells finish (everything a day later).
3. **Budget 24e6 rows (recommended) or 16e6.** At 16e6 a run is about 20 h and the
   standard cell's region gets 1.8e6 rows, slightly fewer than today.

## 8. Risks

- dev-nle's race path may not apply `zerobox`. Found in A0a in minutes; if so it is a
  small EMC2 change with a new gate line, and stage T must be redone for the new commit.
- The point mass at A = 0 is 15% of rows spent on one value. If the standard cell
  regresses (test 4), the shares are the first lever, as in zero-box v2.
- `c_A = 0.05` is reasoned, not measured. Test 2 shows within one run whether resolution
  near zero is adequate.
- Runs cannot be paused and resumed (`train_race.py` has no restore), so a wave that
  starts must be allowed to finish.

## 9. As run (2026-09-28)

**Code (stage A0a), laptop.** `simulation/sim_rdm_a0.R`, `simulation/test_rdm_a0.R`,
`train_race.py --zerobox-A`, exporter (`zerobox_c`, `context_transforms$A = "zerobox"`,
bounds on the network's scale, five golden parameter sets at exactly A = 0),
`port/tests/validate_m7a.R` and `sbc/score_bias_race.R` build the asinh coordinate, launch
script amended (header, data, flag, `ONLY` for a wave, a guard that never restarts a run in
progress). Acceptance: sampler test 13 of 13; default path still bit-identical (first epoch
train 1.1992952823638916, val 0.7444201707839966, as before M7a); five-arm smoke on 1% of
the A0 data: golden <= 7.1e-13, dev-nle `dfun`/`pfun` vs the R port <= 1.4e-13 **including
at A = 0 exactly**, controls 0.00021 and 0.00025. The risk of section 8, that dev-nle's race
path might not apply `zerobox`, is closed: it does.

**Data, tux26.** `~/emc2nn/nle-malte/data/train/rdm24m_a0` (24e6 rows, 240 files, 1.7 GB,
seed 4876) and `data/val/RDM_a0` (1e5 rows, seed 9876), simulated with dev-nle 14c12c79's
EMC2 from `.Rlibs-malte` in under a minute. Sampler test 13 of 13 on both; A = 0 rows
against the closed-form Wald on three blocks of 6e5 rows: KS D <= 0.0016, p 0.41, 0.08, 0.35.
tux26's `arrow` is in `~/emc2nn/Rlib-devnn`, not in `.Rlibs-malte`: run the simulation with
`R_LIBS=~/emc2nn/Rlib-devnn A0_LIB=~/emc2nn/nle/.Rlibs-malte` (the script asserts EMC2
comes from `A0_LIB`). The data are not on the laptop; they are reproducible from the seeds.

**Timing, measured on tux26 at 7 cores per run:** first epoch 156 s (baseline) and 209 s
(affine + decision time, run beside it), validation 11 s. So **a run is about 14 to 18 h,
not the 30 h planned.**

**Wave 1**, launched 08:46 CEST, cores 40 to 74: `ss_base_s801..804` and `ss_aff_s801`.
Expected to finish about 23:00 CEST on 28 September, before zero-box v2 training ends, so
wave 1 does not overlap stage Z.

**Revised schedule.**

| step | duration | expected (CEST) |
|---|---|---|
| wave 1 (baseline arm + 1) | about 14 h | 28 Sep, about 23:00 |
| screen of the baseline arm: the new flow's candidates | about 1 h | 29 Sep, morning |
| remaining 11 of our runs, 10 at a time | about 18 h, twice if 5 at a time | 29 to 30 Sep |
| Malte's 4 runs (environment not yet built on tux26) | under 1 h each | with the above |
| full screen and table | about 2 h | 30 Sep |

The remaining runs overlap stage Z's cells if launched on 29 September; decision 2 of
section 7 (cores during stage Z) is still open for them.

**Wave 2**, launched 11:13 CEST on 28 September at AH's instruction (24 cores): the other
11 of our runs and Malte's 4, six at a time at 4 cores, cores 0 to 23. tux26 was then at
load 103 on 96 cores (another user's 36 R jobs), and epochs slowed to 205 to 266 s, wave 1's
to about 250 s. This supersedes the revised schedule above: wave 1 about 04:30 on
29 September, the baseline screen that morning, wave 2 complete on the morning of
30 September, the full screen and table the same day. Decision 2 of section 7 is moot for
launching; it matters only for how many cores stage Z's cells take while wave 2 runs.

**Correction, 11:51 CEST.** tux26 was not oversubscribed: top shows 25.7% idle. The load
average counts threads, and our runs carry many more threads than they use (a run pinned to
7 cores uses about 3.2, one pinned to 4 uses about 2.4). Epochs are back to 177 s (wave 1)
and 190 to 244 s (wave 2). Expected: wave 1 about 01:00 on 29 September, wave 2 complete
between the late evening of 29 September and the early morning of 30 September.

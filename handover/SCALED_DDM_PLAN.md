# Scaled DDM network: s and a by exact rescaling, t0 by shift, st0 by exact convolution

Decided by AH 2026-10-13: build and train on tux26; st0 by the CDF route; κ box as proposed
(below). Not launched on tux26 until AH says yes to each run (pilot, then full training).

## 1. Why

Every DDM network so far conditions on all eight parameters (v, a, t0, s, Z, SZ, sv, st0),
with s varied over [0.1, 2] in training although every fit fixes s = 1. Three of those inputs
enter the likelihood in closed form:

* **t0** shifts rt: the network should model decision time d = rt − t0.
* **st0** smears d uniformly: L = (1/st0) ∫₀^st0 f(rt − t0 − τ) dτ = [F(rt − t0) − F(rt − t0 − st0)] / st0
  (EMC2's lower-edge window, confirmed numerically in sbc/st0conv_screen.R).
* **a and s** set the scale. Dividing space by a and multiplying time by (s/a)² maps a diffusion
  with (v, a, s, sv, Z, SZ) onto one with boundary 1 and noise 1:

      f(d; v, a, s, sv, Z, SZ) = (s/a)² · g( d·(s/a)² ; κ, η, Z, SZ ),   κ = v·a/s²,  η = sv·a/s²

  Z and SZ are relative to a and do not change; P(R) depends only on (κ, η, Z, SZ).
  Checked against EMC2's dDDM over 300 random sets with s varied (2026-10-13): agreement 5e-5 in
  log density where density > 1, 1.3e-3 where > 0.1; the larger values at density < 1e-2 follow
  the analytic code's own tail precision.

The network therefore needs **four inputs (κ, η, Z, SZ)** and models the unit-scale decision
time u = d·(s/a)². The evidence for doing this:

* a is the parameter most often outside the envelope (cases 2, 3, 4 at both N). Here its time
  scale is exact and it reaches the network only through κ and η.
* st0 was the zero-box's failure (+0.50 to +0.60). Here it is exact. S2 (running) tests the
  convolution on v1.
* the zero box shrinks: η = 0 is sv = 0 exactly; st0 = 0 is the plain density. The network
  learns exact zeros for η and SZ only.
* the support is exact: d ≤ 0 has density 0, so the onset rule needs no special case.

## 2. The network

| item | value |
|---|---|
| inputs | κ (identity), η (asinh(η/0.01), exact 0), Z (probit), SZ (asinh(SZ/0.05), exact 0), then R |
| output | spline flow over log u; classifier for P(R = 2 \| κ, η, Z, SZ) on exact pWDM targets |
| architecture, budget | unchanged: [256, 128] conditioner, 64 bins, batch 1024, 300 epochs, 30 kept checkpoints, 600k draws × 40 rows |
| seeds | 6 × 4 threads (24 cores); more cores do not shorten a run (v1 at 4 threads 29 h, v2 at 12 threads 28.3 h) |

**Box** (AH: "go with your proposal"):

| input | free range | sampling of free draws |
|---|---|---|
| κ | [−10, 10] | 50% uniform on [−10, 10], 50% uniform on [−3, 3] (doubles density where fits live) |
| η | {0} ∪ (0, 8] | 75% log-uniform on [0.01, 8], 25% asinh-uniform on [0, 8] (v2's st0 recipe) |
| Z | [0.1, 0.9] | uniform (as before) |
| SZ | {0} ∪ [0.05, 0.99] | asinh-uniform (as before) |

Regimes: full 55%, η = 0 15%, SZ = 0 15%, both 0 15%. In the original parameters the box covers
any (v, a, s, sv) with |v·a/s²| ≤ 10 and sv·a/s² ≤ 8, any t0, and st0 within EMC2's bound.
The SBC priors (v ≈ 1, a ≈ 0.8, s = 1, sv ≈ 0.1) give κ ≈ 0.4–1.6 and η ≈ 0.05–0.2.
Out-of-box parameters get density 0 (the evaluator's rejection), as now.

The spline's log-u range is set from the pilot data (§4) rather than guessed.

## 3. Evaluation (the "CDF route")

With c = (s/a)², d = rt − t0, u₁ = d·c, u₀ = (d − st0)·c and F the network's defective CDF
p_R·Φ(z(u)) (exactly the integral of its density, since the flow lives on u > 0):

* st0 = 0: L = c · f(u₁)
* st0 > 0: L = [F(u₁) − F(max(u₀, 0))] / st0, with F(0) = 0
* d ≤ 0: L = 0

Two network evaluations per trial (one, at st0 = 0). Very small windows (u₁ − u₀ < 1e-7·u₁)
use the density at the midpoint. The first implementation is an R model function `DDMsc(card)`
(handover/nle-methods/port/R/ddm_scaled.R): DDM() with its likelihood replaced, so EMC2's
transforms, bounds, simulator (rfun) and the zero-exceptions are DDM's own, calling the compiled
flow evaluator in EMC2 dev-nle unchanged. No EMC2 change is needed to train, screen or run SBC.
A compiled version, for speed, follows only if the SBC passes.

**Tests before training:**
1. Oracle test: replace the network with the analytic unit density and CDF (EMC2's dDDM/pDDM at
   a = s = 1); `DDMsc`'s likelihood must equal dDDM with st0 for random parameters, s varied.
2. Card test: the exported card's golden vectors through the R path (as validate_ddm.R).
3. One-epoch smoke run through simulate → train → classifier → export → `DDMsc` → one SBC rep.

## 4. Pipeline and time (24 cores)

| step | where | time |
|---|---|---|
| build + tests 1–3 | laptop | ~1 day |
| pilot: 1 seed, 30 epochs, screen | tux26, **ask first** | ~2–3 h |
| simulate 24e6 rows + choice targets | tux26 | ~0.5 h |
| train 6 seeds × 4 threads | tux26, **ask first** | ~28 h |
| classifier, export ~180 checkpoints, score-bias screen, select | tux26 | ~4–5 h |
| SBC, six network cells (cases 2–5, 7, 8), N = 400, against the existing analytic controls | tux26 | ~4–8 h on the R path |

## 5. Selection and verdict (declared before training)

* Selection: score-bias screen per regime (full, η = 0 i.e. sv = 0, SZ = 0, st0 = 0, all zero),
  rank on mean |g| summed over a, v, Z (not g_a alone: the v1 rule missed g_st0); top 3 to SBC.
* Prediction: st0 clean in every cell; a inside the envelope in cases 2–4; t0 no worse than v1.
* Verdict: the eight-case table again, network against the existing analytic controls.

## 6. Build log

**2026-10-13 15:00–16:00, laptop. Built and tested; nothing on tux26 yet.**

* Simulator: `simulation/scaled.R`, `simulation/sim_ddm_scaled.R` (train | val | choice).
  * Free η starts at 0.01, because EMC2 refuses sv in (0, 0.01). The asinh encoding interpolates down to η = 0.
  * Sobol points are given a seeded random shift. Without it, randtoolbox's disabled scrambling would make the validation draws repeat the training draws.
  * The simulator must run on the dev-nn EMC2 library. On the final-sampler library, `simulate_trials()` puts every stored ll on the floor (the script asserts this doesn't happen).
  * Smoke data: log u spans −11.3 to 0.8, so the spline range is [−12, 2].
* Flow: `--scaled` added to `train_ddm.py`, `retrain_classifier.py` and `export_flow_ddm.py`; `eta` added to the zero-box constants in `data.py`.
  * Old cards' `context_transforms` are unchanged.
  * Training speed: 200k rows run at about 80 batches/s with 4 threads, the same per-step cost as v2, so 300 epochs on 24e6 rows ≈ 24–28 h.
* Likelihood: `port/R/ddm_scaled.R` (`DDMsc`, `ddm_sc_density`, card evaluator, analytic oracle).
* Tests:
  * Oracle (`port/tests/test_ddm_scaled.R`): agreement to 8e-12 at st0 = 0, ≤ 5e-5 with st0 > 0, and exact zeros below t0. **PASS**
  * Card (`port/tests/test_ddm_scaled_card.R`, on a 2-epoch smoke card): log pdf agrees with the golden vectors to 5e-13, CDF to 6e-15. **PASS**
  * Cell wiring (`sbc/scaled_cell.R`, ORACLE=1): the cell's LL equals the analytic control's.
    * In st0only the match is exact.
    * In the full case it differs by −0.25 ± 0.6. That is the control's own precision: EMC2's DDM dfun runs dDDM at precision 0.005, while dDDM at 1e-8 equals the oracle to 1e-4 on a 400-trial set.
    * The registered scaffold model passes SZ as a proportion; the cell converts it to sw.
  * Screen (`sbc/scaled_screen.R`): `sc:oracle` gives g ≈ 0 (≤ 6e-5), as the analytic control does.
  * Two-replicate SBC smoke through `nn_sbc_cell` (st0only): runs end to end.
* Speed (R path, smoke card, N = 400, single thread):
  * full: 0.41 ms against the analytic 1.04 ms.
  * st0only: 0.41 ms against 0.32 ms.
  * A compiled version would remove most of the R overhead.
* tux26 pipeline: `flow/scripts/scaled_pipeline_tux.sh` (STAGE = sim | pilot | train | export).
  * The four edited Python files on tux26 are identical to the local HEAD, so copying the new versions over overwrites nothing.

**2026-10-13 15:49 – 19:07, tux26 (AH approved sim + pilot).**

* Simulation: the training set took 16.1 min, the validation set 1.0 min. 0% of ll at the floor. Log u spans −10.9 to 0.7.
* Pilot `sc_pilot`: 30 epochs in 2.9 h, validation loss −1.436.
* Classifier `clf_sc_big` on the exact P(R) targets: mean |err| 8e-5, max 0.004.
* Export: e27 and e29 exported; e28 failed even at the relaxed tolerance (max 0.028).
  * Port against JAX: p90 |dlog pdf| 1.4e-3, against 1–6e-4 for the zero-box flows. This should be understood before the full export stage.

**2026-10-14 07:30, pilot score-bias screen (laptop, 300 draws, `sbc/scaled_screen/pilot_*.out`; oracle g ≤ 7e-5 in every regime).**

sum |g| over the differenced parameters:

| regime | pilot e29 | v1 zb_s903_e271 | v2 zb2_s901_e274 |
|---|---|---|---|
| full | .026 | .044 | **.008** |
| sv0 | **.008** | .037 | .028 |
| SZ0 | **.006** | .066 | .037 |
| st00 | .025 | **.010** | .016 |
| st0only | **.005** | .030 | .008 |
| SZonly | **.010** | .015 | .051 |
| all0 | **.003** | .017 | .015 |

* st0 g is at or below .001 everywhere (v1 up to .014, v2 up to .010): the exact convolution works.
* The 30-epoch pilot is best in 5 of 7 regimes.
* It loses only where sv and SZ are both free (full, st00): g_a is −.016 to −.018 there. That is the network's own (κ, η, SZ) surface, which the 300-epoch seeds and checkpoint selection address.

**2026-10-14 07:29, tux26: STAGE=train launched (AH approved): 6 seeds × 4 threads, 300 epochs, ETA ≈ 2026-10-15 11:00.**
Export (STAGE=export) needs a separate yes, and the port-tolerance issue above should be checked before it.

**2026-10-14 ~10:30, port-vs-JAX gap resolved (`flow/scripts/port_x64_check.py`, sc_pilot e28, the exporter's golden grid, run on tux26 with 2 threads, AH approved).**
The same network was evaluated as the float64 port, JAX float32 and JAX float64.

| | median | p90 | p99 | max |
|---|---|---|---|---|
| \|JAX f32 − port\| | 3.9e-6 | 1.4e-3 | 8.3e-3 | 2.8e-2 |
| \|JAX f64 − port\| | 7e-15 | 4e-14 | 2e-11 | 1.4e-10 |

The port is exact; the whole gap is JAX float32 rounding.
It sits in the far right tail: log u ≥ −1, with log pdf −12 to −19, which is still above the gate's 1e-10 floor.
Below log u = −1 the f32 error is ~1e-5.
The scaled grid reaches u = 4, deep into that tail. Large |κ| makes it worse (p90 4e-4 at |κ| < 3, 1.4e-3 above).
Not a port defect. For scaled exports the gate should compare with JAX float64.
* Exporter changed (AH approved): `export_flow_ddm.py --ref f64`, the default under `--scaled`, gates the port against JAX float64 at `--port-tol-f64` (1e-8). The float32 gap is still printed, for information.
  * Non-scaled exports keep the original float32 gate (`--ref f32`).
  * Tested on the smoke model: the f64 gap is 2.8e-13 (PASS), the f32 numbers are unchanged, and the golden and weights files are byte-identical to the old export.
  * Synced to tux26, where only the replaced block differed.
  * Verified on tux26 (AH approved): pilot e28, which failed both float32 gates, now exports. Port vs float64 is 1.4e-10 (PASS); the float32 gap is 2.8e-2 as before.

**2026-10-15, tux26 training and export (AH approved each).**
* Training: 6 seeds × 300 epochs, 07:29 on the 14th to 09:21 on the 15th, all exit 0; final val losses −1.4358 to −1.4372.
* Export started automatically at 09:21 and took 5 min: 180 cards (30 per seed). Every card passed the float64 gate (worst 1.8e-10).
  * The "1 failures" in the pipeline summary is the pilot's old e28 line in the shared log.
* Selection screen on the laptop, staged for time:
  * Stage 1: all 180 cards, 7 regimes, 100 draws, 6 cores (`sbc/scaled_screen/run_stage1.sh`, output in `stage1/`).
  * Stage 2: the leaders re-screened at 300 draws.
  * Then the top 3 by sum |g| over v, a and Z, summed over regimes, go to SBC.
* Screen results (stage 2, 300 draws, all 180 cards, oracle |g| ≤ 7e-5 in every regime; `sbc/scaled_select.R` → `sbc/scaled_screen/stage2/ranking.rds`):
  * **Selected: sc_s1106_e245 (0.042), sc_s1101_e276 (0.046), sc_s1103_e272 (0.053).** Score = Σ over regimes of |g_v|+|g_a|+|g_Z|.
  * Stage 1 (100 draws) picked the same top 2 in the same order; its third was s1101_e271, now fifth.
  * The comparators score 0.137 (v2), 0.170 (v1) and 0.082 (the pilot), from their 300-draw pilot screen.
  * The worst single regime for the top 3 is ≤ 0.012, against v2's 0.048 (SZonly).
  * max |g_st0| is 0.0050 over all 180 cards, against 0.010–0.014 for v1/v2.
  * Seeds vary: the best per seed ranges from 0.042 to 0.081.

**2026-10-15 12:36 CEST, SBC on AWS (AH: "use that AWS setup"; rules in ~/.claude/plans/aws/nle_sbc_handover.md).**
* 18 cells: the 3 selected cards × cases 2, 3, 4, 5, 7, 8 (sv0, SZ0, st00, st0only, SZonly, full). N = 400, 500 replicates, network-only against the existing analytic controls in sbc/nci/results_8case.
* Machine: on-demand c7a.32xlarge `i-045ce187ef1199680` (128 cores, $7.03/h), tag Project=nle.
  * Each cell runs 3 data sets × 3 chains.
  * Estimate ≈ 5 h ≈ $37. The cap (MAX_SECONDS 6.5 h) bounds the cost at $46.
  * The driver syncs to `s3://emc2-results-434352151599/nle/scaled_sbc/run1/` every 5 min and terminates the machine at the end.
* EMC2 is dev-nle 142bd497 (the final sampler, same as ~/R/nle-lib and the controls). It is not on GitHub, so it was shipped with `git archive` and built clean.
* Bundle: `sbc/aws/scaled_sbc/bundle` (setup.sh, run_all.sh, smoke.sh).
* Test machine (c7a.4xlarge, ~6 min, ≈ $0.10), now terminated:
  * Oracle wiring is exact.
  * Network vs analytic LL over 200 draws: cor .9999, diff −6.8 ± 7.0.
  * About 80 s per replicate on 3 cores.

- 2026-10-16 **AWS run1 result: the scaled network passes; sc_s1106_e245 is out of the envelope once in 35 parameter tests (chance level).**
  - Run: all 18 cells finished, 0 failed; cells 10:36 → 16:59 UTC. The machine self-terminated after the final sync; nothing tagged Project=nle is left. Cost about 6.45 h × $7.03 ≈ $45, plus $0.10 for the test machine. The ETA of 5.6 h was low.
  - Local copy (summaries only): `sbc/aws/scaled_sbc/run1/`. Table script: `sbc/scaled_8case_table.R`. References are the N = 400 cells (case 4 v1 = `results_zerobox_v1_aa66/st00`).
  - Out of the 95% envelope, N = 400, 500 replicates:

    | Case | old network | analytic control | s1106_e245 | s1101_e276 | s1103_e272 |
    |---|---|---|---|---|---|
    | 2 st0+SZ | a, st0 | none | none | none | none |
    | 3 st0+sv | a, st0 | none | none | a | none |
    | 4 sv+SZ | a, t0 | none | none | t0 | t0 |
    | 5 st0 | t0, st0 | t0 | a | none | a |
    | 7 SZ | Z | none | none | none | none |
    | 8 all | t0, Z (c4, uncorrected) | — | none | a, t0 | t0, Z, st0 |

  - Worst |std bias| for s1106: .13 (t0, case 8); old networks .15–.50. Coverage .922–.972. Min KS p: s1106 .020, s1101 8e-4, s1103 3e-3.
  - Prediction (st0 clean; a in the envelope in cases 2–4; t0 no worse than v1): **met by s1106**. s1101 fails a in case 3; s1103 fails st0 in case 8 (std bias +0.04).
  - Case 8 needs no t0 correction with s1106 (t0 +0.13, in the envelope), which replaces c4 plus the 0.0019 shift.
  - Residual pattern across all three cards: t0 +0.05 to +0.19 and a −0.03 to −0.12, small and mostly inside the envelope.

- 2026-10-16 **AH: adopt sc_s1106_e245 as the single DDM network, called "DDMnle" from now on** (card `handover/nle-methods/port/export/ddm_sc_s1106_e245_weights.json`). PDF of its six SBC cells: `sbc/aws/scaled_sbc/run1/sc_s1106_e245_SBC_6cases.pdf`.
- 2026-10-16 **DDMnle speed vs WienR** (EMC2's DDM, default precision; `BENCH=1 Rscript sbc/scaled_cell.R <regime> <card>`; N = 400, single thread, 200 prior draws, ms per likelihood). The laptop was also running the 8-core LAN SBC at the time.

  | case | WienR | DDMnle (R glue) | speed-up | network call alone | speed-up if compiled (est.) |
  |---|---|---|---|---|---|
  | 2 st0+SZ | 1.19 | 0.52 | 2.3x | 0.16 | ~7x |
  | 3 st0+sv | 0.36 | 0.46 | 0.8x | 0.16 | ~2.2x |
  | 4 sv+SZ | 0.35 | 0.38 | 0.9x | 0.10 | ~3.5x |
  | 5 st0 | 0.34 | 0.41 | 0.8x | 0.16 | ~2.1x |
  | 7 SZ | 0.31 | 0.33 | 0.9x | 0.10 | ~3x |
  | 8 all | 1.18 | 0.45 | 2.6x | 0.16 | ~7x |

  DDMnle runs on EMC2's R likelihood path (`port/R/ddm_scaled.R`), whose glue costs about 0.25–0.35 ms per evaluation. The network call alone (`ddm_ens_eval_trials_cpp`, one parameter set, 400 rows; 800 rows when st0 > 0 needs two CDFs) is the floor a compiled `c_name` path would approach. The "compiled" column is that estimate, not a measurement.
- 2026-10-16 **LAN SBC, case 5, in the LAN's own training box** (AH). The network is HSSM `ddm_uniform_st`, using EMC2-nle's converted card. The control is the analytic DDM in the LAN's parameterisation (`sbc/lan_case5/lan_model.R`). At the truth on one data set the control matches WienR 1e-8 to 0.08; the LAN is 3.1 log units below.
  - Prior (sampled scale), ±4 sd inside the box: v N(0.5, 0.6); log a_lan N(log .75, .2); probit z N(0, .12); log t N(log .55, .15); log st N(log .075, .25).
  - Laptop, 8 cores: network 1×2, control 2×3. Run with `sbc/lan_case5/run.sh`; output in `sbc/lan_case5/results`. Launched 09:53.
  - Per likelihood (400 trials): LAN 0.14 ms (compiled mlp_joint), control 0.42 ms.
  - 3-replicate smoke test: v std bias +4.3 and coverage 0/3. That is three replicates only, but it is an early sign.
  - Rebalanced at 10:08: the LAN's replicates take ~4x longer than the control's (about 2 min each at 2 cores, against 0.75 min at 3), so its sampler works harder despite the faster likelihood. `sbc/lan_case5/run2.sh` waits for the control (~3 h), then restarts the LAN at 4×2 cores, resuming from the saved replicates. Expected end about 16:30.
- 2026-10-16 **DDMnle on EMC2's compiled path** (AH: "move the glue into EMC2's compiled path").
  - EMC2-nle local branch `dev-nle-scaled`, off 142bd497, uncommitted; diff saved as `sbc/ddmnle_bench/emc2_dev-nle-scaled.patch`. Built clean (from a copy of the tracked files) into `~/R/nle-scaled-lib`; `~/R/nle-lib` is untouched.
  - The change:
    - `src/model_NN.h`: a `scaled` flag on the flow_joint native spec. `nn_trial_ll_scaled()` reads v, a, sv, t0, st0, s, Z, SZ from the ParamTable and computes kappa, eta, Z, SZ and c = (s/a)². It issues one network call for every density request (st0 = 0) and every CDF pair (st0 > 0), then forms c·g or [G(u1) − G(u0)]/st0.
    - `src/flow_ddm.cpp` / `nle_native.h`: a new `nle_ddm_native_full()` that returns the log pdf and the CDF together.
    - R side: `ddm_sc_native_reg()` in `port/R/ddm_scaled.R`, and `NATIVE=1` in `sbc/scaled_cell.R`.
  - Compiled agrees with the R reference to ≤ 1.1e-11 in the log-likelihood over 200 prior draws in each of the six cases. So the AWS SBC results (run on the R path) carry over unchanged.
  - Trap found: building the registration inside the model function re-parsed the card JSON on every likelihood batch (about +0.25 ms per particle at 200 particles). It must be built once.
  - **Timings** (N = 400, single thread, 200 draws, ms per likelihood; laptop also running the 8-core LAN SBC):

    | case | WienR | DDMnle R path | DDMnle compiled | speed-up |
    |---|---|---|---|---|
    | 2 st0+SZ | 1.17 | 0.44 | 0.125 | **9.3x** |
    | 3 st0+sv | 0.34 | 0.42 | 0.120 | **2.9x** |
    | 4 sv+SZ | 0.34 | 0.36 | 0.080 | **4.3x** |
    | 5 st0 | 0.36 | 0.41 | 0.125 | **2.9x** |
    | 7 SZ | 0.33 | 0.35 | 0.075 | **4.5x** |
    | 8 all | 1.22 | 0.46 | 0.125 | **9.8x** |

    At N = 4000 (case 4): WienR 3.18, compiled 0.53 (6x), R path 1.59. The cases with st0 > 0 cost about 0.045 ms more because they need two CDF evaluations per trial.

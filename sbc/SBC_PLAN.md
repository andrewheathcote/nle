# SBC plan: neural-network likelihoods in EMC2

Status: finalized 2026-07-10, informed by the laptop precision study
(`appendix_precision_study.md`). Headline inputs: DDMnn information cost is
small (κ ≈ 0.88, crossover ≈ 114 trials at base 100) but it carries stable
systematic biases (t0 ≈ +3%, a ≈ −7%) that trials cannot fix; race-model nn
likelihoods show no such pathology (local noise/signal 0.11–0.17).

## Phase 0 (laptop, ~1–1.5 h): compression arm

**Question.** EMC2's general default floors rts to a 1/60 s grid
(`design_model(compress = TRUE, rt_resolution = 1/60)`), a leading-edge
shift of up to ~17 ms. The SBC machinery itself runs uncompressed
(`compress = FALSE, rt_resolution = 1e-12`), as did the precision study —
so the observed flow bias is NOT compression-induced. But real-world fits
compress, t0 is leading-edge sensitive, and the flow's leading-edge
distortion may interact with flooring differently than the analytic
likelihood.

**OUTCOME (run 2026-07-10, `compression_arm_results.rds`).** Flooring shifts
t0 by −res/2 (−8.4 ms at 1/60, −4 ms at 1/100, nil at 1/1000), identically
for both likelihoods — no flow×compression interaction. The flow's a-bias is
untouched (intrinsic); its t0-bias coincidentally cancels the 1/60 shift at
the studied generating point (do not rely on this). Default 1/60 biases t0
by ~−1.9 posterior sd at N=100 even for the analytic DDM → recommend
rt_resolution ≤ 1/1000 for t0-sensitive work. Phase 1 unchanged
(uncompressed). Details: appendix_precision_study.md §A.4.

**Design.** Reuse `precision_component.R` (per-spec `make_emc_args` +
label-keyed cache). At N = 100 (4 replicate datasets, same seeds as the main
study so uncompressed cells come from cache):

| spec label | model | compress | rt_resolution |
|---|---|---|---|
| DDM (cached) | DDM | FALSE | 1e-12 |
| DDMnn (cached) | DDMnn | FALSE | 1e-12 |
| DDM_r60 | DDM | TRUE | 1/60 |
| DDMnn_r60 | DDMnn | TRUE | 1/60 |
| DDM_r1000 | DDM | TRUE | 1/1000 |
| DDMnn_r1000 | DDMnn | TRUE | 1/1000 |
| DDMnn_r100 | DDMnn | TRUE | 1/100 |

= 20 new fits (12 analytic ≈ 20–26 min each, 8 nn ≈ 2 min each) ≈ 1 h on 9
cores. **Outcome measures:** shift in t0/a posterior medians relative to the
uncompressed cell of the same model, separately for analytic (pure
compression effect — note flooring should bias t0 *down* by ≈ res/2) and nn
(compression × flow interaction). **Decision:** if 1/60 moves t0 by an
amount comparable to its posterior sd (or to the flow bias) while 1/1000
does not, recommend `rt_resolution <= 1/100` for nn fits (and possibly in
general) in the EMC2 docs; SBC phase 1 stays uncompressed either way.

## Phase 1 (NCI): calibration cells

All cells: single participant, vignette designs (`run_sbc_nn.R`), 500
replicates, priors as in the precision study (all mass inside training
boxes; SZ mean qnorm(.2) deviation documented), uncompressed (SBC default).

| cell | trials | purpose | est. wall @ 48 cores |
|---|---|---|---|
| LNR | 100 | SBC machinery ground truth (race) | ~0.5 h |
| LNRnn | 100 | race nn calibration | ~0.5 h |
| RDM | 100 | ground truth | ~0.5 h |
| RDMnn | 100 | race nn calibration | ~0.5 h |
| DDM | 100 | ground truth (expensive cell) | ~4.5 h |
| DDMnn | 100 | headline: nn calibration at base | ~0.5 h |
| DDMnn | 200 | predicted degradation with N | ~0.6 h |
| DDMnn | 400 | ditto (covers precision crossover region too) | ~1 h |

Predictions from the precision study: LNRnn/RDMnn pass (rank ecdfs within
95% bands, coverage ≈ .95); DDMnn shows miscalibrated t0 (ranks shifted low
— truth below posterior mass) and a (shifted high), worsening from N=100 to
400; v/Z/SZ calibrated; st0/sv mildly off at 400.

**OUTCOME (run 2026-07-10 on Gadi; analysis sbc/analyze_sbc.R, summary
sbc/nci/analysis/).** All predictions confirmed: analytic cells fully
calibrated; RDMnn PASS; DDMnn t0 coverage .52/.25/.10 and a .61/.32/.15
across N=100/200/400 (mean ranks → 0/1), st0 degrading (.94→.79), sv KS-
failing from N=200; v/Z/SZ clean throughout. One unpredicted wrinkle:
LNRnn s ranks non-uniform (KS p=1.5e-5, mean rank .43, bias +0.011) with
intact coverage — mild, monitor. Precision-study gates validated as SBC
screening proxies. Appendix §A.7.

Outputs per cell: rank ecdf (`plot_sbc_ecdf`) and histogram, coverage, mean
bias, sd of posterior medians; archived .rds per cell (`sbc/results/`).

PBS setup (to finalize with NCI details): one job per cell,
`ncpus=48, mem=190GB`, walltime from table ×1.5 safety;
`Rscript sbc/run_sbc_nn.R <model> <trials> 500 48 sbc/results`.
Requires dev-nn EMC2 installed in the NCI R library.

## Phase 2 status (2026-07-10): budget arm resolved, architecture arm queued

Step 0 (checkpoint-based, no training) ran the gates across the 50k / 200k /
1M budgets: κ = 0.48 / 0.81 / 0.93 (budget buys precision) but the a/t0
biases have log-log budget slopes +0.26 / +0.38 (capacity floor, not data
limit) — all budgets FAIL G1. Error map (`sbc/edge_error_map.R`): the flow
overshoots the leading edge (+0.8–1.0 log units) and deep tail (+6),
budget-invariantly. **Data arm rejected; architecture arm queued**:
`flow/scripts/train_variants_tonight.sh` trains arch64 (64 spline bins) and
arch64w (64 bins + [9,64,32] conditioner) on the existing 1M data
(~45–90 min each, then auto export + golden validation), classifier copied
from the large budget. Tail overshoot (drives the growing sv/st0 biases)
deferred: leading candidate is training with a censored likelihood (upper
censoring ≈ 10 s) so the flow never models the unsampled deep tail.
Results: `sbc/budget_study_results.rds`; write-up: appendix §A.5–A.6.

**ARCH ARM OUTCOME (2026-07-11, `sbc/arch_study_results.rds`).** arch64
(bins only): edge error unchanged — spline resolution not binding. arch64w
(bins + [9,64,32]): **leading-edge problem solved** — t0 |bias| 0.029→0.005
(std +0.1 sd at N=800), a 0.068→0.009, κ = 1.02. G1 still FAILS on the tail
params: st0 +2.2 sd at N=800 (worse than large's +1.7; residual error
redistributed). Sole remaining blocker = unsampled tail.

**CENSOR ARM OUTCOME (2026-07-12, `sbc/censor_study_results.rds`).**
arch64wC (censored MLE, C=10 s) did NOT fix the tail: st0 +2.7 sd @N800
(vs +2.2 uncensored); t0 slightly better, a/v/Z slightly worse; κ 0.97.
Mechanism: only 0.34% of trials beyond 10 s, and the distortion-driving
region (~1–3 s upper shoulder) is far below the censor. Candidate next
steps: (i) narrow the spline support (range [−5,5] → e.g. [−2.5,2.5],
rt 0.08–12 s) — concentrates all knots on the data range, structural
analogue of the conditioner-capacity win; (ii) seed-replicate arch64w to
separate real variant effects from training noise (single-run caveat on
all variant contrasts); (iii) lower censor combined with EMC2-side UC
censoring of fits.

## Phase 2: training-budget sufficiency methodology

Goal: a reusable yardstick for "budget is sufficient for estimation",
defined as matching the race-model nn quality observed with the current
large budget.

Procedure for each candidate flow (e.g., DDM at a larger budget, trained on
the laptop):
1. Export + golden validation (`export_flow_ddm.py`, `validate_ddm.R`) —
   correctness gate, minutes.
2. Precision component, nn cells only (analytic reference and previous
   budgets all cached): 16 fits ≈ 40 min laptop. Gates, race-calibrated:
   - **G1 (bias):** |standardized bias| ≤ 0.5 posterior sd for every
     parameter at N = 800 (race levels; current DDM large budget fails at
     3.3–3.7 sd for t0/a).
   - **G2 (noise):** local error-noise/signal ratio ≤ 0.2 (race: 0.11–0.17;
     current DDM: 0.53).
   - **G3 (information):** κ ≥ 0.8 for well-identified parameters at
     N ≤ 800.
3. Only if G1–G3 pass: SBC cell (DDMnn' at N=100, optionally 400) on NCI to
   confirm calibration.

The same pipeline doubles as the criterion for any *new* model's flow
(race- or DDM-character) before it is exposed to users.

## Notes / risks

- SBC draws truths from the prior; priors are deliberately tight and
  box-interior. Conclusions are relative to this operating region — that is
  the intended scope (calibration where the flow claims validity).
- Both laptop study kills were consistent with memory pressure or host
  sleep; per-cell caching makes all phases resumable at zero recompute cost.
- The race SBC cells double as the empirical reference levels for the G1–G3
  gates; revisit gate numbers once those cells are in.

## Phase 3 (NCI): calibration of the production ensemble (ens5)

Status: designed 2026-07-14 (`sbc/nci/SBCens.{R,pbs}`), predictions registered
2026-08-31, **not yet run**. Requires EMC2 dev-nn at >= 85dde2a6 on Gadi (the
Phase 1 install predates the ensemble); `SBCens.R` guards with
`stopifnot("ddm_ens_eval_trials_cpp" %in% ls(getNamespace("EMC2")))`.

**Question.** Phase 1 measured the *large-budget single flow*. Production
`DDMnn` is now ens5 — an equal-weight density mixture of five arch64w flows
(seeds 100-104), which on the laptop gates fixed the leading edge (t0, a),
collapsed the seed-noise components as sqrt(k), and left `st0` as the sole
systematic blocker. Phase 3 asks whether SBC confirms that, i.e. whether the
gate improvements translate into restored *calibration* and not merely
smaller point biases.

**Design.** `DDMnn` (= ens5) at N = 100 / 200 / 400, 500 replicates, 48 cores,
uncompressed, same `sbc_specs.R` cells as Phase 1 -> `results_ens5/`. Analytic
DDM and single-flow DDMnn cells from Phase 1 (`results/`) are the references;
no re-run needed.

### Registered predictions

Derived from the ens5 gate study (`sbc/ensemble_study_results.rds`, 4
replicates per cell) via the shift model that reproduced Phase 1 to within
about .05 coverage: for standardized bias b, expected 95% coverage is
`Phi(1.96 - b) - Phi(-1.96 - b)`. (Phase 1 check: b = 1.03 -> .82 obs .79;
b = 2.05 -> .46 obs .52; b = 2.72 -> .22 obs .32; b = 3.57 -> .05 obs .15.)

| par | ens5 std bias @ N=100/200/400 | predicted coverage | Phase 1 single-flow coverage |
|---|---|---|---|
| t0  | +0.06 / +0.57 / -0.21 | .95 / .91 / .94 | .52 / .25 / .10 |
| a   | -0.32 / -0.46 / -0.93 | .94 / .93 / .84 | .61 / .32 / .15 |
| st0 | +0.41 / +0.51 / +1.67 | .93 / .92 / **.64** | .94 / .89 / .79 |
| sv  | +0.01 / +0.01 / +0.03 | .95 / .95 / .95 | .95 / .95 / .96 (KS-fails from N=200) |
| v   | -0.50 / -0.30 / +0.03 | .93 / .94 / .95 | .96 / .97 / .97 |
| Z   | +0.06 / -0.09 / +0.41 | .95 / .95 / .93 | .97 / .95 / .97 |
| SZ  | +0.04 / +0.04 / +0.08 | .95 / .95 / .95 | .95 / .97 / .94 |

Headline predictions, in falsifiable order:

1. **t0 and a calibration is restored.** Coverage >= .90 at every N, rank
   ecdfs inside the 95% bands. This is the main claim: the arch64w
   conditioner plus seed-ensembling fixes the leading-edge pathology that
   dominated Phase 1. Failure here falsifies the gate-as-screen methodology.
2. **sv is repaired.** Phase 1 KS-failed from N=200; ens5 gate bias is ~0 at
   every N, so sv should now pass KS throughout.
3. **st0 is the sole survivor, and it is WORSE than the single flow at
   N=400** (predicted .64 vs observed .79). The residual error redistributed
   into st0 when the leading edge was fixed; ensembling did not touch it
   because it is systematic, not seed noise. A monotone degradation
   .93 -> .92 -> .64 is the signature to look for.
4. **v/Z/SZ stay calibrated** at these N. (Their gate biases only turn
   troublesome at N=800, outside this grid.)

Caveats on the predictions: gate cells use 4 replicates at a single
generating point, so the std biases carry real noise (the seed/range arm put
it at +-1-2 sd at N=800; less at N<=400). SBC draws truths across the prior
region, so systematic agreement matters more than per-cell agreement.

### st0 as the mechanism-study base

The st0 rank ecdf from this campaign is the intended base measurement for the
follow-up mechanism study. Point estimates give one number per cell; the rank
ecdf over 500 prior draws shows the *shape* of the miscalibration, which is
what discriminates the candidate mechanisms:

- **Uniform displacement** (ecdf shifted, parallel to the diagonal) => the
  flow misrepresents the upper shoulder the same way everywhere in the prior
  box, consistent with the budget-invariant edge-map overshoot; the fix is
  architectural or loss-side.
- **Region-dependent** (ecdf S-shaped, or coverage varying with the drawn st0
  / a / t0) => the distortion is local to part of the box, and a targeted
  remedy (importance-weighted training over the shoulder, or a wider prior
  for st0) becomes viable.

Diagnostics to compute alongside the standard ecdf: rank vs drawn st0, rank
vs drawn a, and the shoulder-region (1-3 s) log-density residual from
`sbc/edge_error_map.R` evaluated at each draw's truth. Recall what is already
ruled out: censoring at 10 s (only 0.34% of trials beyond it), spline-range
narrowing (Z regressed, st0 unchanged), and training seed (ensembling
collapsed every other component but not this one).

### Resourcing

Phase 1 single-flow DDMnn cells took 37 / 50 / 75 min (162 min total) at 48
cores. The ensemble evaluates 5 member flows per likelihood call, so budget
3-5x: roughly 8-13 h for the three cells sequentially. **`SBCens.pbs`
currently requests 8 h, which is likely insufficient** — either raise it to
24 h (Gadi `normal` allows 48 h at ncpus <= 48) or split into one job per N
so the cells run concurrently and each is comfortably inside its own limit.

### OUTCOME (run 2026-08-31 on Gadi; analysis sbc/nci/analysis_ens5/)

Three cells, 500 replicates each, 55 / 75 / 112 min wall at 48 cores. The
ensemble cost only 1.49x the single flow, not the 3-5x budgeted — the
duplicate-row cache conditions each member once per unique row, so members
2-5 are nearly free. Walltime headroom was ample.

**Scorecard: mean |error| = .023 over 21 predictions (MC se .010).**

| par | N=100 pred/obs | N=200 pred/obs | N=400 pred/obs |
|---|---|---|---|
| t0  | .95 / **.938** | .91 / **.944** | .94 / **.956** |
| a   | .94 / **.962** | .93 / **.954** | .84 / **.934** |
| st0 | .93 / **.940** | .92 / **.836** | .64 / **.666** |
| sv  | .95 / .950 | .95 / .950 | .95 / .956 |
| v   | .93 / .940 | .94 / .934 | .95 / **.900** |
| Z   | .95 / .942 | .95 / .940 | .93 / .968 |
| SZ  | .95 / .932 | .95 / .934 | .95 / .946 |

**Prediction 1 (t0/a calibration restored): CONFIRMED for t0.** Coverage
.938/.944/.956 with mean rank .425/.425/.453 — improving toward .5. The
leading-edge pathology that gave .52/.25/.10 in Phase 1 is gone. The
gate-as-SBC-screen methodology stands.

**Prediction 2 (sv repaired): CONFIRMED.** .950/.950/.956, KS p .93/.59/.06;
Phase 1 KS-failed from N=200.

**Prediction 3 (st0 sole survivor, worse than the single flow at N=400):
CONFIRMED, quantitatively.** .666 observed against .64 registered (2.7 MC
se), versus the single flow's .788. The predicted monotone collapse
.93 -> .92 -> .64 came out .940 -> .836 -> .666: right shape, right
endpoint, but the onset is one cell earlier than registered (st0 was already
broken at N=200, where .836 missed .92 by 8.6 MC se — the single genuinely
failed prediction).

**Prediction 4 (v/Z/SZ stay calibrated): FAILED for v.** Z and SZ hold, but
v degrades .940/.934/.900 with mean rank .557/.571/.613 and KS p 7e-13 at
N=400. Not anticipated; v was clean throughout Phase 1.

**Unregistered: a's pathology is re-emerging.** Coverage .934 at N=400 is far
better than the predicted .84 (the largest miss, +.094), but mean rank drifts
.523 -> .596 -> .668 with KS p 6.9e-31 — the Phase 1 direction (.872 ->
.942 -> .979) at a much slower rate. **Methodological point: coverage is too
blunt for these two. Both v and a look acceptable on coverage and are
unambiguous on mean rank + KS at 500 replicates.** Gates built on coverage
alone would have passed a flow that is still drifting.

### Mechanism: region-dependent, coupled to a

The N=100 and N=200 cells alone suggested a uniform displacement; N=400
overturns that. Mean st0 rank by quintile of drawn st0, as displacement from
the calibrated analytic DDM cell (.148/.368/.538/.690/.856):

| N | Q1 | Q2 | Q3 | Q4 | Q5 | shape |
|---|---|---|---|---|---|---|
| 100 | -.069 | -.175 | -.177 | -.137 | -.102 | mid-peaked |
| 200 | -.112 | -.224 | -.294 | -.298 | -.302 | plateau |
| 400 | -.114 | -.302 | -.414 | -.515 | -.552 | monotone growing |

The distortion concentrates at **high drawn st0** and sharpens with N.
Corroborating: rank dispersion collapses .975 -> .762 -> .398; delta
(rho minus the analytic twin's) runs -.019 -> -.130 -> -.306; and st0-rank
vs drawn-**a** grows +.103 -> +.138 -> **+.225** (p 3.9e-7, decisively past
Bonferroni over 21 tests), the only cross-parameter correlation significant
at any N.

Note the control that makes this readable: rho against a parameter's OWN
drawn truth is NOT a miscalibration signal — it reaches +0.999 in calibrated
analytic cells, because a weakly identified parameter has posterior ~ prior,
so rank ~ prior CDF of truth (correlated AND uniform). All statements above
are relative to the analytic twin.

**This selects the targeted-remedy branch** of the Phase 3 design over the
architectural one, and explains why censoring at 10 s and range-narrowing
both did nothing: neither touched the region where the error lives. Leading
candidate is importance-weighted training over the high-st0 / upper-shoulder
region. Next diagnostic: `sbc/st0_mechanism.R` (per-draw shoulder residual
against WienR, correlated with the SBC rank).

### Practical guidance

DDMnn/ens5 is fit for use to N ~ 100-200 provided st0 is not of inferential
interest; st0 should not be trusted beyond N=100; v needs watching above
N=200. t0, a, Z, sv, SZ are sound over the studied range on coverage, with
the a/v rank drift noted above as the open caveat.

### Mechanism study outcome (2026-08-31, sbc/st0_mechanism.R, N=400, 500 draws)

Per draw, the ens5 flow's log-density error against WienR (precision 1e-8) at
that draw's own generating parameters, over the upper shoulder (1-3 s) with
leading-edge and bulk controls, plus `kl` = the density-weighted error
(E_true[log f_flow - log f_true] = -KL, the expected per-trial log-likelihood
bias — the inference-relevant statistic, since an unweighted tail residual can
be +5 log units in a region carrying no mass).

Magnitudes: kl +0.026 (sd .010) per trial — ~10 log units of systematic
offset over 400 trials, ample to move a posterior. Unweighted: shoulder
+4.98, edge +0.41, bulk +0.10.

Raw correlations with the st0 rank looked promising — edge +0.319
(p 2.7e-13), bulk -0.291 (p 3.2e-11), shoulder -0.234, edge-minus-bulk
contrast +0.300 — suggesting the flow trades leading-edge against bulk
density and st0 absorbs it.

**That does not survive control. Partialling out drawn st0 AND drawn a,
every residual correlation collapses to zero: edge -0.012 (p .79), bulk
+0.019 (p .67), shoulder +0.001 (p .98), kl -0.040 (p .37).** The raw
associations were mediated by the drawn parameters (rho(edge, drawn st0)
= +0.186, rho(kl, drawn st0) = +0.371). Consistently, the shoulder residual
is flat across drawn-st0 quintiles (+5.17/+5.04/+5.05/+4.71/+4.92) while the
mean rank varies six-fold (.034 -> .304), and kl adds only R^2 .3236 ->
.3299 over st0 + a.

**Conclusion: the likelihood error measured AT the truth — its magnitude, its
regional shape, and its density-weighted total — carries no information about
which draws are miscalibrated, beyond where the draw sits in (st0, a) space.**
This rules out the working hypothesis carried since the censor arm (that the
distortion "lives at the 1-3 s upper shoulder"): that region's error is large
but inert, which is also why censoring at 10 s changed nothing.

What this leaves: posterior displacement is driven not by the error at the
truth but by how the error VARIES as st0 moves — the tilt the flow imposes on
the likelihood along the st0 axis. A residual can be uniformly +5 log units
and displace nothing; a residual with a slope in st0 displaces the mode.
**Next diagnostic: finite-difference d(log f_flow - log f_true)/d(st0) at each
draw (and the same in a), which is cheap and directly predicts displacement.**
Region-targeted importance weighting is NOT clearly indicated on the present
evidence, since no region's error magnitude predicts the failure.

### st0 slope diagnostic — MECHANISM IDENTIFIED (2026-08-31, sbc/st0_slope.R)

Where st0_mechanism.R measured the LEVEL of the flow's error, this measures
its GRADIENT. Define L(theta) = E_true[log f(rt,R|theta)] under the true
defective density at each draw's own truth. For the true model dL/dtheta = 0
at the truth; for the flow it is not, and that gradient is the force that
displaces the posterior mode:
  shift = N*g / (1/psd^2 + N*I),   z = shift * sqrt(1/psd^2 + N*I)
with g = dL_flow/dst0 (central difference, h=.05 on the log scale) and
I = -d2L_true/dst0^2 from WienR (unit Fisher information).

**Estimator validation is mandatory and built in**: the same estimator applied
to the ANALYTIC likelihood must return g ~ 0 at the truth. A first version
weighted the rt grid by density alone; because the grid is log-spaced, that is
not an expectation, and it returned g_analytic = -0.25 (st0) / -1.46 (a) —
producing slope results with the WRONG SIGN. Including the trapezoid interval
widths gives g_analytic = -0.00017 across all three cells. t0 is deliberately
not profiled: it shifts the support boundary, so the difference straddles a
discontinuity and does not validate (g_analytic ~ -0.7).

**Result: a constant per-trial score bias in st0.**

| cell | g_st0 | I_st0 | predicted z | observed b_implied |
|---|---|---|---|---|
| N=100 | +0.0274 | 0.168 | 0.34 | 0.29 |
| N=200 | +0.0267 | 0.164 | 0.59 | 0.98 |
| N=400 | +0.0286 | 0.173 | 1.04 | 1.53 |

g is the same in all three cells (+0.027 to +0.029) — as required, since it is
a property of the likelihood at the truth, not of N. Three independent sets of
500 prior draws agree. The sign is positive: the flow's likelihood prefers
LARGER st0, predicting overestimation, and st0's observed bias is +0.172 with
mean rank .140 (low = overestimate). **The entire degradation curve
.940 -> .836 -> .666 follows from one per-trial gradient**, with the 1-D
profile recovering 60-70% of the displacement; the shortfall is coupling,
which the st0-vs-a rank correlation (+.225) independently indicates.

Across draws the tilt again fails to predict WHICH draws miscalibrate once
drawn st0 and a are controlled (+.013 at N=400). That is not a failure of the
account — g barely varies over the box (quintile range +.022 to +.036). The
bias is systematic, not draw-specific, so it afflicts every draw alike.

**a is coupling-driven, not intrinsic.** g_a = +0.057 but I_a = 5.84, giving
only +0.010 displacement, and with the WRONG sign against a's observed bias
of -0.012. a is dragged as st0 is pushed, which fits its much milder drift and
suggests it resolves if st0 does.

### Implication for the fix

Maximum-likelihood training minimises KL over the training distribution, which
constrains the LEVEL of the density error and places no constraint on its
gradient in theta. A flow can have excellent held-out likelihood and still
carry a systematic score bias — which is what ens5 does. This is also why the
censor arm failed: a uniform offset in log-density, however large, moves the
log-likelihood without moving its argmax.

Candidates, in order of principle:
1. **Score penalty in training** — add a term targeting E_true[d log f_flow /
   d theta] at sampled theta, so the flow is trained to have the right
   stationary point and not merely the right density. Changes the loss in
   flow/scripts/train_ddm.py. Unlike the censor and range arms this targets
   the quantity that actually displaces posteriors.
2. **Post-hoc score correction** — g is near-constant, and the machinery now
   measures it over the box in ~3 min per 500 points; apply a correction to
   the st0 mapping. A patch rather than a cure, and it needs its own SBC.

## Phase 4 (NCI): the identification 2x2 for sv and SZ

Status: designed 2026-09-01. Scripts `sbc/nci/{sbc_specs_id.R, SBCid.R,
SBCid_*.pbs, launch_sbc_id.sh}`. NOT yet run; awaiting the production flow
decision (score-penalty weighting, below).

**Why.** Phases 1-3 evaluated calibration at a single generating point,
sv=0.1 / SZ=0.2, and the score-bias work of 2026-08-31 showed that point is
pathological: the measured unit information is I_sv = 3e-6, so the likelihood
would need ~14 MILLION trials to match the prior. sv's posterior there IS the
prior, so its uniform SBC ranks were close to content-free, and the moment the
score penalty perturbed it, coverage fell to .47. SZ is milder (I = 0.0134,
parity at N ~ 3300) but still prior-dominated at N <= 400.

Crucially, **no design can fix this**: Fisher information is per-trial and
additive, so a difficulty manipulation contributes N x mean(I_c) over cells
that each carry I_sv ~ 0. Measured across v in {0.5,2} and {0,1,2,3}, I_sv
stays at 0.000. Only the generating VALUE of sv moves it (I ~ sv^4), and only
N multiplies it.

**Design.** A 2x2 crossing the sv/SZ generating magnitude with N, which
separates "more trials" from "a value the data can speak to". Ratios below are
likelihood-info / prior-info (>1 = likelihood dominates), measured against the
analytic DDM with prior psd 0.15:

| cell | sv | SZ | N | sv ratio | SZ ratio | reads |
|---|---|---|---|---|---|---|
| P1 | 0.1 | 0.2 | 400  | 0.00 | 0.12 | neither identified (= the Phase 1-3 point) |
| P2 | 0.1 | 0.2 | 3200 | 0.00 | 0.97 | N identifies SZ only |
| P3 | 1.5 | 0.4 | 400  | 0.33 | 0.67 | magnitude alone is not enough |
| P4 | 1.5 | 0.4 | 3200 | 2.61 | 5.38 | both identified |

All DDMnn, 500 replicates, 48 cores, uncompressed. Analytic controls C3 (N=400,
500 reps) and C4 (N=3200, 200 reps) at the sv=1.5/SZ=0.4 point validate the
machinery where it has never been run. sv=2.0 was rejected: its prior would
span log 1.44 against a training-box ceiling of 1.386. `make_sbc_cell_id`
enforces that box check and errors rather than silently sampling outside.

Walltimes: P1/P3 6 h, C3 12 h, P2/P4 16 h, C4 36 h (from the Phase 1 scaling,
DDMnn ~ 25.6 + 0.123N min at 48 cores; analytic ~ 2.8x that per trial).

### Registered predictions

1. **P1 reproduces Phase 3** for the shipped flow, giving a direct read on
   whether the score penalty carried into SBC. This is the link between the
   laptop score-bias measurement and calibration.
2. **sv coverage is uninformative in P1/P2 and informative in P3/P4.** In P1
   and P2, sv ranks should be near-uniform whatever the flow does, because the
   posterior is the prior; treat any sv "pass" there as vacuous. P4 is the
   first real test of sv calibration in this project.
3. **SZ separates P1 from P2.** SZ is the one parameter that becomes
   identified by N alone at the existing generating point, so P2 vs P1 isolates
   the effect of trials from the effect of magnitude.
4. **st0's bias should be smaller at the sv=1.5 cells.** Measured score bias
   over prior draws is +0.0294 at sv=0.1 but +0.0113 at sv=1.0, i.e. the
   pathology we have chased since July is worst exactly where sv is unknowable.
   If P3/P4 show better st0 calibration than P1, that coupling is real.
5. **The penalty's weighting trade should reverse between cells.** Up-weighting
   sv in the penalty helped sv at sv=1.0 (g +0.006) and hurt it at sv=0.1
   (g +0.034); down-weighting did the opposite. P1 and P4 are the two ends of
   that, so whichever weighting ships, one of them is its weak point.

### Race-model cells: the same identification audit

The information calculation applies to any model, so it was run for LNR and
RDM at their Phase 1 generating points (Monte-Carlo curvature of the expected
log-likelihood; note `make_data(n_trials=M)` yields M trials PER DESIGN CELL,
so the per-trial normaliser is nrow(dat), not M - getting this wrong doubles
every value).

| model | par | N*I/prior at N=100 | status |
|---|---|---|---|
| LNR | m | 18.9 | identified |
| LNR | m_lMTRUE | 13.2 | identified |
| LNR | s | 2.35 | borderline |
| LNR | t0 | 0.33 | weak (moving support; unreliable) |
| RDM | B | 5.43 | identified |
| RDM | t0 | 2.87 | borderline (moving support) |
| RDM | v | 2.25 | borderline |
| RDM | v_lMTRUE | 1.66 | borderline |
| **RDM** | **A** | **0.10** | **PRIOR-DOMINATED** |

**RDM's A is the analogue of the DDM's sv**: start-point variability at N=100
carries a tenth of the prior's information, so RDMnn's Phase 1 "PASS" on A was
close to content-free. Unlike sv this is cheap to repair - A reaches parity at
N ~ 1040 with no change to the generating point, where I_sv would need ~14M
trials. Hence cell R1 (RDMnn, N=1000, A ratio 0.96) with analytic control R2.

**LNR was a real test**: every sampled parameter except t0 is at or above
2.35x prior. That matters because Phase 1's one unexplained anomaly was LNR's
s (mean rank .43, bias +.011, KS p 1.5e-5), and the measured score bias
predicts it (g_s = +0.032, 18x its MC error, right sign, never shown the SBC
result). Cells L1 (N=400) and L2 (N=1000) ask whether that anomaly WORSENS
with N as the DDM's st0 did (.94 -> .84 -> .67); L3 is the analytic control.
If it does, LNR becomes the cleanest available demonstration that the score
penalty fixes a real calibration failure - a far smaller target than the DDM.

RDM's own score biases remain INCONCLUSIVE (2-3x MC error, and the estimator
fails its g_analytic ~ 0 validation for B and t0), so no RDM retraining is
justified until that measurement is sound.

Race cells are cheap: analytic RDM/LNR ran at 5.0 and 4.0 min for N=100 x 500
reps in Phase 1, so the controls cost minutes, and RDMnn/LNRnn scale from
56.7 and 29.1 min.

### Open decision this blocks

Which flow ships as DDMnn. Candidates measured 2026-09-01 (predicted coverage,
N=400 / sv=0.1 unless noted): scoreD1d (sv down-weighted) st0 .946 sv .915;
scoreW1d (sv up-weighted) st0 .920 sv .470, but at sv=1.0/N=3000 scoreW1d is
the best of all (every parameter >= .89) while scoreD1d falls to sv .538,
a .500. A middle weight (sv 0.49) is training as scoreM1d. Phase 4 should be
run with ONE production flow, chosen on the sv=1.0 evidence as well as the
sv=0.1 evidence, not on the historical operating point alone.

---

## Phase 5 — the (SZ0, SZ, SZ1) reparameterisation

**Proposal (AH).** Replace (a, Z, SZ) with three positive quantities: SZ0 the
gap below the start-point distribution, SZ its width, SZ1 the gap above, with

    a = SZ0 + SZ + SZ1,   w = (SZ0 + SZ/2)/a,   sw = SZ/a

Motivation: remove EMC2's `Ttransform`, sw = 2·SZprop·min(Z, 1−Z), which is
non-differentiable at Z = 0.5 and couples SZ to both Z and a. a and Z are the
two parameters still failing G1 after every other intervention.

### Registered result 1 — the reparameterisation cannot widen coverage

This is algebraic, not empirical. Since SZ0, SZ1 > 0,

    min(w, 1−w) = min(SZ0 + SZ/2, SZ1 + SZ/2)/a ≥ (SZ/2)/a = sw/2
    ⟹ sw ≤ 2·min(w, 1−w)  for every valid triple.

Confirmed numerically: **0.00%** of 400k triple draws exceed the old ceiling.
EMC2's `Ttransform` is therefore not an arbitrary restriction — it is exactly
the statement that the start-point distribution must fit between the two
boundaries, and *any* parameterisation of a DDM start point obeys it. EMC2's
own bound SZ ∈ [0.01, 0.99] applies to sw *after* the transform and is a model
bound, not a parameterisation artefact.

**Consequence.** The "expand the priors on a and Z" motivation is dead: there is
no additional (a, Z, SZ) region to reach. The only surviving argument for the
reparameterisation is *smoothness* — the conditioner is fed SZ0/SZ/SZ1 directly,
so the min() kink is gone from the input mapping it has to represent.

### Registered result 2 — a fresh triple box would confound the test

Sampling SZ0, SZ, SZ1 ~ log-uniform[0.002, 5] independently (with rejection to
the old a and w ranges) does not reproduce the old prior:

| | 5% | 50% | 95% | sd |
|---|---|---|---|---|
| a, new | 0.080 | 0.883 | 4.302 | log-sd 1.253 |
| a, old | 0.063 | 0.501 | 3.975 | log-sd 1.329 |
| w, new | 0.146 | 0.500 | 0.854 | **0.206** |
| w, old | 0.124 | 0.500 | 0.876 | **0.255** |

The triple box puts 28.3% of w within ±0.05 of 0.5 (uniform would be 12.5%) and
shrinks sd(w) by 19% — i.e. *less* variation in the start point, the opposite of
what the "are we varying them enough?" hypothesis wants. Rejection rate is 44.6%.

**Design decision.** Test the reparameterisation as an exact *re-encoding* of the
existing 16M-row set (`simulation/reencode_triple.py`), not a new simulation.
rt, ll, and the parameter distribution are held byte-identical; the only thing
that changes is the representation the flow is conditioned on. That isolates the
smoothness effect, which is now the entire hypothesis.

### Run

`triple256w`: 16M re-encoded rows, dims [9, 256, 128], 64 bins, 300 epochs,
matching `cap256w` in every other respect (`--triple` selects the context
columns). Comparator is `cap256w`, whose best checkpoint was epoch 72.

**Prediction.** If the a/Z residual is the conditioner struggling with the
Ttransform kink, `triple256w` should improve G1 on a and Z relative to
cap256w's 0.32 / −0.02. If the residual is instead the a–Z trade-off already
measured (cor = −0.644, R² = 0.414), it will not move, and the reparameterisation
should be recorded as a negative result and dropped.

### Gotcha found while building this

`simulation/sim_functions.R` does not set `.libPaths()`, and this machine has a
second EMC2 3.4.1 in the system library whose DDM likelihood returns 0 density
for ordinary in-bounds parameters. `make_data()` still returns sensible rt, so
the only symptom is the entire `ll` column pinned at min_ll = −23.0259. Every
entry-point script must begin with

    if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))

### Dual-encoding support in the export and the ports

The triple run needed real plumbing, because the flow and the choice classifier
end up conditioned on *different* encodings of the same parameter draw: the flow
takes (SZ0, SZ, SZ1), while the classifier is frozen (deliberately held constant
across replications) and still takes EMC2's (a, Z, SZ). They therefore need
different scalers as well.

The design keeps **EMC2 parameters as the port's public interface** and
re-encodes internally for the flow only. So `ddm_eval(fl, theta, ...)` is called
exactly as before and the golden CSV still records EMC2 theta; nothing
downstream (score_bias_draws.R, the SBC drivers) has to know. Changes:

- `flow/scripts/export_flow_ddm.py`: `--triple`; `emc2_to_triple()`; feeds each
  head its own encoding and scaler; writes `context_encoding`,
  `classifier_context_names`, `classifier_scaler`; scaler falls back to
  `out_base` when `DDM_flow` has none.
- `flow/src/flow/config.py`: `joint_log_prob(..., clf_params=None)` — optional
  separate classifier context, defaulting to the old single-encoding behaviour.
- `port/R/flow_ddm.R` and `port/src/flow_ddm.cpp`: `.emc2_to_triple()` /
  `emc2_to_triple()`, applied only when `context_encoding == "triple"`.
- `flow/scripts/train_ddm.py`: `--triple`, and the scaler is now persisted
  *before* training (see below).

Verified: **15/15 golden checks pass for `triple256w`, and 15/15 still pass for
`arch64w`** (old encoding, regression check). Port-vs-JAX max |Δlog pdf| =
1.6e-3, median 7e-6. `score_bias_draws.R` runs on both tags, and the estimator's
analytic control is ~0 (a = +0.0019, everything else < 2e-4).

Morning command: `bash sbc/run_triple_eval.sh 300`.

### Second gotcha: the checkpointer wipes the run directory

`train_loader.pkl` (the context scaler) was written only *after* `train_model`
returned, so any run not taken to its nominal epoch count could not be exported
at all. Moving the write earlier is not sufficient on its own — the checkpointer
**wipes `DDM_flow` when training starts**, deleting anything put there first.
The scaler is therefore written to `out_base`, which is never wiped, and the
exporter falls back to it. Borrowing another run's scaler is not a workaround
whenever the context columns differ, as they do here.

### Phase 5 OUTCOME — negative. Drop the reparameterisation.

`triple256w` trained to convergence (best val −0.12300 at epoch 181, vs
`cap256w` −0.12516 at epoch 72 — the same fit). Classifier held identical
(step 135) so the comparison is clean.

Mean score bias g over 300 SBC prior draws (analytic control ~0 throughout):

| | v | a | Z | sv | SZ | st0 |
|---|---|---|---|---|---|---|
| triple256w | −0.0048 | **−0.0422** | −0.0231 | +0.0005 | −0.0032 | +0.0102 |
| cap256w    | −0.0096 | **+0.0013** | −0.0258 | −0.0002 | +0.0003 | +0.0025 |

**The registered prediction fails.** Removing the Ttransform kink did not
improve a — it made it ~32x worse — and left Z essentially unchanged
(−0.0231 vs −0.0258). st0 and SZ also degrade. Only v improves, and not
materially.

So the a/Z residual is **not** the conditioner struggling with the
non-differentiable `min(Z, 1-Z)` mapping. Together with the algebraic result
that the reparameterisation cannot widen coverage, it now has no motivation
left on either axis. Recorded as a negative result; the a–Z trade-off
(cor = −0.644, R² = 0.414) stands as the explanation for the residual.

Checked and ruled out as explanations for the degradation:
- *Extrapolation*: 0% of the 300 draws fall outside the trained range in
  triple coordinates (log SZ0 draws [−1.57, −0.64] vs training [−7.76, 1.46]).
- *Input collinearity*: max |r| = 0.285 among the three log inputs, condition
  number 2.3 (vs 1.0 for the old orthogonal encoding) — too mild to matter.
- *Undertraining*: triple256w got 181 epochs to cap256w's 72.

**Methodological note worth keeping.** `triple256w` fits the validation set
*better* than cap256w in pure density terms (mean log f − analytic = +0.0065
vs −0.0243) while being 32x worse in the a-score. Density accuracy and score
accuracy are genuinely different objectives; a held-out NLL that matches or
beats the baseline is not evidence of calibration.

### Trap: EMC2 ships its own compiled copy of the port

`sbc/score_bias_draws.R` and the SBC drivers evaluate through
`EMC2:::ddm_eval_trials_cpp`, which is EMC2's **own compiled build** of the
port, not `port/src/flow_ddm.cpp`. It predates the reparameterisation, applies
one scaler to one context for both heads, and so silently fed the triple flow
raw EMC2 parameters — a 6.2 log-unit error that showed up as a spurious score
bias of −2.08 on a and −1.10 on Z. Every golden check still passed, because
those compare the port against JAX, and both used the *exported* scaler.

EMC2 is installed binary-only here, so it cannot be rebuilt; `build_one()` now
routes `context_encoding == "triple"` models through the R port instead. **Any
future change to the port must be mirrored into EMC2 or routed around, or it
will be silently ignored by every SBC and score-bias run.**

## Phase 6 — where the a/Z bias actually lives

Two diagnostics, both cheap, both run 2026-09-10.

### 1. The (v, a, s) scaling redundancy is NOT the problem

The DDM likelihood depends on (v, a, sv, s) only through (v/s, a/s, sv/s) —
verified against WienR to 1e-11 on stored training rows. The training box varies
s over [0.1, 2], so it contains an exact 1-parameter redundancy, and SBC always
evaluates at s = 1. The natural hypothesis is that error along the unconstrained
null direction leaks into g_a.

Tested by adding s to the measured parameters. The null direction in
(v, log a, log s) is (v, 1, 1), so the test is v*g_v + g_a + g_s:

| tag | mean abs g_a | mean abs g_s | v*g_v + g_a + g_s | ratio to g_a |
|---|---|---|---|---|
| arch64w | 0.2144 | 0.1966 | −0.0007 | **0.02** |
| cap256w | 0.0185 | 0.0283 | +0.0048 | 0.26 |
| cap256w_c3 | 0.0206 | 0.0176 | +0.0074 | 0.39 |

g_a and g_s are nearly equal and OPPOSITE, i.e. the error sits in the identified
direction log(a/s), not the null one — the network has learned the invariance to
~2% for the worst model. **Fixing s = 1 in training would buy nothing.**
Hypothesis rejected; do not spend a training run on it.

### 2. The bias is roughly half classifier, and our best numbers are cancellation

log f_joint = log f(rt | R) + log P(R), so g splits exactly. Measured over 150
draws (decomposition verified exact to 5 dp; the flow share is identical across
the two tags, which share a flow):

| | v | a | Z |
|---|---|---|---|
| cap256w joint | −0.0091 | **+0.0010** | −0.0251 |
| — flow | −0.0002 | +0.0121 | −0.0075 |
| — classifier | −0.0089 | −0.0111 | −0.0176 |
| cap256w_c3 joint | +0.0033 | +0.0157 | **+0.0013** |
| — flow | −0.0002 | +0.0121 | −0.0075 |
| — classifier | +0.0035 | +0.0036 | +0.0088 |

Two consequences, both important:

- **cap256w's excellent g_a (+0.0010) is an accident.** It is flow +0.0121
  cancelling classifier −0.0111. The same is true of cap256w_c3's g_Z (+0.0013)
  = flow −0.0075 + classifier +0.0088. Neither reflects an accurate model, and
  neither should be expected to replicate. This is the most likely explanation
  for the seed-to-seed instability we kept seeing in the G1 tables (0.44 / 0.74
  / 0.63 for nominally identical ensembles) — we have partly been selecting
  configurations on lucky cancellation.
- **The classifier carries 70-98% of the v and Z bias**, and a comparable share
  of a. It is a small MLP fitted to choice probability only; nothing ever asked
  its theta-derivatives to be right.

### Implication: fix the classifier's score first

P(R | theta) is available EXACTLY and cheaply from `WienR::pWDM` (defective CDF
at infinity), and unlike the flow's rt-density it is a smooth scalar with no
normalisation constraint and no moving support. So its theta-gradient is an
exact, well-conditioned supervised target — Sobolev training (match value AND
gradient), not a penalty. None of the failure modes that killed the flow score
penalty apply: there is nothing to exploit, and no curvature to destroy.

This is the recommended next experiment. It is hours, not days, because the
classifier is tiny (8 -> 32 -> 16 -> 1), and it attacks the majority share of
the residual. Only once it is fixed is the flow's own residual (a +0.0121,
Z −0.0075) exposed cleanly enough to be worth attacking.

## Phase 7 — the classifier is fixed; the flow's a-residual is now the whole story

### The analytic P(R) target was wrong

`simulation/sim_choice_prob.R` passed `a = exp(a)`, `v = v` straight to `pWDM`
with no s-scaling, but the likelihood depends on v, a, sv only through
v/s, a/s, sv/s. Over the box s in [0.1, 2] the target is wrong by 0.055 mean /
0.83 max. Measured on 20k held-out theta:

| classifier | vs old (wrong) target | vs CORRECTED target |
|---|---|---|
| shipped (BCE on labels) | 0.0553 | **0.0092** |
| c2 (retrained, wrong target) | 0.0001 | **0.0552** |
| c3 (tuned small, wrong target) | 0.0034 | **0.0554** |
| **c4 (retrained, CORRECTED)** | — | **0.000262** |

The shipped classifier was ~6x more accurate than we believed; the "575x
reduction" claim in `retrain_classifier.py` was measuring agreement with a
faulty reference. c2/c3 faithfully learned the bug. (At s ~ 1, where SBC
evaluates, the targets coincide and c3 is genuinely better than shipped,
0.0048 vs 0.0084 — the damage is away from s = 1.)

### The classifier is a VALUE problem, not a gradient problem

    g_clf,j = dP2_hat/dtheta_j * [P2/P2_hat - P1/P1_hat]

The bracket vanishes identically when the value is exact, whatever the
gradient. So Sobolev/gradient training is the wrong tool — plain regression on
an accurate target is sufficient AND necessary. Predicted g_clf from value
error alone matched measurement to the right order before the fact
(cap256w: -0.0061 predicted vs -0.0089 measured).

### Result: prediction confirmed

Value error 0.0092 -> 0.000262 (35x). Classifier share of g fell 13-24x and the
joint bias collapsed onto the flow share:

| | v | a | Z | SZ | st0 |
|---|---|---|---|---|---|
| cap256w joint | −0.0091 | +0.0010 | −0.0251 | +0.0001 | +0.0027 |
| **cap256w_c4 joint** | **+0.0004** | **+0.0125** | **−0.0062** | −0.0017 | +0.0035 |
| flow share (unchanged) | −0.0002 | +0.0121 | −0.0075 | −0.0017 | +0.0035 |
| c4 classifier share | +0.0006 | +0.0005 | +0.0013 | −0.0000 | +0.0000 |

v improved 20x, Z 4x. **a got WORSE, from +0.0010 to +0.0125 — as predicted,
because the flattering cancellation is gone.** cap256w never was accurate on a;
it had a flow error of +0.0121 masked by a classifier error of −0.0111.

Two conclusions:

- **The classifier is done.** Its share is now ~1e-3 or less on every
  parameter, it has an exact cheap target, and the fix is pure regression.
  Notably the FLOW's v bias is −0.0002: the classifier was the entire v
  problem, which vindicates the retraining programme — with the right target.
- **The flow's a-residual (+0.0121) is now the single dominant term**, with
  Z (−0.0075) second. Everything else is <= 0.0035.

### Next

1. Rebuild the 5-member ensemble on `_c4` and re-run the Gadi SBC gates. Every
   G1 number we have is contaminated by classifier error and by cancellation,
   so the table needs redoing before any of it is quoted.
2. Then attack the flow's a-residual specifically — e.g. pair-based derivative
   distillation on the stored `ll` (differences between nearby theta are a
   finite-difference of the true score, needing no analytic score).

A crude single-point extrapolation from arch64w's g-to-G1 ratios suggests c4
would land near G1 a 0.07, Z 0.40, st0 0.15, v 0.05 — inside the 0.5 gate. That
calibration rests on ONE model and different ratios per parameter (5.7 for a,
~64 for Z), so it is a reason to run the SBC, not a result.

### Phase 7 addendum — with the classifier fixed, the ensemble LOSES

Score bias over 300 prior draws, all on the same compiled evaluator:

| | v | a | Z | SZ | st0 |
|---|---|---|---|---|---|
| single cap256w_c4 | +0.0003 | **+0.0136** | −0.0061 | −0.0017 | +0.0034 |
| 5-member ens (c4) | +0.0031 | **−0.0398** | +0.0047 | −0.0003 | −0.0041 |
| 5-member ens (c3) | +0.0060 | −0.0366 | +0.0124 | −0.0000 | −0.0041 |

The ensemble is **3x worse on a than its own members, with the opposite sign**.
That is not a measurement artefact: a density mixture's score is a
density-weighted average of member scores, not the average of them, so
ensembling is free to move the score bias in either direction. The pattern
replicates across c3 and c4 with se <= 0.0007.

The ensemble now wins only on Z and SZ, both small, while losing on the
dominant parameter and costing 4.3x more (3880 vs 900 us/1000). **The single
flow with a correct classifier is the leading production candidate**, which
reverses the Phase 3 recommendation. Phase 3's ensemble advantage was
substantially classifier cancellation.

### Gadi submission (Phase 7)

`DDMnn` evaluates whatever is in `<EMC2>/extdata/flownn/ddm_ens5.rds` and that
name is hardcoded, so the bundle must be swapped in place. Verified locally
end-to-end: swapping changed the DDMnn density from 0.69969 (arch64w, the
bundle actually installed — Phase 3 was NOT cap256w) to 0.82056, matching the R
port on cap256w_c4 to 7e-15.

Files: `sbc/ddm_ens5_single_c4.rds`, `sbc/ddm_ens5_c4.rds`,
`sbc/nci/{SBCc4.R, SBCc4_N*.pbs, launch_sbc_c4.sh}`. The original bundle is
preserved at `sbc/ddm_ens5_ORIGINAL_arch64w.rds`.

**The two configurations cannot run concurrently** — one hardcoded filename,
one EMC2 install. The launcher refuses to swap while jobs are queued or
running, and each job prints the installed member tags at startup so a result
can never be misattributed. Run `single` first: it is the better candidate and
the cheaper one.

## Phase 8 — seed variance dominates, and val loss cannot see it

Flow-only score bias (classifier share removed) across the five cap256w seeds,
100 prior draws each:

| flow seed | best val loss | flow g_a |
|---|---|---|
| cap256w | −0.12516 | **+0.0127** |
| cap256w_s101 | −0.12558 | −0.1333 |
| cap256w_s102 | −0.12580 | +0.0207 |
| cap256w_s103 | −0.12542 | −0.0125 |
| cap256w_s104 | −0.12484 | −0.0841 |

mean −0.0393, sd 0.0668. **Val loss spans 0.001 while g_a spans 0.154, and they
correlate −0.20 (n = 5, i.e. not at all).**

This is the cleanest statement of the project's thesis so far: five models with
indistinguishable held-out likelihood differ by 10x in the quantity that governs
calibration. Maximum likelihood does not constrain the score, and no amount of
held-out NLL will reveal which flow is calibrated. It also means:

- **cap256w's good a-bias (+0.0127) is a lucky draw, not an architecture
  property.** It is the best of five; the expected value for a fresh flow is
  −0.039 +/- 0.067. Reporting +0.0127 as "the model's" bias would be seed
  selection, and a replication would not reproduce it.
- **Correction to Phase 7's addendum.** The ensemble's g_a (−0.0398) is almost
  exactly the mean of its members' flow-only g_a (−0.0393, within 3%). So an
  equal-weight density mixture DOES average member scores here, contrary to what
  I wrote there. The ensemble is not pathological — it is the honest average,
  and the single flow looks better only because cap256w is the good seed.

### Consequence for the running SBC

The `c4single` campaign now on Gadi measures a SELECTED seed. If it passes G1
that is informative but not reproducible: the same recipe re-run with a new seed
would be expected to fail. Selecting a flow on measured g is defensible model
selection, but it must be (a) declared, and (b) done on prior draws independent
of those used for the final report.

### Classifier size: c4 vs c5

| classifier | params | mean abs err | us/call (N=400) | clf share of g (v, a, Z) |
|---|---|---|---|---|
| shipped | 833 | 0.00916 | 660 | −0.0082, −0.0101, −0.0162 |
| c3 (wrong target) | 289 | 0.05538 | 660 | +0.0035, +0.0036, +0.0088 |
| **c4 (big, corrected)** | 134145 | **0.000262** | **1177** | +0.0006, +0.0005, +0.0013 |
| **c5 (small, corrected)** | 289 | 0.004344 | **660** | −0.0012, −0.0013, −0.0022 |

c5 is free and 2x better than shipped; c4 costs 1.8x and makes the classifier
share negligible. c5's joint g_a (+0.0115) looks marginally better than c4's
(+0.0132) but that is cancellation again — its classifier error happens to
oppose the flow's. Prefer c4 while diagnosing (it isolates the flow); c5 is the
sensible production choice if the 1.8x matters.

## Phase 7 RESULT — c4single SBC (tux26, 500 replicates x 3 cells)

Standardised bias (mean error / sd of error across replicates), gate |.| <= 0.5:

| N | | v | a | t0 | Z | sv | SZ | st0 |
|---|---|---|---|---|---|---|---|---|
| 100 | ens5 (arch64w, old clf) | −0.206 | −0.077 | +0.297 | +0.054 | −0.005 | −0.004 | +0.394 |
| 100 | **c4single** | +0.039 | −0.003 | +0.118 | −0.026 | −0.021 | +0.040 | −0.026 |
| 200 | ens5 | −0.248 | −0.356 | +0.295 | −0.058 | +0.051 | −0.006 | **+0.858** |
| 200 | **c4single** | +0.009 | −0.115 | +0.249 | −0.008 | −0.036 | −0.018 | −0.002 |
| 400 | ens5 | −0.366 | **−0.647** | +0.187 | +0.057 | −0.065 | −0.053 | **+1.520** |
| 400 | **c4single** | +0.052 | +0.012 | +0.319 | −0.032 | +0.014 | −0.051 | −0.136 |

**Every cell now passes the bias gate.** The old model failed at N=200 (st0
0.858) and N=400 (a −0.647, st0 +1.520). Improvements at N=400: a 54x, st0 11x,
v 7x. Z was already fine and stays fine.

Coverage is at or above nominal everywhere (0.932–0.970 against 0.95, MC se
0.0097).

### The remaining failure is t0, and it was in our blind spot

t0 is the ONLY parameter that got worse (+0.187 -> +0.319 at N=400) and the only
one whose rank distribution is decisively non-uniform: KS p = 7.4e-01 / 3.3e-04
/ 1.8e-07 at N = 100 / 200 / 400. Mean rank falls 0.485 -> 0.449 -> 0.422, i.e.
t0 is increasingly OVERestimated as N grows — the signature of a fixed score
bias, whose displacement grows ~sqrt(N). Extrapolating, it reaches the 0.5 gate
near N = 800–1600.

**t0 was excluded from every score-bias diagnostic we used to steer** — see the
header of `sbc/score_bias_draws.R`: "t0 excluded: it moves the support boundary
and does not validate". So we optimised the six parameters we could measure, and
the failure migrated into the one we could not. That is a methodological lesson
worth stating in the paper: a calibration diagnostic with a blind spot will
relocate the error rather than remove it.

### Mechanism: a t0/st0 leading-edge trade-off

| N | cor(t0 error, st0 truth) | cor(t0 error, t0 truth) | mean t0 error (LOG scale) |
|---|---|---|---|
| 100 | +0.329 | −0.210 | +0.00197 |
| 200 | +0.318 | −0.214 | +0.00327 |
| 400 | +0.367 | −0.164 | +0.00316 |

Rank-vs-truth: rho(rank_t0, st0) = −0.32, rho(rank_t0, SZ) = +0.30 at N=400.
t0 is overestimated by ~0.28% (0.83 ms at t0 = 0.3 s), and more so when the
true st0 is large. t0 and st0
jointly define the leading edge (support starts at t0, smeared over st0), so a
flow that smooths the sharp onset must trade one against the other — the exact
analogue of the a/Z trade-off, on the time axis. st0's own bias is −0.136,
opposite in sign to t0's +0.319, as a trade-off implies.

### Caveat that must travel with this result

cap256w is the best of five seeds (flow g_a +0.0127 against a seed mean of
−0.039 +/- 0.067, Phase 8). This campaign therefore measures a SELECTED flow.
The result is real for this artefact, but the recipe re-run with a fresh seed
would not be expected to reproduce it. Any claim must be phrased as "a flow
selected on held-out score bias passes SBC", not "this training procedure
produces calibrated flows".

### N=800 follow-up (300 replicates, local, NON-STANDARD replicate count)

| N | reps | v | a | t0 | Z | sv | SZ | st0 |
|---|---|---|---|---|---|---|---|---|
| 100 | 500 | +0.039 | −0.003 | +0.118 | −0.026 | −0.021 | +0.040 | −0.026 |
| 200 | 500 | +0.009 | −0.115 | +0.249 | −0.008 | −0.036 | −0.018 | −0.002 |
| 400 | 500 | +0.052 | +0.012 | +0.319 | −0.032 | +0.014 | −0.051 | −0.136 |
| 800 | 300 | +0.113 | −0.143 | **+0.371** | −0.120 | −0.011 | −0.061 | −0.005 |

**t0 passes the bias gate at N=800 (0.371 < 0.5), and the growth is
DECELERATING, not sqrt(N).** Anchored at N=100, sqrt(N) predicts 0.333 at
N=800; the per-doubling increments are +0.131, +0.070, +0.052, i.e. roughly
logarithmic. Extrapolating that trend, t0 reaches the gate near N ~ 5000, well
beyond realistic designs; a sqrt(N) law would have put it at N ~ 1800. So the
earlier prediction of a gate failure around N = 800-1600 is REFUTED.

The deceleration is itself informative: a fixed score bias produces sqrt(N)
growth indefinitely, so a saturating displacement means the t0 error is being
partly resolved as N grows — consistent with the t0/st0 trade-off unwinding once
st0 becomes well identified.

But the rank distribution is decisively non-uniform from N=200 on
(KS p = 3.3e-04, 1.8e-07, 3.8e-06 at N = 200/400/800), while COVERAGE stays
nominal (0.953 at N=800, vs 0.95). So the practical reading is: **t0 credible
intervals remain valid; t0 point estimates carry a systematic ~0.28% upward
offset.** Strict SBC uniformity fails for t0; interval-based inference does not.

At N=800 several parameters drift up (a −0.143, Z −0.120, v +0.113) but with
300 replicates the se is 0.058, so these are 2-2.5 se — real but small, and all
comfortably inside the gate.

## Phase 9 — can score bias predict SBC bias? Partly, and the exception is informative

Score bias costs a minute; SBC costs hours. If the first-order relation

    delta = (N*I + Sigma0^-1)^-1 * (N*g),   standardised bias_j = delta_j / sqrt(V_jj)

held, we could screen seeds cheaply. The per-trial Fisher information at the SBC
prior mean (computed analytically from WienR, `sbc/predict_sbc_from_g.R`, saved
to `sbc/fisher_ddm.rds`) is positive definite, with per-trial sds
v 2.61, a 0.41, t0 0.14, Z 0.75, sv 590, SZ 8.93, st0 2.55 — so sv is
effectively unidentified, and SZ/st0/sv are prior-dominated at every N we run.

Posterior coupling implied by I^-1: **t0-SZ +0.86**, a-sv +0.82, a-v +0.72,
v-Z −0.64, a-Z −0.44. The t0-SZ figure is notable given Phase 7's finding that
t0's rank correlates with SZ at +0.30.

### Validation against the 7 SBC cells we have

Restricted to the 9 observed effects exceeding 3 Monte Carlo standard errors
(smaller ones are indistinguishable from noise, and correlating a predictor
against noise measures nothing):

overall cor +0.916, sign agreement 67%, and it decomposes cleanly:

| parameter | n | sign | magnitude captured |
|---|---|---|---|
| v | 3 | 3/3 correct | ~60% |
| st0 | 4 | 3/4 (the miss is marginal at 3.0 se) | ~80% |
| **a** | 2 | **0/2 — wrong sign both times** | — |

**The theory works where the error is a uniform displacement, and fails where it
is not.** Phase 7's mechanism diagnostics already showed rho(rank_a, drawn a) =
+0.15, i.e. a's error is REGION-DEPENDENT — it varies with the true value of a —
whereas v and st0 behave like uniform shifts. A first-order displacement theory
cannot represent a distortion whose sign depends on where you are in the
parameter space.

### Consequences

- **g is not a sufficient screen for a**, which is the parameter we care most
  about. Seed screening still needs SBC. The cheap-predictor route is closed for
  now.
- **The a error is qualitatively different from the others.** It is not a score
  bias in the usual sense but a region-dependent distortion. That reframes the
  fix: uniform corrections (a constant offset, a global penalty, ensembling) are
  the wrong shape for it, which is consistent with every one of those having
  failed in Phases 5-8.
- g remains the right tool for LOCALISING error (it isolated the classifier
  share cleanly in Phase 7); it is just not a substitute for the SBC verdict.

### Phase 9b — the a-error is a smooth function of theta, and not of a

Per-draw score bias for `cap256w_c4` (300 draws) regressed on the true
parameters:

| correlate | r with g_a |
|---|---|
| v | +0.633 |
| sv | −0.588 |
| st0 | −0.399 |
| t0 | −0.338 |
| **a** | **+0.004** |

**Joint R^2 = 0.946.** Almost all the draw-to-draw variation in the flow's
a-error is explained by WHERE in parameter space you are — and essentially none
of it by `a` itself. Mean g_a is +0.0136 with sd 0.0190 (cv 1.4), so the error
changes sign across the space.

Tested and refuted: density sharpness. The differential entropy of the analytic
RT density explains 1.0% of g_a (r = +0.102, p = 0.08) against 94% for
v + sv + st0 + t0, so "the flow struggles with peaked densities" is wrong.

PCA on the per-draw g vectors: PC1 47.2%, PC2 36.6%, PC3 10.1% — roughly three
dimensional, not one, so there is no single dominant error mode to characterise.
PC1 loads st0 +0.55, a +0.50 against SZ −0.47, Z −0.38.

**Why a post-hoc correction is harder than it looks.** g_a(theta) being 94.6%
predictable suggests simply learning and subtracting it. But a correction that
is a function of theta ALONE cannot work: adding c(theta) to log f multiplies
the density by e^c, and renormalising over (rt, R) cancels it exactly, leaving
the score unchanged. Any effective correction must reshape the density in rt,
i.e. be a function c(theta, rt) — which is distillation against the analytic
likelihood by another name. That route was tried at 4M rows and lost to
baseline; it has NOT been retried at 16M with the corrected classifier, and on
this evidence it deserves one more attempt.

## Phase 10 — a validated score-bias estimator for t0

`sbc/score_bias_t0.R`. t0 was excluded from every diagnostic we steered by, and
that is where the residual ended up (Phase 7), so closing this was overdue.

**Why the standard estimator fails.** It needs E_true[log f_hat] over the full
support. Raise t0 and the ANALYTIC likelihood assigns zero density to a region
where the truth has mass, so the expectation is -Inf exactly. That is a property
of the analytic reference, not of the flow — the flow splines log(rt) over the
whole line and is positive everywhere, so ITS g_t0 is well defined. The control
was broken, not the measurement.

**The fix.** Evaluate over a fixed region A = [c, Inf) supported by both
perturbations. On a restricted region the score identity gives a known non-zero
target, d/dtheta log P(A | theta), computable from the defective CDF, so

    score bias = g_flow(A) - g_analytic(A)

with g_analytic(A) checkable against the closed form. That check is the
validation the old estimator lacked.

**c must clear the st0 smear, not merely the onset.** With c only 7% above t0 —
inside the smear — the estimator returns -0.202 against a target of +0.138, and
refining the grid from 160 to 4000 points changes nothing: near the leading edge
the density varies on the scale of st0, so a step of H = 0.05 in log t0 is far
too coarse for a central difference. With c past t0*exp(H) + st0 the density is
smooth in t0 and the estimator agrees to under 1%.

Validation over 40 draws: estimator +2.40592, closed form +2.39801, difference
+0.00791 (0.3%), correlation 1.0000.

### First measurement

| tag | score bias on t0 |
|---|---|
| cap256w_c4 | **-0.0694** (se 0.0045) |
| arch64w | **+0.2105** (se 0.0079) |

**t0 is the largest single score bias in the current model** — 5x its bias on a
(+0.0136), in a parameter we had never measured. For arch64w it is comparable to
that model's a-bias (+0.212).

Caveat: this is the score bias restricted to A, not the full-support quantity
that enters the displacement formula, and its sign (-0.069, implying t0 pulled
DOWN) is opposite to the observed SBC bias (+0.371, t0 overestimated). Given
Phase 9 showed the displacement is multivariate and t0 couples to SZ at +0.86,
sign agreement was not expected; but it does mean this measurement cannot yet be
read directly as a predicted direction of bias.

### Phase 10b — t0 is systematic, a is noise, and seed quality is one-dimensional

Score bias on t0 across the five cap256w seeds (100 draws, classifier held at c4;
the estimator validates against the closed form to 0.3%, correlation 1.0000):

| seed | g_a | g_t0 |
|---|---|---|
| cap256w | +0.0127 | −0.0629 |
| s101 | −0.1333 | −0.2083 |
| s102 | +0.0207 | −0.0367 |
| s103 | −0.0125 | −0.1006 |
| s104 | −0.0841 | −0.1499 |

| | mean | sd | mean/sd | all same sign |
|---|---|---|---|---|
| g_a | −0.0393 | 0.0668 | 0.59 | no (mixed) |
| **g_t0** | **−0.1117** | 0.0688 | **1.62** | **yes, all negative** |

**Two different problems, needing two different fixes.**

- **t0's error is SYSTEMATIC.** Every seed has it, with the same sign, and the
  mean is 1.6x the spread. Ensembling, averaging and seed selection cannot
  remove it — even the best seed is at −0.037. It is a property of the method,
  so it needs a training change.
- **a's error is SEED NOISE** around a mean not distinguishable from zero
  (Phase 8: −0.039 +/- 0.030). Selection or averaging is the right shape of fix.

**Seed quality is one-dimensional.** cor(g_a, g_t0) = +0.986 across seeds, and
the Spearman rank correlation is exactly 1.000 — the seeds order identically on
both parameters. A seed that is good on a is good on t0. That makes a single
cheap scalar a plausible screen, which is what a reproducible recipe needs.

### REGISTERED PREDICTION for the running tux26 seed campaign

Ordering by score bias (best to worst): **s102, s103, s101**. If seed quality is
genuinely one-dimensional and g screens it, the N=400 SBC standardised biases
should order the same way, with s101 clearly the worst. A confirmed ordering
makes "train k seeds, screen on g, keep the best" a defensible recipe with a
minute of screening per seed. A scrambled ordering kills the cheap screen and
leaves SBC-per-seed as the only option.

Note this is a weaker and more testable claim than the one Phase 9 refuted:
there, g failed to predict the ABSOLUTE SBC bias on a. Here we only need it to
rank seeds correctly.

## Phase 11 RESULT — the recipe is not reproducible, but the cheap screen works

Four seeds, identical architecture/data/classifier, N=400, 500 replicates:

| seed | v | a | t0 | Z | sv | SZ | st0 | worst | verdict |
|---|---|---|---|---|---|---|---|---|---|
| cap256w | +0.052 | +0.012 | +0.319 | −0.032 | +0.014 | −0.051 | −0.136 | t0 +0.319 | PASS |
| s102 | +0.168 | +0.434 | −0.260 | −0.236 | −0.075 | +0.080 | −0.042 | a +0.434 | PASS |
| s103 | +0.119 | −0.301 | +0.518 | −0.171 | −0.013 | +0.003 | −0.254 | t0 +0.518 | **FAIL** |
| s101 | +0.068 | **−1.493** | +0.480 | +0.382 | +0.089 | −0.100 | −0.048 | a −1.493 | **FAIL** |

**Half the seeds fail.** An unscreened flow from this recipe has roughly a 50%
chance of failing calibration, and the failure can be catastrophic: s101's bias
on a is −1.493, three times the gate. This settles the question the writeup left
open — **we do not have a reproducible recipe**, and no claim should be made
that this training procedure produces calibrated flows.

### But the registered prediction was confirmed

Predicted ordering from score bias alone (Phase 10b): s102 best, s103 middle,
s101 worst. Observed ordering: **s102, s103, s101. CONFIRMED.**

And within this model family the relationship is quantitative, not merely
ordinal:

| | correlation |
|---|---|
| g_a vs SBC bias on a | **+0.980** |
| g_a vs worst abs bias | **−0.989** |
| g_t0 vs SBC bias on t0 | −0.671 |

**This does not contradict Phase 9.** There, g failed to predict the ABSOLUTE
SBC bias when comparing across different models (arch64w ensemble vs cap256w
single) — the g-to-displacement map is multivariate and model-dependent. Within
a family that differs only by seed, that map is held fixed, and g_a tracks the
SBC bias on a almost perfectly. The screen is valid for ranking seeds of one
architecture; it is NOT valid for comparing architectures.

### The recipe this implies

1. Train k flows differing only in seed.
2. Screen each on g_a — one minute per seed, no SBC needed.
3. Keep the best; confirm that one with a single SBC run.

With cor = −0.989 between g_a and worst-case bias, and a ~50% base failure rate,
k = 4-5 gives a high chance that the selected flow passes. This must be reported
as selection: the claim is "a flow selected on held-out score bias passes SBC",
and the draws used to screen must be independent of those used to report.

### What screening cannot fix

t0's bias is systematic across seeds (Phase 10b: all five negative, mean 1.6x
the spread). Screening picks the least-bad t0, not a good one — s103 fails ON
t0 (+0.518), and cap256w's +0.319 is its worst parameter. So screening addresses
the a-variance and leaves the t0-bias untouched. A training change is still
needed there, and distillation against the analytic likelihood is the only
intervention we have that reshapes the density in rt rather than applying a
uniform correction.

## Phase 12 — simulator "bridge crossing" hypothesis: tested and refuted

Suggestion from colleagues: a discrete-step simulator attributes every boundary
crossing to the END of the step, and misses crossings that occur and return
within a step, biasing first-passage times long. Right order of magnitude for
the t0 residual, so worth checking.

**Refuted on three independent grounds.**

1. **There is no time stepping.** `EMC2:::rDDM` calls
   `WienR::rWDM(..., method = "p-ars")` — piecewise adaptive rejection sampling,
   exact in time. No steps, so no missed bridge crossings.

2. **The sampler is not detectably biased** at the precision EMC2 uses
   (5e-3). 200,000 trials at the SBC prior mean, against precision 1e-10 and
   against the analytic CDF:

   | | P(upper) | mean rt | max abs ECDF−CDF |
   |---|---|---|---|
   | precision 5e-3 (EMC2 default) | 0.68606 | 0.47491 | 0.00260 |
   | precision 1e-10 | 0.68758 | 0.47530 | 0.00242 |
   | analytic | 0.68732 | — | — |

   The mean rt differs by 0.38 ms against a Monte Carlo se of 0.56 ms, P(upper)
   by 1.2 se, and both deviate from the analytic CDF by ~0.0025 against a KS
   band of 0.0037. Consistent with exact.

3. **The control shows the OPPOSITE pattern to a simulator bias.** On the same
   simulated data, same pipeline, same prior (N=100, 500 replicates):

   | likelihood | t0 standardised bias | raw t0 error |
   |---|---|---|
   | **DDM, analytic** | **−0.015** | −0.08 ms |
   | DDMnn flow (cap256w_c4) | +0.118 | +0.59 ms |
   | DDMnn flow (arch64w) | +1.879 | — |

   This is the decisive one, and it is a logical point rather than a numerical
   one. If the SIMULATOR were biased, the flow — trained on simulated data —
   would have learned that same biased distribution and would be CONSISTENT with
   data drawn from it, while the exact analytic likelihood would be mismatched
   and would show the bias. We observe exactly the reverse: the analytic
   likelihood recovers t0 with no bias, and the flow does not. A simulator
   artefact predicts the opposite of what we see.

**Conclusion: the t0 bias is in the flow's likelihood, not in the data.** The
analytic cell also serves as a validation of the whole SBC pipeline — priors,
ranks, coverage — since an exact likelihood run through it comes out calibrated
on every parameter (worst |bias| 0.082 on Z).

## Phase 13 — WHERE the t0 error lives, and why MLE cannot see it

Flow log-density error against distance past the onset, in units of st0,
weighted by the true density (cap256w_c4, 150 draws):

| (rt−t0)/st0 | mass | mean error |
|---|---|---|
| 0 – 0.25 | 0.0003 | +0.0126 |
| **0.25 – 0.5** | 0.0058 | **+0.1194** |
| 0.5 – 0.75 | 0.0200 | +0.0393 |
| 0.75 – 1 | 0.0382 | +0.0246 |
| 1 – 1.5 | 0.1250 | −0.0002 |
| 1.5 – 2 | 0.1399 | −0.0359 |
| 2 – 4 | 0.3633 | −0.0033 |
| 4 – 8 | 0.2340 | +0.0185 |
| > 8 | 0.0735 | −0.0271 |

**The flow over-predicts density inside the st0 smear**, by up to +0.12 in log
density, and is near-unbiased beyond it. Over-predicting mass just past the
onset is exactly the direction that lets a later t0 fit the data, i.e. it biases
t0 UP — which is what SBC shows (+0.371 at N=800, +0.0027 log = 0.8 ms).

### Why maximum likelihood cannot fix this

Weighting each bin's error by its mass, the entire leading-edge defect
contributes about **0.0014 to the mean log-likelihood**. The seed-to-seed spread
in validation loss is **0.001** (Phase 8). So the training signal for the region
that determines t0 is the same size as seed noise, and smaller than the
difference between two runs that are supposedly identical.

This is the sharpest quantitative statement of the project's thesis so far, and
it should go in the paper: it is not that MLE is indifferent to calibration in
some abstract sense, it is that **the region carrying the information about t0
holds 6% of the probability mass and contributes ~0.1% of the log-likelihood**.
No amount of training on this objective will prioritise it.

It also predicts which interventions can work: anything that AMPLIFIES the
leading-edge signal (reweighting) or REMOVES the need to learn the onset
location (modelling rt − t0 instead of rt). And it explains why every uniform
intervention we tried failed on t0.

## Project plan (agreed 2026-09-11)

### Paper structure if routes 2 and 3 fail without a new lead

1. **Report the general recipe**, with a full account of the pitfalls and blind
   alleys. The negative results are a genuine contribution: score penalty,
   ensembling, SZ0/SZ/SZ1 reparameterisation, s = 1, geometry features, extra
   capacity, more trials per draw, density-sharpness, simulator bridge-crossing,
   and the two measurement traps (the s-scaling target bug and the
   flow/classifier cancellation) all cost real time and are all documented here.
2. **Justify the ~0.28% t0 correction**, if it can be justified.
3. **Distillation**, and if it works, ship the artefact as a validated ~10x
   speedup others can use.

### Model scope

DDM first, locked off. Then **RDM only — LNR is dropped.** LNR's analytic
likelihood is already cheap (lognormal race), so a neural surrogate buys little;
RDM is the expensive one and the one worth accelerating. Neither race model has
a choice classifier (response probability emerges from the racing densities), so
the classifier half of the DDM work does not transfer; the score-bias
diagnostic, the seed-variance question and the SBC machinery all do.

### Next project (noted for context)

Analytic-likelihood distillation for a case where the analytic likelihood is too
slow to use in practice. Goal is a fast validated ARTEFACT, not a general
recipe, so the seed problem is handled by screening rather than solved. Expected
to be much quicker given everything learned here.

**One risk to flag early:** if the analytic likelihood is too slow to USE, it may
also be too slow to DISTIL AGAINST — distillation needs it evaluated on millions
of training rows. Mitigations worth planning for: evaluate on fewer theta with
many rt points each (the flow needs coverage of theta-space more than of rt
given a fixed theta); precompute once and reuse across seeds; or train on
simulation and use the analytic only for the score-bias screen, which needs
~300 draws rather than millions.

## Phase 14 — half the "seed variance" is epoch-to-epoch variance inside one run

Ten checkpoints from a SINGLE run (cap256w_ckpt, same seed and config as
cap256w), all on the converged plateau — epochs 161-237, val loss spread 0.004:

| | mean g_a | sd | range |
|---|---|---|---|
| within one run (10 checkpoints) | −0.0442 | **0.0314** | −0.112 to −0.002 |
| between runs (5 seeds, best-val) | −0.0393 | 0.0668 | −0.133 to +0.021 |

Within-run sd is **47% of between-run sd**; the within-run range covers **72%**
of the between-run range; and the two means agree (−0.044 vs −0.039).

Decomposing, the observed between-run spread of best-val checkpoints contains
the within-run spread: sd_seed = sqrt(0.0668^2 - 0.0314^2) = 0.059. So genuine
seed-to-seed variation is 0.059 and checkpoint-to-checkpoint variation on the
plateau is 0.031 — about 22% of the total variance.

**The model is not converged in any sense that matters.** Validation loss moves
by 0.004 across these epochs while g_a swings by 0.11. The optimiser is
wandering on a plateau that is flat in NLL and emphatically not flat in score —
which is Phase 13's finding (the leading edge is ~0.1% of the objective) seen
from the optimiser's side.

### A much cheaper recipe

Checkpoint selection gets 72% of the diversity of seed selection for 1/5 of the
compute. Training five seeds costs ~5 x 12 h; training ONE run while retaining
30 checkpoints costs 12 h, and screening them on g costs minutes. The best
checkpoint found inside this single run (e161, g_a = −0.0020) is better in
magnitude than cap256w's +0.0127, which was the best of five seeds.

Revised recipe: **train one run, retain many checkpoints, screen them on g,
confirm the winner with one SBC run.** Seeds can be added on top if the budget
allows, but they are no longer the first lever.

Caveat: checkpoints from one run are correlated in ways five independent seeds
are not, so the effective diversity is probably lower than the 72% range figure
suggests. The claim needs the same SBC confirmation seeds got — screening on g
ranks reliably WITHIN an architecture (Phase 11), and a checkpoint series is the
same architecture, so the screen should transfer; that is an assumption, not yet
a result.

### Also motivates weight averaging

Wandering on a flat plateau is the exact situation where averaging weights
across checkpoints (SWA) helps: it lands at a central point rather than an
arbitrary one. Unlike density-mixture ensembling — which provably just averages
member scores (Phase 8) — weight averaging produces a single network at a
different point in parameter space, and is not ruled out by that result. It is
now cheap to test: the 30 checkpoints already exist.

### Phase 14b — weight averaging fails too: averaging converges to the mean, selection finds the best

Ten checkpoints from one run, all sharing scaler, spline and classifier
(verified identical before averaging):

| model | g_a |
|---|---|
| weight average (SWA) | **−0.04728** |
| density mixture of the same 10 | −0.04464 |
| arithmetic mean of member g_a | −0.04427 |
| best single checkpoint (e161) | **−0.00201** |

The SWA model is a valid density (total mass 0.99999) and fits the validation
set as well as its members (gap to analytic +0.0049 against +0.0037 for e208),
so nothing is broken — it simply lands where the mixture lands, marginally
worse than the member mean.

**The general statement, now supported by two independent mechanisms:
averaging converges to the member mean; only selection finds the best member.**
A density mixture averages scores (Phase 8). Averaging weights reaches the same
place by a different route. Since the member mean is −0.044 and the best member
is −0.002, **selection beats averaging by a factor of 22 here**, at lower cost
— one network to evaluate instead of ten.

This was worth testing rather than assuming: weight averaging is a genuinely
different operation from density mixing (single network, different point in
parameter space, not ruled out by the Phase 8 result), and the plateau-wandering
of Phase 14 is exactly the situation SWA exists for. It still fails. Add it to
the blind alleys.

**Consolidated recipe.** Train one run, retain many checkpoints, SELECT on g —
do not average in any form — and confirm the winner with one SBC run.

## Phase 15 — edge-weighted loss at amp=30: FAILS, and not as a trade-off

`cap256w_edge30`: identical to cap256w_ckpt except the loss reweights each row
by 1 + 30*exp(-u), u = (rt-t0)/st0. 300 epochs, 10 checkpoints.

Score bias across its 10 plateau checkpoints: **g_a = +1.36 to +1.79**
(mean +1.53), against the plain-loss run's mean of −0.044. That is 35x worse in
magnitude and sign-flipped. g_Z −0.28 to −0.43 against −0.013; g_v −0.05 to
−0.08 against −0.007.

**It did not even improve the edge**, so this is not a bulk-for-edge trade:

| (rt−t0)/st0 | plain loss | edge amp=30 |
|---|---|---|
| 0 – 0.5 | +0.115 | +0.745 |
| 0.5 – 1 | +0.031 | +0.803 |
| 1 – 2 | −0.019 | +0.463 |
| 2 – 4 | −0.004 | −0.208 |
| > 4 | +0.008 | −1.040 |

Worse in every band. The best edge-weighted validation loss came at epoch 44 and
degraded thereafter — a run that never converged. At amp = 30 the effective
sample size collapses onto a few edge rows; the gradient stops describing the
distribution and the model fails to learn the bulk at all (tail error −1.04).

Implementation checked, not assumed: context indices (t0 = 2, st0 = 7) correct,
`data` is raw rt, and amp = 0 reproduces `loss_fn` to 1e-5. The failure is real.

**amp = 30 was my choice and it was too aggressive.** Phase 13 showed the edge
carries ~0.1% of the log-likelihood, and I picked an amplification to make it
~3%, without considering that the same weight starves the other 99%. A milder
setting (amp 3-10) may still be viable; this result rules out the setting, not
the idea. But it is now a sweep, not a single run.

### The sweep is newly affordable

The batch-size benchmark (`sbc/bench_batch.log`, measured on an idle machine):

| batch | ms/step | rows/s | rel | cores |
|---|---|---|---|---|
| 128 (default) | 1.32 | 97,213 | 1.0x | 3.6 |
| 1024 | 3.46 | 295,859 | **3.0x** | 3.8 |
| 16384 | 48.48 | 337,988 | 3.5x | 5.1 |

Batch 1024 is 3x faster per epoch (16M rows: 165 s -> 54 s). Core use barely
moves (3.6 -> 3.8), so the gain is amortised per-step overhead, NOT recruiting
idle cores — my earlier framing was wrong.

Since one run uses under 4 cores, three runs fit on this laptop concurrently. An
amp sweep {0, 3, 10} at batch 1024 — with its own plain-loss control at the same
batch, since batch size changes the trajectory — costs one ~5 h window rather
than three 13 h ones.

## Phase 16 — the edge weight is dead; BATCH SIZE is the first thing to move the mean

Three concurrent runs, identical but for the edge amplification, all batch 1024,
300 epochs, 10 checkpoints each.

### Edge weighting: monotonically harmful, no useful window

| amp | 0 | 3 | 10 | 30 (batch 128) |
|---|---|---|---|---|
| mean g_a | +0.0086 | +0.4709 | +0.9908 | +1.53 |

Clean dose-response. The idea is dead across the range, not merely mis-tuned at
my original amp = 30. Route 3 closed.

### But the control arm is the best model of the project

Plain loss, batch 1024 vs batch 128 (both 10 plateau checkpoints):

| | mean g_a | sd | mean g_t0 | best val |
|---|---|---|---|---|
| **batch 1024** | **+0.0086** | **0.0098** | **+0.031** | **−0.12528** |
| batch 128 | −0.0443 | 0.0313 | −0.060 | −0.12455 / −0.12516 |

- **5x smaller mean on a, 3.2x tighter spread, and the best validation loss of
  any run** — so this is not a fit-for-calibration trade.
- Full vector at batch 1024: v −0.0107, a +0.0086, Z −0.0167, sv +0.0011,
  SZ +0.0015, st0 +0.0010, t0 +0.031.
- **This is the first intervention in the whole project to shift the MEAN**
  rather than the spread. Everything else — ensembling, weight averaging,
  reparameterisation, capacity, edge weighting — moved the spread or made
  things worse. And it arrived as a side effect of a change made for speed.

### Both problem parameters CROSSED ZERO between batch 128 and 1024

| | g_a | g_t0 |
|---|---|---|
| batch 128 | −0.044 | −0.060 |
| batch 1024 | +0.009 | +0.031 |

Linear interpolation puts the a-crossing at ~83% and the t0-crossing at ~66% of
the way from 128 to 1024 — i.e. both near batch 500-700. **A batch sweep is now
the most promising open lead in the project**, and it is cheap: batch 1024 is 3x
faster per epoch than 128, three runs fit concurrently, and a 300-epoch run is
~5 h. A sweep of {256, 512, 768} is one overnight window.

### Caveat on attribution

Batch size is not a clean single factor here. peak_lr was held at 1e-4 while
total_steps fell from 37.5M to 4.69M, so the learning-rate schedule changed too,
and convergence moved later (best val at epoch 262 vs 179 and 73). The honest
statement is that this CONFIGURATION is better, not that batch size per se
causes it. A sweep that varies batch with lr held fixed keeps the same
confound — which is acceptable for finding a good operating point, but not for
a mechanistic claim in the paper.

## Phase 17 — analytic control at N=400: the t0 drift is the FLOW's

tux26, 500 replicates per cell (the registered configuration), ~4 h on 72 cores.

| | v | a | t0 | Z | sv | SZ | st0 |
|---|---|---|---|---|---|---|---|
| **N=100** analytic, Gadi | −0.041 | +0.066 | **−0.015** | +0.082 | −0.058 | −0.008 | −0.056 |
| **N=100** analytic, tux26 | +0.020 | +0.086 | **+0.005** | +0.007 | +0.048 | −0.021 | +0.082 |
| **N=100** flow cap256w_c4 | +0.039 | −0.003 | **+0.118** | −0.026 | −0.021 | +0.040 | −0.026 |
| **N=400** analytic, tux26 | +0.042 | +0.086 | **−0.038** | −0.026 | +0.042 | +0.005 | −0.003 |
| **N=400** flow cap256w_c4 | +0.052 | +0.012 | **+0.319** | −0.032 | +0.014 | −0.051 | −0.136 |

### Two things established

**1. tux26 reproduces Gadi.** The two independent N=100 analytic cells agree on
every parameter within Monte Carlo error (se = 1/sqrt(500) = 0.045; largest
discrepancy is t0, 0.020, under half an se). The SBC pipeline — priors, ranks,
coverage, the lot — is now validated across two machines and two installs.

**2. The analytic likelihood does NOT drift on t0, and the flow does.**

| N | analytic t0 | flow t0 | difference |
|---|---|---|---|
| 100 | +0.005 | +0.118 | +0.113 |
| 400 | −0.038 | +0.319 | +0.357 |

The analytic stays inside +/- 1 se of zero at both N while the flow grows
3-fold. The gap triples. **The t0 bias is a property of the learned likelihood,
not of the simulator, the prior, the rank computation or anything else in the
pipeline** — which is the precondition the paper needs before attributing it to
the flow and justifying a ~0.28% correction.

Note the analytic's worst parameter is a at +0.086 at BOTH N (1.9 se), small and
non-growing — consistent with Monte Carlo noise rather than a real defect, but
worth not over-claiming: "the exact likelihood is calibrated to within ~0.09 at
500 replicates" is the defensible statement, not "exactly zero".

### Phase 16b — CORRECTION: the batch-1024 advantage, with classifiers matched

Phase 16 compared `b1k_amp0` against `cap256w_ckpt`, both carrying the OLD
copied classifier, and reported batch 1024 as 5x better on mean g_a. That
comparison is internally valid but the wrong reference: the model that actually
passed SBC is `cap256w_c4`, which uses the corrected-target classifier. Splicing
c4 into the batch-1024 checkpoints (`sbc/splice_classifier.R`) gives the fair
comparison:

| par | b1k_amp0 + c4 (mean, sd over 10 ckpts) | cap256w_c4 |
|---|---|---|
| v | −0.0008 (0.0005) | +0.0003 |
| **a** | **+0.0209** (0.0098) | **+0.0136** |
| Z | +0.0030 (0.0021) | −0.0061 |
| sv | +0.0012 (0.0001) | −0.0001 |
| SZ | −0.0005 (0.0005) | −0.0017 |
| st0 | +0.0019 (0.0019) | +0.0034 |
| t0 | +0.031 | −0.064 |

Batch 1024 is better on Z, st0, SZ and t0, and slightly WORSE on a on average —
though its best checkpoint (`b1k_amp0_e216_c4`) reaches g_a +0.0051, beating
cap256w_c4's +0.0136.

**The 5x headline does not survive the correction.** The advantage is real but
modest and parameter-dependent. The lesson for the record: always splice the
production classifier BEFORE comparing flows, because the classifier
contributes a share of g comparable to the flow's own (Phase 7), and comparing
flows carrying different classifiers measures the classifier as much as the flow.

The zero-crossing argument for a batch sweep is unaffected: it rests on the
old-classifier comparison of the same two arms, where both a and t0 changed sign
between batch 128 and 1024, and the classifier contribution is common to both.

## Phase 18 — RDM: state of play, and a problem with the rationale

### Where the RDM stands

- **One flow only**: `rdm_large` (Julian's original). Architecture is small —
  context [v, B, t0, s, A], a SINGLE hidden layer [5 -> 32] into 97 spline
  outputs, 32 bins. Compare the DDM's current [9 -> 256 -> 128] into 193
  outputs with 64 bins. The RDM flow has never had the capacity or budget work
  the DDM received.
- **SBC at N=100 only**, 500 replicates:

  | | v | v_lMTRUE | B | A | t0 |
  |---|---|---|---|---|---|
  | RDM analytic | −0.031 | +0.013 | +0.041 | −0.050 | +0.039 |
  | RDMnn flow | −0.031 | +0.060 | −0.009 | +0.024 | **+0.152** |

  Same signature as the DDM: **t0 is the worst parameter and it is
  overestimated**, 4x the analytic's. No N=200/400/800 cells, so we do not know
  whether it drifts with N as the DDM's does (+0.118 -> +0.371).
- **No training data on disk.** `data/train/*/RDM` does not exist locally; the
  RDM flow would have to be re-simulated before any retraining.
- **No race-model score-bias tooling.** `score_bias_draws.R` and
  `score_bias_t0.R` are DDM-specific.

### The rationale problem: an RDM neural likelihood is 80x SLOWER

Measured directly (`EMC2:::dRDM` vs `EMC2:::dRDMnn`, 2000 accumulator-rows):

| | per call |
|---|---|
| RDM analytic | **490 us** |
| RDM flow | **39,380 us** |

and the same pattern in SBC wall time (N=100, 500 reps):

| model | analytic | flow | ratio |
|---|---|---|---|
| DDM | 68.4 min | 37.2 min | **0.54x — flow WINS** |
| LNR | 4.0 min | 29.1 min | 7.2x slower |
| RDM | 5.0 min | 56.7 min | 11.4x slower |

**The neural likelihood only pays for the DDM**, whose analytic likelihood is an
infinite series with precision control. The race models have closed-form
first-passage densities (Wald for RDM, lognormal for LNR) that are already
cheap, and the flow costs roughly the same regardless of model — worse for a
race, since each trial needs a density AND survivors for every accumulator.

**This contradicts the stated reason for keeping RDM over LNR** ("RDM is the
expensive one"). On this evidence RDM and LNR analytics cost about the same
(5.0 vs 4.0 min) and NEITHER is worth accelerating.

### What this means for scope

An RDM element cannot be justified as a speedup. It can be justified as a
**generality demonstration** — that the calibration method transfers to a model
with a different architecture and no choice classifier — which is a legitimate
and arguably stronger claim for a methods paper than a second speed result.
That reframing changes what "refinement" means: the target is a calibrated RDM
flow that demonstrates the recipe, not a production artefact anyone would
deploy for speed.

Caveat on the timings: they are for one design (2 accumulators, 1 condition,
single participant). Designs with many accumulators or conditions could shift
the ratio, and the analytic RDM's cost grows with accumulators — but so does the
flow's, and from a much higher base.

## Phase 19 — RDM pipeline built, and a step-size bug that also affects the DDM

Everything needed to run the RDM refinement is now in place and smoke-tested
end to end (simulate -> train -> export -> score bias):

- `simulation/sim_rdm_16m.R` — single-accumulator RDM data at 16e6 rows, the
  scale the DDM uses. `rdm_large` (the only existing RDM flow) had 1e6. Note the
  race simulator supports n_trials = 1 only, so the budget is pure
  theta-diversity, unlike the DDM's 40 trials per draw.
- `flow/scripts/train_race.py` — race training, no classifier. Defaults encode
  the DDM findings: batch 1024, [256,128], 64 bins, 30 checkpoints retained.
  The existing RDM flow is a single 5->32 layer with 32 bins.
- `flow/scripts/export_flow.py` — now takes arbitrary `--budget` tags plus
  `--step`/`--tag-suffix` for checkpoint screening, and falls back to the run
  root for the scaler (the checkpointer wipes the checkpoint dir).
- `sbc/score_bias_race.R` — race score bias, assembling
  L = f_R(rt) * prod_{j != R} S_j(rt) from the port's `log_pdf` and `log_sf`,
  with the analytic control and a restricted-region t0 term.

### The step-size bug

The race estimator's analytic control came out at −0.003 to −0.005 with the
DDM's H = 0.05 — that is 25-40% of the flow signal it was meant to measure. It
is not truncation (identical at Tmax 20/100/500 s) and not resolution
(identical at 200/800 grid points). It is O(H^2) central-difference truncation:

| H | control for v |
|---|---|
| 0.050 | −0.003163 |
| 0.020 | −0.000506 |
| 0.010 | −0.000126 |
| 0.005 | −0.000032 |

Textbook quadratic. At H = 0.01 the control is ~1% of signal, so the race
estimator now defaults there (`SBR_H`).

**This affects the DDM measurements too.** `score_bias_draws.R` uses H = 0.05,
and its analytic control for a is +0.00187 — 14% of cap256w_c4's measured
g_a of +0.0136. Since the offset is common to every model measured the same
way, it does NOT affect the rankings that the seed-screening result rests on
(Phase 11), and it does not change any sign or conclusion. But absolute DDM
g values carry roughly a +0.002 offset on a, and should be re-measured at
H = 0.01 before anything is quoted numerically in the paper.

### First RDM measurement

| tag | v | v_lMTRUE | B | A | t0 |
|---|---|---|---|---|---|
| rdm_large | −0.0082 | −0.0036 | **+0.0229** | +0.0030 | **+0.0189** |

Threshold (B) and non-decision time (t0) dominate — the same pair as the DDM,
where a and t0 are the problem parameters. That is a useful generality signal in
itself: the failure mode is not DDM-specific.

## Phase 20 — the batch sweep, and a CORRECTION to Phase 16's headline

Four batch sizes, one training run each, 10 plateau checkpoints per run, plain
loss, same architecture and seed:

| batch | mean g_a | within-run sd | mean g_t0 | abs sum |
|---|---|---|---|---|
| 128 | −0.0443 | 0.0313 | −0.067 | 0.111 |
| 512 | −0.0273 | 0.0142 | **+0.011** | **0.038** |
| 768 | −0.0332 | 0.0085 | −0.032 | 0.065 |
| 1024 | **+0.0086** | 0.0098 | +0.036 | 0.045 |

**The zero-crossing prediction is refuted.** Both intermediate arms sit on the
batch-128 side and 768 is MORE negative than 512, so g_a is not monotone in
batch and the crossing is between 768 and 1024, not the 500-700 that linear
interpolation from two points suggested. Predicted and recorded before the run;
wrong.

### CORRECTION: the mean effect is not established

Phase 16 reported batch 1024 as shifting the MEAN of the score bias — "the first
intervention in the whole project to shift the mean rather than the spread".
That claim does not survive this sweep.

Each batch arm is a SINGLE training run. Run-to-run variation of g_a, measured
at batch 128 over five seeds (Phase 8), has sd 0.0668. The spread of the four
arm means here is 0.0529 — **0.79 of one run-to-run sd**. The arm differences,
including the 1024-vs-128 difference that Phase 16 headlined, are comfortably
within what two runs of the SAME configuration differ by. Non-monotonicity is
exactly what one expects when reading noise as signal.

**What IS established: batch size reduces the WITHIN-run spread.** 0.0313 at
128 against 0.0098 at 1024 is a 10.3x variance ratio, against an F(9,9) critical
value of 5.35 at p = 0.01, and the pattern is consistent across all four arms
(0.031, 0.014, 0.009, 0.010). That is a real effect measured within runs, where
run-to-run variation cannot contaminate it.

So the defensible statement is: **larger batches make a run's checkpoints more
homogeneous; whether they move the mean is unknown.** Establishing the latter
needs replication — 3+ seeds at each of two batch sizes, which is 6 runs. At
batch 1024 (~5 h each, ~4 cores) that is one overnight window on a machine that
can hold three concurrently, or a single window on tux26.

### What still holds from Phase 16

- Batch 1024 is ~3x faster per epoch than 128 (measured, `sbc/bench_batch.log`).
- The batch-1024 run has the best validation loss of any run (−0.12528).
- Edge weighting is dead: monotone harm, +0.009 / +0.471 / +0.991 / +1.53 at
  amp 0 / 3 / 10 / 30 — those arms differ by far more than run-to-run variation.

## Phase 21 — RDM refined: the DDM's lessons transfer, 2-7x better on every parameter

`rdm_b1k`: 16e6 rows (vs `rdm_large`'s 1e6), conditioner [5,256,128] (vs a
single 5->32 layer), 64 spline bins (vs 32), batch 1024, 30 checkpoints.
Score bias over 100 SBC prior draws, analytic control ~1e-4:

| | v | v_lMTRUE | B | A | t0 |
|---|---|---|---|---|---|
| rdm_large | −0.0087 | −0.0041 | +0.0228 | +0.0032 | +0.0188 |
| **rdm_b1k** | **−0.0012** | **+0.0018** | **+0.0065** | **+0.0007** | **+0.0057** |
| factor | 7.5x | 2.2x | 3.5x | 4.8x | 3.3x |

Better on every parameter, including t0 and B — the two that dominate the RDM
exactly as a and t0 dominate the DDM. **The DDM recipe transfers**, which is the
generality claim the RDM element exists to support, and it cost one simulation
and one training run because the blind alleys were already mapped.

### Two bugs found on the way, both mine

**1. Context column order.** `train_race.py` used ["v","B","t0","A","s"];
`export_flow.py`'s registry declares ["v","B","t0","s","A"] — A and s
transposed. The port builds theta from the exported `context_names`, so it fed
the network permuted inputs. Measured score bias was v +5.15, B −9.33,
t0 −5.13 — roughly 500x rdm_large's, and near-identical across all ten
checkpoints, which is the signature of a systematic wiring fault rather than a
training failure. The network was fine; only its metadata disagreed.

Fixed both ways: `train_race.py` now uses the exporter's canonical order, and
`export_flow.py` gained `--context` to declare a different order for a run
already trained. The uniformity across checkpoints was the tell — a badly
trained model varies between checkpoints, a mis-wired one does not.

**2. The port-vs-distrax guard is calibrated for the old architecture.**
`export_flow.py` asserts max |dlog_pdf| < 5e-3 between the float64 port and
float32 distrax. The new RDM flow sits at 6.2e-3 to 2.0e-2 across its 30
checkpoints — systematically above, not one unlucky draw:

| | median | p90 | p99 | max | frac>1e-3 |
|---|---|---|---|---|---|
| rdm_large | 2.4e-06 | 9.2e-05 | 1.4e-03 | 3.1e-03 | 0.012 |
| rdm_b1k | 9.2e-06 | 3.3e-04 | 4.9e-03 | 7.2e-03 | 0.063 |

The comparison runs float64 port vs float32 distrax, so this measures FLOAT32's
rounding, not the port's error — and a network with ~15x the parameters and
twice the bins accumulates proportionally more. Corroborating: max |dz| is
2.1e-5 against a 5e-4 threshold, so the spline inversion is accurate and only
the log-density, which sums log-determinants, drifts. Verified independently
that the exported density integrates to 1.00000 (rdm_large: 0.99996).

`export_flow.py` now prints the distribution rather than the max alone, and
takes `--port-tol`. The threshold should be revisited for any architecture
larger than the original.

### Phase 21b — RDM checkpoint screen: the recipe works on the race model too

29 plateau checkpoints of `rdm_b1k` screened on score bias (200 draws each),
against `rdm_large`:

| par | mean over ckpts | sd | best ckpt | rdm_large |
|---|---|---|---|---|
| v | −0.00372 | 0.00615 | −0.00013 | −0.00854 |
| v_lMTRUE | −0.00001 | 0.00453 | −0.00019 | −0.00398 |
| B | +0.00950 | 0.01099 | −0.00011 | +0.02209 |
| A | +0.00111 | 0.00139 | +0.00017 | +0.00315 |
| t0 | +0.00658 | 0.00793 | −0.00006 | +0.01850 |

Total |g|: **best checkpoint 0.0072**, mean over checkpoints 0.0278,
`rdm_large` 0.0563. So the retrained flow is **2x better on average and 7.8x
better when screened** — and screening buys a further 3.9x over taking an
arbitrary checkpoint, exactly the pattern the DDM showed (Phase 14: select,
never average).

Best checkpoint `rdm_rdm_b1k_e242`: v +0.0018, v_lMTRUE +0.0039, B +0.0014,
A +0.0002, **t0 −0.00006**. Every parameter under 0.004, and t0 — the RDM's
worst parameter in `rdm_large` at +0.0185 — is essentially zero.

**This is the strongest single piece of evidence that the method generalises.**
The race model has no choice classifier, a different likelihood structure
(winner density x survivors) and different parameters, and the same recipe —
more data, more conditioner capacity, more bins, large batch, retain
checkpoints and screen on score bias — produces the same kind of improvement.

One checkpoint (step 246) was rejected by the export guard on max |dz| =
2.3e-03, 110x the typical 2.1e-05 and 4.6x its threshold, while its log-density
distribution was unremarkable. That is an ill-conditioned spline inversion at
some theta, not float32 drift — the guard catching a real defect in one
checkpoint. Screening 29 rather than 30 costs nothing.

### Phase 21c — the RDM accuracy gain costs 13x in speed

Measured via EMC2's compiled path, 2000 accumulator-rows:

| | per call | vs analytic |
|---|---|---|
| RDM analytic (`dRDM`) | 467 us | 1x |
| `rdm_large` flow (5->32, 32 bins, ~3.3k params) | 37,200 us | 80x |
| **`rdm_b1k` flow (5->256->128, 64 bins, ~59k params)** | **484,367 us** | **1037x** |

18x the parameters, 13x the time. So the 7.8x score-bias improvement was bought
with a 13x slowdown. For the DDM that trade is still worth it (the flow beats
its analytic); for the RDM the analytic was already 80x faster than the OLD
flow, so this only widens a gap that was never going to close.

That does not undermine the RDM element — it was reframed in Phase 18 as a
GENERALITY demonstration rather than a speedup, and Phase 21b delivers that.
But it does mean:

- the RDM comparison against Malte's likelihood must be on CALIBRATION, not
  speed, and should say so explicitly;
- an RDM SBC cell is now expensive. `RDMnn` at N=100 / 500 reps was 2,722
  core-minutes with `rdm_large`; at 13x that is ~35,400 core-minutes —
  **50-74 h on the laptop's 8 cores**, so it is a cluster job.

A cheaper middle architecture (e.g. [64,32] with 64 bins) was never tested for
the RDM and might capture most of the calibration gain at a fraction of the
cost. Worth one run before committing cluster time to the [256,128] version,
since nothing about the RDM requires the DDM's capacity.

### Setting up the RDM SBC

`EMC2:::dRDMnn` HARDCODES `theta = log(cbind(v, B, t0, s, A))` and ignores the
export's `context_names`, so a flow trained on a different column order cannot
be dropped in. `sbc/permute_context.R` re-expresses a flow in a target order by
permuting the scaler entries and the first layer's weight ROWS — exact, not an
approximation (verified: max |dlog_pdf| difference 4.9e-15).

Ready: `sbc/rdm_b1k_best.rds` (from checkpoint e242, canonical order), field-
compatible with the bundled `rdm_large.rds`. Install by replacing
`<EMC2>/extdata/flownn/rdm_large.rds`; the original is preserved at
`sbc/rdm_large_ORIGINAL.rds`.

### Phase 21d — RDM capacity: [64,32] gets most of the gain at a sixth of the cost

The [256,128] conditioner was transplanted from the DDM without testing whether
a race model needs it. Testing that (one 3.5 h run, 20 checkpoints screened):

| flow | total abs g | vs rdm_large | us/2000 rows | vs rdm_large | SBC @ 8 cores |
|---|---|---|---|---|---|
| rdm_large [5->32, 32 bins] | 0.05626 | 1.0x | 37,200 | 1.0x | 5.7 h |
| **rdm_small [64,32, 64 bins]** | **0.01255** | **4.5x** | **86,667** | **2.3x** | **13.2 h** |
| rdm_b1k [256,128, 64 bins] | 0.00721 | 7.8x | 484,367 | 13.0x | 73.8 h |

Per parameter, best checkpoints:

| | v | v_lMTRUE | B | A | t0 |
|---|---|---|---|---|---|
| rdm_small | −0.0035 | −0.0018 | +0.0030 | +0.0008 | −0.0034 |
| rdm_b1k | +0.0018 | +0.0039 | +0.0014 | +0.0002 | −0.0001 |
| rdm_large | −0.0085 | −0.0040 | +0.0221 | +0.0032 | +0.0185 |

**[64,32] captures 58% of the calibration gain for 18% of the speed cost.**
Both retrained flows fix the B and t0 problems that dominate `rdm_large`;
[256,128] is better on t0 specifically (−0.0001 vs −0.0034), which matters
because t0 is the parameter that fails SBC.

The screening spread is wider for the small model (best 0.0126, mean 0.0417,
worst 0.0973) than the large one, so checkpoint selection matters MORE at lower
capacity — consistent with the DDM finding that larger batches (and, it now
appears, larger networks) make a run's checkpoints more homogeneous.

**Recommendation: use [64,32] as the RDM reference artefact.** 13.2 h of SBC on
the laptop is affordable where 73.8 h is not, the calibration is 4.5x better
than the existing flow, and the RDM's role is a generality demonstration rather
than a production likelihood (it remains ~185x slower than the analytic). If
the SBC shows t0 still failing, [256,128] is the fallback and needs cluster time.

## Phase 22 — RESOLVED: batch size is a variance tool, not a mean tool

Five seeds at batch 1024 (tux26, 10 plateau checkpoints each), against the five
batch-128 seeds of Phase 8/10b. This is the replication Phase 20 said was needed.

### g_a

| | batch 128 | batch 1024 |
|---|---|---|
| mean of run-means | −0.0393 | −0.0362 |
| between-seed sd | 0.0668 | **0.0227** |

Mean shift **+0.0031, t = 0.10** — nothing. Variance ratio **8.7x**, against
F(4,4) critical 6.39, significant at p < 0.05.

### g_t0

| seed | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| g_t0 | −0.0911 | −0.0704 | −0.1023 | −0.0838 | −0.0074 |

| | batch 128 | batch 1024 |
|---|---|---|
| mean | −0.1117 | −0.0710 |
| between-seed sd | 0.0688 | 0.0374 |

Mean shift **+0.0407, t = 1.16** — not significant. Variance ratio 3.4x, below
the 6.39 critical value, so the variance reduction is suggestive here but not
established at n = 5. **All ten checkpoints are negative.**

### Conclusions

1. **Batch size does NOT shift the mean of either parameter.** The Phase 16
   claim is dead, and now for a second independent reason: the single b1k_amp0
   run gave g_a = +0.0086 (2.0 sd above the seed mean) and g_t0 = +0.031
   (2.7 sd away). Both were lucky draws from one run. Phase 20 suspected this
   from the arm-spread arithmetic; this confirms it with replication.

2. **Batch size DOES reduce variance**, strongly for a (8.7x) and suggestively
   for t0 (3.4x). Combined with the within-run effect (Phase 20: 0.031 -> 0.010),
   larger batches make both checkpoints AND seeds more homogeneous. That is
   genuinely useful — it makes screening reliable — but it is not a fix.

3. **The tighter variance exposes a systematic bias in `a` that was hidden.**
   At batch 1024, mean g_a = −0.0362 with se 0.0102: **t = 3.6**. At batch 128
   the same mean sat in a spread three times wider (t = 1.3), which is why
   Phase 8 called it indistinguishable from zero. `a` has a real systematic
   component of about −0.036; the noise was concealing it. |mean|/sd rises from
   0.59 to 1.59.

4. **t0 remains systematically negative at every batch size and every seed** —
   10/10 checkpoints here, 5/5 seeds at batch 128, |mean|/sd of 1.9. Nothing
   tried in this project has moved it.

### Consequence for the recipe and the paper

Selection handles neither mean. Screening finds the best checkpoint of a
distribution whose CENTRE is displaced, and larger batches tighten that
distribution around the same displaced centre — which paradoxically makes the
best-of-N slightly WORSE as N-effective falls, while making results more
reproducible.

So the honest recipe is: **large batch for reproducibility and speed, checkpoint
screening to pick the best of the remaining spread, and an explicit statement
that a systematic residual remains on a (−0.036) and t0 (−0.071).** Item 2 of
the paper plan — justifying the ~0.28% t0 correction — is now the main route,
with distillation (item 3) the only untried intervention that could move a mean.

## Phase 23 — accuracy against EMC2's DEFAULT-precision analytic

Our estimator has always used WienR at precision 1e-8 as truth, which is right
for a calibration reference. But EMC2's `dDDM` defaults to **5e-3**, so that is
what users actually run against. Scoring the 5e-3 analytic with the same
estimator, 1e-8 as truth, 120 draws:

| par | mean g | se |
|---|---|---|
| v | −0.00007 | 0.00004 |
| a | +0.00011 | 0.00017 |
| Z | −0.00011 | 0.00021 |
| sv | −0.00000 | 0.00000 |
| SZ | +0.00010 | 0.00020 |
| st0 | −0.00016 | 0.00012 |

**Essentially zero on every parameter** — yet its POINTWISE error is large:
max |log f(5e-3) − log f(1e-8)| per draw has median 0.33 and max 0.51.

### The point this makes

The default-precision analytic is **pointwise much worse than our flow** (0.33
against ~0.12 at the flow's worst region, and a mean gap of 0.0065) and
**calibration-wise much better** (|g| ~1e-4 against +0.013 on a and −0.063 on
t0). The two orderings are opposite.

The reason is WHERE the error sits. WienR's low-precision error is series
truncation, largest where the density is tiny and the integration weight is
therefore negligible; it does not correlate with theta, so it cancels in the
expectation and in its derivative. The flow's error is concentrated at the
leading edge, which carries little mass but all the information about t0, and it
varies systematically with theta.

This is the project's thesis stated in reverse, and it is the cleanest
demonstration of it we have: **a model can be pointwise sloppy and perfectly
calibrated, or pointwise tight and biased.** Held-out likelihood, MSE and
max-error all fail to distinguish them.

### Consequence for the claims

Against current practice (EMC2 at its default), the flow is **4.0-5.8x faster
and less accurate in the sense that matters**. The defensible claim is a
characterised trade, not a free win:

- speed: 4.0x (c4 classifier) or 5.8x (c5), rising to 59-86x if a tight analytic
  is required — the flow's cost is fixed while the analytic's scales with
  precision;
- cost: t0 point estimates carry ~0.28% of bias (0.8 ms at t0 = 0.3 s);
  intervals remain valid.

Given the stated priority of accuracy over speed, **c4 is the right classifier
for the shipped artefact** despite costing 45% of the speedup, and the paper
should present the speed/accuracy relationship rather than a single multiplier.

## Phase 24 — DISTILLATION WORKS: the first intervention to move the mean

Three arms, batch 1024, corrected c4 classifier, target = the analytic `ll`
already stored per row (joint, converted to conditional via the classifier).
Five plateau checkpoints each; baseline is plain MLE at batch 1024 over five
seeds (Phase 22).

| beta | arm | mean g_a | mean g_t0 | abs sum |
|---|---|---|---|---|
| 0 | plain MLE (5 seeds) | −0.0362 | −0.0710 | 0.1072 |
| 1 | dist_b1 | **−0.0090** | −0.0598 | 0.0688 |
| 10 | dist_b10 | +0.0151 | −0.0328 | **0.0478** |
| inf (nll_weight 0) | dist_pure | +0.0469 | +0.0249 | 0.0718 |

**Both parameters move monotonically positive with beta, and both cross zero.**
g_a crosses between beta = 1 and 10; g_t0 between beta = 10 and pure.

### Why this is believable where the batch claim was not

The batch claim rested on a single run differing from another single run by less
than the between-seed sd. Here:

- `dist_pure` shifts g_a by **3.7 between-seed sd** and g_t0 by **2.6 sd** —
  individually beyond the noise that sank the batch result.
- The three arms order **monotonically in beta on both parameters
  independently**. Three runs landing in the correct order on two correlated-
  but-distinct quantities is not a plausible accident.
- The direction is the one the mechanism predicts: a pointwise target pulls the
  fit toward the analytic, and overshooting past zero as beta grows is what an
  increasingly dominant regression term should do.

### The trade-off, stated honestly

No tested beta zeroes both. beta = 10 is the best total (0.0478, versus 0.107 for
MLE and 0.077 for cap256w_c4), but g_a and g_t0 cross zero at different beta, so
a sweep would trade one against the other rather than fixing both. Since t0 is
the parameter that fails SBC, a beta between 10 and pure is the place to look —
accepting g_a around +0.03 to bring g_t0 to ~0.

### What it does and does not establish

It **does** show the residual is not architectural: the same network, same data
and same batch size reach a different centre when the objective changes. Every
previous intervention moved the spread; this moves the centre.

It **does not** give a general method. Distillation requires the analytic
likelihood, so it applies where one exists but is slow — the DDM, and the next
project — and not to genuinely intractable models. For those, the honest
position remains the one Phase 22 left: selection and large batch control the
spread, and a systematic residual remains.

### Next

1. A beta sweep between 10 and pure (say 30, 100) to find where g_t0 ~ 0,
   with the g_a cost measured rather than assumed.
2. Replicate the chosen beta across 3 seeds before believing the mean — the
   lesson from Phase 20/22.
3. SBC on the selected model: score bias predicts rank within an architecture,
   but the verdict is SBC.

## Phase 25 — the MATCHED batch replication: two Phase 22 conclusions corrected

Phase 22 compared five batch-1024 seeds trained on tux26 against the five
batch-128 seeds of Phase 8/10b — which were trained on a different machine, at a
different time, with a different surrounding configuration. This is the matched
control: **five seeds per arm, same machine, same seeds (101-105), same
[256,128]/64-bin architecture, same 300 epochs, same 10 retained plateau
checkpoints, identical `variant_config.json` except `batch`.**

Protocol, following Phase 16b: the c4 classifier (`cap256w_c4`) was spliced into
all 100 exports before measuring. The runs were trained carrying the copied
`large` classifier, whose choice probability is **13x worse** than c4 on the SBC
prior draws (mean |P(R) err| 0.0113 vs 0.00085, measured directly). Comparing
flows across different classifiers measures the classifier.

Estimators: `score_bias_draws.R` at **H = 0.01** (analytic control a = +0.00008)
and `score_bias_t0.R` for t0, 200 prior draws each. The seed is the unit of
replication throughout — checkpoints within a run are not independent.

### 1. The mean: no batch effect (Phase 22 conclusion 1 CONFIRMED)

| | batch 128 | batch 1024 | diff | t | p |
|---|---|---|---|---|---|
| g_a | −0.0427 | −0.0259 | +0.0168 | 0.85 | 0.43 |
| g_t0 | −0.0084 | −0.0696 | −0.0612 | −1.96 | 0.094 |

Only `SZ` reaches nominal significance (+0.0028, t = −3.11, p = 0.016), which
does not survive six-parameter correction and is on a prior-dominated nuisance
parameter.

### 2. CORRECTION: the 8.7x variance ratio does not survive

| | Phase 22 (mixed machines) | Phase 25 (matched) |
|---|---|---|
| b128 between-seed sd, g_a | 0.0668 | **0.0380** |
| b1024 between-seed sd, g_a | 0.0227 | 0.0225 |
| variance ratio | **8.7x, p < 0.05** | **2.84x, p = 0.34** |

The b1024 arm reproduces exactly (0.0227 -> 0.0225), so the entire change is in
the b128 arm: its spread nearly halves once it is measured on runs matched to
the b1024 arm. **Roughly half of what Phase 22 attributed to batch size was
between-machine/between-era heterogeneity.** The t0 ratio is 2.79x (p = 0.34),
also not significant.

Batch 1024 still buys reproducibility and a 3x epoch-time saving, so the
recommendation is unchanged — but the effect size claimed for it was inflated,
and "8.7x" must not be quoted.

### 3. The `a` residual, properly replicated

Pooled over all ten seed-runs with the seed as the unit:

    g_a = −0.0343   sd 0.0307   t = −3.53   p = 0.006   89/100 checkpoints negative

Phase 22 reached t = 3.6 from one arm of five. This is the same conclusion on
twice the independent runs, and it is the strongest evidence in the project that
the `a` residual is real rather than a lucky draw.

Also significant but inconsequential: v +0.0027 (t = 4.44) and sv +0.0008
(t = 4.44) — sv has per-trial sd 590 and is prior-dominated at every N we run.
Z, SZ and st0 are indistinguishable from zero.

### 4. CORRECTION: t0's systematic negativity is NOT established

Phase 22 conclusion 4 read "t0 remains systematically negative at every batch
size and every seed". Against the matched b128 arm that is false:

| arm | s1 | s2 | s3 | s4 | s5 | arm mean | n negative |
|---|---|---|---|---|---|---|---|
| b1024 | −0.092 | −0.064 | −0.108 | −0.071 | −0.014 | **−0.070** | 49/50 |
| b128 | −0.034 | −0.021 | **+0.032** | −0.086 | **+0.068** | **−0.008** | 27/50 |

Two of five matched b128 seeds are POSITIVE, and the arm mean is
indistinguishable from zero. Pooled over ten seed-runs: −0.039, t = −2.18,
**p = 0.057**.

t0 is robustly negative *at batch 1024* (49/50 checkpoints), but that is a
property of batch 1024, not a universal one. The earlier "nothing has moved t0"
framing was built on an arm that was not matched.

### 5. The cheap predictor still cannot rank models — now shown for t0 too

Ranking all 106 measured flows by predicted standardised SBC bias
(delta = (N I + Sigma0^-1)^-1 N g) puts `b1k_amp0_e216_c4` far ahead (worst
parameter 0.128 at N=800) of production `cap256w_c4` (0.604, nominally failing).
**That ranking should not be acted on.** For `cap256w_c4` = `c4single`, the only
model with both a measured g and an observed SBC curve, the prediction has the
wrong SIGN on t0:

| N | predicted t0 | observed t0 (Phase 7) |
|---|---|---|
| 100 | −0.152 | +0.118 |
| 400 | −0.376 | +0.319 |
| 800 | −0.604 | +0.371 |

Magnitudes are the right order; the sign is inverted. Phase 9 established the
same failure for `a` (0/2 sign correct) and attributed it to the error being
region-dependent rather than a uniform displacement. **t0 now joins `a`.** Since
`a` or t0 is the worst parameter for nearly every model in the ranking, the
ranking inherits their unreliability. It is a hypothesis generator; SBC remains
the verdict.

### 6. RETRACTED: the gate failures are NOT a batch effect

First reading: 7 of 50 b128 checkpoints tripped the exporter's 5e-3
float32/float64 port gate and no b1024 checkpoint did, suggesting batch 128
produces occasionally sharper flows. Re-running the gate check over ALL 100
checkpoints (`flow/scripts/gate_check.sh`) refutes it:

| | n > 5e-3 | median max |dlog_pdf| | worst | median frac>1e-3 |
|---|---|---|---|---|
| batch 1024 | 3/50 | 1.87e-03 | **3.29e-02** | 0.0037 |
| batch 128 | 7/50 | 2.05e-03 | 2.04e-02 | 0.0041 |

7/50 against 3/50 is Fisher p = 0.32, the medians are indistinguishable, and
b1024's worst outlier is LARGER. The apparent effect came from comparing a
freshly-run gate check against b1024 exports produced earlier under different
conditions — the same mixed-provenance error that inflated the variance ratio in
item 2, made twice in one phase.

What survives is a SEED effect, not a batch effect: b128 s4 accounts for 5 of
the 7 failures, and b1024 s3 for 2 of its 3. Individual seeds produce flows with
sharper regions; batch size does not predict which.

The 7 were exported with `--port-tol 2.5e-2` and retained rather than dropped,
which remains correct: all had p99 and frac>1e-3 better than
`rep_b1024_s1_e216`, which passed, and dropping them would have removed the
roughest checkpoints from one arm only.

### Consequence for the paper

Item 2 of the paper plan (justify the ~0.28% t0 correction) is unaffected — that
rests on Phase 7's observed SBC curve, not on g. What changes is the recipe
section: batch size should be sold as reproducibility plus speed, with the
variance claim stated as "a factor of ~3, not significant at five seeds per
arm", and the t0 residual described as batch-dependent rather than universal.

## Phase 26 — RDM SBC: `rdm_small` PASSES on every parameter and every criterion

tux26, 48 cores (cores_per_chain 16 x 3 chains), N = 100, 500 replicates,
117.9 min. Bundle recorded at launch: `budget=rdm_small step=195 bins=64
context=v,B,t0,s,A`, and the original `rdm_large.rds` was restored afterwards on
both machines (verified by comparison against `sbc/rdm_large_ORIGINAL.rds`).

| par | std bias | mean rank | KS p | coverage | gate |
|---|---|---|---|---|---|
| v | −0.024 | 0.509 | 0.91 | 0.958 | PASS |
| v_lMTRUE | −0.026 | 0.501 | 0.93 | 0.948 | PASS |
| B | +0.004 | 0.502 | 0.61 | 0.970 | PASS |
| A | −0.075 | 0.519 | 0.17 | 0.960 | PASS |
| t0 | **−0.013** | 0.520 | 0.27 | 0.960 | PASS |

Worst standardised bias is 0.075 against a 0.5 gate; every rank distribution is
uniform; coverage is 0.948-0.970 against nominal 0.95 (MC se 0.0097).

### Against the two references, computed identically

| model | v | v_lMTRUE | B | A | t0 |
|---|---|---|---|---|---|
| `rdm_large` (Julian's) | −0.031 | +0.060 | −0.009 | +0.024 | **+0.152** |
| RDM analytic | −0.031 | +0.013 | +0.041 | −0.050 | +0.039 |
| **`rdm_small` (new)** | −0.024 | −0.026 | +0.004 | −0.075 | **−0.013** |

KS p, same order: t0 goes 0.058 -> 0.912 (analytic) -> 0.273, and v_lMTRUE's
marginal non-uniformity in the old flow (0.038) is also resolved (0.932).

**t0 improves 12x and now sits closer to zero than the ANALYTIC's own t0 bias at
this replicate count.** This is Phase 21's recommendation confirmed: [64,32] with
the DDM's recipe (batch 1024, 64 bins, checkpoint screening on score bias) is
sufficient, and `rdm_b1k` is not needed.

### Why this matters beyond the RDM

1. **The recipe transfers.** Everything carried over except the classifier,
   which a race model does not have: capacity, bin count, batch size, and
   checkpoint screening. A recipe that only worked on the model it was developed
   on would be worth little; this is the generality evidence the paper needs.

2. **The DDM's t0 failure does NOT reproduce here** (KS p 0.27, std bias −0.013).
   That is evidence for the mechanism proposed in Phase 7: the DDM's t0 error
   comes from a t0/st0 leading-edge trade-off, and **the RDM has no st0 at all**.
   A hard support boundary with nothing to trade against is apparently easier for
   a flow to represent than a smeared one. The competing explanation — that
   flows generically mis-handle the leading edge — predicts a t0 failure here
   and does not survive.

3. It closes the RDM branch. Scenario 1 of the contingency plan; no further RDM
   compute is required.

### Caveat that must travel with this result

`rdm_small` was SELECTED by checkpoint screening on score bias from 30 retained
checkpoints, and the screening spread is wide at this capacity (best 0.0126,
mean 0.0417, worst 0.0973). This SBC therefore validates a selected checkpoint,
not the architecture on average. That is the intended use of the recipe — select,
then verify — but it means a fresh [64,32] run would need its own screening and
should not be assumed to land here. N = 100 only; larger N is untested for the
RDM, and the DDM's t0 problem only became visible from N = 200.

## Phase 27 — the t0 correction: it is 0.28%, not 3 ms, and it WORKS

### First, a units error in our own record

Every "~3 ms" in this document and in `HANDOVER_PLAN.md` was wrong by ~3.5x, and
has been corrected. The Phase 7 mechanism table reported the mean t0 error as
"+0.00197 s", but those values are on the SAMPLED (log) scale, not in seconds.
Reading them as seconds gives "2-3 ms"; converting properly gives

    mean bias +0.0028 on log t0  =  0.28%  =  0.83 ms at a typical t0 of 0.3 s

The document contradicted itself — Phase 13's control table already carried the
correct "+0.59 ms" for the same model at N=100 — and the inconsistency went
unnoticed because nothing depended on the number until we tried to apply it.
**Lesson for the paper: state the scale on every table of parameter-space
quantities.** Everything in this project is on the sampled scale except where it
is explicitly not, and t0, a, sv and st0 are all log-transformed.

### The correction is multiplicative, not additive

Because the bias is constant on the log scale, the correction is

    t0_corrected = t0_hat * exp(-0.00276) = t0_hat * 0.99724

A fixed millisecond subtraction is the wrong shape: it would under-correct fast
responders and over-correct slow ones.

### It transfers across N, tested out of sample

`sbc/test_t0_correction.R`. Bias columns are recomputed exactly. Ranks are not
stored as posterior draws, so they are recomputed under a normal-posterior
approximation, s = -bias/qnorm(rank) per replicate, then
rank' = Phi(-(bias-delta)/s). The reconstruction reproduces the published
uncorrected mean ranks (0.485/0.448/0.420) exactly, which is the check that it
is reading the right quantity; the corrected ranks remain an approximation.

delta is estimated LEAVE-ONE-N-OUT, so each cell is corrected using only the
other cells:

| N | std bias before | after | KS p before | after |
|---|---|---|---|---|
| 100 | +0.118 | −0.064 | 0.44 | 0.031 |
| 200 | +0.249 | +0.051 | **7.2e-04** | **0.33** |
| 400 | +0.319 | +0.053 | **6.7e-08** | **0.93** |
| 800 | +0.371 | −0.019 | **2.0e-06** | **0.40** |

Per-N delta: 0.00197 / 0.00327 / 0.00316 / 0.00266 — stable, no trend after
N=100. **The correction removes the rank non-uniformity that made strict SBC
fail at N = 200, 400 and 800**, and shrinks the standardised bias ~6x.

At N=100 it slightly OVER-corrects (p = 0.031): the bias is genuinely smaller
there, so a single global delta is mildly too aggressive at small N. Reporting a
single constant is defensible; claiming it is exact at every N is not.

### Status of paper item 2

This is the justification that item 2 asked for, with one caveat: it is
calibrated on the same four SBC cells it is tested on, in the leave-one-N-out
sense only. A genuinely independent test would be a new cell at an N not in the
set — N=200 for the RDM analogue, or N=1600 for the DDM.

## Phase 28 — RDM at N=400: t0 is FLAT, and the DDM contrast is decisive

tux26, 24 cores (cores_per_chain 8 x 3), N = 400, 500 replicates, 866.9 min.
Bundle verified at launch and the original restored on exit (exit 0).

| par | std bias | mean rank | KS p | coverage | gate |
|---|---|---|---|---|---|
| v | +0.034 | 0.496 | 0.81 | 0.946 | PASS |
| v_lMTRUE | −0.026 | 0.502 | 0.71 | 0.948 | PASS |
| B | +0.041 | 0.487 | 0.33 | 0.940 | PASS |
| A | +0.090 | 0.475 | 0.054 | 0.946 | PASS |
| **t0** | **−0.001** | 0.518 | 0.11 | 0.944 | PASS |

Coverage is 0.940-0.948 against nominal 0.95 (MC se 0.0097) — slightly low
across the board but within 1 se everywhere.

### The result this cell was run to get

N = 400 is exactly where the DDM's t0 failure became decisive. Same estimator,
same gate, same replicate count:

| N | DDM `cap256w_c4` t0 | RDM `rdm_small` t0 |
|---|---|---|
| 100 | +0.118 (KS p 3.9e-01) | −0.013 (KS p 2.7e-01) |
| 200 | +0.249 (KS p **3.3e-04**) | — |
| 400 | +0.319 (KS p **1.8e-07**) | **−0.001** (KS p 1.1e-01) |
| 800 | +0.371 (KS p **3.8e-06**) | — |

**The DDM's t0 bias grows monotonically and its rank distribution collapses;
the RDM's is flat at zero and stays uniform.** This is not a power difference —
both cells have 500 replicates at the same N, and the DDM's effect is 300x its
own standard error at N=400.

### What it establishes

1. **The RDM is confirmed at the N where the DDM broke.** Phase 26's pass was
   not an artefact of N=100 being underpowered. Scenario 1 stands; no further
   RDM compute is required, and no t0 correction is needed for the RDM.

2. **The t0/st0 mechanism survives its sharpest test.** Phase 7 proposed that
   the DDM's t0 error comes from a leading-edge trade-off between t0 and st0.
   The RDM has NO st0: its support boundary is a hard jump with nothing to trade
   against. The mechanism predicts no t0 failure here, and there is none, at
   either N. The rival explanation — that normalizing flows generically
   mis-handle a density's leading edge — predicts a failure that grows with N in
   BOTH models, and is refuted.

3. **A cost: the independent test of the t0 correction is no longer available
   from the RDM.** Phase 27's correction is calibrated and tested on the same
   four DDM cells (leave-one-N-out). The RDM would have provided a clean
   out-of-sample check had it shown the same bias — but it has no bias to
   correct. An independent DDM cell at an N outside {100,200,400,800} remains
   the honest way to close Phase 27.

### The one parameter worth watching

`A` moved from −0.075 (N=100) to +0.090 (N=400), with KS p 0.054 — the only
marginal value in the table, and a sign flip. Both are comfortably inside the
gate and 0.054 is unremarkable across five parameters, but note that A is by far
the best-identified parameter here (sd of posterior medians 0.0165 against
~0.09 for the rest), so a small absolute error shows up larger in standardised
terms. If an RDM cell is ever run at higher N, A is the parameter to watch, not
t0.

## Phase 29 — REFUTED: fixing st0 does NOT fix t0

tux26, 24 cores, N=400, 500 replicates each. Cell A holds st0 at the prior mean
(0.05), Cell B at the training-box floor (0.01). Both use the shipped
`cap256w_c4` flow with st0 moved from the sampled set into `constants`, so the
design, priors and replicate count match the estimated-st0 cell exactly.

| par | st0 ESTIMATED | st0 FIXED 0.05 | st0 FIXED 0.01 |
|---|---|---|---|
| v | +0.052 (p 0.38) | +0.080 (p 0.071) | +0.156 (p 7.4e-03) |
| a | +0.012 (p 0.74) | −0.072 (p 0.20) | **+0.448 (p 6.8e-17)** |
| **t0** | **+0.319 (p 1.8e-07)** | **+0.269 (p 8.5e-07)** | **+0.293 (p 8.5e-11)** |
| Z | −0.032 (p 0.64) | **−0.148 (p 5.0e-05)** | **−0.405 (p 2.8e-12)** |
| sv | +0.014 (p 0.74) | −0.005 (p 0.84) | −0.019 (p 0.39) |
| SZ | −0.051 (p 0.12) | +0.030 (p 0.81) | −0.154 (p 1.2e-02) |
| st0 | −0.136 (p 3.3e-02) | — | — |

### The hypothesis and its fate

Phase 7 proposed a **t0/st0 leading-edge trade-off**: t0 and st0 jointly set
where the density turns on, so a flow that smooths the onset trades one against
the other. Phase 28 supported it by analogy — the RDM has no st0 and no t0 bias.

The prediction is direct: remove st0 from the ESTIMATION problem and the
trade-off has no partner, so the t0 bias should collapse.

**It does not.** At st0 fixed to its true value the t0 bias goes +0.319 ->
+0.269, a 16% reduction, with the rank distribution still decisively non-uniform
(KS p 8.5e-07). The bias is essentially intact.

**The posterior-trade-off account of the t0 bias is therefore wrong.** What
remains is the simpler explanation: the flow mis-fits the leading edge, that
error biases t0 directly, and st0 is a passenger rather than a cause. Phase 7's
observation that st0's bias is opposite in sign to t0's is then a CONSEQUENCE of
both reading the same mis-fitted edge, not evidence of a trade.

### Where the error goes instead

Fixing st0 does not remove the error, it **relocates** it. With st0 estimated,
Z is clean (−0.032, p 0.64); with st0 fixed at 0.05, Z degrades to −0.148
(p 5.0e-05). The estimated st0 was absorbing error that now lands on Z. This is
the same "error migrates to wherever it is not being watched" pattern as the
original blind spot, and it is the third time this project has seen it.

### Cell B is worse across the board, and is confounded

At st0 = 0.01 every parameter degrades: a +0.448 (p 6.8e-17, nearly at the 0.5
gate), Z −0.405, v +0.156, and coverage falls below nominal for t0 (0.914) and
a (0.932) against an MC se of 0.0097. **st0 = 0.01 is the exact lower bound of
the training box**, so this confounds "near-hard edge" with "box boundary" as
flagged when the cell was designed. It cannot distinguish them. A cell at
st0 = 0.02 would separate the two, and is worth running before any claim is made
about behaviour near zero.

### Consequences

1. **The paper's mechanism section must be rewritten.** The t0/st0 trade-off is
   not the explanation. The RDM's clean t0 therefore needs a different account:
   candidates are that its [64,32] flow fits its own leading edge better, that
   the race assembly f_R * prod S_j is more forgiving of edge error, or that t0
   is simply better identified there. None of these is yet tested.
2. **"Fix st0 and the artefact is safe" is FALSE** and must not be recommended.
   Fixing st0 leaves t0 biased and makes Z worse; fixing it near zero is worse
   still.
3. **The st0 = 0 retrain is not undermined — if anything it matters more.**
   Cells A and B evaluate the EXISTING 8-parameter flow with st0 pinned; they do
   not test a flow trained without st0. A dedicated 7-parameter flow spends all
   its capacity on the hard-edge case instead of covering st0 in [0.01, 0.49],
   and Cell B shows the current artefact is poor near its st0 floor. The retrain
   is now the experiment that decides whether the leading edge is learnable at
   all, rather than a convenience build.
4. **The multiplicative t0 correction (Phase 27) is unaffected** and is now the
   only demonstrated remedy for the shipped artefact.

## Phase 30 — RETRACTED: distillation does not move the mean either

Phase 24 reported distillation as "the first intervention to move the mean",
resting on three single runs (beta = 1, 10, pure) that ordered monotonically on
both g_a and g_t0. The sweep and its replication kill that claim.

### The sweep (beta = 20, 30, 50; 10 checkpoints each, seed 100)

| beta | g_a | g_t0 |
|---|---|---|
| 0 (MLE) | −0.0259 | −0.0696 |
| 1 | −0.0114 | −0.0624 |
| 10 | +0.0137 | −0.0317 |
| **20** | +0.0342 | **−0.0836** |
| **30** | +0.0231 | **+0.0159** |
| 50 | +0.0146 | +0.0216 |
| pure | +0.0461 | +0.0265 |

The monotonicity already fails at beta = 20, which is worse than beta = 1,
beta = 10 AND plain MLE, despite a checkpoint sd of only 0.0057.

### The replication (beta = 30, three fresh seeds)

| seed | g_a | g_t0 |
|---|---|---|
| 100 (the sweep run) | +0.0231 | **+0.0159** |
| 301 | −0.0645 | −0.0879 |
| 302 | +0.0204 | −0.0823 |
| 303 | −0.0645 | −0.1417 |

Three fresh seeds give g_t0 mean **−0.1040, sd 0.0328**. The sweep's own run sits
**+3.7 sd** away from that mean. It was a lucky draw, exactly as `b1k_amp0` was
in Phase 22.

### Against plain MLE, replicated on both sides

| | beta = 30 (4 runs) | plain MLE (5 seeds) | diff | t | p |
|---|---|---|---|---|---|
| g_t0 | −0.0740 (sd 0.0656) | −0.0696 (sd 0.0358) | −0.0044 | −0.12 | 0.91 |
| g_a | −0.0214 (sd 0.0498) | −0.0259 (sd 0.0225) | +0.0046 | +0.17 | 0.87 |

**Distillation moves neither mean.** Phase 24 is retracted in full.

### Why Phase 24 looked convincing

Its defence was "three runs landing in the correct order on two correlated-but-
distinct quantities is not a plausible accident". The arithmetic says otherwise:
the spread across beta = 1/10/pure was 0.0889, while the between-seed sd at
FIXED beta is 0.0328. Three draws from a distribution that wide land in
monotone order by chance 1 time in 6, and there were two quantities to look at.
The pattern was well inside what noise produces.

### The lesson, now paid for three times

Batch size (Phases 16 -> 20 -> 22 -> 25) and distillation (24 -> 30) both
produced a confident single-run claim that replication destroyed. The invariant:
**an intervention's effect must be compared against the between-SEED spread at
fixed settings, measured, before it is believed.** Every arm in this project
that was believed on one run per condition has since been retracted; no arm
replicated across >= 3 seeds has been.

For the paper this belongs in the appendix as a methodological finding, not a
confession: with flow training, run-to-run variance at fixed hyperparameters is
comparable to the effect sizes anyone will want to report, so single-run
comparisons of training interventions are uninformative in this domain.

### What survives

- Selection and large batch control the SPREAD. Nothing tried controls the
  CENTRE.
- The systematic residual stands: g_a −0.0343 (t = −3.53, p = 0.006, 10 seeds),
  g_t0 batch-dependent and pooled −0.039 (p = 0.057).
- **Phase 27's multiplicative t0 correction is the only demonstrated remedy**,
  and Phase 29 removed the st0-fixing alternative.
- The direct-regression baseline (g_a +0.0080, g_t0 +0.0063) remains the best
  calibrated route measured — and is now the ONLY route that beats plain MLE on
  the centre. It has not been seed-replicated, and on this phase's evidence
  that replication is mandatory before any claim is made for it.

## Phase 31 — the direct-regression baseline REPLICATES, and it has no residual

Four seeds (reg_b1k plus 401/402/403), identical settings, measured with the
same estimators at H = 0.01 over 200 SBC prior draws. Analytic controls: a
+0.00007 for the six-parameter estimator; the t0 estimator agrees with its
closed form to 0.3%.

| | g_a | g_t0 |
|---|---|---|
| reg_b1k | +0.0076 | +0.0063 |
| reg_s401 | −0.0182 | −0.0215 |
| reg_s402 | +0.0058 | +0.0014 |
| reg_s403 | +0.0063 | −0.0267 |
| **mean (sd)** | **+0.0004 (0.0124)** | **−0.0101 (0.0164)** |

**Neither mean differs from zero** (p = 0.96 and p = 0.31). This is the first
claim in the project to survive seed replication.

### Against the plain-MLE flow (5 seeds, same estimator)

| | regression | flow | diff | t | p |
|---|---|---|---|---|---|
| g_a | +0.0004 | −0.0259 | +0.0263 | 2.22 | 0.065 |
| g_t0 | −0.0101 | −0.0696 | +0.0595 | **3.30** | **0.017** |

The regression net is **significantly better on t0**, the parameter that fails
SBC, and its between-seed sd is ~2x smaller on both parameters. Against
distillation at beta = 30 it is better on t0 by +0.064 (p = 0.145, n small).

### Normalisation, the thing a flow gets for free

Total mass over rt and R, which a flow makes exactly 1 by construction:

| seed | mean | sd across theta |
|---|---|---|
| reg_b1k | 1.00003 | 0.00190 |
| reg_s401 | 1.00710 | 0.00167 |
| reg_s402 | 1.00006 | 0.00217 |
| reg_s403 | 1.00084 | 0.00185 |

A CONSTANT offset is harmless — it shifts every log-likelihood equally. What
matters is the variation with theta, sd(log Z) ~ 0.002, which is an order of
magnitude below the score biases being compared. s401's 0.7% mean offset is
therefore not a defect; its theta-variation is the lowest of the four.

### What this means, stated carefully

**The flow's calibration residual is not intrinsic to the problem.** A plain
MLP regressing on the analytic log-density, with no classifier, no
normalisation guarantee and no spline machinery, has no detectable systematic
bias on either parameter — where the flow has g_a = −0.0343 at p = 0.006 over
ten seeds. Whatever causes the flow's residual is a property of the flow, not
of the DDM's density.

**But the comparison is not free.** Direct regression:
- requires the analytic likelihood to train, so it can only COMPRESS something
  you already have. It is useless for a genuinely intractable model, which is
  the case NLE exists for;
- gives no CDF, so race models (f_R * prod S_j) and censoring are out of reach
  without a second network with nothing tying dF/drt to f;
- is not normalised, though the measured theta-variation is small enough not to
  matter here.

So the honest framing for the paper is a **division of labour**, not a winner:
where an analytic exists but is too slow, regression is the better artefact —
simpler, better calibrated, one forward pass. Where no analytic exists, or a
CDF is needed, the flow is the only option and its residual is the price.

That is a more useful result than "our flow works", and it directly answers the
reviewer question this baseline was built to answer.

### Consequence for the st0 = 0 build

The st0 = 0 artefact is a compression of an analytic that exists, which is
exactly the regime where regression wins. The build now running trains flows;
a regression arm on the same data costs ~2 h against the flows' ~10 h and
should be run alongside, with the speed of both measured. Deciding between them
on calibration AND speed is cheap, and the flow is not the obvious default.

## Phase 32 — st0 = 0 FAILS catastrophically, and finally explains t0

> **RETRACTED (Phase 37, 2026-09-23).** This cell simulated its data at
> **st0 = 0.05**, not 0: `DDMnn`'s `p_types` default for st0 is −2.996, where the
> analytic `DDM`'s is −Inf. The flow was trained at st0 = 0, so the verdict below
> measures a model mismatch, not the flow. Phase 34's analytic control used the
> `DDM` model and therefore simulated at st0 = 0, which is why it passed and
> appeared to exonerate the pipeline. Corrected, the SAME flow PASSES. See
> Phase 37 before reading any number in this section.


tux26, 24 cores, N=400, 500 replicates, 173.8 min. Flow `z0_s503_e282`, selected
from 90 checkpoints across 3 seeds on total |g| = 0.064.

| par | coverage | mean bias | std bias | |
|---|---|---|---|---|
| v | 0.944 | +0.0087 | +0.094 | |
| **a** | **0.156** | +0.0600 | **+2.233** | FAILS |
| **t0** | **0.000** | +0.0414 | **+3.144** | FAILS |
| **Z** | 0.868 | −0.0188 | **−0.621** | FAILS |
| **SZ** | 0.906 | −0.0794 | **−0.546** | FAILS |
| sv | 0.930 | +0.0134 | +0.085 | |

**t0 coverage is 0.000** — the truth was outside the credible interval in all 500
replicates. That is not a calibration problem, it is a broken likelihood.

### It is not the data, and not the bulk of the density

The training target is EXACT: stored `ll` against WienR at st0 = 0, 500 sampled
rows, mean / median / max error all **0.000000**.

The flow is excellent where the mass is. Error against the analytic, by distance
past t0, for one SBC draw:

| rt − t0 (s) | mean err | max \|err\| | true mass |
|---|---|---|---|
| 0.001–0.003 | **+20.300** | 41.8 | **0.00000** |
| 0.003–0.010 | +1.602 | 6.58 | 0.00043 |
| 0.010–0.030 | +0.077 | 0.61 | 0.05345 |
| 0.030–0.100 | −0.014 | 0.047 | 0.39802 |
| 0.100–0.300 | +0.017 | 0.043 | 0.47097 |
| 0.300–1.000 | −0.115 | 0.378 | 0.11707 |
| 1.000–3.000 | +2.905 | 9.04 | 0.00043 |

**87% of the mass is fitted to ±0.02 log units.** The error is confined to the
first few milliseconds past t0, where the true mass rounds to zero.

### The mechanism, at last

With st0 = 0 the support boundary is hard: the analytic density is **exactly
zero** below t0, so log f = −Inf. A spline flow over log rt has positive density
everywhere and cannot produce −Inf. Measured on the selected flow, at the true
t0 = 0.2876 s:

| rt | flow log f | analytic |
|---|---|---|
| 0.2013 (70% of t0) | −10.70 | −Inf |
| 0.2848 (99% of t0) | −9.41 | −Inf |
| 0.2934 (102%) | −4.99 | finite |

Total flow mass below the true t0 is **0.0000** — so in MASS terms the flow is
right to within rounding.

**But inference is driven by log-density, not by mass.** When the sampler
proposes a t0 above some observed rt, the analytic penalty is infinite and t0 is
pinned below min(rt). The flow's penalty is about 10 log units per offending
trial — finite, and easily outweighed by a better fit to the other 399. So t0 is
pushed up, `a` compensates, and Z/SZ follow. Exactly the observed sign and
ordering.

### What this settles

1. **The t0 bias is a hard-boundary problem, not a trade-off.** Phase 29
   refuted the t0/st0 posterior trade-off; this supplies the replacement. In the
   8-parameter model st0 > 0 makes the true density genuinely smooth at the
   onset, so the flow's smoothness is not a mis-specification and the bias stays
   survivable (+0.319). Remove the smear and the same flow becomes unusable.

2. **"Train it without st0" is not a fix — it is the worst case.** The st0 = 0
   artefact must not ship. Phase 29 already removed fixing st0 at evaluation
   time; this removes retraining without it.

3. **A density can be correct to 1e-11 in mass and still destroy a boundary
   parameter.** Held-out likelihood, MSE and total-mass checks all pass this
   flow. Only SBC catches it. This is the thesis of the paper in its most
   extreme form, and it is worth leading with.

4. **It predicts where else this bites**: any parameter defining a support
   boundary, in any model, approximated by a density with unbounded support.
   The RDM's clean t0 is now the anomaly needing explanation, not the DDM's
   failure — its edge is gentler and its race assembly carries survivor
   information, but neither has been tested.

### Consequence

A flow intended to support inference about a boundary parameter needs the
boundary built in, not learned: a hard floor at t0 in the evaluator, an
explicitly truncated base distribution, or a parameterisation in (rt − t0).
None of these is tested, and the last is the natural next experiment.

## Phase 32b — CORRECTION to Phase 32's mechanism

Phase 32 asserted that the st0 = 0 failure works through the support boundary:
that a proposed t0 above an observed rt costs infinity under the analytic but
only ~10 log units under the flow, so t0 drifts up. **That specific mechanism is
not supported, and the claim was made without testing it.**

### The test it fails

`sbc/test_t0_truncation.R` simulates a dataset at a known truth, then profiles
the log-likelihood in t0 with everything else held at the truth, for the
analytic, the flow, and the flow with the true support enforced
(`ll = -Inf` when `t0 > rt`). Peak displacement from the truth, ms, 8 draws:

| flow | analytic | flow | flow + truncation |
|---|---|---|---|
| `cap256w_c4` (st0 > 0) | +0.87 | +1.18 | **+1.18** |
| `z0_s503_e282` (st0 = 0) | +0.69 | +0.86 | **+0.86** |

Truncation changes nothing for either flow, because **the profile maximum sits
comfortably below min(rt) and the constraint never binds**. And the st0 = 0
flow's marginal t0 displacement is +0.86 ms — against an SBC bias of +0.0414 in
log units, which at t0 = 0.3 s is **+12.6 ms**. The marginal profile is off by a
fifteenth of the SBC effect.

Note also that the analytic itself carries +0.69 to +0.87 ms of finite-sample
MLE bias here, so the flow's *excess* over the analytic is only ~0.2-0.3 ms.

### Not a wiring bug either

The EMC2 bundle path (`ddm_ens_eval_trials_cpp` via `.theta_DDMnn7`) and the R
port agree to **1.1e-14** on matched theta and rt. The SBC result is real.

### What is actually established

1. The st0 = 0 flow fails SBC catastrophically: t0 +3.144 with coverage 0.000,
   a +2.233, Z -0.621, SZ -0.546, while v and sv are clean.
2. The training target is exact (0.000000 against WienR over 500 rows).
3. The flow fits 87% of the probability mass to +-0.02 log units.
4. The flow assigns finite log-density where the analytic is -Inf (log f ~ -9.5
   at 99% of t0), with total mass below t0 of 9e-6 for the 8-parameter flow.
5. **The failure is a JOINT displacement of (t0, a, Z, SZ), not a marginal t0
   problem.** Holding the others at the truth removes almost all of it. The
   four parameters that move are exactly those the Fisher information couples
   (t0-SZ +0.86, a-Z -0.44); v and sv, which are uncoupled, are unaffected.

The mechanism is therefore in the joint geometry, and remains **unidentified**.
A profile in one parameter cannot see it; the next diagnostic is a full MLE over
all parameters, flow versus analytic, on matched simulated data.

### The lesson, which is the same one as Phases 22 and 30

A plausible mechanism was written up as established on the strength of a
consistent story rather than a test, one commit after the test would have taken
ten minutes. The prior two retractions were about believing single runs; this
one is about believing an explanation. **Register the prediction a mechanism
makes, then test it, before writing it up as the finding.**

### Status of the two proposed fixes

- **Enforcing the support** (`ll = -Inf` when `t0 > rt`) is correct -- it is a
  true property of the model, costs nothing, and removes a class of failure that
  can occur. But it is measured here NOT to bind at the likelihood maximum for
  either flow, so it is insurance rather than a remedy. Worth adding to the
  evaluator regardless.
- **Parameterising in (rt - t0)** is untouched by this correction and is now the
  more promising route. Modelling the decision time d = rt - t0 puts the
  boundary at a FIXED location, and under the flow's log transform the boundary
  maps to log d -> -Inf, which a spline over the real line represents naturally
  rather than having to learn. t0 then enters only through an exact shift with
  unit Jacobian. Untested.

## Phase 33 — the distortion is GLOBAL, and score bias cannot see it

> **RETRACTED (Phase 37, 2026-09-23).** This cell simulated its data at
> **st0 = 0.05**, not 0: `DDMnn`'s `p_types` default for st0 is −2.996, where the
> analytic `DDM`'s is −Inf. The flow was trained at st0 = 0, so the verdict below
> measures a model mismatch, not the flow. Phase 34's analytic control used the
> `DDM` model and therefore simulated at st0 = 0, which is why it passed and
> appeared to exonerate the pipeline. Corrected, the SAME flow PASSES. See
> Phase 37 before reading any number in this section.


Phase 32b left the st0 = 0 failure unexplained: SBC gives t0 +3.144 with
coverage 0.000, but neither a t0 profile nor a joint MLE started at the truth
reproduces it. Two further diagnostics, both on `z0_s503_e282`.

### Joint MLE over all free parameters, started at the truth (40 draws, N=400)

| par | analytic | flow | flow − analytic | t |
|---|---|---|---|---|
| v | +0.01021 | −0.00735 | −0.01755 | −1.43 |
| a | +0.00060 | +0.00531 | **+0.00471** | **2.99** |
| t0 | +0.00570 | +0.00694 | +0.00124 | 0.51 |
| Z | +0.00086 | +0.01050 | **+0.00964** | **2.77** |
| SZ | +0.07621 | +0.16903 | +0.09282 | 1.26 |
| sv | −0.00184 | −0.03129 | −0.02945 | −0.41 |

t0 excess is **+0.37 ms** against an SBC bias of +12.6 ms. `a` and `Z` are
significantly displaced but small. The likelihood surface AT the truth is
essentially fine.

### Multi-start: is there a better optimum elsewhere? (12 draws x 13 starts)

Starts dispersed at 2x the prior sd:

| | t0 displacement (ms) | |
|---|---|---|
| started at the truth | +2.83 | |
| best of 13 starts | **+4.42** | nll better by **0.855** on average |

**There are better-scoring optima away from the truth, and they sit at higher
t0.** A truth-started optimiser never visits them; MCMC does. This closes part
of the gap between the MLE tests and SBC, and points at the real character of
the fault.

### What this establishes

**The flow's defect is not local to the truth — it is in how its error varies
ACROSS parameter space.** Near any given theta the density is excellent (87% of
mass to +-0.02 log units, Phase 32). But log f̂(rt | theta) as a function of
theta for FIXED data is distorted, creating spurious likelihood gains at
displaced parameter values.

**This is a limit of the score-bias diagnostic, and it belongs in the paper.**
g = d/dtheta E_true[log f̂] is evaluated AT the truth: it is a local derivative,
and a small g at every theta does not constrain the shape of the likelihood
surface between thetas. The posterior is a global object. So:

- score bias is **necessary but not sufficient**;
- it ranks checkpoints within an architecture (rho = -0.989) because they share
  a surface shape, and fails across architectures because they do not;
- **only SBC probes the surface the posterior actually explores.**

That also explains the pattern in Phase 9b — g_a is a smooth function of
(v, sv, st0, t0) with joint R^2 = 0.946 and essentially no dependence on `a`
itself. The error was always a property of WHERE you are in the space.

### Still unaccounted for

Multi-start recovers +4.42 ms of the SBC's +12.6 ms. The remainder is not yet
explained. Candidates: the posterior mean/median differs from any mode because
the flow's surface is skewed; or the sampler's exploration reaches regions
neither optimiser start covered. A direct check -- running the EMC2 sampler on
one simulated dataset and comparing its t0 posterior against the analytic's --
would settle it and is the obvious next step.

## Phase 34 — the analytic control PASSES: the st0=0 failure is the flow's

> **RETRACTED (Phase 37).** The control was not matched to the cell it was
> controlling: it used the analytic `DDM` model, whose st0 default is 0, while
> the flow cell's `DDMnn` default put st0 at 0.05. Both cells were run with st0
> omitted from the formula, so they were fitting DIFFERENT data-generating
> models. The conclusion "the failure is entirely the flow's" does not follow.


The st0 = 0 flow cell (Phase 32) was run without an analytic control, breaking
the rule Phase 13 exists to enforce. Running it now, same design, same priors,
same sampler, N=400, 500 replicates, 313.1 min on 24 cores:

| par | ANALYTIC std bias | coverage | FLOW std bias | coverage |
|---|---|---|---|---|
| v | +0.039 | 0.948 | +0.094 | 0.944 |
| a | **+0.048** | 0.950 | **+2.233** | 0.156 |
| **t0** | **+0.017** | **0.946** | **+3.144** | **0.000** |
| Z | +0.018 | 0.970 | −0.621 | 0.868 |
| SZ | −0.016 | 0.962 | −0.546 | 0.906 |
| sv | +0.092 | 0.952 | +0.085 | 0.930 |

**The analytic recovers every parameter essentially perfectly** — worst
standardised bias 0.092, coverage 0.946–0.970 against nominal 0.95.

So st0 = 0 inference is NOT intrinsically hard, the hard support boundary does
NOT defeat the sampler, and the tight t0 posterior is not the problem.
**The failure in Phase 32 is entirely the flow's.**

### A single dataset had suggested otherwise, and was wrong

`sbc/test_sampler_posterior.R` fitted one simulated dataset with both
likelihoods and found the ANALYTIC missing t0 by 3.32 posterior sd, with the
flow only 1.09 ms worse. That reading — that the failure might be intrinsic —
does not survive 500 replicates: it was one unlucky draw from a distribution
whose true mean bias is +0.0001. **n = 1 diagnostics set hypotheses, they do not
settle them**, which is the same lesson as Phases 22, 30 and 32b in a fourth
costume.

### Consequences

1. **Phase 32's headline stands**: the st0 = 0 flow is unusable and must not
   ship. Phase 32b's correction to its *mechanism* also stands — the fault is
   not the support boundary.
2. **Phase 33's account is the live one**: the flow's error varies across
   parameter space, distorting the likelihood SURFACE away from the truth, where
   a local diagnostic like score bias cannot see it.
3. **The (rt − t0) reparameterisation is now clearly warranted.** With the
   analytic exonerated, the target is squarely the flow's representation, and
   the strongest structural fix is to remove t0 from the conditioning set
   entirely: train on the decision time d = rt − t0, so the network never sees
   t0 and cannot distort the surface in the t0 direction. t0 then enters only
   through an exact shift with unit Jacobian.
4. It also sharpens the paper: an exact likelihood and an approximation that is
   correct to ±0.02 log units over 87% of the mass give *completely* different
   SBC verdicts on the same data. That is the thesis, with the cleanest control
   in the project behind it.

## Phase 35 — decision-time (rt − t0) ALSO fails, and shows t0 is a victim

> **RETRACTED (Phase 37, 2026-09-23).** This cell simulated its data at
> **st0 = 0.05**, not 0: `DDMnn`'s `p_types` default for st0 is −2.996, where the
> analytic `DDM`'s is −Inf. The flow was trained at st0 = 0, so the verdict below
> measures a model mismatch, not the flow. Phase 34's analytic control used the
> `DDM` model and therefore simulated at st0 = 0, which is why it passed and
> appeared to exonerate the pipeline. Corrected, the SAME flow PASSES. See
> Phase 37 before reading any number in this section.


tux26, 24 cores, N=400, 500 replicates, 287.2 min. Flow `dt_s703_e167`, selected
from 90 checkpoints across 3 seeds on total |g| = 0.0240 — the best screening
number of any DDM build in the project, with g_t0 = −0.0034 against the shipped
flow's −0.0620.

The design, priors, N and replicate count match Phase 34's analytic cell exactly,
so that cell is the control and no second run was needed.

| model | v | a | t0 | Z | SZ | sv |
|---|---|---|---|---|---|---|
| **analytic (control)** | +0.039 | +0.048 | **+0.017** | +0.018 | −0.016 | +0.092 |
| st0 = 0 flow | +0.094 | +2.233 | +3.144 | −0.621 | −0.546 | +0.085 |
| **decision-time flow** | +0.318 | **+1.665** | **+2.490** | **−0.877** | **−0.850** | −0.199 |

Coverage for t0: analytic 0.946, st0=0 0.000, decision-time **0.002**.

### The hypothesis, and what actually happened

Dropping t0 from the conditioning set was meant to be structural: the network
never sees t0, so it *cannot* distort the likelihood surface in the t0
direction, and t0 re-enters only as an exact shift with unit Jacobian
(golden-validated at 1.0e-11; the wrapper agrees with the R port to 9.8e-15).

**t0 is still displaced by +2.490 with coverage 0.002.** Only modestly better
than the st0=0 build, and `Z`, `SZ` and `v` are all WORSE.

### What that establishes, and it is the useful part

**t0's displacement cannot be caused by the flow's representation of t0,
because in this build the flow has no representation of t0 at all.** It must
arrive through the other parameters: the flow's error in the decision-time
density displaces (a, Z, SZ), and t0 follows through the Fisher coupling
(t0-SZ +0.86, a-Z −0.44).

So **t0 is the victim, not the cause** — the parameter where a distortion
originating elsewhere surfaces most visibly, because its posterior is 3-20x
tighter than the others and it is coupled to all of them.

That reframes the whole t0 investigation, including for the SHIPPED
8-parameter flow: its +0.319 is very likely secondary too. Chasing t0
representations -- the st0/t0 trade-off (Phase 29), the support boundary
(Phase 32b), the conditioning set (here) -- has now failed three times, and
this phase explains why all three were looking in the wrong place.

### Why the screening number was so misleading

`dt_s703_e167` had the best total |g| of any DDM checkpoint measured (0.0240,
with g_t0 = −0.0034) and failed SBC catastrophically. This is Phase 33's point
in its sharpest form: **score bias is a derivative at the truth and says nothing
about the likelihood surface the posterior explores.** Screening on it selects
confidently among models that may all be unusable.

### Where this leaves the DDM

- The shipped `cap256w_c4` (st0 estimated) remains the only DDM artefact that
  passes SBC, with the Phase 27 multiplicative correction for t0.
- Three reduced-parameterisation builds -- st0 fixed, st0 = 0, decision-time --
  have now all failed. **None should ship.**
- The remaining hypothesis is that the fault is accuracy in the *shape* of the
  density as a function of (a, Z, SZ), not anything about t0. Testing it means
  measuring surface distortion directly rather than at the truth: e.g. the
  flow-vs-analytic log-likelihood difference profiled over the coupled
  parameters on FIXED data, which is cheap and has never been done.

## Phase 36 — surface distortion measured directly: 1D profiles do not explain it either

> **RETRACTED (Phase 37, 2026-09-23).** This cell simulated its data at
> **st0 = 0.05**, not 0: `DDMnn`'s `p_types` default for st0 is −2.996, where the
> analytic `DDM`'s is −Inf. The flow was trained at st0 = 0, so the verdict below
> measures a model mismatch, not the flow. Phase 34's analytic control used the
> `DDM` model and therefore simulated at st0 = 0, which is why it passed and
> appeared to exonerate the pipeline. Corrected, the SAME flow PASSES. See
> Phase 37 before reading any number in this section.


Phase 35 left the fault located "in the other parameters". This measures the
likelihood SURFACE, which is what Phase 33 says score bias cannot see. For a
FIXED simulated dataset,

    Delta(theta) = log L_flow(data | theta) - log L_analytic(data | theta)

A perfect approximation makes Delta constant. Any VARIATION tilts the surface
and displaces the mode. Profiled over +-3 prior sd per parameter, others held at
the truth, 12 datasets x 400 trials (`sbc/test_surface_distortion.R`).

| par | shipped `cap256w_c4` — **PASSES** | decision-time `dt_s703_e167` — **FAILS** |
|---|---|---|
| | peak shift (sd) / tilt | peak shift (sd) / tilt |
| v | +0.125 / 0.86 | +0.012 / 0.43 |
| a | +0.000 / 20.89 | +0.025 / 18.40 |
| t0 | +0.025 / 25.04 | +0.012 / 29.85 |
| Z | −0.100 / 1.46 | −0.038 / 1.94 |
| **SZ** | **−0.687** / 0.96 | **−0.762** / 1.07 |
| **sv** | **−1.800** / 0.51 | **−1.350** / 0.78 |
| st0 | +0.012 / 1.84 | — |

### Two findings, one of them inconvenient

**1. The distortion is concentrated in the variability parameters.** Both flows
shift the profile peak of `sv` by 1.4-1.8 prior sd and `SZ` by ~0.7, while
`v`, `a`, `t0` and `Z` move by under 0.13 sd. The flows fit the decision-time
shape well and the variability structure poorly. That is consistent with
Phase 35: t0's displacement arrives through its neighbours, and `SZ` is the
parameter the Fisher information couples to t0 at +0.86.

**2. It does not discriminate pass from fail.** The SHIPPED flow, which passes
SBC, has a LARGER `sv` peak shift (−1.800) than the decision-time flow that
fails (−1.350), and comparable shifts everywhere else. So a 1D profile with the
other parameters held at the truth does not predict the SBC verdict.

The likely reason the large sv/SZ shifts are survivable at all is that both are
prior-dominated (per-trial sd 590 for sv), so the prior pins them however tilted
the likelihood is. The shifts are real and inconsequential.

### The honest state

Three diagnostics have now failed to predict the SBC verdict: score bias at the
truth (Phase 33), the joint MLE from the truth including multi-start (Phase 33),
and 1D surface profiles from the truth (here). **Every one of them conditions on
the truth in the other parameters.** The distortion that matters is genuinely
multivariate — it lives in directions a one-at-a-time profile cannot reach.

The next diagnostic that could work has to move several parameters at once:
profile Delta along the leading eigenvectors of the Fisher information, or along
the line joining the truth to the flow's own posterior mean from a fitted SBC
replicate. Both are cheap; neither has been done.

### For the paper

This strengthens rather than weakens the thesis. It is one thing to say
held-out likelihood does not predict calibration; it is another to show that
score bias, joint MLE, multi-start optimisation and surface profiling all fail
to predict it as well, while SBC separates a usable likelihood from an unusable
one on the same data. **The practical recommendation is unchanged and now much
better evidenced: run SBC, with its analytic control. Nothing cheaper
substitutes.**

## Phase 37 — the st0 = 0 and decision-time failures were a DESIGN DEFAULT, not the flows

The largest retraction in the project, and it was found by a diagnostic that
disagreed with SBC rather than by rechecking SBC.

### How it was found

A new diagnostic (`sbc/test_posterior_shift.R`) computes the flow's posterior
directly on fixed data — importance-reweighting draws from a defensive mixture
of the analytic and flow Laplace approximations, so the prior is included and
every parameter moves together, which is what Phases 33 and 36 said the failing
diagnostics could not do. Over 60 prior datasets it reports SBC's own statistic,
`mean(posterior mean − truth) / sd`, for both likelihoods on the same data.

| flow | predicted t0 | observed SBC t0 |
|---|---|---|
| `cap256w_c4` (passes SBC) | **+0.28** | **+0.319** |
| `z0_s503_e282` (st0 = 0) | +0.12 to +0.20 | +3.144 |
| `dt_s703_e167` (decision time) | +0.27 to +0.55 | +2.490 |

It reproduces the shipped flow's SBC bias and contradicts the other two by a
factor of 15. Unchanged by a 12-start search for distant modes, and unchanged by
simulating with `WienR::rWDM` instead of grid inversion. The analytic control of
the diagnostic reads ~0 (within ±0.13 at 60 datasets).

Ruled out, in order: the laptop wrapper (EMC2's `DDMnn7` path and the R port
agree to 1e-9 ACROSS theta, not merely at the truth); the bundle tux26 ran (its
logs record `z0_s503` step 282, md5 identical to the laptop's); tux26's own
compiled evaluator (identical log-densities to the laptop port, ~1e-12, on fixed
inputs); and the sampler (a local 3-replicate cell reproduced the failure exactly
— t0 +5.44, coverage 0.000 — in 3.2 minutes).

### The cause

**`DDMnn`'s `p_types` default for `st0` is −2.996 on the sampled scale (= 0.05 s).
The analytic `DDM`'s is −Inf (= 0).** Omitting `st0` from a design formula
therefore gives st0 = 0.05 under a flow design and st0 = 0 under the analytic
design. Both st0 = 0 drivers omitted it, so:

- the FLOW cells simulated data with a 50 ms uniform non-decision smear and
  evaluated it with flows trained at st0 = 0;
- the ANALYTIC control cell simulated at st0 = 0 and evaluated at st0 = 0, so it
  passed — **the control was not matched to the cell it was controlling.**

A second entry compounds it: `DDMnn`'s `bound$minmax` confines st0 to
[0.01, 0.49] with no `exception`, where `DDM` carries `c(sv = 0, SZ = 0, st0 = 0)`.
Simply naming `st0 = -Inf` in `constants` therefore makes every replicate fail
out of bounds until the exception is added too.

Four independent confirmations:

| check | flow design (`DDMnn7`) | analytic design (`DDM`) |
|---|---|---|
| `mapped_pars(...)[1, "st0"]` | **0.05** | 0 |
| simulated mean rt (20k trials) | 0.4755 | 0.4514 |
| `WienR::rWDM` reference | 0.4744 at st0 = 0.05 | 0.4519 at st0 = 0 |
| KS on 100k simulated trials | max abs ECDF−CDF **0.0020** at st0 = 0.05 (0.1453 at st0 = 0) | — |

And the tell that generalises: on EMC2-simulated data from these cells, the
**analytic** MLE is displaced +13.4 ms in t0, where matched data give +1.0 ms.
The approximation and the exact likelihood were displaced TOGETHER, which no
defect of the approximation can produce.

### The corrected result

Same flow, same design, same priors, N = 400, laptop, 12 replicates:

| | v | a | t0 | Z | SZ | sv | t0 coverage |
|---|---|---|---|---|---|---|---|
| st0 omitted (as Phase 32 ran it) | −0.16 | **+2.13** | **+5.44** | −0.45 | −0.57 | +1.19 | **0.000** |
| st0 named a constant | −0.161 | **+0.038** | **+0.346** | −0.032 | +0.068 | −0.003 | **1.000** |

**The st0 = 0 flow passes.** Confirmed at 500 replicates (tux26, 166.5 min):

| par | std bias | coverage | mean rank | KS p |
|---|---|---|---|---|
| v | −0.089 | 0.946 | 0.526 | 0.036 |
| a | +0.228 | 0.944 | 0.435 | 6.8e-07 |
| t0 | +0.276 | 0.936 | 0.430 | 1.4e-07 |
| Z | +0.282 | 0.956 | 0.426 | 1.2e-06 |
| SZ | +0.065 | 0.960 | 0.481 | 0.129 |
| sv | +0.003 | 0.960 | 0.502 | 0.72 |

Every parameter is inside the gate, and coverage is nominal throughout
(0.936-0.960 against 0.95, MC se 0.0097). Rank uniformity fails for `a`, `t0`
and `Z` (mean ranks ~0.43, i.e. mild systematic OVERestimation), which is the
same signature the shipped 8-parameter flow shows on t0 alone (+0.319,
KS 1.8e-07, coverage nominal). So the honest verdict is: **intervals valid,
point estimates carrying a small upward offset on three parameters** -- a
usable artefact of the same character as the shipped one, and nothing like the
retracted "coverage 0.000".

### The decision-time cell, also at 500 replicates

tux26, 168.2 min, same correction (`st0` named a constant):

| par | std bias | coverage | mean rank | KS p |
|---|---|---|---|---|
| v | +0.050 | 0.966 | 0.491 | 0.57 |
| a | −0.073 | 0.928 | 0.520 | 0.12 |
| **t0** | **+0.392** | **0.816** | 0.423 | 8.5e-13 |
| Z | −0.108 | 0.952 | 0.537 | 0.0029 |
| SZ | −0.005 | 0.952 | 0.505 | 0.88 |
| sv | +0.050 | 0.950 | 0.487 | 0.47 |

Against the retracted +2.490 / coverage 0.002, t0 is now +0.392 -- inside the
gate, but with **coverage 0.816 against nominal 0.95** (MC se 0.0097), which is
lower than the displacement alone predicts (0.932), so the t0 posterior is too
narrow as well as displaced. The build is therefore calibrated at the gate level
and still the weakest of the three on t0; it does not supersede the shipped flow.

**A 12-replicate preview of this cell showed v at −0.934 (t = −3.24, p = 0.008)
and it did not survive**: v is +0.050 at 500 replicates, with KS p 0.57 and
coverage 0.966. Twelve replicates estimate the standardised bias too poorly to
act on, even when the accompanying t-test looks decisive -- the same lesson as
Phases 22/30 in yet another costume, and it was caught this time only because
the confirmation was already queued.

### The cheap diagnostic, scored against all three corrected cells

| flow | posterior-shift prediction for t0 | SBC t0 (500 reps) |
|---|---|---|
| `cap256w_c4` | +0.28 | +0.319 |
| `z0_s503_e282` | +0.12 to +0.20 | +0.276 |
| `dt_s703_e167` | +0.27 to +0.55 | +0.392 |

Three flows, three cells, agreement within ~0.1 on every one -- and on `v` for
the decision-time flow (+0.02 to +0.11 predicted, +0.050 observed) where a
12-replicate SBC said −0.934. At one minute per 7-parameter flow against ~168
minutes per cell, this is the screening instrument the project lacked.

### What this retracts

- **Phase 32** (st0 = 0 "fails catastrophically", "a density can be correct to
  1e-11 in mass and still destroy a boundary parameter") — the failure was the
  data, not the density.
- **Phase 32b's** residue: the mechanism it corrected was never operative.
- **Phase 33** ("the distortion is GLOBAL, and score bias cannot see it") — the
  joint MLE and multi-start tests were run against the same mismatched cell.
- **Phase 34** ("the failure is entirely the flow's") — unmatched control.
- **Phase 35** (decision-time build "also fails", "t0 is a victim, not a cause")
  — the `DDMnn6` driver omits st0 the same way. Re-run in progress.
- **Phase 36**, in part: its comparison of a "passing" against a "failing" flow
  compared two flows that both pass.

What SURVIVES: Phase 29 (fixing st0 at evaluation time does not fix t0 — those
cells name st0 explicitly), every 8-parameter DDM cell and every RDM cell (their
formulas name all parameters, so no default is used), and the score-bias results,
which never went through EMC2 designs.

### The methodological lesson, which is the sharpest one yet

Previous lessons were about believing single runs (Phases 22, 30) and believing
untested mechanisms (32b). This one is about controls:

**A control is only a control if it differs from the treatment in exactly one
respect.** The analytic cell differed in two: the likelihood AND, invisibly, the
simulated model. It therefore certified a pipeline that was broken, for two
weeks, across four phases, while every cheap diagnostic that contradicted it was
recorded as a failure of the diagnostic.

The corollary is worth stating positively: **when a cheap diagnostic disagrees
with an expensive one, do not assume the cheap one is wrong.** The posterior-shift
estimator was right and cost one minute per flow against three to five hours.

# Appendix: Posterior precision and bias of the neural-network DDM likelihood

## A.1 Rationale

The neural likelihood estimator for the full diffusion decision model (DDM)
approximates the joint density of response and response time with two small
neural networks (Section X of the main text): a conditional rational-quadratic
spline flow for the response-time distribution and a multilayer-perceptron
classifier for the choice probability. Any such approximation perturbs the
likelihood surface. This appendix quantifies the inferential consequences for
a single participant, asking two questions. First, does the approximation
behave like a loss of information — such that its cost can be expressed as the
number of additional trials required to recover the precision of the exact
likelihood? Second, is there a systematic (bias-like) component that
additional trials cannot repair?

## A.2 Methods

### Likelihoods

The exact reference was EMC2's analytic DDM (`DDM`), whose density is computed
by the WienR package (Hartmann & Klauer, 2021) at precision 5e-3. A control
analysis (not shown; see repository) re-evaluated reference log-likelihoods at
precisions 1e-4 to 1e-8: the choice of precision changed summed
log-likelihoods by sd 0.28 across the parameter draws used below, two orders
of magnitude too small to affect any conclusion, and all headline statistics
were identical to three decimals when recomputed against the finest-precision
reference.

The approximate likelihood (`DDMnn`) used the flow and classifier trained at
the largest simulation budget of Wuth (2026) (1,000,000 training trials; flow:
one hidden layer of 32 and one of 16 units conditioning a 32-bin spline on the
8 model parameters and the response; classifier: hidden layers of 32 and 16
units; 3,330 weights in total), evaluated by a C++ port validated to ~1e-12
against the original JAX implementation. Parameter vectors outside the flow's
training region are assigned zero likelihood; all parameter values in this
study lay well inside that region.

### Design, priors, and data

The design was the single-participant DDM of the EMC2 simulation-based
calibration vignette: a two-choice task with all eight parameters
intercept-only, the diffusion coefficient fixed (s = 1), leaving seven sampled
parameters (v, a, t0, Z, sv, SZ, st0). Priors were independent normals on the
sampled scale (log for a, t0, sv, st0; probit for Z, SZ) with the vignette's
means and standard deviations, except that the SZ prior mean was moved from
qnorm(.05) to qnorm(.20) because the former lies exactly on the boundary of
the flow's training region:

| parameter | prior mean | prior sd |
|---|---|---|
| v | 1 | 0.15 |
| a | log(0.8) | 0.15 |
| t0 | log(0.3) | 0.10 |
| Z | qnorm(0.5) | 0.05 |
| sv | log(0.1) | 0.15 |
| SZ | qnorm(0.2) | 0.15 |
| st0 | log(0.05) | 0.15 |

The data-generating parameter vector was fixed at the prior mean. Four
replicate datasets were simulated (exact sampler, WienR::rWDM) at each of
N = 100, 200, 400, and 800 trials. Each dataset was fitted by both
likelihoods (a paired design: the nn-versus-analytic contrast is computed on
identical data). Because analytic fits were ~12 times slower (Table A.3), the
analytic reference was fitted at N = 100 and 200 only; the information model
below bridges to larger N.

### Fitting

Fits used EMC2's non-hierarchical sampler with the package's
simulation-based-calibration settings: three chains run to convergence
criteria of minimum effective sample size 100 and maximum Gelman–Rubin
statistic 1.1, no response-time compression (rt_resolution = 1e-12). Note
that these settings disable EMC2's default compression (which floors response
times to a 1/60 s grid); the biases reported below therefore cannot arise
from compression.

### Measures

For each fit we recorded, for every parameter on the sampled scale, the
posterior standard deviation, the 95% credible-interval width, and the bias
of the posterior median relative to the generating value. Posterior standard
deviations were aggregated across replicates as root-mean-squares.

The information-loss question was formalized by the approximation
posterior precision = prior precision + N × i, where i is the information per
trial. For each parameter, the analytic unit information i_DDM was pooled
over its two trial counts, the nn unit information i_nn(N) was computed at
each N, and their ratio κ(N) = i_nn(N) / i_DDM expresses the flow's
information extraction per trial relative to the exact likelihood; the trial
count at which the nn likelihood matches the precision of the exact
likelihood at a base of N₀ trials is approximately N₀ / κ.

All computations are reproducible from the repository
(sbc/precision_component.R, sbc/run_precision_study_ddm.R); every fit is
cached, so extending the study (e.g., with flows trained at other budgets)
re-uses all completed cells.

## A.3 Results

### Precision

At the calibration-study operating point (N = 100) posterior standard
deviations under the two likelihoods were equal to within a few percent for
all seven parameters (ratios nn/analytic 0.87–1.02), and similarly at
N = 200 (0.92–1.04). Table A.1 gives the full precision surface. The three
between-trial variability parameters (sv, SZ, st0) remained essentially at
their priors under both likelihoods at these trial counts, as expected for
weakly identified DDM parameters under realistic single-subject data.

**Table A.1.** Root-mean-square posterior sd (sampled scale) by parameter,
likelihood, and trial count. Analytic reference fitted at N ≤ 200.

| par | N=100 nn (an) | N=200 nn (an) | N=400 nn | N=800 nn |
|---|---|---|---|---|
| v  | .127 (.127) | .112 (.111) | .097 | .079 |
| a  | .038 (.039) | .028 (.028) | .020 | .015 |
| t0 | .015 (.017) | .012 (.013) | .0088 | .0074 |
| Z  | .041 (.040) | .036 (.035) | .031 | .025 |
| sv | .154 (.154) | .156 (.156) | .155 | .153 |
| SZ | .154 (.155) | .155 (.156) | .153 | .156 |
| st0| .146 (.145) | .124 (.131) | .115 | .098 |

For the well-identified parameters (v, a, t0, Z) the information ratio κ was
0.85–1.05 at N ≤ 200, declining gently with N (e.g., v: 0.99, 0.98, 0.84,
0.80 across the four trial counts) as sharpening posteriors expose
approximation error. The overall median κ was 0.88, implying that matching
the exact likelihood's precision at a 100-trial base requires approximately
114 trials under the neural likelihood — an information cost of about 14%.
κ estimates for sv, SZ, and st0 are unstable because both likelihoods carry
near-zero information about these parameters at the reference trial counts
(the ratio of two near-zero quantities).

### Bias

The information-loss description is incomplete. Table A.2 decomposes the
bias of the posterior median by trial count. Under the analytic likelihood
all standardized biases were at most 0.4 posterior sd, consistent with
correct inference at these replicate counts. Under the neural likelihood,
two parameters showed absolute biases that were stable across trial counts —
t0 overestimated by ≈ +0.03 log units (≈ +3%) and a underestimated by
≈ −0.07 log units (≈ −7%) — so that their standardized biases grew with N
(t0: +2.0 sd at N = 100 to +3.3 sd at N = 800; a: −1.8 sd to −3.7 sd). The
variability parameters st0 and sv showed absolute biases that increased with
N (st0: +0.06 to +0.17; sv: +0.02 to +0.14). Estimates of v, Z, and SZ were
unbiased throughout (≤ 0.6 sd).

**Table A.2.** Mean bias of the posterior median (sampled scale;
standardized bias in parentheses).

| par | DDM N=100 | DDM N=200 | DDMnn N=100 | N=200 | N=400 | N=800 |
|---|---|---|---|---|---|---|
| t0 | −.003 (−0.2) | +.004 (+0.3) | +.030 (+2.0) | +.035 (+3.0) | +.026 (+2.9) | +.025 (+3.3) |
| a  | −.011 (−0.3) | −.005 (−0.2) | −.069 (−1.8) | −.075 (−2.7) | −.072 (−3.6) | −.056 (−3.7) |
| st0| +.012 (+0.1) | −.036 (−0.3) | +.056 (+0.4) | +.055 (+0.4) | +.118 (+1.0) | +.169 (+1.7) |
| sv | −.000 (−0.0) | +.000 (+0.0) | +.021 (+0.1) | +.032 (+0.2) | +.082 (+0.5) | +.135 (+0.9) |
| v  | −.046 (−0.4) | −.010 (−0.1) | −.051 (−0.4) | −.009 (−0.1) | +.029 (+0.3) | −.049 (−0.6) |
| Z  | +.007 (+0.2) | −.000 (−0.0) | −.000 (−0.0) | −.004 (−0.1) | +.008 (+0.3) | +.007 (+0.3) |
| SZ | −.001 (−0.0) | +.001 (+0.0) | −.002 (−0.0) | −.011 (−0.1) | −.033 (−0.2) | −.034 (−0.2) |

The joint direction of the two stable biases — boundary separation down,
non-decision time up — is the signature of a likelihood that slightly
misplaces the leading edge of the response-time distribution, with the
sampler compensating along the well-known a–t0 trade-off.

### Computation time

**Table A.3.** Median wall time per fit (single subject, one core, to the
convergence criteria above).

| N | analytic DDM | DDMnn |
|---|---|---|
| 100 | 26.0 min | 2.3 min |
| 200 | 57.0 min | 3.4 min |
| 400 | — | 5.1 min |
| 800 | — | 8.9 min |

## A.4 Response-time compression

EMC2's default data processing floors response times to a 1/60 s grid
(`rt_resolution = 1/60`, `compress = TRUE`), predicting a one-sided
leading-edge shift of res/2 ≈ 8.3 ms on average. Because the main study (and
EMC2's calibration machinery) disable compression, a supplementary arm
refitted the N = 100 datasets under both likelihoods at resolutions 1/60,
1/1000, and (neural likelihood only) 1/100.

Compression shifted the t0 posterior median by −0.0285 log units (≈ −8.4 ms)
at 1/60 and −0.0135 (≈ −4 ms) at 1/100, matching the res/2 prediction, with
no effect at 1/1000 and no effect on any other parameter; the shift was
identical for the analytic and neural likelihoods (no interaction). Three
consequences. First, the neural likelihood's boundary-separation bias was
unchanged under every compression condition, confirming it is intrinsic to
the flow rather than a discretization artifact. Second, at the default
resolution the flow's +3% t0 bias and compression's −2.9% shift cancelled
almost exactly at the generating point studied (residual bias +0.001, 0.1
posterior sd); this coincidence of magnitudes should not be mistaken for
unbiasedness, as the two errors vary independently across parameter values
and display refresh rates. Third, the default resolution biases t0 by ≈ −1.9
posterior sd at N = 100 even under the *exact* likelihood; for
non-decision-time-sensitive applications we recommend
`rt_resolution <= 1/1000`, which removed the bias while retaining part of
compression's computational benefit (analytic fits: 20 min at 1/1000 vs 26
min uncompressed vs 6 min at 1/60).

## A.5 Where the approximation error lives

To localize the source of the systematic biases, the flow's joint
log-density was compared directly against the exact likelihood (WienR at
precision 1e-8) across the response-time axis at the generating point, for
flows trained at all three simulation budgets (50k, 200k, and 1M training
trials). The error is highly structured. In the bulk of the distribution
(0.1 s to 0.7 s beyond non-decision time, where nearly all probability mass
lies) mean absolute errors were small (0.14–0.31 log units). At the leading
edge (within 100 ms of non-decision time) the flow systematically
*overshoots* the density, by +0.73 to +1.00 log units on average; in the far
tail (beyond 0.7 s, where the true density falls to e−10 and below) it
overshoots by ≈ +6 log units, decaying far too slowly. The leading-edge
overshoot explains the joint a–t0 bias directly: given data with a sharp
onset, a likelihood with excess early mass is best accommodated by shifting
non-decision time later and narrowing the boundary separation.

Critically, both error components are essentially constant across the 20×
budget range. For the tail this has a simple mechanism: training is maximum
likelihood on simulated trials, and regions of density e−10 receive no
training samples at any realistic budget. For the leading edge it points to
a resolution limit of the 32-bin spline over the wide log-rt support rather
than to data scarcity.

## A.6 Training-budget scaling

The precision-study design (Section A.2) was repeated with the flows
trained at the small (50k) and medium (200k) budgets, fitted to the same
datasets (the analytic reference and large-budget cells being identical to
those of the main study). Budget cleanly improved the *stochastic* quality
of the approximation: the information ratio κ rose from 0.48 (small) to
0.81 (medium) to 0.93 (large), and idiosyncratic biases of the weaker flows
(e.g., a −3.9 sd start-point bias at the small budget) disappeared with
more data. The *systematic* leading-edge biases did not improve: pooled
over trial counts, the absolute a and t0 biases had log–log slopes against
budget of +0.26 and +0.38 respectively — no evidence of the −0.5 slope that
statistical scaling would produce, and consistent with the budget-invariant
error map of Section A.5. All three budgets failed the bias criterion
(|standardized bias| ≤ 0.5 posterior sd at N = 800; worst parameters 3.7–3.9
sd), each on a different worst parameter.

The two results together imply that simulation budget purchases precision
but not unbiasedness: beyond roughly the medium budget, the residual error
is a capacity/structure floor of the network, not a data limit. Two
remedies follow and are evaluated in [Section X]: finer spline resolution
(64 bins, optionally with a wider conditioner) targeting the leading edge;
and, for the tail — whose practical signature is the slowly growing sv/st0
biases — training with an explicitly censored likelihood (e.g., an upper
censoring bound of 10 s), so that the flow is never asked to model the
unsampled deep tail at all. The censored-likelihood approach is deferred to
future work; the architecture manipulations are reported next.

Two variants were trained on the unchanged large-budget data: 64 spline
bins with the original conditioner (arch64), and 64 bins with a wider
conditioner (hidden layers 64 and 32 units; arch64w). Doubling the spline
resolution alone left the leading-edge overshoot unchanged, identifying the
conditioner's capacity to place the knots, not their number, as the binding
constraint. The wider conditioner halved the leading-edge overshoot,
halved the bulk error, and reduced the tail overshoot by two log units.
The inferential consequences were decisive for the leading-edge biases:
pooled absolute bias fell from 0.029 to 0.005 (t0) and from 0.068 to 0.009
(a), with standardized t0 bias at N = 800 dropping from +3.3 to +0.1 sd,
and the information ratio reaching κ = 1.02 (no measurable information
loss). However, the residual tail error now dominates: the st0 bias grew
to +2.2 sd at N = 800 (from +1.7 under the original architecture), keeping
the variant short of the overall bias criterion. The leading-edge problem
is thus solved by conditioner capacity; the remaining blocker is the
unsampled-tail problem, for which the censored-likelihood remedy is the
designated next step.

## A.7 Simulation-based calibration

Single-participant SBC (500 replicates per cell, truths drawn from the
Section-A.2 priors, uncompressed data) was run for all three analytic
models and their neural twins at N = 100 trials, and additionally for the
neural DDM at N = 200 and 400. Calibration was assessed per parameter by
the coverage of the 95% credible interval (Monte Carlo se ≈ 0.010), the
mean normalized rank (0.5 when calibrated), and a Kolmogorov–Smirnov test
of rank uniformity (Bonferroni-corrected over 47 tests).

All three analytic cells were fully calibrated (coverage 0.946–0.972, all
KS tests non-significant), validating the SBC machinery and priors. The
neural RDM passed on all parameters. The neural LNR passed on coverage for
all parameters but showed a detectably non-uniform rank distribution for
the shape parameter s (mean rank 0.433, KS p = 1.5e−5; mean bias +0.011 on
the log scale) — a mild distortion with negligible interval-coverage
consequences at this trial count, consistent with the LNR flow having the
largest error-noise ratio among the race models.

The neural DDM failed catastrophically and exactly as the precision study
predicted. Non-decision time coverage fell from 0.52 (N = 100) to 0.25
(N = 200) to 0.10 (N = 400) with mean ranks of 0.07 → 0.01 (truth below
essentially the entire posterior: overestimation); boundary separation fell
from 0.61 to 0.15 with mean ranks 0.87 → 0.98 (underestimation). The
tail-driven distortions emerged with N as predicted: st0 coverage 0.94 →
0.79 (mean bias growing +0.05 → +0.14), and sv rank uniformity failing from
N = 200 despite nominal coverage. Drift rate, start point, and start-point
variability remained calibrated at every N. Mean biases integrated over the
prior region (t0 +0.034, a −0.064 at N = 100) closely match the
single-point estimates of Table A.2, indicating the leading-edge distortion
is approximately uniform over the training region studied.

Methodologically, the correspondence between the cheap laptop precision
study (Sections A.3–A.6) and the full SBC supports using the former as a
screening gate for candidate flows, reserving SBC for confirmation.

## A.8 Conclusions

The neural DDM likelihood's stochastic approximation error behaves as a
modest information loss (≈ 14% additional trials at a 100-trial base) and its
fits are an order of magnitude faster than the exact likelihood. However, it
also carries a systematic distortion, concentrated on boundary separation and
non-decision time, whose absolute size (≈ 3–7% of the parameter value at the
generating point studied) does not diminish with trial count: additional data
sharpen the posterior around a slightly wrong value. The error maps to a
specific, budget-invariant overshoot of the density at the leading edge of
the response-time distribution (Section A.5), and increasing the simulation
budget improves precision but not this bias (Section A.6) — motivating
architectural rather than data remedies. Consequences for calibration are
examined by simulation-based calibration in Appendix X.

Limitations: a single generating point (the prior mean) and four replicate
datasets per cell; bias estimates at the generating point are 4–6 standard
errors from zero, but their variation across the parameter space is not
characterized here (the calibration study, which integrates over the prior,
addresses this).

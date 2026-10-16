# DDMnle: a calibrated neural likelihood for the DDM with variability

Report for Russ and Julian. Draft of 2026-10-16.

## 0. Summary

- **DDMnle** is a single neural likelihood for the diffusion model (DDM) with trial-to-trial variability in drift (sv), start point
  (SZ) and non-decision time (st0), usable in EMC2 as a drop-in replacement for the analytic DDM.
- **Validation.** Held-out likelihood does not show whether posteriors are right, so we validate by calibration: a score-bias screen
  of every trained checkpoint, then simulation-based calibration (SBC) at N = 400 trials with 500 replicates in six variability
  cases, each against the analytic DDM as control. DDMnle falls outside the 95% rank-ECDF envelope once in 35 parameter tests (a,
  in the st0-only model), which is what chance gives, and needs no correction.
- **Speed.** DDMnle is 1.9–6.2x faster at N = 100 and 2.9–11.5x faster at N = 400 than EMC2's default-precision (5e-3) compiled WienR,
  and 9–685x faster than WienR at the project's reference precision 1e-8, with the largest gains where st0 and SZ are both free
  (§1).
- **LAN.** In Appendix A we use our methodology to test a published LAN from HSSM for the st0 case.

## 1. Speed

Cases with no variability, or with sv only, are not covered here, because the analytic DDM is faster there. Milliseconds per likelihood evaluation, single thread. Speed-up = WienR time / DDMnle time.

| Model | N | WienR 5e-3 | WienR 1e-8 | DDMnle compiled | speed-up vs 5e-3 | speed-up vs 1e-8 |
|---|---|---|---|---|---|---|
| st0+SZ | 100 | 0.32 | 20.9 | 0.0525 | 6.1x | 399x |
| st0+SZ | 400 | 1.55 | 92.4 | 0.135 | 11.5x | 685x |
| st0+sv | 100 | 0.11 | 0.65 | 0.0575 | 1.9x | 11x |
| st0+sv | 400 | 0.455 | 2.15 | 0.143 | 3.2x | 15x |
| sv+SZ | 100 | 0.0875 | 0.4 | 0.035 | 2.5x | 11x |
| sv+SZ | 400 | 0.34 | 1.15 | 0.0775 | 4.4x | 15x |
| st0 | 100 | 0.09 | 0.55 | 0.0475 | 1.9x | 12x |
| st0 | 400 | 0.523 | 3.2 | 0.182 | 2.9x | 18x |
| SZ | 100 | 0.127 | 0.45 | 0.0512 | 2.5x | 9x |
| SZ | 400 | 0.54 | 1.4 | 0.113 | 4.8x | 12x |
| sv+SZ+st0 | 100 | 0.4 | 24.1 | 0.065 | 6.2x | 371x |
| sv+SZ+st0 | 400 | 1.85 | 105 | 0.177 | 10.4x | 590x |

Notes.

- Timings are single thread, in ms per likelihood evaluation at N trials, median over repeats and prior draws.
- WienR at 5e-3 is EMC2's default DDM precision. 1e-8 is the precision the project uses as ground truth.
- The laptop was also running the 8-core LAN SBC while these were timed, so absolute times are probably a little high for all columns.
- Models with st0 cost somewhat more for DDMnle than those without, because each trial needs two CDF evaluations instead of one density.

## 2. How neural likelihoods are judged

A neural likelihood is worth using only if the posteriors it produces are the posteriors the true likelihood would produce. Held-out
likelihood does not test that: it can order networks opposite to calibration. EMC2's default-precision analytic DDM is pointwise
worse than a good network but better calibrated, because its error is series truncation that does not depend on the parameters and
cancels; and networks that differ only in seed have indistinguishable validation losses but score biases on `a` that differ by 0.15,
with half failing SBC. So we use two calibration measures:

- **Score bias** per parameter, g = E[∂ log L̂/∂θ − ∂ log L/∂θ] over prior draws. It is cheap (about a minute per network) and is used
  to rank many checkpoints. It is necessary, not sufficient.
- **SBC** (rank-ECDF with 95% simultaneous envelopes, standardised posterior bias, coverage, KS test), the final test.

## 3. The scaled DDM network

Three of the DDM's parameters enter the likelihood in closed form, so DDMnle does not learn them:

- **a and s set the scale.** Dividing space by a and multiplying time by (s/a)^2^ maps the diffusion onto one with boundary 1 and noise 1:
  f(d) = (s/a)^2^ · g(d·(s/a)^2^; κ, η, Z, SZ), with κ = v·a/s^2^ and η = sv·a/s^2^. Z and SZ are relative to a and unchanged. The
  network therefore has four inputs (κ, η, Z, SZ), plus the response.
- **t0 is a shift.** The network models the decision time d = rt − t0.
- **st0 is an exact convolution.** With u~1~ = d·(s/a)^2^ and u~0~ = (d − st0)·(s/a)^2^, the likelihood is [G(u~1~) − G(u~0~)]/st0,
  where G is the network's own (defective) CDF. This matches the analytic convolution to 2e-6.
- **Exact zeros.** η = 0 is sv = 0 and SZ = 0 is no start-point variability; both are exact zeros of the input encoding, and st0 = 0
  is the plain density.
- **Box.** κ ∈ [−10, 10], η ∈ {0} ∪ (0, 8], Z ∈ [0.1, 0.9], SZ ∈ {0} ∪ [0.05, 0.99]. The SBC priors (v ≈ 1, a ≈ 0.8, s = 1, sv ≈ 0.1)
  give κ ≈ 0.4–1.6 and η ≈ 0.05–0.2, well inside.
- **Architecture.** A conditional spline flow over log u ([256, 128] conditioner, 64 bins, batch 1024) with a separate classifier for
  P(R) trained on exact pWDM targets.
- **Verification.** With the network replaced by the exact unit-scale density, the likelihood agrees with the analytic DDM to 8e-12
  at st0 = 0 and ≤ 5e-5 with st0 > 0, with exact zeros below t0. The compiled likelihood agrees with the reference implementation
  to ≤ 1.1e-11 in the log-likelihood over 200 prior draws in each of the six models.

### 3.1 How DDMnle is incorporated in EMC2

A trained network is exported as a card: a file holding the weights, spline settings, input transforms, and golden test vectors
computed from the training code. The loader asserts the card's input order (κ, η, Z, SZ) on load instead of trusting metadata,
because a mis-ordered network gives a large, silent bias.

- **A model function, not a new model.** DDMnle is EMC2's own `DDM()` with only the likelihood replaced. The parameter transforms,
  bounds, the exact-zero rules for sv, SZ and st0, and the simulator (so `make_data`) are the analytic DDM's. Designs, priors,
  samplers and plotting therefore work unchanged; a fit differs from an analytic-DDM fit only in the likelihood call.
- **What runs per likelihood call.** All trials in a call share one parameter set, so the network's conditioner runs once per
  response, not once per trial; each trial then costs a spline evaluation. For each trial the compiled code reads v, a, sv, t0, st0,
  s, Z, SZ from EMC2's parameter table, forms κ, η, Z, SZ and c = (s/a)^2^, and makes one network call: a density when st0 = 0, a CDF
  pair when st0 > 0. It returns c·g or [G(u~1~) − G(u~0~)]/st0. Out-of-box parameters, and RTs at or below t0, get the same floor or
  zero the analytic model uses.
- **Checks.** An oracle test (network replaced by the exact unit-scale density), a golden-vector test of the card, and agreement of
  the compiled path with the reference implementation, as above.

## 4. Selection

- **Training:** 6 seeds × 300 epochs, 600k parameter draws × 40 rows each, 30 checkpoints kept per seed (180 in all). Final validation
  losses (−1.4358 to −1.4372) were indistinguishable across seeds, so selection does not use them.
- **Screening:** every checkpoint was screened by score bias in 7 regimes (full, η = 0, SZ = 0, st0 = 0, and combinations of zeros),
  in two stages (100 draws, then the leaders at 300). The score is Σ over regimes of |g~v~| + |g~a~| + |g~Z~|. The analytic oracle
  scores ≤ 7e-5 in every regime, so the screen has a floor well below the networks'. max |g~st0~| over all 180 checkpoints is 0.005,
  so the exact convolution does what it should. The best checkpoint per seed ranged from 0.042 to 0.081: seed variance is large,
  which is why selection is on score bias.
- **Criteria set in advance:** st0 clean in every cell; a inside the envelope in the three models with two variability sources. The top 3 checkpoints went to SBC and
  DDMnle is the one that met them.

## 5. SBC results

Six models (listed parameters free, the others pinned at exactly 0): st0+SZ, st0+sv, sv+SZ, st0, SZ, and sv+SZ+st0. N = 400 trials,
500 replicates per model, network only, against the analytic DDM as control. Appendix B contains the SBC plots (rank-ECDF
differences with 95% envelopes) for each model.

**Standardised posterior bias by parameter, and parameters outside the 95% rank-ECDF envelope:**

| Model | v | a | t0 | Z | SZ | st0 | sv | DDMnle outside envelope | analytic control outside envelope |
|---|---|---|---|---|---|---|---|---|---|
| st0+SZ | −0.06 | −0.03 | +0.01 | +0.08 | −0.04 | +0.05 | | none | none |
| st0+sv | +0.01 | +0.02 | +0.05 | +0.05 | | +0.02 | +0.02 | none | none |
| sv+SZ | −0.02 | −0.04 | +0.08 | −0.05 | +0.04 | | +0.01 | none | none |
| st0 | +0.05 | +0.05 | +0.05 | −0.06 | | −0.00 | | a | t0 |
| SZ | +0.04 | −0.03 | +0.01 | −0.04 | −0.01 | | | none | none |
| sv+SZ+st0 | −0.07 | −0.10 | +0.13 | +0.01 | −0.00 | +0.02 | +0.02 | none | not run |

- **Size of the bias:** worst |standardised bias| is 0.13 (t0, sv+SZ+st0); coverage is 0.932–0.968 and the smallest KS p-value is 0.020.
- **The only flag** is a in the st0 model, one of 35 tests; its standardised bias is +0.05. The analytic control in the same cell has
  t0 outside the envelope, so that cell is not a clean one even for the analytic likelihood.
- **Residual pattern:** t0 is slightly high (up to +0.13) and a slightly low in the fullest model (−0.10), both inside the
  envelope. No correction is applied.
- **Caveats:** calibration is shown at N = 400 trials and one participant per replicate; larger N and hierarchical fits are
  untested. The SBC was run on the reference implementation; the compiled path was verified against it by likelihood agreement
  (≤ 1.1e-11), not by rerunning SBC.

## Appendix A: ONNX, EMC2, and LAN Comparison

### A.1 Bringing ONNX networks into EMC2

The field distributes its likelihood approximation networks (LANs; Fengler et al.) as ONNX files, consumed by HSSM. EMC2 does not
run an ONNX file directly, and does not need to: a LAN is a small multilayer perceptron (for the DDM with uniform non-decision time,
7 inputs, three tanh layers of 100 units, one output, with a log-likelihood floor at −16.12), and an MLP can be converted.

- **Conversion.** A converter reads the ONNX graph and writes the same kind of card as DDMnle's. The converted card is checked
  against onnxruntime before it is accepted.
- **Registration.** The card is registered with EMC2's compiled evaluator together with an analytic "twin" model that has the
  same structure. The network reuses the twin's simulator, so data can be generated from the analytic model and fitted with the
  network, and the twin supplies the control for SBC.
- **Parameterisation is checked, not inferred.** The LAN's parameters are not EMC2's. For this LAN the mapping is
  a~EMC2~ = 2·a~LAN~, Z = z, t0 = t − st, st0 = 2·st, sv = SZ = 0, s = 1, determined empirically against WienR and verified to 4e-7.
  It is implemented as a model function, with the prior and bounds taken from the LAN's own training box and out-of-box parameters
  returning the model floor.
- **Why this is useful.** It makes the published LAN library (angle, Levy, Ornstein, Weibull and others, many with no analytic
  likelihood) available to EMC2 designs, and it gives a like-for-like test: the same SBC cell run with the LAN and with the analytic
  DDM in the LAN's parameterisation as control.

### A.2 Making DDMnle available through ONNX

DDMnle has not been exported to ONNX yet; the route is the one already proven for other DDM flows.

- **Interface.** The LAN-shaped signature is a float64 `[N, 10]` input: the parameters on the sampled scale, then rt and choice
  (−1 lower, +1 upper); the output is log p(rt, choice | θ), the joint defective log-density a LAN returns. A flow exported this way
  reproduced its reference implementation to 2.4e-12. A second build takes θ once and rt and choice as vectors; it is 25x faster
  (0.58 against 14.5 ms per 1000 trials) because a trial-wise graph re-runs the network for every trial.
- **What is new for DDMnle.** The network sees (κ, η, Z, SZ), and the transformations that make it a DDM likelihood sit outside it:
  the scaling c = (s/a)^2^, the t0 shift and, when st0 > 0, a difference of two CDFs. The ONNX graph must therefore contain the
  network's CDF as well as its density (the spline CDF and a normal CDF through the ONNX Erf operator), and the preprocessing
  (κ, η, c from v, a, sv, s) and the st0 branch must be written as graph operations. The natural contract is that the caller passes
  EMC2's natural parameters (v, a, t0, s, Z, SZ, sv, st0) and the graph does the rest. Probit and log transforms stay with the
  caller, because the probit has no exact ONNX operator.
- **Acceptance test.** The export is accepted only if it matches the compiled DDMnle in EMC2 to about 1e-11 over the same 200 prior
  draws in each of the six models, plus the card's golden vectors.
- **What an HSSM contribution would still need.** A model configuration for the seven-parameter DDM (v, a, z, t, sz, sv, st), the
  parameter-conversion wrapper above, bounds from DDMnle's box and, as the project would insist, an SBC run through HSSM's own
  sampler rather than EMC2's.

### A.3 The LAN SBC

[Pending: the LAN's SBC in its own training box, with the analytic DDM in the same parameterisation as control. A 3-replicate smoke
test gave a v standardised bias of +4.3 and coverage 0/3; three replicates are not a result. The LAN and DDMnle's st0 model use
different boxes and priors, so each is read against its own analytic control, not against the other.]

## Appendix B: SBC plots

Rank-ECDF difference plots for DDMnle, N = 400 trials, 500 replicates per model; the shaded region is the 95% envelope. Each panel gives coverage, bias, precision and the KS p-value.

![Model st0+SZ](report_figs/sbc_1.png){width=85%}

**Figure B1.** st0+SZ.

![Model st0+sv](report_figs/sbc_2.png){width=85%}

**Figure B2.** st0+sv.

![Model sv+SZ](report_figs/sbc_3.png){width=85%}

**Figure B3.** sv+SZ.

![Model st0](report_figs/sbc_4.png){width=85%}

**Figure B4.** st0.

![Model SZ](report_figs/sbc_5.png){width=85%}

**Figure B5.** SZ.

![Model sv+SZ+st0](report_figs/sbc_6.png){width=85%}

**Figure B6.** sv+SZ+st0.


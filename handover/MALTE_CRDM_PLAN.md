# Malte Lüken's neural race flows: assessment, RDM comparison, CRDM in dev-nle

Written 2026-09-28 from a reading of
`github.com/maltelueken/neural-race-model-estimation` (branch `development`,
head fae8add, 2026-09-25) and its shared library `github.com/maltelueken/eamax`
(v0.1.1). Nothing here has been run: the repository ships code and configs but
**no trained weights**, so every statement about his flows' accuracy is his
(from the notebooks' stored output), and every statement about ours is from
`docs/METHODS.md`. Sections: 1 what he built; 2 which tweaks are worth taking
(question a); 3 the RDM comparison (b); 4 CRDM in dev-nle, the applicable
tests and the real-data study (c); 5 what is needed and in what order.

---

## 1. What he built

**Model code** lives in `eamax` (JAX): analytic Wald accumulator, a race
assembled from per-accumulator `log_pdf` / `log_sf`, a conflict-pulse
accumulator with a grid simulator and a Volterra (Fortet) reference density,
NUTS and tempered-SMC samplers. The paper repository adds the two
parameterisations, priors, the hybrid CRDM race and an experimental flow
variant (`confrdm_jax/flows_affine.py`) that is what the final runs use.

**The flow.** One rational-quadratic spline over log decision time, standard
normal base, conditioned by an MLP on one accumulator's parameters — the same
family as our `flow_race`. The race is assembled analytically from the flow's
density and survivor, as ours is. Differences from `rdm_small`:

| | ours (`rdm_small`) | Malte, final configuration |
|---|---|---|
| what the flow models | rt, with t0 a conditioning input | **decision time**; t0 is an exact shift applied by the race |
| inputs | `v, B, t0, s, A` on the sampled (log) scale, standardised by the training data's moments | `v, s, b` (RDM) or `v, amp, tau, s, b` (CRDM), as `(log(x + eps) − loc) / scale` with loc, scale in closed form from the uniform box |
| start-point variability `A` | yes | **none** — his accumulator is a plain Wald |
| spline | 64 bins on a fixed [−5, 5] in log rt, identity tails | 12 bins on [−5, 5] in a **standardised** log time: the conditioner also emits a per-context location and scale (`affine`), so tails are log-normal with the context's own scale |
| conditioner | [64, 32], GELU(tanh) | [128, 128], GELU(tanh), 3·12+3 outputs |
| data | 16e6 stored rows, one trial per θ draw, analytic ll stored | **online**: 64 θ draws × 500 trials simulated fresh every step, 100k steps (6.4e6 θ, 3.2e9 trials), nothing stored, no validation set |
| optimiser | AdamW, warmup-cosine to 1e-4 peak, batch 1024, 300 epochs, 30 checkpoints | AdamW, cosine decay from 5e-3, global gradient-norm clip 5.0, one final checkpoint |
| loss | NLL (+ optional score penalty) | right-censored NLL (non-crossing trials scored by log S(t_max)) |
| precision | float32 training, float64 port | float64 throughout |
| outside the box | likelihood refused (floor) | inputs **clipped** to the box |
| selection | 30 checkpoints screened on score bias | none; single run, single seed |
| validation | exact port check, score bias with analytic control, posterior shift, SBC with matched control | KL / W1 / KS against the analytic (RDM) or Volterra (CRDM) density on a 45- / 540-point θ grid; parameter recovery (100 datasets × N = 50…1000, NUTS; hierarchical, 20 subjects, tempered SMC); classifier two-sample test between neural and analytic posteriors |

Training boxes (natural scale, uniform): RDM `v [0, 8]`, `s, b [0.25, 3.5]`;
CRDM `v [0, 8]`, `amp [0, 1]`, `tau [0.01, 0.5]`, `s, b [0.25, 3]`.

**The CRDM.** A two-accumulator RDM in which the accumulator that the
distractor points at carries a transient gamma-shaped pulse (shape 2) added to
its drift: mean path `M(t) = v t + amp · (e t / tau) · exp(−t / tau)`, peak
height `amp`, time scale `tau` — the DMC's automatic activation moved into a
race. Parameters `v_c_intercept, v_c_slope, amp, tau, s_true, b, t0`, all
log-linked; the non-target's noise is fixed at 1. At `amp = 0` it **is** the
RDM, so the two are nested. Only the pulsed accumulator needs the expensive
density, so his likelihood is a *hybrid race*: flow on the pulsed accumulator,
closed-form Wald on the other.

**His numbers** (stored notebook output; ms per parameter set over a 2000-point
rt grid, on his hardware):

| | neural | reference | ratio |
|---|---|---|---|
| Wald (RDM accumulator) | 0.169 | 0.072 analytic | flow 2.4x **slower** |
| CRDM, Volterra dt = 0.05 | 0.186 | 0.92 | 4.9x |
| CRDM, Volterra dt = 0.005 | 0.192 | 7.97 | 42x |
| CRDM, Volterra dt = 0.0005 | 0.196 | 99.4 | 506x |

Posterior means from the neural and analytic RDM likelihoods correlate
0.998–0.9999 across parameters and N. His own timing agrees with ours on the
point that matters for framing: **for the RDM the flow is slower than the
closed form; the speed case is the CRDM.**

---

## 2. Which tweaks are worth taking (a)

Ranked by what I expect them to buy *for calibration*, which is where our
flows are limited. "Test" means an ablation in our pipeline, on the RDM, where
training is cheap and the analytic control is exact.

| # | tweak | verdict | why |
|---|---|---|---|
| 1 | **Per-context location and scale on log time** (`affine`) | **Test first — the most promising** | Our residual is a *score* error: the θ-derivative of log f̂ is slightly wrong, val loss is blind to it, and it varies 10x across seeds. The diffusion's exact symmetries (rescaling time, rescaling space) are shifts in log-input space, and with an affine stage the flow's location is *linear* along them — the score in those directions is right by construction instead of learned through 64 moving knots. We found more bins do not help ("spline resolution never bound"), which says the binding constraint is how the density moves with θ, not its shape; that is exactly what this stage takes off the spline. Side benefit: 12 bins instead of 64 shrinks the output layer from 193 to 39. |
| 2 | **t0 as an exact shift** (decision-time flow) | **Adopt for race models** | One fewer conditioning dimension and the support boundary is exact rather than learned. We built this for the DDM (`dt_s703`), where it was convicted by the st0-default bug and later exonerated; dev-nle already supports it (`pre = "t0"`). For the RDM there is no st0, so nothing is lost. It does not remove our DDM t0 offset by itself — that is the onset's *shape* under st0 — but zero-box v1 shows the offset is not intrinsic (t0 +0.08 there against +0.32 shipped). |
| 3 | **Right-censored loss** | **Adopt when the simulator has a horizon** | Irrelevant to our current flows (exact samplers, no t_max). Required for the next project — distilling likelihoods that are only simulable on a grid — where dropping non-crossing trials biases the density by −log(1 − q(θ)), which is θ-dependent and therefore a score bias. |
| 4 | **Trim degenerate corners; clip the gradient norm** | **Relevant to zero-box** | He measured gradient norms of 19–420 from near-deterministic corners (s → 0) against 0.5–7 elsewhere, and a run that degraded under them. Our zero-box flow has exactly such a corner by design — st0 → 0, a sharp onset — and v1 lost st0 resolution wherever st0 is free. This is a second candidate mechanism beside training density (§11 of the zero-box plan), and the two predict differently: if v2 (density fixed) still fails on st0, gradient clipping or a per-row loss cap is the next lever. Cheap to add to `train_ddm.py`. |
| 5 | **float64 for the export cross-check** | **Adopt (small)** | Two of 90 zero-box exports failed the port gate on single-point float32 outliers (2.9e-2) with healthy percentiles. Running the JAX side of the gate in x64 removes the false failures without touching training. |
| 6 | Online simulation, no stored data | **Not as is** | His 64 θ per step × 500 trials spends samples where we measured no return (no gain beyond ~40 trials per draw) and gives *less* θ-diversity than our RDM set (6.4e6 vs 16e6 draws). What is attractive is the absence of epochs: half our "seed" variance is epoch-to-epoch. Whether a cosine decay to ~0 on fresh data narrows the score-bias spread is an empirical question the seed study in §3 answers; it needs a JAX sampler, which exists for the Wald and not for the full DDM. |
| 7 | log(x + eps) inputs | **Already have the equivalent** | We condition on log parameters, and zero-box uses asinh(x / c) for the same purpose (finite at 0, log-like above). Nothing to take; note for the port that his eps version is a distinct transform. |
| 8 | [128, 128] conditioner | **No** | We measured [64, 32] at 58% of the calibration gain of [256, 128] for 18% of the cost, and evaluation cost is what makes the RDM flow 185x slower than the closed form. |
| 9 | Clipping inputs to the box at inference | **No, for EMC2** | It makes the likelihood flat outside the box, so a posterior that wants to leave the box is silently given the edge's density. That suits gradient samplers (his reason: NUTS and SMC collapse on extrapolation spikes); EMC2's samplers are gradient-free and dev-nle's contract is to refuse, which is the honest failure. |
| 10 | KL / W1 / KS density metrics; C2ST on posteriors | **Complementary, not a substitute** | Density-level agreement does not predict calibration — our seeds have identical val loss and 10x different score bias. C2ST is a global version of our posterior-shift screen without its direction or its per-parameter reading. Worth reporting his KL beside our g on the same flows, because the comparison itself is a result. |

The untested assumption in his pipeline, from where we stand: **one run, one
seed, no selection.** Everything we learned says a single flow is one draw
from a wide distribution of score biases. His tweaks may have narrowed that
distribution — that would be the most valuable thing to learn from him — or
his artefact may be a lucky or unlucky draw. §3's seed study is how to find out.

---

## 3. Comparing his RDM with ours (b)

### 3.1 What makes the comparison fair

- **No start-point variability in his.** The common model is the RDM with
  `A = 0`. EMC2's analytic RDM has the exact Wald special case
  (`exception = c(A = 0)`), so the matched control exists. Our flow's box has
  `A ≥ 1e-4`; it is evaluated at `A = 1e-4` as a named constant (an error of
  order A/B ~ 1e-4 in the threshold, far below anything measured).
- **Common region.** Intersection of the boxes: `v ≤ 5`, `s ∈ [0.25, 2]`,
  `b = B ∈ [0.25, 3]`, `t0 ∈ [0.05, 1]`. Our standard RDM SBC cell
  (`sbc/nci/sbc_specs.R`: v 1.5 and 2.24 by match, B 1, t0 0.2, s fixed at 1,
  prior sd 0.1 on the sampled scale) sits well inside it once `A` (0.3 there)
  is set to 0, so the comparison cell is our standard cell minus `A`: four
  sampled parameters, `v`, `v_lMTRUE`, `B`, `t0`. The design is EMC2's
  (`v ~ lM`, log-linked); the network only ever sees each accumulator's
  natural-scale `v`, `s`, `b`, so his additive drift parameterisation does not
  enter.
- **Same evaluator, same sampler.** Both flows run through dev-nle's compiled
  `flow_race` path and EMC2's sampler, so differences are the flows', not
  JAX-vs-C++ or NUTS-vs-DE-MCMC.
- **Calibration, not speed**, as the methods document already says; speed is
  reported, in the same evaluator, as a secondary table.

### 3.2 Steps

| step | what | acceptance | compute |
|---|---|---|---|
| B0 | Obtain his RDM conditioners: the **final** one (affine, log inputs, two hidden layers, 12 bins, 128, clip, narrow box, 100k steps) and the **plain** one (500k steps), each with its sidecar. Failing that, retrain from his SLURM specs. | checkpoint loads in his code; densities reproduce his KL table | none, or hours on tux26 CPU |
| B1 | Converter, Orbax checkpoint + sidecar → our JSON card (`flow/scripts/malte_to_card.py`): MLP layers, spline settings, and three new card fields — `affine` (last two outputs: loc, raw scale; scale = softplus(raw + c) + 1e-3), `input_scaling {eps, loc, scale}`, `pre = "t0"`. Golden vectors written from **his** code in float64. | card round-trips; golden file covers box interior, edges, both tails | minutes |
| B2 | R port first (`port/R/flow_race.R`: affine stage, log-eps input transform), as the readable spec; then the same two additions in dev-nle's `flow_race.cpp` (a prompt for the EMC2 session, with the R port and golden vectors as its acceptance test). | R port vs golden < 1e-10; dev-nle `dfun` vs R port < 1e-7 (the usability gate's check 2) | minutes |
| B3 | Register: `register_nn_model(card, twin = RDM, exceptions = c(A = 0), pre = "t0")`, with `b ≡ B`. Assert `mapped_pars()` for every parameter not sampled. | the Phase 37 checks | — |
| B4 | **Score bias** over SBC-prior draws, both flows and the analytic control, H = 0.01; t0 by the restricted-region estimator. | control < 1e-3 | minutes, laptop |
| B5 | **Posterior-shift screen**, 48–60 datasets, both flows. | control within ±0.15 | ~30 min, laptop |
| B6 | **SBC**: the A = 0 cell at N = 100 and N = 400, 500 replicates, each flow with its matched analytic control (one control serves both flows: same cell). Report standardised bias against the 0.5 gate, KS on ranks, coverage. | — | ~6 cell-runs; about half a day on tux26 at 72 cores |
| B7 | **Speed** in the compiled evaluator: µs per 2000 accumulator-rows, shared and trial-varying θ, against `rdm_small` and the closed form. | — | minutes |
| B8 | **Seed study** — the one that answers (a). Train four seeds of each of: our pipeline as is; ours + affine; ours + decision-time; ours + both; and his configuration as published. Screen every retained checkpoint on score bias. Outcome: the *distribution* of \|g\| per arm, not a best case. | registered before the numbers: an arm "helps" if its median \|g\| is below our baseline's best quartile | RDM training is cheap: ~1 day on tux26 for the 20 runs |

### 3.3 What to expect, so the result can be read against it

His density-level accuracy will likely look better than ours (more capacity,
affine stage, 200x more samples) and his posterior means correlate 0.999 with
the analytic's. Neither is calibration. Our prior from the DDM and RDM work
is that a single unselected run sits somewhere in a wide score-bias
distribution; `rdm_small` is a *selected* checkpoint and that caveat travels
with it. So the informative outcomes are: (i) his single run passes SBC with
uniform ranks — then the tweaks narrowed the distribution and B8 should show
it; (ii) it passes the gate but fails ranks on some parameter — the usual
unselected-flow result, and selection is what we add; (iii) it is better on
t0 by construction (exact shift) whatever else happens.

---

## 4. CRDM in dev-nle (c)

### 4.1 What dev-nle needs

1. **The evaluator additions from B2** (affine stage, log-eps inputs,
   decision-time shift) — shared with the RDM comparison, so they are built
   once.
2. **A hybrid race.** dev-nle's `flow_race` evaluates the network for every
   accumulator. Two routes:
   - *all-flow*: evaluate the CRDM flow on both accumulators with `amp = 0`
     on the one without the pulse. No evaluator change — the flow was trained
     down to `amp = 0`, where it is a plain Wald — but the unpulsed accumulator
     then carries approximation error it need not, and costs a network
     evaluation where a closed form is 2.4x cheaper.
   - *hybrid* (his design, **recommended**): network on the pulsed accumulator,
     EMC2's own Wald on the other. Needs a per-accumulator switch in the race
     assembly; exact where it can be.
   The all-flow route is the fallback and a useful check: the two must agree
   to the flow's error at `amp = 0`, which is measurable against the closed
   form.
3. **The design.** Which accumulator is pulsed is a property of the trial
   (where the distractor points), exactly as which accumulator matches is. In
   EMC2 terms: a second match factor, `lD = (lR == distractor response)`, built
   by a design function beside `lM`; `amp ~ 0 + lD` with the `lDFALSE` level a
   named constant at exact zero. That is the zero-box situation again —
   `amp = 0` on the sampled (log) scale is −Inf — and the same machinery
   (`p_types` −Inf, bound exception at 0, transform finite at zero) applies.
   `tau` is shared; `A = 0` is a named constant.
4. **No analytic twin in EMC2**, so registration uses the hand-written twin
   route (`twin = function() list(type, p_types, transform, rfun)`), which
   needs:
   - **`rfun`, a CRDM simulator in C++**: exact mean path plus Brownian
     increments on a grid, the Brownian-bridge crossing correction, within-step
     dequantisation (his `first_passage_from_mean`, ~30 lines). Validated
     against his JAX simulator (two-sample KS on 1e5 draws per θ) and against
     the Volterra CDF.
   - **a reference density, `dCRDM_volterra`, in C++**: the Fortet recursion,
     solved **once per distinct parameter row** and interpolated over that
     row's trials (his code re-solves per trial, which is why it looks slower
     than it is). This is the "analytic control" for every test below.
     Checked for convergence in dt (0.002, 0.001, 0.0005) and against the
     closed-form Wald at `amp = 0`, where it must agree.

### 4.2 Which of our tests apply

| test | applies? | control |
|---|---|---|
| exact port check (dev-nle `dfun` vs R port vs JAX golden) | yes | — |
| nesting check: flow at `amp = 0` vs closed-form Wald | yes — **new, and exact** | analytic |
| total mass | yes (1 by construction per accumulator; the race's defective masses must sum to 1) | — |
| score bias, all of `v, amp, tau, s, b` | yes | Volterra; seconds per draw |
| t0 score bias (restricted region) | yes; exact shift, so this tests the onset's shape | Volterra |
| posterior-shift screen | yes | Volterra in the importance weights: a few thousand evaluations per dataset |
| SBC, network | yes, N = 400 (200 per congruency), 500 replicates | — |
| SBC, matched control | **feasible but slow**: two Volterra solves per likelihood evaluation, ~25–50 ms at dt = 0.002 — the cost class of the regression-net cell (3.8 days for 250 replicates at 48 cores). Run one cell at 250 replicates. | Volterra |
| choice-classifier decomposition | no (race model) | — |
| zero regimes | only `amp = 0`, covered by the nesting check | — |
| **new: simulator-step sensitivity** | his flows were trained on simulations at dt = 0.05, 0.005 and 0.0005; score all three against the same reference. The training simulator's discretisation is an error source our pipeline never had. | Volterra at 0.0005 |

### 4.3 The real-data study

**Question.** Does the conflict pulse improve the account of conflict-task
data over a standard RDM in which congruency moves drift rate (and possibly
threshold)?

**Models**, all with `A = 0` so the two are nested at `amp = 0`:
M0 RDM, `v ~ congruency × lM` — analytic, for speed; M0b as M0 with
`B ~ congruency`; M1 CRDM, congruency enters only through the pulse.
Hierarchical, fitted in EMC2 on dev-nle.

**Two controls the study needs because one likelihood is approximate:**

1. **A null comparison.** Fit M0 twice, once with EMC2's analytic RDM and
   once with Malte's RDM *flow*. Any difference between those two fits in
   information criteria or marginal likelihood is the approximation's
   contribution and nothing else. It sets the scale below which an M1-vs-M0
   difference is not evidence. Per-trial log-likelihood error accumulates
   linearly: a KL of 1e-3 nats is 10 nats over 20 subjects × 500 trials —
   the size of many published model-selection differences.
2. **Box occupancy.** Report, per subject, how much posterior mass lies
   within 5% of the training box's edges (`amp ≤ 1`, `tau ∈ [0.01, 0.5]`,
   `s, b ∈ [0.25, 3]`, `v ≤ 8`). A fit that leans on an edge is a fit to the
   box, and dev-nle will refuse outside it.

**Comparison.** Posterior predictive first — delta plots and conditional
accuracy functions, which are where the pulse should earn its place (fast
errors on incongruent trials; negative-going delta in Simon-type tasks) —
then DIC/BPIC and bridge-sampling Bayes factors, each read against the null
comparison.

**Open:** which data sets. The study needs conflict tasks with enough trials
per subject per congruency to resolve the fast tail; flanker and Simon sets
with different delta-plot signatures would be the informative pair.

---

## 5. What is needed, and in what order

**From Malte:** the trained conditioner directories with sidecars — RDM final
and plain; CRDM at dt = 0.0005, 0.005, 0.05 (final configuration). Without
them the fallback is retraining: the RDM flows are cheap; the CRDM at
dt = 0.0005 simulates 64 × 500 paths × 8000 steps per training step and was a
GPU job of up to 8 h.

**Decisions (AH, 2026-09-28):** checkpoints requested from Malte
(`~/Downloads/julian/EMAIL_TO_MALTE.md`); **hybrid race** in dev-nle; the
real-data study of §4.3 is **deferred**, and with it the choice of fitting
parameterisation and of data sets. Still open: whether to also report our
standard A-free cell for `rdm_small` beside the A = 0 comparison (default:
yes, it exists already). The work is staged for cleared-context sessions in
`handover/MALTE_STAGES.md`.

**Sequence.** tux26 is committed to zero-box v2 until about 30 Sep, and
compute is tux26-only until 1 Oct, so the order is by what needs no compute:

| when | work | needs |
|---|---|---|
| now | B1 converter, B2 R-port additions, golden vectors; C++ simulator and Volterra reference with their own tests | checkpoints (B1/B2 golden vectors); nothing else |
| now | prompt for the EMC2 session: affine stage, log-eps transform, hybrid race, `lD` design function, hand-written-twin registration for CRDM | the R port as spec |
| after zero-box v2's cells | B4–B7 (screens are laptop-scale; SBC cells ~half a day at 72 cores) | tux26 |
| after 1 Oct | B8 seed study; CRDM screens and SBC (network cell ~hours; Volterra control ~days at reduced replicates) | compute |
| then | §4.3 real-data study | data sets; the null comparison first |

**Model staging** (as in the zero-box plan): converter, simulator port and
launch scripts are Sonnet-grade against written acceptance tests; the hybrid
race and the `lD` / exact-zero design in dev-nle are trap-dense and want
Opus/Fable; reading the seed study and the null comparison is judgement.

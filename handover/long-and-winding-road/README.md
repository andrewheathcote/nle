# The Long and Winding Road

Everything we tried that did not work, with the code and the numbers, so a
sceptical reader can check any of it.

This is not a confession. Several of these are ideas a competent person would
have tried in the same order, and two of them were *published in our own notes
as successes* before replication killed them. The failures are properties of the
method; the retractions are properties of how we tested. Both are worth knowing
before you repeat them.

`SBC_PLAN.md` is the full audit trail — 36 phases, including every claim we
withdrew and why. Each entry below names the phase so you can read the original
and the retraction together.

---

## A. Ideas that do not work

| directory | idea | what happens |
|---|---|---|
| `score-penalty/` | add `‖E[score]‖²` to the loss to drive score bias to zero | satisfiable by flattening `log f` in θ. Since `I = −E[∇² log f]`, flattening destroys curvature (κ 1.027 → 0.46). Unbounded versions are exploited outright. **Phase 5** |
| `ensembling/` | average several flows; errors should cancel | a density mixture's score is the **mean** of member scores (−0.0398 vs member mean −0.0393), so it cannot beat its members — at 4.3× the cost. **Phase 6** |
| `weight-averaging/` | plateau-wandering is exactly what SWA is for | lands where the mixture lands (−0.0473). Averaging converges to the member mean by either route; only *selection* finds the best member (22× better). **Phase 6b** |
| `reparameterisation/` | (SZ0, SZ, SZ1) removes EMC2's `Ttransform` kink at Z = 0.5 | cannot widen coverage — `sw ≤ 2 min(w, 1−w)` is algebraically forced — and made bias on `a` 32× worse. **Phases 10–12** |
| `edge-weighting/` | upweight the leading edge, where the error is | monotone harm: +0.009 / +0.471 / +0.991 / +1.53 at amplitude 0 / 3 / 10 / 30. Upweighting collapses the effective sample size. **Phase 15** |
| — | fix the diffusion coefficient `s`; the box carries an exact (v, a, s) redundancy | the network already respects the invariance: the derivative along the null direction is 2% of the bias on `a`. **Phase 8c** |
| — | more capacity, more spline bins | bins never bind; [512,256] was worse than [256,128]. **Phase 8** |
| — | more trials per parameter draw | no gain beyond ~40. **Phase 17** |

## B. Claims we made and then retracted

These cost the most, and all four have the same root: **believing something
measured once.**

| directory | claim | how it died |
|---|---|---|
| `batch-and-seed/` | batch 1024 shifts the mean of the score bias, 5× better | replication over 5 seeds: mean shift **t = 0.10**. Against a *matched same-machine* arm the variance ratio is 2.84× (p = 0.34), not the 8.7× first reported — half of it was provenance. **Phases 16 → 20 → 22 → 25** |
| `distillation/` | distilling toward the analytic moves the mean; three arms ordered monotonically on two quantities | three fresh seeds at the selected β gave −0.104 where the sweep's own run gave +0.016 — **3.7 sd away**. Against plain MLE: p = 0.91. Three draws from the measured between-seed spread land in monotone order by chance 1 in 6. **Phases 24 → 30** |
| `st0-routes/` | the t0 bias is a t0/st0 leading-edge trade-off | holding st0 fixed barely moves it (+0.319 → +0.269) and *relocates* the error: `Z` degrades from −0.032 (p 0.64) to −0.148 (p 5e−05). **Phases 7 → 29** |
| `st0-routes/` | …then: it is the hard support boundary | enforcing `ll = −∞` when `t0 > rt` changes **nothing** — the likelihood maximum sits below min(rt) and the constraint never binds. **Phases 32 → 32b** |

## C. Three builds that failed SBC

All in `st0-routes/`, with `results/` holding the SBC objects. Each was a
reasonable structural response to the t0 problem, and each was worse than the
artefact it meant to replace.

| build | t0 std bias | t0 coverage |
|---|---|---|
| **analytic control** | **+0.017** | **0.946** |
| st0 fixed at 0.05 (evaluation-time) | +0.269 | — |
| st0 = 0 (retrained, 7 parameters) | +3.144 | 0.000 |
| decision-time `d = rt − t0` (6 parameters) | +2.490 | 0.002 |

The last is the informative one. The flow **never sees t0**, so its displacement
cannot originate there — it arrives through the other parameters, and t0 is
where it surfaces because its posterior is 3–20× tighter than the others and it
couples to all of them. **t0 is the victim, not the cause.** Phase 35.

## D. Diagnostics that do not predict the SBC verdict

`measurement-traps/`. This is the most transferable part of the directory.

| diagnostic | why it fails |
|---|---|
| score bias at the truth | a derivative at **one point**; says nothing about the surface a posterior explores. The decision-time flow had the **best** total \|g\| of any DDM build (0.0240) and coverage 0.002 |
| joint MLE from the truth | lands +0.37 ms from the analytic's on a build that fails catastrophically |
| multi-start optimisation | finds better optima away from the truth, but recovers only a third of the effect |
| 1D surface profiles | shows the distortion is in `sv`/`SZ` — but the flow that **passes** has a *larger* `sv` shift (−1.800) than the one that fails (−1.350) |

All four condition on the truth in the other parameters. The distortion that
matters is multivariate.

**The conclusion the project rests on: run SBC, with its analytic control.
Nothing cheaper substitutes.** We skipped the control once and spent a day
explaining a number that was entirely the flow's.

## E. What we would do differently

1. **Measure the between-seed spread before believing any intervention.** Every
   claim that rested on one run per condition was retracted; none replicated
   across ≥ 3 seeds was.
2. **Treat a mechanism as a hypothesis until its prediction is tested.** Two of
   the four retractions above are explanations that fit the data and were wrong.
   One died to a ten-minute test written one commit too late.
3. **Make the analytic control part of the estimator, not an optional check** —
   for the score-bias estimator *and* for SBC.
4. **Don't transplant hyperparameters between models.** [256,128] came from the
   DDM; the RDM only needed [64,32], which captures 58% of the gain for 18% of
   the cost.

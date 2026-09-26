# The recipe

Five steps, each with the measured effect that justifies it. Where a step
contradicts an obvious alternative, the alternative and its number are given.

---

## 1. Simulate at scale

**16 × 10⁶ rows, one trial per parameter draw.**

The budget buys θ-diversity, not trials per θ. Measured: no gain beyond ~40
trials per draw, and at P ≥ 25k the two axes trade freely.

Simulation is deterministic from the recorded seed — verified bit-for-bit,
16,000,000 rows, all columns identical, in 11.9 minutes. **Regenerate rather
than deposit**, and ship a checksum manifest so a user can prove their
regeneration matches.

## 2. Train ONE run at large batch, retaining many checkpoints

**Batch 1024, 30 checkpoints retained.**

Batch 1024 is 3× faster per epoch than 128 and makes a run's checkpoints more
homogeneous. It does **not** move the mean of the score bias — a claim of 5×
improvement was retracted after replication (t = 0.10), and the variance ratio
is 2.84× (p = 0.34) against a matched arm, not the 8.7× first reported.

Retain 30, not 10: roughly half the apparent *seed* variance is epoch-to-epoch
variance on the plateau, so screening a window from one run is ~5× cheaper than
training multiple seeds for the same spread.

## 3. SELECT the best checkpoint on score bias — never average

| approach | result |
|---|---|
| density-mixture ensemble | the mixture's score is the **mean** of member scores, so it cannot beat its members — at 4.3× the cost |
| weight averaging (SWA) | lands in the same place as the mixture |
| **selection** | **22× better than averaging on the DDM, 3.9× on the RDM** |

Averaging converges to the member mean by either route. Only selection finds the
best member. Screen on draws independent of those used to report.

## 4. Confirm with SBC — and always run the analytic control

Score bias ranks checkpoints *within* an architecture (ρ = −0.989 with SBC
rank). It does **not** rank across architectures: the first-order displacement
it implies has the wrong SIGN for `a` and for `t0` on the one flow whose SBC
curve is independently known (see `TRAPS.md`).

Before spending hours on a cell, run `evaluation/test_posterior_shift.R`
(~1 min per 7-parameter flow): it importance-reweights analytic-posterior draws
to the flow's posterior and returns SBC's own statistic. It reproduced the
shipped flow's t0 bias (+0.28 predicted, +0.319 observed), and a disagreement
between it and an SBC cell is how the st0 default bug was found.

**Run the same SBC cell with the analytic likelihood.** It is the only thing
that separates "the approximation is wrong" from "the inference problem is
hard". We skipped it once and spent a day explaining a number that turned out to
be entirely the flow's — the analytic's worst standardised bias on the same
design was 0.092.

Report: standardised bias against a |·| ≤ 0.5 gate, rank uniformity (KS), and
coverage.

## 5. Replicate across ≥ 3 seeds before believing any training intervention

Not boilerplate. Run-to-run variance at fixed hyperparameters is comparable to
any effect worth reporting.

**Every claim in this project that rested on one run per condition was later
retracted. None replicated across ≥ 3 seeds was.**

Two worked examples, both retracted: batch size shifting the mean (single run,
5× apparent, t = 0.10 on replication) and distillation shifting the mean (three
arms ordering monotonically, which three draws from the measured between-seed
spread do by chance 1 time in 6; the selected β's own run sat 3.7 sd from its
replication mean).

---

## What the artefacts look like when this is followed

| model | SBC | note |
|---|---|---|
| RDM `rdm_small` [64,32] | passes N=100 and N=400 | worst \|std bias\| 0.090, coverage 0.940–0.970 |
| DDM `cap256w_c4` | passes N=100–800 | t0 carries a +0.28% offset; see below |
| DDM regression baseline | **passes the gate** at N = 400, 250 reps (2026-09-27): worst +0.437 (`Z`); ranks non-uniform on v, Z, sv, SZ | t0 clean (−0.04); screen predicted Z +0.35 (hit), under-predicted v/sv/SZ by ~0.2 |
| DDM st0=0 | passes the gate | worst +0.282 at 500 reps after the st0-default fix (`TRAPS.md`); ranks non-uniform on a, t0, Z, coverage nominal |

**[64,32] captures 58% of the calibration gain of [256,128] for 18% of the
cost** on the RDM. Do not transplant architecture between models without
measuring — [256,128] came from the DDM and was only tested afterwards.

---

## The t0 correction (DDM, st0 estimated)

The residual bias is constant on the **log** scale, so the correction is
multiplicative:

```
t0_corrected = t0_hat * exp(-0.00276) = t0_hat * 0.99724
```

That is 0.28%, or 0.83 ms at a typical t0 = 0.3 s. A fixed millisecond
subtraction is the wrong shape — it would under-correct fast responders and
over-correct slow ones.

Validated leave-one-N-out (each cell corrected using only the others), it
removes the rank non-uniformity that made strict SBC fail:

| N | KS p before | after |
|---|---|---|
| 200 | 7.2e−04 | 0.33 |
| 400 | 6.7e−08 | 0.93 |
| 800 | 2.0e−06 | 0.40 |

Coverage was nominal throughout even before correction: **intervals were always
valid; only point estimates carried the offset.**

Caveat: calibrated and tested on the same four cells, leave-one-N-out only. An
independent cell at an N outside {100, 200, 400, 800} would close it.

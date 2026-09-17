# Censored-likelihood training for the DDM flow

Goal: remove the unsampled-tail distortion (the sole G1 blocker after
arch64w fixed the leading edge; st0 +2.2 sd at N = 800) by never asking the
flow to model the deep tail's *shape*, only its total mass.

## Two facts that shape the design

1. **The training data is already tail-truncated.** Max rt in the 1M
   training set is 20 s (simulator-side cutoff): the flow was trained on
   truncated data with an untruncated objective, so the missing tail mass
   is necessarily misallocated by normalization — a latent inconsistency
   the censored objective repairs for free.
2. **A 10 s censor is informative, not vacuous**: 0.34% of training trials
   (3,357) lie beyond it, so the censored term receives real gradient
   signal about P(rt ≥ C | context).

## Objective

With censor C (default 10 s), each training trial contributes

- rt < C:  log q(rt | context)          (unchanged density term)
- rt ≥ C:  log SF_q(C | context)        (survivor lump at the censor)

where SF_q(C) = 1 − Φ(g⁻¹(log C)) comes from the same single spline
inversion the evaluator already uses — no new machinery. The validation
loss uses the same objective so best-checkpoint selection is consistent.

The flow remains a proper density on (0, ∞). What changes is what the loss
constrains: exact shape below C, total mass above C, shape above C
unconstrained. Expected effects: (a) capacity and normalization pressure
concentrate below C where the data live; (b) the truncated-simulator
inconsistency disappears for the [10 s, 20 s] band; (c) the density between
~1 s and C — where the tail overshoot currently sits and where slow-context
observations occur — is now competing for normalization mass only against
correctly-lumped tail mass.

## Downstream consequences (deliberately: none)

The exporter, golden harness, R/C++ ports, EMC2 models, and gates are all
unchanged — censoring is purely a training-objective modification.
Documented caveat: the trained flow's density *above* C is unconstrained,
so fits to data containing rt ≥ C should censor those observations (EMC2's
UC machinery, e.g. UC = 10); in our study designs generated rts never
approach 10 s at the operating points used.

Second-order limitation: because the simulator truncated at 20 s, the
learned lump P(rt ≥ 10 | context) is really P(10 ≤ rt < 20 | context) for
contexts with true mass beyond 20 s (extreme slow corners of the box). If
that ever matters, the simulation should be regenerated with the censor
applied at source; for the operating region of interest the discrepancy is
negligible.

## Implementation

- `flow/src/flow/training.py`: `loss_fn_censored` (density term for
  observed, survivor term at C for censored, min-prob guard for invalid);
  `train_step_censored` / `eval_step_censored`; `censor=None` threaded
  through `train_model` → `train_one_epoch` / `validate_model` with the
  default reproducing the existing path exactly.
- `flow/scripts/train_ddm.py`: `--censor` flag, recorded in
  `variant_config.json`.
- Candidate: `arch64wC` = arch64w architecture (64 bins, [9, 64, 32])
  trained with `--censor 10`. Gates as usual; success criterion is st0/sv
  joining t0/a inside G1 while κ stays ≈ 1.

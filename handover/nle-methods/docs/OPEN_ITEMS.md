# Open items at handover

Status 2026-09-27. Nothing is outstanding; the rest are judgement calls that
do not block the document.

---

## 1. CLOSED: the direct-regression baseline's SBC cell

This was the one number the document promised and did not have; it has it now.

**CLOSED 2026-09-27.** The cell finished at 00:46 CEST after 3.8 days
(`results/sbc/reg_s402/DDMreg_N400_R250.rds`, sha256 in `ARTEFACTS.sha256`).
It passes the gate (worst +0.437 on `Z`, t0 −0.036) and fails rank uniformity
on v, Z, sv, SZ. The screen's prediction was right on the ranking (Z worst, t0
clean; Z within 0.09) and under-predicted v, sv, SZ, st0 by ~0.2 sd. Written to
the four places: `docs/METHODS.md` §4, `docs/RECIPE.md`, `README.md`,
`../../sbc/SBC_PLAN.md` Phase 38. Nothing in this package is now outstanding.

---

## 2. Judgement calls, not blockers

- **Which DDM artefact ships.** `cap256w_c4` remains the recommendation. The
  st0 = 0 build (`z0_s503_e282`) is now a real alternative: it passes the gate
  at 500 replicates with nominal coverage, and fails rank uniformity on `a`, t0
  and `Z` where the shipped flow fails on t0 alone. Neither is clean.
- **CLOSED — an independent cell for the t0 correction** (2026-09-24): N = 250,
  300 replicates, predicted delta 0.002764 against observed 0.002765; corrected
  standardised bias +0.000, KS p 0.22. Phase 27 is closed.
- **REOPENED 2026-10-13 — the t0 correction on the final sampler.** An exact test (draws saved,
  `sbc/t0corr_exact.R`, N = 800, 500 reps) finds that the fixed delta 0.00276 over-corrects: mean rank .534,
  KS p .007, out of the envelope. The rank-centring shift in that cell is about 0.0019 (in-sample). The
  2026-09-24 check used the normal-posterior rank approximation, which is not reliable here. An
  out-of-sample exact test of about 0.0019 is pending. Details are in ZERO_BOX_RETRAIN_PLAN.md, progress log.
- **`mlp_forward` through BLAS, at the next EMC2 rebase.** The evaluator's
  fixed cost is ~0.87 ms per call, all of it scalar-loop matrix multiplication
  in `port/src/flow_ddm.cpp` (§9 of `EMC2_INTEGRATION.md`); `dgemv_` should take
  it toward ~0.2 ms, and batching the conditioner would also close most of the
  57x gap on trial-varying designs (§10). **Deliberately deferred**: the flow
  integration is to be redone against the current `dev` EMC2, and this change
  needs both compiled copies rebuilt and the golden vectors re-run, so it
  belongs in that step rather than as a patch to the present build.
- **An ONNX path in EMC2 — for the published LANs, not for our flows.** Our
  flows do not need it (§9: the ONNX marginal cost equals the C++ port's, so it
  would buy only the fixed overhead that the BLAS fix addresses anyway). The
  There are two real arguments. **Capability**: `lnccbrown/HSSM` ships LANs for `angle`, `levy`,
  `ornstein`, `weibull`, `gamma_drift`, `ddm_sdv`, `ddm_uniform_st` and several
  race and LBA variants, and EMC2 has analytic likelihoods for almost none of
  them. An ONNX reader would make all of those available to EMC2 designs at
  once. **And trial-varying designs where no analytic exists**: a batched
  runtime evaluates a per-trial network at 14.6 ms per 1000 trials where a
  scalar per-row loop takes 839 (§10 of `EMC2_INTEGRATION.md`). For the DDM that
  is moot — WienR costs 5.2 ms — but for `angle`, `weibull` or any future flow
  for an intractable model there is nothing to fall back on, and this is the
  difference between a usable regression design and an unusable one. Batching
  the conditioner in `flow_ddm.cpp` wins the same thing for our own models; ONNX
  gets it for free for models we did not build. Two things to know before starting: those files have a **rank-1
  signature** (one trial per call), so they need either a per-call loop
  (measured: 8.0 ms per 1000 trials, call-overhead dominated) or a small graph
  edit to accept a batch dimension; and **their calibration is unknown** — the
  one we scored carries g on log a of −0.122, an order of magnitude worse than
  our flow's over the same region (`METHODS.md` §3.8), so anything reached this
  way needs its own SBC before it is trusted.
- **A compiled evaluator for the regression net.** Would turn a 5-day cell into
  a 3-hour one and make the baseline a practical artefact rather than a
  comparison. Note that a naive C++ loop will not beat reference BLAS by much;
  this needs an optimised BLAS or a vectorised library, and RcppEigen is not
  installed on tux26.
- **Phases 33 and 36 could be re-measured** on correctly specified cells. Their
  conclusions are already marked as resting on mis-specified ones, and §6 no
  longer depends on them, so this is for completeness only.

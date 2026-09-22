# Handover plan — neural likelihoods for EAMs

Three deliverables for Julian: a methods document, a clean OSF-ready code
directory, and a separate directory backing the blind-alley appendix.

Working notes live in `sbc/SBC_PLAN.md` (~3170 lines, 36 phases). That file is
the audit trail, not a deliverable — everything below is distilled from it, and
it should travel with the handover as provenance but not as reading.

---

## 1. The document

Ordered RDM-first deliberately: the race model introduces the flow method with
no classifier, so the core machinery is presented once, cleanly. The DDM section
then inherits it and adds only what is genuinely DDM-specific.

### 1.1 Introduction — what problem a neural likelihood solves

The honest framing, which the evidence forces:

| model | analytic | flow | verdict |
|---|---|---|---|
| DDM | 6,300–11,400 us/1000 | 660 us/1000 | flow wins ~10-17x |
| LNR | 4.0 min per SBC cell | 29.1 min | flow 7x SLOWER |
| RDM | 5.0 min per SBC cell | 56.7 min (185x per call at current capacity) | flow slower |

A neural likelihood pays only where the analytic is expensive — the DDM, whose
density is an infinite series with precision control. The race models have
closed-form first-passage densities. So the RDM appears here as a **generality
demonstration**, not a speedup, and the paper should say so in the first
paragraph of that section rather than let a reviewer find it.

### 1.2 RDM — the method, presented once

- **Training data.** Single-accumulator first-passage samples with the analytic
  log-density stored per row. Budget: 16e6 rows. One trial per parameter draw,
  so the budget is pure theta-diversity.
- **Architecture.** Spline flow over log rt conditioned on theta; conditioner
  MLP; no classifier — the race likelihood is assembled at inference as
  `f_winner(rt) * prod_j S_j(rt)`, and the flow supplies both density and
  survivor.
- **Capacity.** [64,32] with 64 bins. Justify empirically: it captures 58% of
  the calibration gain of [256,128] for 18% of the cost. Do NOT present
  [256,128] as the recommendation — it was transplanted from the DDM and tested
  only afterwards.
- **Evaluation — the part to foreground.** This is the paper's methodological
  contribution and deserves more space than the architecture:
  - *Score bias* `g = d/dtheta E_true[log f_hat]`, zero for a correct
    likelihood, measured over draws from the SBC prior. Cheap (minutes).
  - *Its analytic control*: the same estimator on the analytic likelihood must
    return ~0. Without this the estimator cannot be trusted, and ours was
    silently biased until the control was checked (see traps).
  - *SBC* as the verdict: rank uniformity, coverage, standardised bias.
  - *The relationship between them*: score bias ranks models **within an
    architecture** but does not predict absolute SBC bias **across**
    architectures, because the displacement is multivariate (`I^-1 g`).
- **Results.** rdm_small vs rdm_large: total |g| 0.0126 vs 0.0563 (4.5x), with
  the improvement concentrated in B and t0, the two worst parameters. **SBC now
  confirms it at N = 100 AND N = 400** (500 replicates each): every parameter
  inside the gate, every rank distribution uniform, coverage 0.940-0.970. t0
  standardised bias −0.013 and −0.001 — a 12x improvement on rdm_large's +0.152
  and closer to zero than the ANALYTIC's own +0.039. Comparison against Malte's
  RDM likelihood on **calibration**.

### 1.3 DDM — inherits the above, adds the choice classifier

- **What is different.** The likelihood factorises as
  `P(R|theta) x f(rt|theta,R)`, so a separate choice classifier is needed. The
  flow never sees it during plain-MLE training.
- **The classifier matters more than it looks.** It carries 70-98% of the score
  bias on v and Z. Its target is available exactly and cheaply (`WienR::pWDM`),
  and — importantly — it is a **value** problem, not a gradient problem:

      g_clf,j = dP2/dtheta_j * [P2/P2_hat - P1/P1_hat]

  which vanishes identically when the value is right, whatever the gradient.
  So plain regression on an accurate target is both sufficient and necessary;
  gradient matching is the wrong tool.
- **Reporting the components.** Always report the flow and classifier shares of
  the score bias separately. A composite model can show near-zero aggregate bias
  while both components are substantially wrong (measured: `a` = +0.0010 was
  flow +0.0121 cancelling classifier −0.0111), and that cancellation does not
  survive a change of seed.
- **Results.** cap256w_c4 passes SBC on every parameter from N=100 to N=800;
  worst is t0 at +0.371 (N=800) against a 0.5 gate. Coverage nominal throughout,
  so intervals are valid and point estimates carry a ~0.28% offset
  (0.8 ms at t0 = 0.3 s). Analytic
  control at N=100 and N=400 establishes the drift is the flow's, not the
  pipeline's.
- **The residual, stated plainly.** A systematic mean of −0.0343 on `a`
  (t = −3.53, p = 0.006 over ten matched seed-runs, 89/100 checkpoints negative)
  survives selection and batch size, which act only on the spread. The t0
  residual is NOT universal: it is robust at batch 1024 (−0.070, 49/50) but
  absent in the matched batch-128 arm (−0.008, 2 of 5 seeds positive), pooled
  −0.039 at p = 0.057. Distillation is the one intervention that moves either
  centre.

### 1.4 The recipe

1. Simulate at scale (16e6 rows).
2. Train ONE run at large batch, retaining many checkpoints.
3. **Select** the best checkpoint on score bias. Never average — averaging
   converges to the member mean by either route (density mixture or weight
   averaging), while selection finds the best member: 22x better on the DDM,
   3.9x on the RDM.
4. Confirm the selected model with one SBC run.
5. Report the selection explicitly, and screen on draws independent of those
   used to report.

With justification for each step and its measured effect size.

### 1.5 Bringing it into EMC2 — for Julian specifically

- The `dev-nn` branch (`ampl-psych/EMC2`, commit 85dde2a6) carries the compiled
  port and the `DDMnn` / `RDMnn` / `LNRnn` model functions.
- Weights live in `<EMC2>/extdata/flownn/*.rds`; the filenames are **hardcoded**
  in `ddmnn_get()` / `flownn_get()`, so swapping a model means replacing the
  file, not adding one.
- **`dRDMnn` hardcodes the context order** `log(cbind(v, B, t0, s, A))` and
  ignores the export's `context_names`. Use `permute_context.R` for a flow
  trained on any other order.
- Install into a dedicated library, never the default user library: the dev-nn
  build reports the same version (3.4.1) as the plain one, so they overwrite
  each other silently.
- How to rebuild a bundle, swap it, and verify (`make_ens5_rds.R`, the total-mass
  check, the golden-vector validation).

### 1.6 Appendix — "The Long and Winding Road"

Tone: advice to someone about to have the same idea, not confession. Each entry
states the idea, why it is appealing, what was measured, and why it fails. Two
sections, because they fail in different ways.

**A. Ideas that do not work**

| idea | why it appeals | what happens |
|---|---|---|
| Score penalty in the loss | drives E[score] to zero directly | satisfiable by flattening log f in theta; since I = −E[grad^2 log f], flattening destroys curvature (kappa 1.027 -> 0.46). Unbounded versions are exploited outright |
| Density-mixture ensembling | averaging should cancel error | the mixture's score is the MEAN of member scores (−0.0398 vs member mean −0.0393), so it cannot beat its members, at 4.3x the cost |
| Weight averaging (SWA) | plateau-wandering is what SWA is for | lands at the same place as the mixture (−0.0473). Averaging converges to the mean by either route; only selection finds the best member |
| SZ0/SZ/SZ1 reparameterisation | removes a non-differentiable kink | cannot widen coverage — `sw <= 2 min(w, 1−w)` is algebraically forced — and made bias on `a` 32x worse |
| Fixing the diffusion coefficient | the box carries an exact (v,a,s) redundancy | the network already respects the invariance: the derivative along the null direction is 2% of the bias on `a` |
| Edge-weighted loss | the leading edge is where the error is | monotone harm: +0.009 / +0.471 / +0.991 / +1.53 at amp 0/3/10/30. Upweighting collapses the effective sample size |
| More capacity / more bins | the obvious first move | spline bins never bind; [512,256] was worse than [256,128] |
| More trials per parameter draw | more data per theta | no gain beyond 40 |
| Larger batch to shift the bias | one run showed a 5x improvement | replication over 5 seeds: mean shift t = 0.10, and against a MATCHED same-machine arm the variance ratio is 2.84x (p = 0.34), not the 8.7x first reported. It buys reproducibility and 3x epoch speed, not calibration |

**B. Ways to get a wrong number and not know it**

More valuable than section A, and genuinely objective — these are properties of
the method, not of any particular team.

| trap | how it presents | how to detect |
|---|---|---|
| Score-bias estimator with no analytic control | plausible numbers, wrong by 25-40% | run the estimator on the analytic likelihood; it must return ~0 |
| Finite-difference step too large | O(H^2) truncation masquerading as signal | halve H; the control should fall 4x. At H=0.05 the control was 14% of the DDM signal, at H=0.01 under 1% |
| Omitting the integration measure | E[.] computed without trapezoid widths | the analytic control returns −0.25 instead of 0 |
| Support-boundary parameters (t0) | estimator returns −Inf or silently wrong | restrict to a region past the smear where the target is `d/dtheta log P(A)`, not 0 |
| Comparing models with different auxiliary components | measures the classifier as much as the flow | splice a common classifier before comparing |
| Aggregate bias hiding component cancellation | near-zero total, both parts wrong | report flow and classifier shares separately |
| An analytic target computed wrongly | the "exact" reference is wrong, so a good model looks bad | our P(R) target omitted the s-scaling: mean error 0.055, and the original classifier was 6x better than we recorded |
| Context column order | ~500x worse score bias, uniform across checkpoints | uniformity across checkpoints is the tell — a badly trained model varies, a mis-wired one does not |
| Two installs of the same package version | one silently returns wrong likelihoods | check `find.package()`, not the version string |
| Port-vs-reference guards calibrated on a small model | a bigger network trips them on float32 accumulation | report the error DISTRIBUTION, not the max; check `dz` separately from `dlog_pdf` |

**C. What we would do differently** — one short section: measure before
transplanting hyperparameters between models; replicate before believing a
single-run difference; and treat the analytic control as part of the estimator,
not an optional check.

---

## 2. Clean directory (OSF-ready)

`handover/nle-methods/`. Everything needed to reproduce the method and nothing
else. Target: a reader can go from empty directory to a validated artefact.

```
nle-methods/
  README.md                  quickstart, the recipe, what each directory is for
  LICENSE
  simulation/
    designs.R                parameter boxes and EMC2 designs
    sim_functions.R          simulate_data and the per-model simulators
    sim_rdm.R                RDM single-accumulator training data
    sim_ddm.R                DDM training data (extracted from sim_code.R)
    sim_choice_prob.R        CORRECTED analytic P(R) with the s-scaling
  flow/                      python package + training/export
    pyproject.toml uv.lock README.md
    src/flow/                config, data, training, checkpoints
    scripts/train_race.py    RDM/LNR
    scripts/train_ddm.py     DDM
    scripts/retrain_classifier.py
    scripts/json_to_classifier_ckpt.py
    scripts/export_flow.py   race export
    scripts/export_flow_ddm.py
    scripts/bench_batch.py   batch-size/throughput benchmark
  port/                      float64 R + C++ evaluators
    R/flow_race.R R/flow_ddm.R
    src/flow_race.cpp src/flow_ddm.cpp
    tests/validate.R tests/validate_ddm.R
  evaluation/
    score_bias_race.R        race score bias + t0 + analytic control
    score_bias_draws.R       DDM score bias
    score_bias_t0.R          DDM t0 via the restricted region
    score_bias_regress.R     direct-regression baseline + normalisation check
    analyze_sbc.R            calibration tables, rank ECDF, mechanism diagnostics
    sbc_specs.R              SBC cell definitions
    run_sbc.R                generic SBC driver (generalised from SBCc4.R)
    permute_context.R        re-express a flow in another context order
    splice_classifier.R      put a common classifier into several exports
    make_bundle.R            build the EMC2 extdata RDS
  artefacts/
    rdm_best_weights.json    the selected RDM flow
    ddm_best_weights.json    the selected DDM flow + classifier
    bundles/                 EMC2-installable .rds files
  results/                   SBC .rds backing every claim in the document
  docs/
    RECIPE.md                the five-step recipe with effect sizes
    EMC2_INTEGRATION.md      section 1.5 as a standalone how-to
    TRAPS.md                 appendix B as a checklist
```

Excluded and why: the ~40 one-off `run_*_study_ddm.R` drivers, superseded
simulation variants, 236 export files (keep ~4), and the blind-alley code (which
goes to directory 3).

## 3. Blind-alley directory

`handover/long-and-winding-road/`. Backs every claim in the appendix, so a
sceptical reader can check any of them.

```
long-and-winding-road/
  README.md                  index: each alley -> its code, data and the number
  score-penalty/             loss_fn_score, lambda sweep, curvature measurements
  ensembling/                make_ens_target*.R, ensemble exports, the mean-of-scores result
  weight-averaging/          make_swa.R and the comparison against the mixture
  reparameterisation/        sim_reparam.R, reencode_triple.py, triple exports
  edge-weighting/            loss_fn_edge and the amp sweep
  batch-and-seed/            the 4-arm sweep, the 5-seed replication, variance decomposition
  measurement-traps/         the analytic-control demos, H-sweep, cancellation split,
                             the s-scaling target comparison, context-order demo
  results/                   the SBC and score-bias .rds for the above
  SBC_PLAN.md                the full audit trail, 36 phases
```

---

## 4. Prerequisites before handover

Ordered by whether they block the document. Status as of 2026-09-22.

**Blocking**

1. ~~Re-measure the DDM score bias at H = 0.01.~~ **DONE.** All headline numbers
   re-measured at H = 0.01 (analytic control on `a` now +0.00008, was +0.00187).
2. ~~RDM SBC cell on the selected [64,32] flow.~~ **DONE, twice** — N = 100 and
   N = 400, 500 replicates each, both passing on every criterion (Phases 26, 28).
3. ~~Settle the DDM artefact.~~ **SETTLED: ship `cap256w_c4`** (st0 estimated,
   Phase 27 multiplicative t0 correction). It is the only DDM flow that passes
   SBC. The direct-regression baseline remains the best-calibrated route on score
   bias (|g_a| + |g_t0| = 0.0143 vs 0.0738), but score bias does not predict the
   SBC verdict (Phases 33, 36), so that is not grounds to ship it unverified.
4. ~~The st0 = 0 build.~~ **FAILED**, as did every reduced parameterisation:
   st0 fixed (0.05, 0.01), st0 = 0 (Phases 32-34, analytic control passes so the
   failure is the flow's), and decision-time rt - t0 (Phase 35). None ships.
   All are documented in `long-and-winding-road/`.

**Non-blocking but worth having**

5. ~~Finish the batch-128 arm on tux26 for a same-machine variance control.~~
   **DONE** (Phase 25) — and it corrected two earlier claims.
6. Decide whether the direct-regression baseline is a section or an appendix
   item. It now has both g_a and g_t0 measured and wins on both, so it needs a
   considered answer rather than a footnote.
7. ~~One end-to-end rehearsal.~~ **DONE** (commit b092a2b): a clean copy
   surfaced 21 unresolved path references, all fixed; 0 unresolved references
   and 0 parse failures across 104 files, and `score_bias_draws.R` reproduces
   cap256w_c4's numbers inside the copy.

**Corrections folded in since this plan was written**

- The "~3 ms" t0 offset was a units error: the Phase 7 table's values are on the
  LOG scale, so the offset is 0.28% = 0.83 ms at t0 = 0.3 s. Nine occurrences
  fixed across this file and `SBC_PLAN.md`.
- The 8.7x batch variance ratio does not survive a matched control (2.84x,
  p = 0.34).
- The export-gate failures are a seed effect, not a batch effect (Fisher
  p = 0.32).
- LNR is out of scope; the paper covers DDM and RDM only.
- `SBCddm7.R` (and its `_analytic`/`6` siblings) look priors up BY NAME:
  `prior()` takes `pmean` positionally and `sampled_pars()` follows formula
  order, so a (Z, SZ, sv) formula would transpose two priors. Checked
  2026-09-22: every stored DDM SBC result has priors aligned to its columns.

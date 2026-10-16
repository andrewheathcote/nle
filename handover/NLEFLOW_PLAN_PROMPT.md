Run this in auto mode; if you are not in auto mode, stop at once and say so.

You are starting a planning stage for Andrew Heathcote (AH) on the repo andrewheathcote/nle, branch `dev-sbc-calibration`
(already checked out on this cloud instance). Question to answer: **is it worth building a separate R package, `nleFlow`**, that

1. learns a new neural likelihood flow (our recipe: conditional spline flow + choice classifier, score-bias screening, SBC),
2. exports flows and LANs to EMC2 cards and to ONNX, and
3. converts ONNX LANs (HSSM) to EMC2 cards?

The output is a plan and a recommendation (build / do not build / build a smaller thing), written to `handover/NLEFLOW_PLAN.md`.
Do not write package code in this stage beyond throw-away probes needed to answer a question.

## Read first (all in the repo)
- `handover/REPORT_RUSS_JULIAN.md`: what DDMnle is and how it is validated; Appendix A covers ONNX/EMC2/LAN.
- `handover/nle-methods/docs/METHODS.md`, `EMC2_INTEGRATION.md`, `RECIPE.md`, `TRAPS.md`: the recipe and the traps
  (context-order, dual-install, sampled vs natural scale, float32 vs float64 export gates).
- `handover/SCALED_DDM_PLAN.md`: the scaled network and its compiled EMC2 path.
- Training and export code: `handover/nle-methods/flow/` (Python/JAX: `train_ddm.py`, `retrain_classifier.py`,
  `export_flow_ddm.py`, `export_onnx_ddm.py`, `onnx_check.py`), `simulation/`, `port/R/` (R port, `ddm_scaled.R`) and `port/tests/`.
- `handover/emc2/`: EMC2 `dev-nle` source (tarball, version 3.4.1, install into its own library) and the compiled-DDMnle patch.
  The tarball contains the card format, the loader/evaluator in C++, `register_nn_model`, and `inst/scripts/onnx_to_card.py`
  (the ONNX-MLP to card converter used for the HSSM LAN in `sbc/lan_case5/`).
- `handover/MALTE_*` and `handover/EMC2_IMPLEMENTATION_BRIEF.md` for the other flow families we ingest (Malte's RDM/CRDM) and
  the ONNX-reader argument.

## What the plan must settle
1. **Scope and audience.** Who would use it (EMC2 users training their own models? HSSM users wanting EMC2? us only?) and what
   they would do that they cannot do now. Overlap with EMC2 itself, with HSSM/LANfactory and with BayesFlow. Where should each
   piece live (nleFlow, EMC2, a Python package)?
2. **Architecture.** Training is Python/JAX. Options: R package wrapping Python via reticulate; a Python package with an R
   front end; R-only export/convert tools plus a documented training recipe. Which parts must be R (card writer, validator,
   converters, SBC/score-bias diagnostics) and which can stay Python. Dependency and CRAN/installation burden, including
   compiled code and the "dedicated library" trap.
3. **Interfaces.** The card format (version it, document it), the model specification a user supplies (simulator, parameter
   transforms and bounds, context order), ONNX signatures (trial-wise vs shared-theta), the ONNX-to-card converter's supported
   graph patterns, and how validation (golden vectors, score bias, SBC) is built in so that a card cannot be exported without passing.
4. **What is reusable now** from the repo, what needs generalising away from the DDM/RDM (hard-coded context order, regimes,
   box, classifier), and what is missing (e.g. DDMnle's ONNX export with CDF difference for st0).
5. **Effort, risk, maintenance.** Rough work breakdown in days, the main risks (format drift against EMC2 `dev-nle` which is not
   yet on GitHub, ONNX operator coverage, JAX/numpy version churn), and who maintains it.
6. **Recommendation** with the smallest useful first milestone, and the decision AH needs to make next.

## Rules
- Auto mode. Do not touch tux26, Gadi or AWS; ask AH first about anything that costs money or compute.
- Do not push to the EMC2 repository (ampl-psych/EMC2) or any remote other than this repo. Work on a new branch
  `nleflow-plan` and push only that; do not push to `main`.
- Weights beyond the committed cards are not available here; do not try to retrain DDMnle (about 28 h on 24 cores).
- Cite file paths for every claim about the existing code; verify by reading, not from memory.
- Show AH the outline of `NLEFLOW_PLAN.md` before writing long prose. Report faithfully: what you checked, what you did not.
- Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- At the end hand over ONE next stage: a model (Sonnet 4.6 or above) and a prompt, in auto mode.

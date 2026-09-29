# SBC: malte_rdm_aff_log_h2_k12_clip_100k.json vs RDM (M6a a0 malte_rdm_aff_log_h2_k12_clip_100k)

| | |
|---|---|
| **Date** | 2026-09-29 |
| **Branch / commit** | `dev-sbc-calibration` @ `?` (EMC2 3.4.1.9001 @ /Users/andrew/R/nle-sbc-lib/EMC2) |
| **Models** | malte_rdm_aff_log_h2_k12_clip_100k.json (neural likelihood, flow_race, sha256 abd249f8fcda21fc8f1283abef6abb7047a8221b0e719be6c385101aff550e3b) and control RDM; single subject |
| **Why** | TODO |
| **Where run** | Gadi |
| **Verdict** | TODO (not run yet) |
| **Report** | none |

## Design
Factors: subjects = 1; subjects = 1; S = left/right; Rlevels: left/right; formula: v ~ lM, B ~ 1, t0 ~ 1, A ~ 1, s ~ 1; constants (sampled scale): s = 0, A = -Inf. 400 trials per replicate.

## Priors
Independent normal on the sampled scale (built by `nn_cell()`); the same prior for the control.

| parameter | mean | sd |
|---|---|---|
| v |  0.405 | 0.1 |
| v_lMTRUE |  0.400 | 0.1 |
| B |  0.000 | 0.1 |
| t0 | -1.609 | 0.1 |

Natural-scale range at +-4 sd vs the training region:

| parameter | cell | prior | region | inside |
|---|---|---|---|---|
| v | lM=TRUE | [1.271,  3.94] | [    0,     8] | TRUE |
| v | lM=FALSE | [1.005, 2.238] | [    0,     8] | TRUE |
| s |  | [    1,     1] | [ 0.25,   3.5] | TRUE |
| B |  | [0.6703, 1.492] | [ 0.25,   3.5] | TRUE |

## Conditions
One cell: 250 replicates x 400 trials, network and control. Seeds: run_sbc() defaults.

## Fitting
run_sbc() arguments: cores_per_chain = 48L; cores_for_chains = 1L.

## Result summary
TODO (bundle prepared, not run yet).

## Files
`bundle/`: `cell.rds` (design and prior of both cells), `run_cell.R` (re-runs the cell), `session.txt`
(EMC2 build, BLAS, artefact line, sessionInfo). `results/`: `sbc_nn.RData`, `sbc_control.RData`
(run_sbc() output: SBC, prior_alpha), `summary.csv`, `ecdf_nn.pdf`, `ecdf_control.pdf`.

## Not covered
Hierarchical models; parameter regions outside this prior (the rest of the training region); other
designs (factors, covariates); censored or truncated data (refused for neural likelihoods);
identifiability (SBC passes by construction for an unidentified parameter: add parameter recovery
and likelihood profiles).

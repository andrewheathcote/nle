# SBC: malte_crdm_aff_log_h2_k12_clip_100k_dt0.0005.json vs CRDM (M6a crdm_net malte_crdm_aff_log_h2_k12_clip_100k_dt0.0005)

| | |
|---|---|
| **Date** | 2026-09-29 |
| **Branch / commit** | `dev-sbc-calibration` @ `?` (EMC2 3.4.1.9001 @ /Users/andrew/R/nle-sbc-lib/EMC2) |
| **Models** | malte_crdm_aff_log_h2_k12_clip_100k_dt0.0005.json (neural likelihood, flow_race, sha256 1241725b20439475f5a37ac64d58212c181c08488b01c1e15835092b4f7ebbe0) and control CRDM; single subject |
| **Why** | TODO |
| **Where run** | Gadi |
| **Verdict** | TODO (not run yet) |
| **Report** | none |

## Design
Factors: subjects = 1; S = left/right; D = left/right; Rlevels: left/right; formula: v ~ lM, B ~ 1, t0 ~ 1, tau ~ 1, amp ~ 0 + lD, A ~ 1, s ~ 1; constants (sampled scale): amp_lDFALSE = -Inf, A = -Inf, s = 0. 400 trials per replicate.

## Priors
Independent normal on the sampled scale (built by `nn_cell()`); the same prior for the control.

| parameter | mean | sd |
|---|---|---|
| v |  0.693 | 0.15 |
| v_lMTRUE |  0.500 | 0.10 |
| B |  0.000 | 0.15 |
| t0 | -1.204 | 0.10 |
| tau | -2.303 | 0.20 |
| amp_lDTRUE | -0.916 | 0.15 |

Natural-scale range at +-4 sd vs the training region:

| parameter | cell | prior | region | inside |
|---|---|---|---|---|
| v | lM=TRUE | [1.603, 6.782] | [    0,     8] | TRUE |
| v | lM=FALSE | [1.098, 3.644] | [    0,     8] | TRUE |
| amp | lD=TRUE | [0.2195, 0.7288] | [    0,     1] | TRUE |
| amp | lD=FALSE | [    0,     0] | [    0,     1] | TRUE |
| tau |  | [0.04493, 0.2226] | [ 0.01,   0.5] | TRUE |
| s |  | [    1,     1] | [ 0.25,     3] | TRUE |
| B |  | [0.5488, 1.822] | [ 0.25,     3] | TRUE |

## Conditions
One cell: 500 replicates x 400 trials, network and control. Seeds: run_sbc() defaults.

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

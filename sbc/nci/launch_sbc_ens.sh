#!/bin/bash
# Phase 3: SBC of the production ens5 DDMnn likelihood.
# Run from the directory holding these files (/scratch/tg53/ajh575/emc2nn/nci).
#
# Prerequisite (once): EMC2 dev-nn at >= 85dde2a6 installed into the R 4.5.0
# library on Gadi — the Phase 1 install predates the ensemble.
#   module load gcc/12.2.0 intel-compiler/2021.10.0 R/4.5.0
#   R CMD INSTALL /path/to/EMC2      # dev-nn checkout
# SBCens.R aborts on a stale build (stopifnot on ddm_ens_eval_trials_cpp).
#
# Submits the three cells as concurrent jobs (8/10/16 h). The all-in-one
# fallback is SBCens.pbs (24 h, runs all three sequentially).
#
# Track with:  qstat -u $USER    (job names SBCens100/200/400)
# Results land in ./results_ens5/DDMnn_N<trials>_R500.rds; logs SBCens_N*.log
# Predictions registered in sbc/SBC_PLAN.md, Phase 3.

set -e
for job in SBCens_N100.pbs SBCens_N200.pbs SBCens_N400.pbs; do
  id=$(qsub "$job")
  echo "submitted $job -> $id"
done

# NCI: SBC of the production 5-seed ensemble DDM likelihood (DDMnn = ens5
# after EMC2 dev-nn commit 85dde2a6) at N = 100, 200, 400.
# Reuses the vignette-derived cell specs; results land in results_ens5/.
#
# Usage:  Rscript SBCens.R [trials ...]
#   no args -> runs all three cells sequentially (SBCens.pbs, 24 h)
#   args    -> runs just those cells, e.g. `Rscript SBCens.R 400`
#              (SBCens_N100/N200/N400.pbs run the cells concurrently)
# Predictions for this campaign are registered in sbc/SBC_PLAN.md, Phase 3.

rm(list = ls())
library(EMC2)
source("sbc_specs.R")

stopifnot("ddm_ens_eval_trials_cpp" %in% ls(getNamespace("EMC2")))  # right build?

args <- commandArgs(trailingOnly = TRUE)
trials_grid <- if (length(args)) as.integer(args) else c(100L, 200L, 400L)
stopifnot(!anyNA(trials_grid), all(trials_grid > 0))
cat("ens5 SBC cells to run:", paste(trials_grid, collapse = ", "), "\n")

for (trials in trials_grid)
  run_sbc_cell("DDMnn", trials, replicates = 500, cores = 48,
               out_dir = "results_ens5")

cat("\nAll requested ens5 SBC cells complete.\n")

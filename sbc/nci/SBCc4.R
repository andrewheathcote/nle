# SBC of the corrected-classifier DDM likelihood (c4) at N = 100, 200, 400.
#
# DDMnn evaluates whatever sits in <EMC2>/extdata/flownn/ddm_ens5.rds, and that
# name is hardcoded. So the bundle MUST be swapped before submitting, and the
# single-flow and 5-member runs CANNOT run concurrently against the same EMC2
# install - they would read each other's weights. Run one, then the other.
#
# Usage:  Rscript SBCc4.R <out_dir> [trials ...]
#   e.g.  Rscript SBCc4.R results_c4single 400
# Predictions registered in sbc/SBC_PLAN.md, Phase 7.

rm(list = ls())

# The dev-nn EMC2 lives in its own library so it cannot be clobbered by, or
# clobber, the plain EMC2 in the default user library. That is not fastidious:
# the dev-nn build WAS overwritten by a plain 3.4.1 in the default library,
# which is why this campaign could not start. Prepending here means only these
# jobs see it.
.lib <- Sys.getenv("EMC2NN_LIB", "")
if (nzchar(.lib)) {
  stopifnot(dir.exists(.lib))
  .libPaths(c(.lib, .libPaths()))
}
library(EMC2)
cat("EMC2:", find.package("EMC2"), as.character(packageVersion("EMC2")), "\n")
source("sbc_specs.R")

stopifnot("ddm_ens_eval_trials_cpp" %in% ls(getNamespace("EMC2")))  # right build?

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) >= 1)
out_dir <- args[1]
trials_grid <- if (length(args) > 1) as.integer(args[-1]) else c(100L, 200L, 400L)
stopifnot(!anyNA(trials_grid), all(trials_grid > 0))

# record WHICH bundle is installed, so a result can never be misattributed
b <- readRDS(system.file("extdata", "flownn", "ddm_ens5.rds",
                         package = "EMC2", mustWork = TRUE))
cat("installed bundle:", length(if (is.null(b$members)) list(b) else b$members),
    "member(s):", paste(b$member_tags, collapse = ", "), "\n")
cat("out_dir:", out_dir, " cells:", paste(trials_grid, collapse = ", "), "\n")

# Gadi hardcoded 48 (one node). Elsewhere, take it from SBC_CORES or leave one
# core free. run_sbc parallelises over replicates, so more cores is ~linear.
n_cores <- as.integer(Sys.getenv("SBC_CORES", "0"))
if (is.na(n_cores) || n_cores < 1)
  n_cores <- max(1L, parallel::detectCores() - 1L)
cat("cores:", n_cores, "\n")

# 500 is the registered default; SBC_REPS lowers it for exploratory cells where
# wall time binds (se of the standardised bias scales as 1/sqrt(reps): 0.045 at
# 500, 0.058 at 300). Anything below 500 is NOT the registered configuration and
# must be reported with its replicate count.
n_reps <- as.integer(Sys.getenv("SBC_REPS", "500"))
stopifnot(!is.na(n_reps), n_reps > 0)
cat("replicates:", n_reps, if (n_reps != 500L) " (NON-STANDARD)" else "", "\n")

for (trials in trials_grid)
  run_sbc_cell("DDMnn", trials, replicates = n_reps, cores = n_cores, out_dir = out_dir)

cat("\nAll requested c4 SBC cells complete.\n")

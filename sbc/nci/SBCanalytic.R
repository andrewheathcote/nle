# Analytic DDM control cell — does the EXACT likelihood stay calibrated on t0 at
# the N where the flow drifts? At N=100 the analytic shows t0 bias -0.015 against
# the flow's +0.118; the flow grows to +0.371 by N=800 and we have no analytic
# trend to compare. This closes that off.
#
#   Rscript SBCanalytic.R <out_dir> <trials> [reps]
# Env: SBC_CORES
rm(list = ls())
.lib <- Sys.getenv("EMC2NN_LIB", "")
if (nzchar(.lib) && dir.exists(.lib)) .libPaths(c(.lib, .libPaths()))
library(EMC2)
source("sbc_specs.R")
a <- commandArgs(trailingOnly = TRUE)
out_dir <- a[1]; trials <- as.integer(a[2])
reps <- if (length(a) > 2) as.integer(a[3]) else 500L
cores <- as.integer(Sys.getenv("SBC_CORES", "4"))
cat(sprintf("ANALYTIC DDM  trials=%d  reps=%d  cores=%d  -> %s\n",
            trials, reps, cores, out_dir))
if (reps != 500L) cat("  NOTE: non-standard replicate count; se scales as 1/sqrt(reps)\n")
run_sbc_cell("DDM", trials, replicates = reps, cores = cores, out_dir = out_dir)
cat("\nanalytic control cell complete.\n")

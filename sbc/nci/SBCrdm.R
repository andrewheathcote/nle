# SBC for the refined RDM flow.
#
# RDMnn evaluates whatever sits in <EMC2>/extdata/flownn/rdm_large.rds and that
# name is hardcoded, so the bundle must be swapped in place. The original is
# preserved at sbc/rdm_large_ORIGINAL.rds.
#
#   Rscript sbc/nci/SBCrdm.R <out_dir> <trials> [reps]
# Env: SBC_CORES
rm(list = ls())
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
library(EMC2)
setwd("sbc/nci"); source("sbc_specs.R"); setwd("../..")
a <- commandArgs(trailingOnly = TRUE)
out_dir <- a[1]; trials <- as.integer(a[2])
reps <- if (length(a) > 2) as.integer(a[3]) else 500L
cores <- as.integer(Sys.getenv("SBC_CORES", "8"))
# record WHICH flow is installed so a result cannot be misattributed
b <- readRDS(system.file("extdata", "flownn", "rdm_large.rds", package = "EMC2", mustWork = TRUE))
cat(sprintf("installed RDM flow: budget=%s  step=%s  bins=%d  context=%s\n",
            b$budget, as.character(b$checkpoint_step), b$spline$num_bins,
            paste(b$context_names, collapse = ",")))
cat(sprintf("RDMnn SBC  trials=%d  reps=%d  cores=%d -> %s\n", trials, reps, cores, out_dir))
run_sbc_cell("RDMnn", trials, replicates = reps, cores = cores, out_dir = out_dir)
cat("\nRDM SBC complete.\n")

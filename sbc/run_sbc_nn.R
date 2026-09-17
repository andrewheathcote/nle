# Single-participant SBC for the nn-likelihood models and their analytic
# twins — laptop entry point. Cell definitions are shared with the NCI
# scripts (sbc/nci/sbc_specs.R); see sbc/SBC_PLAN.md for the full design.
#
#   Rscript sbc/run_sbc_nn.R <model> <trials> <replicates> <cores> [out_dir]
#   e.g.  Rscript sbc/run_sbc_nn.R DDMnn 100 50 9 sbc/results
#
# Models: LNR, LNRnn, RDM, RDMnn, DDM, DDMnn.

# prefer the dedicated nn-EMC2 library (nle/.Rlibs) so installs of other
# EMC2 branches into the default library cannot affect these studies
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(EMC2))
source("sbc/nci/sbc_specs.R")

args <- commandArgs(trailingOnly = TRUE)
model_name <- if (length(args) >= 1) args[1] else "DDMnn"
trials <- if (length(args) >= 2) as.integer(args[2]) else 100L
replicates <- if (length(args) >= 3) as.integer(args[3]) else 500L
cores <- if (length(args) >= 4) as.integer(args[4]) else 9L
out_dir <- if (length(args) >= 5) args[5] else "sbc/results"

run_sbc_cell(model_name, trials, replicates, cores, out_dir)

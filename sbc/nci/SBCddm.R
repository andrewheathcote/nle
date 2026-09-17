# NCI: the slow cell — analytic DDM at N = 100, 500 replicates.
# ~26 min per replicate on one core (WienR likelihood): ~4.5 h on 48 cores.

rm(list = ls())
library(EMC2)
source("sbc_specs.R")

run_sbc_cell("DDM", trials = 100, replicates = 500, cores = 48,
             out_dir = "results")

cat("\nAnalytic DDM SBC cell complete.\n")

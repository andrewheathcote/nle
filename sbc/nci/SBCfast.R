# NCI: all fast SBC cells (race models + nn models), run sequentially;
# each cell parallelizes over datasets with cores_per_chain = 48.
# The slow analytic-DDM cell runs separately (SBCddm.R / SBCddm.pbs).
#
# Estimated total ~4-5 h on 48 cores (see sbc/SBC_PLAN.md).

rm(list = ls())
library(EMC2)
source("sbc_specs.R")

REPLICATES <- 500
CORES <- 48
OUT <- "results"

cells <- list(
  list(model = "LNR", trials = 100),
  list(model = "LNRnn", trials = 100),
  list(model = "RDM", trials = 100),
  list(model = "RDMnn", trials = 100),
  list(model = "DDMnn", trials = 100),
  list(model = "DDMnn", trials = 200),
  list(model = "DDMnn", trials = 400)
)

for (cell in cells)
  run_sbc_cell(cell$model, cell$trials, REPLICATES, CORES, OUT)

cat("\nAll fast SBC cells complete.\n")

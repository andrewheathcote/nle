# Validation set for the st0 = 0 build.
#
# REQUIRED, not optional. The 8-parameter val set at data/val/DDM was generated
# with st0 sampled from [0.01, 0.49], so its rt values and its stored `ll` both
# reflect a smeared leading edge. Scoring an st0 = 0 flow against it measures
# the model against the wrong distribution -- and because the retained
# checkpoint window is chosen by best val_loss, it would actively select
# checkpoints that OVER-SMOOTH the leading edge, which is the exact failure
# mode this build exists to test.
#
#   Rscript handover/st0_zero/sim_val_st0zero.R
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
source("simulation/sim_functions.R")
source("simulation/designs_st0zero.R")

VAL_SEED <- 1656      # same seed as the 8-parameter val set (simulation/sim_code.R)
VAL_SIZE <- 10000
out <- file.path("data", "val", "DDM_st0zero")
cat(sprintf("=== validation: %d draws -> %s ===\n", VAL_SIZE, out))
# n_params = VAL_SIZE/2 with n_trials = 1 per stimulus gives VAL_SIZE rows,
# matching simulation/sim_code.R's DDM validation call exactly.
simulate_data(param_namesDDM0, lowerDDM0, upperDDM0, designDDM0,
              n_params = VAL_SIZE / 2, transforms = transformsDDM0,
              out_dir = out, chunk_size = 1e5, seed = VAL_SEED,
              n_trials = 1)
cat("=== done ===\n")

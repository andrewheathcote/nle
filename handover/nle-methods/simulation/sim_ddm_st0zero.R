# Training data for the st0 = 0 DDM flow: 400k draws x 40 rows = 16e6 rows.
#
# Identical to simulation/sim_ladder16m.R except for the 7-parameter design.
# Same seed, same draw count, same rows-per-draw, so the only difference from
# the shipped 8-parameter training set is that st0 is 0 rather than sampled.
#
# Run from the nle repo root:
#   Rscript handover/st0_zero/sim_ddm_st0zero.R
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
source("simulation/sim_functions.R")
source("handover/st0_zero/designs_st0zero.R")

TRAIN_SEED <- 3876        # same seed as the 8-parameter build
N_PARAMS   <- 400000L
N_TRIALS   <- 20L         # per stimulus -> 40 rows each -> 16,000,000 rows

out <- file.path("data", "train", "st0zero400k", "DDM")
cat(sprintf("=== %d draws x %d rows = %d rows -> %s ===\n",
            N_PARAMS, 2 * N_TRIALS, N_PARAMS * 2 * N_TRIALS, out))
t0 <- proc.time()[3]
simulate_data(param_namesDDM0, lowerDDM0, upperDDM0, designDDM0,
              n_params = N_PARAMS, transforms = transformsDDM0,
              out_dir = out, chunk_size = 1e5, seed = TRAIN_SEED,
              n_trials = N_TRIALS)
cat(sprintf("=== done in %.1f min ===\n", (proc.time()[3] - t0) / 60))

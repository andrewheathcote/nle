# Score-augmented VALIDATION set for --score-match. The standard val set has no
# score_<par> columns, and the loader requests the same target columns for both
# splits. Different seed from training (VAL_SEED), 1000 draws x 40 rows.
source("simulation/sim_functions_score.R")
source("simulation/designs.R")
VAL_SEED <- 1656
out <- file.path("data", "val_score", "DDM")
cat(sprintf("=== 1000 draws x 40 rows -> %s ===\n", out))
t0 <- proc.time()[3]
simulate_data(param_namesDDM, lowerDDM, upperDDM, designDDM,
              n_params = 1000L, transforms = transformsDDM,
              out_dir = out, chunk_size = 1e5, seed = VAL_SEED,
              n_trials = 20L, sim_fun = simulate_trials_score, n_cores = 3L)
cat(sprintf("=== done in %.1f min ===\n", (proc.time()[3] - t0) / 60))

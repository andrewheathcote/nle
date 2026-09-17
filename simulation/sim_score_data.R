# Training set with true per-trial scores, for the exact-score-matching test.
# Matches the tpt40 configuration (25,000 draws x 40 rows = 1,000,000) so it is
# directly comparable to the flow already trained on that data.
#
# Run from the nle repo root:  Rscript simulation/sim_score_data.R

source("simulation/sim_functions_score.R")
source("simulation/designs.R")

TRAIN_SEED <- 3876
out <- file.path("data", "train", "score40", "DDM")
cat(sprintf("=== 25000 draws x 40 rows -> %s ===\n", out))
t0 <- proc.time()[3]
simulate_data(param_namesDDM, lowerDDM, upperDDM, designDDM,
              n_params = as.integer(Sys.getenv("SCORE_N_PARAMS", "25000")), transforms = transformsDDM,
              out_dir = out, chunk_size = 1e5, seed = TRAIN_SEED,
              n_trials = 20L, sim_fun = simulate_trials_score,
              n_cores = 3L)   # leave cores for the 4M training runs
cat(sprintf("=== done in %.1f min ===\n", (proc.time()[3] - t0) / 60))

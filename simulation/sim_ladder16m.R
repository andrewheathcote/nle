# Budget ladder at fixed trials-per-draw (N = 40 rows per parameter set).
#
# tpt40 (25k draws x 40 = 1M rows) matched arch64w (500k x 2 = 1M) on every
# diagnostic - inside the seed band on all six parameters, worst G1 bias 1.81
# vs 2.23, kappa 1.014 vs 1.027 - at 1/20th the parameter-space coverage. So
# within P >= 25k the two axes trade freely, which makes N=40 affordable at
# large P. This rung asks whether more TOTAL data now lowers the bias.
#
# That question is open rather than settled: the July budget arm found bias
# insensitive to a 20x data increase (kappa 0.48/0.81/0.93 but a/t0 bias
# slopes +0.26/+0.38, "capacity floor"), but it was run at 32 spline bins,
# before arch64w. tpt40's result suggests the floor has moved.
#
# Run from the nle repo root:  Rscript simulation/sim_ladder.R

source("simulation/sim_functions.R")
source("simulation/designs.R")

TRAIN_SEED <- 3876
N_PARAMS <- 400000L
N_TRIALS <- 20L          # per stimulus -> 40 rows each -> 4,000,000 rows

out <- file.path("data", "train", "tpt40p400k", "DDM")
cat(sprintf("=== %d draws x %d rows = %d rows -> %s ===\n",
            N_PARAMS, 2 * N_TRIALS, N_PARAMS * 2 * N_TRIALS, out))
t0 <- proc.time()[3]
simulate_data(param_namesDDM, lowerDDM, upperDDM, designDDM,
              n_params = N_PARAMS, transforms = transformsDDM,
              out_dir = out, chunk_size = 1e5, seed = TRAIN_SEED,
              n_trials = N_TRIALS)
cat(sprintf("=== done in %.1f min ===\n", (proc.time()[3] - t0) / 60))

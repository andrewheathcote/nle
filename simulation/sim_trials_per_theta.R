# Trials-per-parameter-draw arm: does the STRUCTURE of the training budget
# matter, holding the total number of simulated trials fixed?
#
# Every existing dataset has exactly 2 rows per parameter draw (n_trials = 1
# per stimulus, 2 stimuli). The budget arm varied the NUMBER of draws
# (25k/100k/500k) at 2 rows each, so it never tested this axis.
#
# Three arms, all 1,000,000 rows, i.e. identical simulation cost:
#   500,000 x  2   (the existing `large` set)
#    25,000 x 40
#    10,000 x 100
#
# Prediction (to be falsified): more trials per draw will NOT help and may
# hurt. The training objective is E_theta E_rt[-log f], whose estimator
# variance is Var_theta(inner)/n_theta + E[Var_rt]/N_total; at fixed N_total,
# raising trials/draw leaves the second term alone and inflates the first.
# For a conditional density over an 8-dim box, theta COVERAGE is what
# generalisation needs: 500k^(1/8) = 6.9 points per dimension vs 10k^(1/8)
# = 3.2. The contrary intuition is that with only 2 draws the flow learns
# nothing about the SHAPE of p(.|theta) at any theta and must rely entirely
# on smoothness across theta - which could matter where the density moves
# fast in theta (t0, st0 shift the leading edge sharply).
#
# Run from the nle repo root:  Rscript simulation/sim_trials_per_theta.R

source("simulation/sim_functions.R")
source("simulation/designs.R")

TRAIN_SEED <- 3876          # same seed as the main training sets
TOTAL <- 1e6                # rows, matched to the `large` budget

ARMS <- list(
  list(tag = "tpt40",  n_trials = 20L),   # 25,000 draws x 40 rows
  list(tag = "tpt100", n_trials = 50L)    # 10,000 draws x 100 rows
)

for (a in ARMS) {
  n_params <- TOTAL / (2 * a$n_trials)    # 2 stimuli per trial
  out <- file.path("data", "train", a$tag, "DDM")
  cat(sprintf("\n=== %s: %d draws x %d trials/stimulus = %d rows -> %s ===\n",
              a$tag, n_params, a$n_trials, n_params * 2 * a$n_trials, out))
  t0 <- proc.time()[3]
  simulate_data(
    param_namesDDM,
    lowerDDM,
    upperDDM,
    designDDM,
    n_params = n_params,
    transforms = transformsDDM,
    out_dir = out,
    chunk_size = 1e5,
    seed = TRAIN_SEED,
    n_trials = a$n_trials
  )
  cat(sprintf("=== %s done in %.1f min ===\n", a$tag, (proc.time()[3] - t0) / 60))
}

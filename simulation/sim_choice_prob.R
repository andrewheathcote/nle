# Analytic choice probability P(R=2 | theta) for classifier training.
#
# The choice classifier currently learns P(R|theta) by binary cross-entropy on
# the simulated R labels - i.e. from Bernoulli draws, originally two per theta.
# But P(R|theta) is available EXACTLY: it is the defective CDF at infinity,
# WienR::pWDM. Regressing on it replaces a coin flip per theta with the true
# probability, which should be far more sample-efficient.
#
# This matters because the shipped classifier (never retrained since July) has
# a measured systematic bias of +0.0089 in P(R=2), consistent across all
# quartiles of drawn v. Since v governs choice proportions, that bias forces
# the fitted v DOWN - the direction and roughly half the magnitude of the v
# residual that is now the only parameter outside the G1 gate.
#
# Run from the nle repo root:  Rscript simulation/sim_choice_prob.R

if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(WienR))
suppressPackageStartupMessages(library(arrow))
source("simulation/sim_functions.R")
source("simulation/designs.R")

N_TRAIN <- as.integer(Sys.getenv("CP_N_TRAIN", "200000"))
N_VAL   <- as.integer(Sys.getenv("CP_N_VAL",   "20000"))
BIG_RT  <- 50            # pWDM at large rt = the choice probability

p_upper <- function(p) {
  w <- pnorm(p[["Z"]])
  pWDM(BIG_RT, response = "upper", a = exp(p[["a"]]), v = p[["v"]], w = w,
       t0 = exp(p[["t0"]]), sv = exp(p[["sv"]]),
       sw = 2 * pnorm(p[["SZ"]]) * min(w, 1 - w),
       st0 = exp(p[["st0"]]), precision = 1e-8)$value
}

make_set <- function(n, seed, out) {
  pm <- param_matrix(n, param_namesDDM, lowerDDM, upperDDM,
                     transforms = transformsDDM, seed = seed, Sobol = TRUE)
  pm <- as.data.frame(pm)
  t0 <- proc.time()[3]
  pm$p_R2 <- vapply(seq_len(n), function(i) p_upper(pm[i, ]), numeric(1))
  dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
  write_parquet(pm, out)
  cat(sprintf("  %-28s n=%d  P(R=2) range %.4f-%.4f  %.1f min\n",
              out, n, min(pm$p_R2), max(pm$p_R2), (proc.time()[3] - t0) / 60))
}
cat("=== analytic choice probabilities ===\n")
make_set(N_TRAIN, 3876, "data/choice_prob/train.parquet")
make_set(N_VAL,   1656, "data/choice_prob/val.parquet")

# Analytic choice-probability targets for the ZERO-BOX classifier.
#
# P(R = upper | theta) from WienR::pWDM at the defective CDF's limit, with the
# s-scaling (the likelihood depends on v, a, sv only through v/s, a/s, sv/s --
# omitting it was the Phase 7 bug) and with exact zeros passed as zeros: theta
# arrives on EMC2's sampled scale with -Inf at a zero, and exp(-Inf) = 0,
# pnorm(-Inf) = 0 are the right natural values. t0 and st0 do not affect P(R).
#
# Draws use the same stratified sampler as the training data (different seeds),
# so the classifier sees every regime in the same proportions. Columns are the
# EMC2-scale theta plus p_R2, which is what retrain_classifier.py --zerobox reads.
#
#   Rscript simulation/sim_choice_prob_zerobox.R [n_train] [n_val]
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
if (dir.exists("../../.Rlibs")) .libPaths(c(normalizePath("../../.Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(WienR); library(arrow); library(parallel)})
source("simulation/sim_functions.R"); source("simulation/designs.R"); source("simulation/zerobox.R")
a <- commandArgs(trailingOnly = TRUE)
N_TRAIN <- if (length(a) > 0) as.integer(a[1]) else 200000L
N_VAL   <- if (length(a) > 1) as.integer(a[2]) else 20000L
CORES <- as.integer(Sys.getenv("SIM_CORES", "8"))
BIG_RT <- 1e6

p_upper <- function(p) {
  w <- pnorm(p[["Z"]]); sc <- exp(p[["s"]])
  SZn <- pnorm(p[["SZ"]])                                   # -Inf -> 0
  pWDM(BIG_RT, response = "upper", a = exp(p[["a"]]) / sc, v = p[["v"]] / sc, w = w,
       t0 = exp(p[["t0"]]), sv = exp(p[["sv"]]) / sc,        # -Inf -> 0
       sw = 2 * SZn * min(w, 1 - w), st0 = exp(p[["st0"]]),  # -Inf -> 0
       precision = 1e-8)$value
}
make_set <- function(n, seed, split) {
  pm <- param_matrix_zerobox(n, param_namesDDM, lowerDDM, upperDDM, transformsDDM, seed = seed)
  reg <- attr(pm, "regime"); pm <- as.data.frame(pm)
  t0 <- proc.time()[3]
  pv <- unlist(mclapply(seq_len(n), function(i) p_upper(pm[i, ]), mc.cores = CORES))
  stopifnot(length(pv) == n, all(is.finite(pv)), max(pv - 1, 0) < 1e-8, max(-pv, 0) < 1e-8)
  pm$p_R2 <- pmin(pmax(pv, 0), 1); pm$regime <- reg
  dir.create("data/choice_prob", recursive = TRUE, showWarnings = FALSE)
  out <- sprintf("data/choice_prob/%s_zerobox.parquet", split)
  write_parquet(pm, out)
  cat(sprintf("  %-40s n=%d  P(R=2) range %.4f-%.4f  %.1f min\n", out, n, min(pv), max(pv),
              (proc.time()[3] - t0) / 60))
  # per-regime sanity: P(R) must be identical whether st0 is 0 or not, so the
  # all-zero and st0=0 strata should span the same range as their neighbours
  print(round(tapply(pm$p_R2, pm$regime, function(x) c(min = min(x), max = max(x))), 3)[1:2])
}
cat("=== zero-box analytic choice probabilities ===\n")
make_set(N_VAL, 20260927L, "val")
make_set(N_TRAIN, 20260928L, "train")

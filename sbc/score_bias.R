# Score bias of an nn likelihood, for ANY model family.
#
# The Phase 3 st0 result (sbc/st0_slope.R) showed that what displaces a
# posterior is not the level of the flow's density error but its GRADIENT in
# theta: for the true model dE_true[log f]/dtheta = 0 at the truth, and any
# departure is a systematic force on the mode. That diagnostic used a
# WienR-weighted rt grid, which is DDM-specific. This one is model-agnostic.
#
# Monte-Carlo estimator. Simulate M trials from the TRUE (analytic) model at
# theta, then
#   g = [ ll(theta + h) - ll(theta - h) ] / (2h M)
# using the SAME simulated data for both evaluations (common random numbers),
# so the paired difference is far less noisy than either term.
#
# Validation: the identical estimator applied to the ANALYTIC likelihood must
# return g ~ 0 (the score has zero expectation at the truth). Reported
# alongside every flow estimate; treat a flow g as meaningful only when it is
# large relative to its analytic counterpart and to the reported MC error.
#
# Usage:
#   Rscript sbc/score_bias.R [families] [M] [reps]
#   Rscript sbc/score_bias.R LNR,RDM,DDM 20000 3

if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(EMC2))
source("sbc/nci/sbc_specs.R")

args <- commandArgs(trailingOnly = TRUE)
fams <- if (length(args) >= 1) strsplit(args[1], ",")[[1]] else c("LNR","RDM","DDM")
M    <- if (length(args) >= 2) as.integer(args[2]) else 20000L
REPS <- if (length(args) >= 3) as.integer(args[3]) else 3L
H    <- 0.05

ll_at <- function(p, model_fn, dat, des) {
  dadm <- EMC2:::design_model(dat, des, compress = FALSE,
                              rt_resolution = 1e-12, verbose = FALSE)
  m <- attr(dadm, "model")()
  EMC2:::calc_ll_R(p, m, dadm)
}

score <- function(p, des, dat) {
  dadm <- EMC2:::design_model(dat, des, compress = FALSE,
                              rt_resolution = 1e-12, verbose = FALSE)
  m <- attr(dadm, "model")()
  sapply(names(p), function(nm) {
    pp <- p; pp[nm] <- pp[nm] + H
    pm <- p; pm[nm] <- pm[nm] - H
    (EMC2:::calc_ll_R(pp, m, dadm) - EMC2:::calc_ll_R(pm, m, dadm)) / (2 * H * nrow(dat))
  })
}

cat(sprintf("Score bias per trial, M = %d simulated trials, %d reps, h = %g\n",
            M, REPS, H))
cat("g_analytic must be ~0 (validation); g_flow is the systematic force on the mode.\n")

all_res <- list()
for (fam in fams) {
  cell_a <- make_sbc_cell(fam)
  cell_n <- make_sbc_cell(paste0(fam, "nn"))
  p <- setNames(cell_a$pmean, names(sampled_pars(cell_a$design)))
  ga <- gf <- NULL
  for (r in seq_len(REPS)) {
    set.seed(1000 + r)
    dat <- make_data(p, cell_a$design, n_trials = M)   # truth = analytic model
    ga <- rbind(ga, score(p, cell_a$design, dat))
    gf <- rbind(gf, score(p, cell_n$design, dat))
  }
  cat(sprintf("\n=== %s (M=%d) ===\n", fam, M))
  cat(sprintf("  %-10s %12s %12s %12s %10s\n",
              "par", "g_analytic", "g_flow", "MCse(flow)", "ratio"))
  for (j in seq_along(p)) {
    mse <- if (REPS > 1) sd(gf[, j]) / sqrt(REPS) else NA_real_
    cat(sprintf("  %-10s %+12.5f %+12.5f %12.5f %10s\n",
                names(p)[j], mean(ga[, j]), mean(gf[, j]), mse,
                if (is.finite(mse) && mse > 0)
                  sprintf("%.1f", abs(mean(gf[, j])) / mse) else "-"))
  }
  all_res[[fam]] <- list(p = p, g_analytic = ga, g_flow = gf)
}
saveRDS(all_res, "sbc/score_bias_results.rds")
cat("\nSaved sbc/score_bias_results.rds\n")

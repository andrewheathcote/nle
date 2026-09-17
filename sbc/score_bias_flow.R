# Score bias of a specific EXPORTED DDM flow (not the packaged DDMnn/ens5).
#
# Same Monte-Carlo estimator as sbc/score_bias.R: simulate M trials from the
# TRUE (analytic) DDM at theta, then g = [ll(theta+h) - ll(theta-h)]/(2hM)
# with common random numbers. Validation: the identical estimator on the
# ANALYTIC likelihood must return g ~ 0 (the score has zero expectation at the
# truth). t0 is reported but NOT trustworthy — it moves the support boundary,
# and the analytic control fails there in both this and the grid estimator.
#
# Usage:
#   Rscript sbc/score_bias_flow.R <tag> [<tag2> ...] [M] [reps]
#   Rscript sbc/score_bias_flow.R score01d arch64w 40000 5
# Reference (sbc/score_bias_results.rds, ens5): st0 +0.02542, a +0.05327,
# Z -0.02595, v -0.00850, SZ -0.00616, sv +0.00106.

if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(EMC2))
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")

args <- commandArgs(trailingOnly = TRUE)
num  <- suppressWarnings(as.numeric(args))
tags <- args[is.na(num)]
nums <- num[!is.na(num)]
M    <- if (length(nums) >= 1) as.integer(nums[1]) else 40000L
REPS <- if (length(nums) >= 2) as.integer(nums[2]) else 5L
H    <- 0.05
stopifnot(length(tags) >= 1)

des_of <- function(model) design(
  factors = list(subjects = 1, S = c("left", "right")),
  Rlevels = c("left", "right"),
  formula = list(v ~ 1, a ~ 1, t0 ~ 1, s ~ 1, Z ~ 1, sv ~ 1, SZ ~ 1, st0 ~ 1),
  constants = c(s = log(1)), model = model, verbose = FALSE)

# DDMnn whose dfun/pfun come from one exported flow file
make_local <- function(path) {
  fl <- ddm_load(path); ptr <- EMC2:::ddm_build(fl); force(ptr)
  function() {
    m <- DDMnn()
    m$dfun <- function(rt, R, pars)
      EMC2:::ddm_eval_trials_cpp(ptr, EMC2:::.theta_DDMnn(pars), rt, as.integer(R))$pdf
    m$pfun <- function(rt, R, pars)
      EMC2:::ddm_eval_trials_cpp(ptr, EMC2:::.theta_DDMnn(pars), rt, as.integer(R))$cdf
    m
  }
}

p <- c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5),
       sv = log(.1), SZ = qnorm(.2), st0 = log(.05))

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

des_a <- des_of(DDM)
cat(sprintf("Score bias per trial (M = %d, %d reps, h = %g)\n\n", M, REPS, H))
out <- list()
for (tag in tags) {
  path <- sprintf("port/export/ddm_%s_weights.json", tag)
  if (!file.exists(path)) { cat("missing:", path, "\n"); next }
  des_n <- des_of(make_local(path))
  ga <- gf <- NULL
  for (r in seq_len(REPS)) {
    set.seed(1000 + r)
    dat <- make_data(p, des_a, n_trials = M)
    ga <- rbind(ga, score(p, des_a, dat))
    gf <- rbind(gf, score(p, des_n, dat))
  }
  cat(sprintf("=== %s ===\n", tag))
  cat(sprintf("  %-6s %12s %12s %12s %8s\n", "par", "g_analytic", "g_flow",
              "MCse(flow)", "|g|/se"))
  for (j in seq_along(p)) {
    mse <- sd(gf[, j]) / sqrt(REPS)
    cat(sprintf("  %-6s %+12.5f %+12.5f %12.5f %8.1f%s\n", names(p)[j],
                mean(ga[, j]), mean(gf[, j]), mse, abs(mean(gf[, j])) / mse,
                if (names(p)[j] == "t0") "   <- invalid (moving support)" else ""))
  }
  cat("\n")
  out[[tag]] <- list(g_analytic = ga, g_flow = gf)
}
saveRDS(out, "sbc/score_bias_flow_results.rds")
cat("Saved sbc/score_bias_flow_results.rds\n")

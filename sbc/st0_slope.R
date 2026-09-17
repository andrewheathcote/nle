# st0 slope diagnostic: does the flow TILT the expected log-likelihood along
# the st0 axis, and does that tilt predict the SBC rank?
#
# Rationale. sbc/st0_mechanism.R showed that the flow's log-density error at
# the truth — its magnitude, its regional shape, and its density-weighted
# total — carries no information about which draws are miscalibrated once
# drawn st0 and a are controlled. That is expected if the error is roughly
# flat in st0: a uniform offset shifts the log-likelihood but not its
# argmax. What displaces a posterior is the DERIVATIVE of the error.
#
# Define the expected per-trial log-likelihood under the true DGP,
#   L(theta) = E_true[log f(rt, R | theta)],
# with the expectation taken over the true defective density at the draw's own
# truth. For the true model dL/dtheta = 0 at the truth (KL is minimised
# there). For the flow it is not, and that nonzero gradient is exactly the
# force displacing the mode:
#   per-trial displacement ~ g / I,  standardized ~ g * sqrt(N / I)
# where g = dL_flow/dtheta at the truth and I = -d2L_true/dtheta2 is the unit
# Fisher information (from the analytic likelihood).
#
# Usage:
#   Rscript sbc/st0_slope.R [cell] [res_dir] [out_dir] [n_draws]
# Default:
#   Rscript sbc/st0_slope.R DDMnn_N400 sbc/nci/results_ens5 sbc/nci/analysis_ens5 500

if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
source("port/R/flow_race.R")
source("port/R/flow_ddm.R")
suppressPackageStartupMessages(library(EMC2))

args    <- commandArgs(trailingOnly = TRUE)
cell    <- if (length(args) >= 1) args[1] else "DDMnn_N400"
res_dir <- if (length(args) >= 2) args[2] else "sbc/nci/results_ens5"
out_dir <- if (length(args) >= 3) args[3] else "sbc/nci/analysis_ens5"
n_draws <- if (length(args) >= 4) as.integer(args[4]) else 500L
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
Ntr <- as.integer(sub("^.*_N", "", cell))

TAGS <- c("arch64w", "arch64w_s101", "arch64w_s102", "arch64w_s103",
          "arch64w_s104")
members <- lapply(TAGS, function(tg)
  ddm_load(sprintf("port/export/ddm_%s_weights.json", tg)))
ens   <- EMC2:::ddm_build_ensemble(members)
BOX   <- members[[1]]$bounds_sampled

x     <- readRDS(file.path(res_dir, paste0(cell, "_R500.rds")))
ranks <- as.matrix(x$SBC$rank$alpha)
truth <- as.matrix(x$SBC$med$alpha) - as.matrix(x$SBC$bias$alpha)
n_draws <- min(n_draws, nrow(truth))
cat(sprintf("cell %s (N=%d trials): %d draws\n\n", cell, Ntr, n_draws))

# flow order [v, a, t0, s, Z, SZ, sv, st0]; SBC gives [v,a,t0,Z,sv,SZ,st0]
FLOW_IDX <- c(v = 1L, a = 2L, t0 = 3L, s = 4L, Z = 5L, SZ = 6L, sv = 7L, st0 = 8L)
to_flow <- function(tr)
  c(tr[["v"]], tr[["a"]], tr[["t0"]], 0, tr[["Z"]], tr[["SZ"]],
    tr[["sv"]], tr[["st0"]])

analytic_lpdf <- function(th, rt, R) {
  w  <- pnorm(th[FLOW_IDX[["Z"]]])
  sw <- 2 * pnorm(th[FLOW_IDX[["SZ"]]]) * min(w, 1 - w)
  WienR::dWDM(rt, response = ifelse(R == 1, "lower", "upper"),
              a = exp(th[FLOW_IDX[["a"]]]), v = th[FLOW_IDX[["v"]]], w = w,
              t0 = exp(th[FLOW_IDX[["t0"]]]), sv = exp(th[FLOW_IDX[["sv"]]]),
              sw = sw, st0 = exp(th[FLOW_IDX[["st0"]]]),
              precision = 1e-8)$logvalue
}
flow_lpdf <- function(th, rt, R) {
  Th <- matrix(th, nrow = length(rt), ncol = length(th), byrow = TRUE)
  EMC2:::ddm_ens_eval_trials_cpp(ens, Th, rt,
                                 rep(as.integer(R), length(rt)))$log_pdf
}

# Expected per-trial log-likelihood at theta, under weights fixed at the truth.
# `which` selects flow or analytic. Returns NA if the shifted theta leaves the
# training box (the flow floors there, which would corrupt the derivative).
in_box <- function(th) all(th >= BOX$lower & th <= BOX$upper)
Lbar <- function(th, grid, fun) {
  if (identical(fun, flow_lpdf) && !in_box(th)) return(NA_real_)
  num <- 0; den <- 0
  for (R in 1:2) {
    lp <- fun(th, grid$rt, R)
    w  <- grid$w[[R]]
    ok <- is.finite(lp) & is.finite(w) & w > 0
    if (!any(ok)) next
    num <- num + sum(w[ok] * lp[ok]); den <- den + sum(w[ok])
  }
  if (den <= 0) NA_real_ else num / den
}

H <- 0.05                                     # step on the sampled (log) scale
slope <- function(th0, par, grid) {
  j  <- FLOW_IDX[[par]]
  tp <- th0; tp[j] <- tp[j] + H
  tm <- th0; tm[j] <- tm[j] - H
  # flow gradient at the truth
  g  <- (Lbar(tp, grid, flow_lpdf) - Lbar(tm, grid, flow_lpdf)) / (2 * H)
  # analytic curvature -> unit Fisher information for this parameter
  l0 <- Lbar(th0, grid, analytic_lpdf)
  lp <- Lbar(tp,  grid, analytic_lpdf)
  lm <- Lbar(tm,  grid, analytic_lpdf)
  I  <- -(lp - 2 * l0 + lm) / (H^2)
  c(g = g, I = I)
}

cat("computing per-draw slopes...\n")
t_start <- proc.time()[3]
out <- data.frame()
for (i in seq_len(n_draws)) {
  tr  <- truth[i, ]
  th0 <- to_flow(tr)
  t0n <- exp(tr[["t0"]])
  rt  <- t0n + exp(seq(log(1e-3), log(8), length.out = 120))
  # The weights must carry the integration MEASURE, not just the density: the
  # grid is log-spaced, so sum(w * lp)/sum(w) is only an expectation once each
  # point is weighted by its interval width. Without this the estimator is
  # badly biased — validated by requiring g ~ 0 for the ANALYTIC likelihood at
  # the truth (the score has zero expectation there). Omitting dr gave
  # g_analytic = -0.25 (st0) and -1.46 (a); including it gives -0.0002 and
  # +0.0019.
  dr  <- c(diff(rt)[1], (head(diff(rt), -1) + tail(diff(rt), -1)) / 2,
           tail(diff(rt), 1))
  w   <- lapply(1:2, function(R) exp(analytic_lpdf(th0, rt, R)) * dr)
  grid <- list(rt = rt, w = w)
  s_st0 <- slope(th0, "st0", grid)
  s_a   <- slope(th0, "a",   grid)
  # validation: the same estimator applied to the ANALYTIC likelihood must
  # return ~0 at the truth. Carried per draw so a silent regression is visible.
  gchk_st0 <- (Lbar(replace(th0, FLOW_IDX[["st0"]], th0[FLOW_IDX[["st0"]]] + H),
                    grid, analytic_lpdf) -
               Lbar(replace(th0, FLOW_IDX[["st0"]], th0[FLOW_IDX[["st0"]]] - H),
                    grid, analytic_lpdf)) / (2 * H)
  out <- rbind(out, data.frame(
    draw = i, rank_st0 = ranks[i, "st0"], rank_a = ranks[i, "a"],
    st0 = tr[["st0"]], a = tr[["a"]], t0 = tr[["t0"]], v = tr[["v"]],
    g_st0 = s_st0[["g"]], I_st0 = s_st0[["I"]],
    g_a   = s_a[["g"]],   I_a   = s_a[["I"]],
    gchk_st0 = gchk_st0))
  if (i %% 100 == 0)
    cat(sprintf("  %d/%d  (%.0f s)\n", i, n_draws, proc.time()[3] - t_start))
}
cat(sprintf("done in %.1f min\n\n", (proc.time()[3] - t_start) / 60))

# Predicted displacement of the posterior mode. st0 is weakly identified and
# the prior is tight (psd .15 on the sampled scale), so the prior precision is
# NOT negligible: at N=400, 1/.15^2 = 44 against N*I ~ 168. Posterior precision
# is prior + likelihood, and the mode shifts by (N * g) / precision.
#   shift = N*g / (1/psd^2 + N*I);  posterior sd = 1/sqrt(1/psd^2 + N*I)
# CAVEAT: this is a one-dimensional profile. The real posterior is 7-dim and
# part of the st0 displacement is absorbed by a and t0, so |z| here is an
# UPPER bound on the displacement SBC actually sees. What the diagnostic is
# for is the ACROSS-DRAW correlation with rank, which is scale-free.
PSD <- c(st0 = .15, a = .15)                  # sbc_specs.R DDM priors
prec <- function(I, psd) 1 / psd^2 + Ntr * I
out$shift_st0 <- Ntr * out$g_st0 / prec(out$I_st0, PSD[["st0"]])
out$z_st0     <- out$shift_st0 * sqrt(prec(out$I_st0, PSD[["st0"]]))
out$shift_a   <- Ntr * out$g_a   / prec(out$I_a,   PSD[["a"]])
out$z_a       <- out$shift_a   * sqrt(prec(out$I_a,   PSD[["a"]]))

f <- function(v) sprintf("%+.4f", mean(out[[v]], na.rm = TRUE))
cat("Per-draw summaries (mean):\n")
cat(sprintf("  g_st0 %s   I_st0 %s   shift_st0 %s   z_st0 %s\n",
            f("g_st0"), f("I_st0"), f("shift_st0"), f("z_st0")))
cat(sprintf("  g_a   %s   I_a   %s   shift_a   %s   z_a   %s\n",
            f("g_a"), f("I_a"), f("shift_a"), f("z_a")))
cat(sprintf("  usable draws: %d of %d\n", sum(is.finite(out$z_st0)), nrow(out)))
cat(sprintf("  VALIDATION mean g_st0 on the ANALYTIC likelihood: %+.5f",
            mean(out$gchk_st0, na.rm = TRUE)))
cat(if (abs(mean(out$gchk_st0, na.rm = TRUE)) < 0.01) "  (OK, ~0)\n\n" else
      "  ** NOT ~0: estimator is biased, do not trust the slopes **\n\n")
# NOTE: t0 is deliberately not profiled. It shifts the support boundary, so a
# finite difference straddles a discontinuity and the estimator does not
# validate there (g_analytic ~ -0.7 rather than 0).

sp <- function(a, b) suppressWarnings(cor(a, b, method = "spearman",
                                          use = "complete.obs"))
spp <- function(a, b) suppressWarnings(
  cor.test(a, b, method = "spearman")$p.value)
pr <- function(y, x, ctrl) {
  ok <- is.finite(y) & is.finite(x) & Reduce(`&`, lapply(ctrl, is.finite))
  cd <- as.data.frame(lapply(ctrl, function(z) rank(z[ok])))
  ry <- residuals(lm(rank(y[ok]) ~ ., data = cd))
  rx <- residuals(lm(rank(x[ok]) ~ ., data = cd))
  ct <- cor.test(ry, rx)
  sprintf("%+.3f (p %.1e)", ct$estimate, ct$p.value)
}

cat("Does the flow's st0 slope predict the st0 rank?\n")
cat(sprintf("  %-12s %-20s %-22s %-22s\n", "predictor", "raw",
            "| drawn st0", "| drawn st0 + a"))
for (v in c("g_st0", "z_st0", "shift_st0")) {
  cat(sprintf("  %-12s %+.3f (p %.1e)   %-22s %-22s\n", v,
              sp(out[[v]], out$rank_st0), spp(out[[v]], out$rank_st0),
              pr(out$rank_st0, out[[v]], list(st0 = out$st0)),
              pr(out$rank_st0, out[[v]], list(st0 = out$st0, a = out$a))))
}
cat("\nSame for a (the other drifting parameter):\n")
for (v in c("g_a", "z_a")) {
  cat(sprintf("  %-12s %+.3f (p %.1e)   %-22s\n", v,
              sp(out[[v]], out$rank_a), spp(out[[v]], out$rank_a),
              pr(out$rank_a, out[[v]], list(a = out$a, st0 = out$st0))))
}

cat("\nPredicted vs observed displacement, by quintile of drawn st0:\n")
q <- cut(out$st0, quantile(out$st0, 0:5/5), include.lowest = TRUE,
         labels = paste0("Q", 1:5))
cat(sprintf("  %-4s %10s %10s %10s\n", "Q", "z_st0", "meanRank", "g_st0"))
for (lv in levels(q))
  cat(sprintf("  %-4s %+10.3f %10.3f %+10.4f\n", lv,
              mean(out$z_st0[q == lv], na.rm = TRUE),
              mean(out$rank_st0[q == lv]),
              mean(out$g_st0[q == lv], na.rm = TRUE)))

pdf(file.path(out_dir, sprintf("%s_st0_slope.pdf", cell)), width = 10, height = 5)
op <- par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
plot(out$z_st0, out$rank_st0, pch = 16, col = "#00000044",
     xlab = "predicted displacement (posterior sd)", ylab = "SBC rank of st0",
     main = sprintf("%s: rank vs predicted shift", cell))
abline(h = .5, lty = 2, col = "grey40"); abline(v = 0, lty = 2, col = "grey40")
plot(out$st0, out$g_st0, pch = 16, col = "#00000044",
     xlab = "drawn st0 (sampled scale)", ylab = "flow st0 slope g",
     main = "likelihood tilt vs drawn st0")
abline(h = 0, lty = 2, col = "grey40")
par(op); dev.off()

saveRDS(out, file.path(out_dir, sprintf("%s_st0_slope.rds", cell)))
cat(sprintf("\nSaved to %s\n", out_dir))

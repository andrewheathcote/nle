# st0 mechanism study: does the ens5 flow's likelihood error, measured at
# each SBC draw's own generating parameters, explain that draw's st0 rank?
#
# Phase 3 established that the st0 miscalibration is region-dependent —
# displacement grows with drawn st0, and st0 rank correlates with drawn a
# (rho +.225 at N=400). This script tests the proposed cause directly:
# per draw, the flow's joint log-density error against WienR over the upper
# shoulder (~1-3 s), correlated with that draw's SBC rank.
#
# If the shoulder residual predicts the rank, the distortion is a local
# density error and importance-weighted training over that region is the
# indicated fix. If it does not, the error is elsewhere and the coupling to
# a is doing the work.
#
# Usage:
#   Rscript sbc/st0_mechanism.R [cell] [res_dir] [out_dir] [n_draws]
# Default:
#   Rscript sbc/st0_mechanism.R DDMnn_N400 sbc/nci/results_ens5 sbc/nci/analysis_ens5 500

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

# ---- ens5 ensemble pointer (the shipped production likelihood) ----------
TAGS <- c("arch64w", "arch64w_s101", "arch64w_s102", "arch64w_s103",
          "arch64w_s104")
members <- lapply(TAGS, function(tg)
  ddm_load(sprintf("port/export/ddm_%s_weights.json", tg)))
ens <- EMC2:::ddm_build_ensemble(members)

# ---- SBC draws ---------------------------------------------------------
x     <- readRDS(file.path(res_dir, paste0(cell, "_R500.rds")))
ranks <- as.matrix(x$SBC$rank$alpha)
truth <- as.matrix(x$SBC$med$alpha) - as.matrix(x$SBC$bias$alpha)
n_draws <- min(n_draws, nrow(truth))
idx <- seq_len(n_draws)
cat(sprintf("cell %s: %d draws, pars %s\n\n", cell, n_draws,
            paste(colnames(truth), collapse = " ")))

# ---- rt regions --------------------------------------------------------
# "shoulder" is the 1-3 s band the Phase 3 analysis implicates; edge and bulk
# are carried as controls so a shoulder-specific effect can be distinguished
# from a global density offset.
SHOULDER <- c(1.0, 3.0)
rt_shoulder <- seq(SHOULDER[1], SHOULDER[2], length.out = 25)

# flow parameter order is [v, a, t0, s, Z, SZ, sv, st0] on the SAMPLED scale;
# SBC columns are [v, a, t0, Z, sv, SZ, st0] with s constant at log(1) = 0.
to_flow <- function(tr)
  c(tr[["v"]], tr[["a"]], tr[["t0"]], 0, tr[["Z"]], tr[["SZ"]],
    tr[["sv"]], tr[["st0"]])

# WienR takes natural scale. SZ is the RAW probit-scale parameter the flow was
# trained on; the analytic DDM's Ttransform remaps it to 2*SZ*min(w, 1-w),
# which is what dWDM's `sw` expects (matches port/R/flow_ddm.R and
# sbc/edge_error_map.R).
analytic_lpdf <- function(tr, rt, R) {
  w  <- pnorm(tr[["Z"]])
  sw <- 2 * pnorm(tr[["SZ"]]) * min(w, 1 - w)
  WienR::dWDM(rt, response = ifelse(R == 1, "lower", "upper"),
              a = exp(tr[["a"]]), v = tr[["v"]], w = w,
              t0 = exp(tr[["t0"]]), sv = exp(tr[["sv"]]),
              sw = sw, st0 = exp(tr[["st0"]]),
              precision = 1e-8)$logvalue
}

# NOTE: `ens` is an ensemble XPtr (class ddm_flow_ensemble) and must be
# evaluated with ddm_ens_eval_trials_cpp, which is trial-wise and so wants a
# theta MATRIX with one row per rt. Passing it to the single-model
# ddm_eval_cpp reinterprets the struct and returns garbage.
resid_region <- function(tr, rt) {
  th <- to_flow(tr)
  d <- numeric(0)
  for (R in 1:2) {
    keep <- rt > exp(tr[["t0"]])           # below t0 the density is 0
    if (!any(keep)) next
    rr <- rt[keep]
    Th <- matrix(th, nrow = length(rr), ncol = length(th), byrow = TRUE)
    fl <- EMC2:::ddm_ens_eval_trials_cpp(ens, Th, rr,
                                         rep(1L, length(rr)) * as.integer(R))$log_pdf
    an <- analytic_lpdf(tr, rr, R)
    ok <- is.finite(fl) & is.finite(an)
    d <- c(d, (fl - an)[ok])
  }
  if (!length(d)) return(c(mean = NA_real_, mean_abs = NA_real_))
  c(mean = mean(d), mean_abs = mean(abs(d)))
}

# Density-weighted log-density error over the WHOLE distribution. This is the
# inference-relevant statistic: E_true[log f_flow - log f_true] = -KL, i.e.
# the expected per-trial log-likelihood bias the sampler actually sees. An
# unweighted tail residual can be +6 log units in a region carrying no mass
# and still be irrelevant; this weights every error by how often it is met.
kl_resid <- function(tr) {
  t0n <- exp(tr[["t0"]])
  rt  <- t0n + exp(seq(log(1e-3), log(8), length.out = 120))
  th  <- to_flow(tr)
  num <- 0; den <- 0
  for (R in 1:2) {
    Th <- matrix(th, nrow = length(rt), ncol = length(th), byrow = TRUE)
    fl <- EMC2:::ddm_ens_eval_trials_cpp(ens, Th, rt,
                                         rep(as.integer(R), length(rt)))$log_pdf
    an <- analytic_lpdf(tr, rt, R)
    ok <- is.finite(fl) & is.finite(an)
    if (!any(ok)) next
    w   <- exp(an[ok])                     # true defective density = weights
    num <- num + sum(w * (fl[ok] - an[ok]))
    den <- den + sum(w)
  }
  if (den <= 0) NA_real_ else num / den
}

# ---- per-draw residuals ------------------------------------------------
cat("computing per-draw shoulder residuals (this takes a few minutes)...\n")
t_start <- proc.time()[3]
out <- data.frame()
for (i in idx) {
  tr <- truth[i, ]
  sh <- resid_region(tr, rt_shoulder)
  # controls: edge and bulk, defined relative to this draw's own t0
  t0n <- exp(tr[["t0"]])
  ed  <- resid_region(tr, seq(t0n + .005, t0n + .10, length.out = 15))
  bu  <- resid_region(tr, seq(t0n + .12, t0n + .70, length.out = 15))
  out <- rbind(out, data.frame(
    draw = i, rank_st0 = ranks[i, "st0"],
    st0 = tr[["st0"]], a = tr[["a"]], t0 = tr[["t0"]], v = tr[["v"]],
    sh_mean = sh[["mean"]], sh_abs = sh[["mean_abs"]],
    edge_mean = ed[["mean"]], bulk_mean = bu[["mean"]],
    kl = kl_resid(tr)))
  if (i %% 100 == 0)
    cat(sprintf("  %d/%d  (%.0f s)\n", i, n_draws, proc.time()[3] - t_start))
}
cat(sprintf("done in %.1f min\n\n", (proc.time()[3] - t_start) / 60))

# ---- does the shoulder residual explain the rank? ----------------------
sp <- function(a, b) suppressWarnings(cor(a, b, method = "spearman",
                                          use = "complete.obs"))
sp_p <- function(a, b) suppressWarnings(
  cor.test(a, b, method = "spearman")$p.value)

cat("Region residuals (flow minus WienR, log units):\n")
for (rg in c("kl", "sh_mean", "sh_abs", "edge_mean", "bulk_mean"))
  cat(sprintf("  %-10s mean %+7.3f   sd %6.3f\n", rg,
              mean(out[[rg]], na.rm = TRUE), sd(out[[rg]], na.rm = TRUE)))

cat("\nSpearman correlations with the st0 SBC rank:\n")
cat(sprintf("  %-12s %8s %11s\n", "predictor", "rho", "p"))
for (v in c("kl", "sh_mean", "sh_abs", "edge_mean", "bulk_mean", "st0", "a")) {
  cat(sprintf("  %-12s %+8.3f %11.2e\n", v,
              sp(out[[v]], out$rank_st0), sp_p(out[[v]], out$rank_st0)))
}

# Partial: does the shoulder residual add anything beyond drawn st0 and a?
fit0 <- lm(rank_st0 ~ st0 + a, data = out)
fit1 <- lm(rank_st0 ~ st0 + a + kl, data = out)
cat("\nDoes the density-weighted residual add to drawn st0 + a?\n")
print(anova(fit0, fit1))
cat(sprintf("\n  R^2: %.4f (st0+a)  ->  %.4f (+ shoulder residual)\n",
            summary(fit0)$r.squared, summary(fit1)$r.squared))
cat("  (kl = density-weighted log-density error = -KL per trial)\n")

# ---- residual by drawn-st0 quintile ------------------------------------
q <- cut(out$st0, quantile(out$st0, 0:5 / 5), include.lowest = TRUE,
         labels = paste0("Q", 1:5))
cat("\nShoulder residual by quintile of drawn st0:\n")
cat(sprintf("  %-4s %10s %10s %10s\n", "Q", "sh_mean", "sh_abs", "meanRank"))
for (lv in levels(q))
  cat(sprintf("  %-4s %+10.3f %10.3f %10.3f\n", lv,
              mean(out$sh_mean[q == lv], na.rm = TRUE),
              mean(out$sh_abs[q == lv], na.rm = TRUE),
              mean(out$rank_st0[q == lv])))

pdf(file.path(out_dir, sprintf("%s_st0_shoulder_residual.pdf", cell)),
    width = 10, height = 5)
op <- par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
plot(out$kl, out$rank_st0, pch = 16, col = "#00000044",
     xlab = "density-weighted log-density error (-KL per trial)",
     ylab = "SBC rank of st0", main = sprintf("%s: rank vs weighted error", cell))
abline(h = .5, lty = 2, col = "grey40")
plot(out$st0, out$kl, pch = 16, col = "#00000044",
     xlab = "drawn st0 (sampled scale)", ylab = "weighted log-density error",
     main = "weighted error vs drawn st0")
abline(h = 0, lty = 2, col = "grey40")
par(op); dev.off()

saveRDS(out, file.path(out_dir, sprintf("%s_st0_mechanism.rds", cell)))
cat(sprintf("\nSaved to %s\n", out_dir))

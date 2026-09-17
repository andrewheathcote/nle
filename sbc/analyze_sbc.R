# Analysis of an NCI SBC campaign: calibration tables, rank-ECDF plots, and
# (for the nn cells) rank-vs-truth diagnostics that discriminate a uniform
# displacement from a region-dependent distortion.
#
# Per cell x parameter: 95% CI coverage (MC se ~ 0.0097 at 500 reps), mean
# rank (0.5 = calibrated; < 0.5 = posterior sits above truth, i.e.
# overestimation), KS test of ranks against uniform, mean bias of the
# posterior median, and the shift b implied by the observed coverage.
#
# Usage:
#   Rscript sbc/analyze_sbc.R [res_dir] [out_dir] [focus_par]
# Defaults reproduce the Phase 1 analysis:
#   Rscript sbc/analyze_sbc.R sbc/nci/results sbc/nci/analysis st0
# Phase 3 (ens5):
#   Rscript sbc/analyze_sbc.R sbc/nci/results_ens5 sbc/nci/analysis_ens5 st0
#
# Cells are discovered from the *_R500.rds files present, so the same script
# serves a full campaign or a single cell as it lands.

if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(EMC2))

args      <- commandArgs(trailingOnly = TRUE)
res_dir   <- if (length(args) >= 1) args[1] else "sbc/nci/results"
out_dir   <- if (length(args) >= 2) args[2] else "sbc/nci/analysis"
focus_par <- if (length(args) >= 3) args[3] else "st0"
# analytic twins live in the Phase 1 directory; they calibrate the
# rank-vs-own-truth correlation, which is an identification artefact, not a
# miscalibration signal (a weakly identified parameter has posterior ~ prior,
# so rank ~ prior CDF of truth: strongly correlated AND uniform). Verified on
# the calibrated Phase 1 cells, where rho reaches +0.999 for sv/SZ.
ref_dir   <- if (length(args) >= 4) args[4] else "sbc/nci/results"
stopifnot(dir.exists(res_dir))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("res_dir = %s\nout_dir = %s\nfocus   = %s\nref_dir = %s\n\n",
            res_dir, out_dir, focus_par, ref_dir))

# ---- discover cells, ordered model-then-N -------------------------------
files <- list.files(res_dir, pattern = "_R500\\.rds$")
if (!length(files)) stop("No *_R500.rds cells found in ", res_dir)
cells <- sub("_R500\\.rds$", "", files)
mdl   <- sub("_N[0-9]+$", "", cells)
nn    <- as.integer(sub("^.*_N", "", cells))
ord   <- c("LNR", "LNRnn", "RDM", "RDMnn", "DDM", "DDMnn")
cells <- cells[order(match(mdl, ord), nn)]
cat("cells:", paste(cells, collapse = ", "), "\n\n")

# ---- registered Phase 3 predictions (SBC_PLAN.md) ----------------------
# predicted 95% coverage for DDMnn = ens5, by parameter and N
PRED <- list(
  t0  = c(`100` = .95, `200` = .91, `400` = .94),
  a   = c(`100` = .94, `200` = .93, `400` = .84),
  st0 = c(`100` = .93, `200` = .92, `400` = .64),
  sv  = c(`100` = .95, `200` = .95, `400` = .95),
  v   = c(`100` = .93, `200` = .94, `400` = .95),
  Z   = c(`100` = .95, `200` = .95, `400` = .93),
  SZ  = c(`100` = .95, `200` = .95, `400` = .95))
is_phase3 <- grepl("ens5", res_dir)

# coverage of a 95% interval when the posterior is displaced by b posterior
# sd: Phi(1.96 - b) - Phi(-1.96 - b). Inverted here to report the implied b.
cov_from_b <- function(b) pnorm(1.96 - b) - pnorm(-1.96 - b)
b_from_cov <- function(cv) {
  if (!is.finite(cv) || cv >= cov_from_b(0)) return(0)
  tryCatch(uniroot(function(b) cov_from_b(b) - cv, c(0, 8))$root,
           error = function(e) NA_real_)
}

# ---- analytic reference: rho(rank, own truth) in a calibrated cell -----
# Keyed by analytic model name; the nn twin is compared against it.
ref_rho <- list()
if (dir.exists(ref_dir)) {
  for (rf in list.files(ref_dir, pattern = "^(LNR|RDM|DDM)_N[0-9]+_R500\\.rds$")) {
    y  <- readRDS(file.path(ref_dir, rf))
    rr <- as.matrix(y$SBC$rank$alpha)
    tt <- as.matrix(y$SBC$med$alpha) - as.matrix(y$SBC$bias$alpha)
    key <- sub("_N[0-9]+_R500\\.rds$", "", rf)
    ref_rho[[key]] <- sapply(colnames(rr), function(p)
      suppressWarnings(cor(rr[, p], tt[, p], method = "spearman")))
  }
  cat("analytic reference cells:", paste(names(ref_rho), collapse = ", "), "\n\n")
}

all_rows <- data.frame()
diag_rows <- data.frame()
for (cell in cells) {
  x      <- readRDS(file.path(res_dir, paste0(cell, "_R500.rds")))
  ranks  <- as.matrix(x$SBC$rank$alpha)
  med    <- as.matrix(x$SBC$med$alpha)
  bi     <- as.matrix(x$SBC$bias$alpha)
  truth  <- med - bi                      # verified: colMeans(truth) == pmean
  cov    <- apply(x$SBC$coverage$alpha, 2, mean)
  bias   <- apply(bi, 2, mean)
  sd_med <- apply(med, 2, sd)
  Ncell  <- as.integer(sub("^.*_N", "", cell))

  for (j in seq_len(ncol(ranks))) {
    p  <- colnames(ranks)[j]
    ks <- suppressWarnings(ks.test(ranks[, j], "punif"))
    pr <- if (is_phase3 && !is.null(PRED[[p]]) &&
              as.character(Ncell) %in% names(PRED[[p]]))
            PRED[[p]][[as.character(Ncell)]] else NA_real_
    all_rows <- rbind(all_rows, data.frame(
      cell = cell, N = Ncell, par = p,
      coverage = cov[j], predicted = pr,
      mean_rank = mean(ranks[, j]), ks_p = ks$p.value,
      mean_bias = bias[j], sd_med = sd_med[j],
      b_implied = b_from_cov(cov[j])))
  }

  # ---- rank-vs-truth diagnostics (nn cells only) ----------------------
  # Shape of the miscalibration, per the Phase 3 mechanism design:
  #   uniform displacement -> |mean rank - .5| large, dispersion ~ 1
  #   region-dependent     -> rank correlates with a drawn parameter, and/or
  #                           dispersion departs from 1 (S-shaped ecdf)
  if (grepl("nn$", sub("_N[0-9]+$", "", cell)) && focus_par %in% colnames(ranks)) {
    r <- ranks[, focus_par]
    disp <- var(r) / (1 / 12)
    twin <- sub("nn$", "", sub("_N[0-9]+$", "", cell))   # DDMnn -> DDM
    for (against in colnames(truth)) {
      rho <- suppressWarnings(cor(r, truth[, against], method = "spearman"))
      pv  <- suppressWarnings(cor.test(r, truth[, against],
                                       method = "spearman")$p.value)
      # reference only defined for the own-truth correlation
      rr <- if (against == focus_par && !is.null(ref_rho[[twin]]) &&
                focus_par %in% names(ref_rho[[twin]]))
              ref_rho[[twin]][[focus_par]] else NA_real_
      diag_rows <- rbind(diag_rows, data.frame(
        cell = cell, N = Ncell, focus = focus_par, against = against,
        rho = rho, p = pv, rho_ref = rr, delta = rho - rr,
        mean_rank = mean(r), dispersion = disp))
    }
    # mean rank by quintile of the focus parameter's own drawn value
    q  <- cut(truth[, focus_par], quantile(truth[, focus_par], 0:5 / 5),
              include.lowest = TRUE, labels = paste0("Q", 1:5))
    qm <- tapply(r, q, mean)
    cat(sprintf("%s: mean rank of %s by quintile of drawn %s:  %s\n",
                cell, focus_par, focus_par,
                paste(sprintf("%s=%.3f", names(qm), qm), collapse = "  ")))
  }

  pdf(file.path(out_dir, paste0(cell, "_ecdf.pdf")), width = 10, height = 6)
  ok <- tryCatch({plot_sbc_ecdf(x$SBC$rank); TRUE},
                 error = function(e) tryCatch({plot_sbc_ecdf(x$SBC$rank$alpha); TRUE},
                                              error = function(e2) FALSE))
  dev.off()
  if (!ok) file.remove(file.path(out_dir, paste0(cell, "_ecdf.pdf")))

  # rank vs drawn value, focus parameter, with binned means
  if (focus_par %in% colnames(ranks)) {
    pdf(file.path(out_dir, sprintf("%s_%s_rank_vs_truth.pdf", cell, focus_par)),
        width = 10, height = 5)
    op <- par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
    for (against in c(focus_par, "a")) {
      if (!against %in% colnames(truth)) next
      tv <- truth[, against]; r <- ranks[, focus_par]
      plot(tv, r, pch = 16, col = "#00000033", ylim = c(0, 1),
           xlab = sprintf("drawn %s (sampled scale)", against),
           ylab = sprintf("SBC rank of %s", focus_par),
           main = sprintf("%s: %s rank vs %s", cell, focus_par, against))
      abline(h = 0.5, lty = 2, col = "grey40")
      br <- cut(tv, quantile(tv, 0:8 / 8), include.lowest = TRUE)
      bm <- tapply(r, br, mean); bx <- tapply(tv, br, mean)
      lines(bx, bm, type = "b", pch = 19, col = "firebrick", lwd = 2)
    }
    par(op); dev.off()
  }
}
rownames(all_rows) <- NULL

# ---- flags: coverage outside 3 MC se of .95; KS at Bonferroni ----------
se_cov   <- sqrt(.95 * .05 / 500)
alpha_ks <- .05 / nrow(all_rows)
all_rows$flag <- ifelse(abs(all_rows$coverage - .95) > 3 * se_cov |
                          all_rows$ks_p < alpha_ks, "**FAIL**", "")

cat(sprintf("\n%-13s %-5s %8s %6s %9s %10s %9s %7s  %s\n",
            "cell", "par", "coverage", "pred", "meanRank", "KS p", "bias",
            "b_impl", "flag"))
for (i in seq_len(nrow(all_rows)))
  cat(sprintf("%-13s %-5s %8.3f %6s %9.3f %10.2e %+9.4f %7s  %s\n",
              all_rows$cell[i], all_rows$par[i], all_rows$coverage[i],
              ifelse(is.na(all_rows$predicted[i]), "-",
                     sprintf("%.2f", all_rows$predicted[i])),
              all_rows$mean_rank[i], all_rows$ks_p[i], all_rows$mean_bias[i],
              ifelse(is.na(all_rows$b_implied[i]), "-",
                     sprintf("%.2f", all_rows$b_implied[i])),
              all_rows$flag[i]))

cat("\nCell verdicts (all parameters pass coverage & KS):\n")
for (cell in cells) {
  sub <- all_rows[all_rows$cell == cell, ]
  cat(sprintf("  %-13s %s   (worst coverage %.3f [%s], min KS p %.1e [%s])\n",
              cell, if (any(sub$flag != "")) "FAIL" else "PASS",
              sub$coverage[which.max(abs(sub$coverage - .95))],
              sub$par[which.max(abs(sub$coverage - .95))],
              min(sub$ks_p), sub$par[which.min(sub$ks_p)]))
}

# ---- scorecard against the registered predictions ----------------------
if (is_phase3 && any(!is.na(all_rows$predicted))) {
  sc <- all_rows[!is.na(all_rows$predicted), ]
  sc$err <- sc$coverage - sc$predicted
  cat("\nRegistered-prediction scorecard (Phase 3, SBC_PLAN.md):\n")
  cat(sprintf("  %-5s %-5s %9s %9s %8s\n", "par", "N", "predicted", "observed", "err"))
  for (i in seq_len(nrow(sc)))
    cat(sprintf("  %-5s %-5d %9.2f %9.3f %+8.3f\n",
                sc$par[i], sc$N[i], sc$predicted[i], sc$coverage[i], sc$err[i]))
  cat(sprintf("\n  mean |error| = %.3f over %d predictions (MC se %.3f)\n",
              mean(abs(sc$err)), nrow(sc), se_cov))
}

# ---- DDMnn trend with N ------------------------------------------------
ddmnn <- all_rows[grepl("^DDMnn_", all_rows$cell), ]
if (nrow(ddmnn)) {
  cat("\nDDMnn calibration trend with N:\n")
  for (p in unique(ddmnn$par)) {
    sub <- ddmnn[ddmnn$par == p, ]
    sub <- sub[order(sub$N), ]
    cat(sprintf("  %-4s coverage: %-22s mean rank: %-22s (N=%s)\n", p,
                paste(sprintf("%.3f", sub$coverage), collapse = " / "),
                paste(sprintf("%.3f", sub$mean_rank), collapse = " / "),
                paste(sub$N, collapse = "/")))
  }
}

# ---- focus-parameter mechanism summary ---------------------------------
if (nrow(diag_rows)) {
  rownames(diag_rows) <- NULL
  cat(sprintf("\nMechanism diagnostics for %s (rank vs drawn truth):\n", focus_par))
  cat(sprintf("  %-13s %-6s %8s %10s %8s %8s %10s %11s\n",
              "cell", "vs", "rho", "p", "rho_ref", "delta", "meanRank",
              "dispersion"))
  for (i in seq_len(nrow(diag_rows)))
    cat(sprintf("  %-13s %-6s %+8.3f %10.2e %8s %8s %10.3f %11.3f\n",
                diag_rows$cell[i], diag_rows$against[i], diag_rows$rho[i],
                diag_rows$p[i],
                ifelse(is.na(diag_rows$rho_ref[i]), "-",
                       sprintf("%+.3f", diag_rows$rho_ref[i])),
                ifelse(is.na(diag_rows$delta[i]), "-",
                       sprintf("%+.3f", diag_rows$delta[i])),
                diag_rows$mean_rank[i], diag_rows$dispersion[i]))
  cat("\n  Reading (dispersion = var(rank)/(1/12); 1 = uniform spread):\n")
  cat("  * rho against the focus parameter's OWN truth is NOT a signal on its\n")
  cat("    own - it reflects identification, and reaches +0.999 in calibrated\n")
  cat("    analytic cells. Read `delta` (rho - analytic twin) instead: delta\n")
  cat("    near 0 means the rank still tracks truth as it should; delta going\n")
  cat("    negative with N means the posterior is being driven by likelihood\n")
  cat("    error rather than by the data.\n")
  cat("  * |mean rank - .5| large, dispersion ~ 1, no cross-parameter rho\n")
  cat("    => uniform displacement over the prior box (architectural fix).\n")
  cat("  * cross-parameter rho significant, or dispersion far from 1\n")
  cat("    => region-dependent distortion (targeted remedy viable).\n")
  saveRDS(diag_rows, file.path(out_dir, "mechanism_diagnostics.rds"))
}

saveRDS(all_rows, file.path(out_dir, "sbc_summary.rds"))
cat(sprintf("\nTable saved to %s; ECDF and rank-vs-truth PDFs alongside.\n", out_dir))

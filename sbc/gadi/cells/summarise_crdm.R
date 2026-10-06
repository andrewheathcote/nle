# Summary of the M6 CRDM SBC cells (Gadi 2026-10-06, EMC2 dev-nle 142bd497 = aa66e11d): the network cell
# (hybrid race, dt 0.0005 card, N = 400, 500 replicates) and the two Volterra controls (CRDM() at dt 2e-3,
# 250 replicates; dt 5e-4, 50 replicates). Run on the Mac (never R/4.5.0 for a statistic).
#   Rscript sbc/gadi/cells/summarise_crdm.R
# Per parameter and cell: standardised bias, coverage, ECDF envelope (nn_sbc_summary). Network against each
# control, unpaired (different prior draws): standardised-bias difference against sqrt(1/n1 + 1/n2), coverage by
# a two-proportion z test, rank distributions by a two-sample KS test. "Beyond chance" = Bonferroni over the
# 6 parameters x 3 tests of one comparison (p < .05 / 18).
.libPaths(c(path.expand("~/R/nle-lib"), .libPaths())); suppressPackageStartupMessages(library(EMC2))
root <- path.expand("~/Downloads/julian/nle/sbc/gadi/cells")
ld <- function(f) if (file.exists(f)) { e <- new.env(); load(f, e); e$SBC }
rd <- function(f) if (file.exists(f)) readRDS(f)
S <- list(network = ld(file.path(root, "crdm_net_N400/results_aa66/sbc_nn.RData")),
          `ctl dt2e-3` = rd(file.path(root, "crdm_ctl_dt2e-3/results_aa66/joined.rds")),
          `ctl dt5e-4` = rd(file.path(root, "crdm_ctl_dt5e-4/results_aa66/joined.rds")))
S <- Filter(function(x) !is.null(x) && nrow(x$rank$alpha) > 0, S)
tab <- do.call(rbind, lapply(names(S), function(k) cbind(cell = k, nn_sbc_summary(S[[k]]))))
tab$gate <- ifelse(abs(tab$std_bias) <= .5 & tab$in_envelope, "ok", "FLAG")
num <- sapply(tab, is.numeric); tab2 <- tab; tab2[num] <- lapply(tab2[num], signif, 3)
cat("=== per cell ===\n"); print(tab2, row.names = FALSE)
write.csv(tab2, file.path(root, "summary_crdm.csv"), row.names = FALSE)

sbv <- function(x) { b <- as.matrix(x$bias$alpha); colMeans(b) / apply(b, 2, sd) }
cmp <- do.call(rbind, lapply(setdiff(names(S), "network"), function(k) {
  a <- S$network; b <- S[[k]]; n1 <- nrow(a$rank$alpha); n2 <- nrow(b$rank$alpha)
  do.call(rbind, lapply(colnames(a$rank$alpha), function(p) {
    d <- sbv(a)[[p]] - sbv(b)[[p]]; zs <- d / sqrt(1 / n1 + 1 / n2)
    c1 <- mean(a$coverage$alpha[, p]); c2 <- mean(b$coverage$alpha[, p]); cp <- (c1 * n1 + c2 * n2) / (n1 + n2)
    zc <- (c1 - c2) / sqrt(cp * (1 - cp) * (1 / n1 + 1 / n2))
    ks <- suppressWarnings(ks.test(a$rank$alpha[, p], b$rank$alpha[, p]))$p.value
    data.frame(control = k, parameter = p, n_net = n1, n_ctl = n2, d_std_bias = d, p_std_bias = 2 * pnorm(-abs(zs)),
               d_coverage = c1 - c2, p_coverage = if (is.finite(zc)) 2 * pnorm(-abs(zc)) else 1, p_ks_ranks = ks)
  }))
}))
cmp$min_p <- pmin(cmp$p_std_bias, cmp$p_coverage, cmp$p_ks_ranks)
cmp$beyond_chance <- cmp$min_p < .05 / 18
num <- sapply(cmp, is.numeric); cmp2 <- cmp; cmp2[num] <- lapply(cmp2[num], signif, 3)
cat("\n=== network vs each control (unpaired) ===\n"); print(cmp2, row.names = FALSE)
write.csv(cmp2, file.path(root, "compare_crdm.csv"), row.names = FALSE)
for (k in unique(cmp$control)) cat(sprintf("DISAGREE network vs %s: %s (smallest p %.3g; Bonferroni threshold %.4f)\n", k,
  if (any(cmp$beyond_chance[cmp$control == k])) "YES" else "no", min(cmp$min_p[cmp$control == k]), .05 / 18))
cat(sprintf("SBC GATE: %s\n", if (all(tab$gate == "ok")) "all cells ok (|std bias| <= .5, ranks in envelope)" else
  paste("FLAG in", paste(unique(paste(tab$cell, tab$parameter)[tab$gate != "ok"]), collapse = ", "))))

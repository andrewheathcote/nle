# Does a fixed t0 correction fix the SBC failure, and does one value work at
# every N?
#
# The bias is constant on the SAMPLED (log) scale, not in milliseconds, so the
# correction is MULTIPLICATIVE: t0_corrected = t0_hat * exp(-delta).
#
# Bias/standardised bias are recomputed exactly. RANKS are not stored as
# posterior draws, so they are recomputed under a normal-posterior approximation:
#   rank = P(draw < truth) = Phi(-bias/s)  =>  s = -bias / qnorm(rank)
# recovers each replicate's posterior sd, and shifting the posterior by -delta
# gives rank' = Phi(-(bias-delta)/s). Exact for Gaussian posteriors; the ranks
# below are therefore approximate where the true posterior is skewed, while the
# bias columns are exact.
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
fs <- c(Sys.glob("sbc/nci/results_c4single/*_R500.rds"),
        "sbc/nci/results_c4single_N800/DDMnn_N800_R300.rds")
D <- lapply(fs, function(f) { x <- readRDS(f)
  list(N = x$trials, bias = x$SBC$bias$alpha$t0, rank = x$SBC$rank$alpha[, "t0"]) })
names(D) <- sapply(D, function(d) d$N)

stats <- function(b, r) {
  r <- r[is.finite(r)]
  ks <- suppressWarnings(ks.test(r, "punif")$p.value)
  c(std_bias = mean(b)/sd(b), mean_rank = mean(r), ks_p = ks, n_rank = length(r))
}
shifted <- function(d, delta) {
  s <- -d$bias / qnorm(pmin(pmax(d$rank, 1e-6), 1 - 1e-6))
  s[!is.finite(s) | s <= 0] <- NA
  list(b = d$bias - delta, r = pnorm(-(d$bias - delta)/s))
}

deltas <- sapply(D, function(d) mean(d$bias))
cat(sprintf("per-N mean log-scale bias: %s\n", paste(sprintf("N%s=%.5f", names(deltas), deltas), collapse="  ")))
cat(sprintf("pooled delta = %.5f  (t0 multiplied by %.5f; %.2f ms at t0=0.3s)\n\n",
            mean(deltas), exp(-mean(deltas)), mean(deltas)*300))

for (lbl in c("UNCORRECTED", "pooled delta", "LEAVE-ONE-N-OUT (honest)")) {
  cat(sprintf("=== %s ===\n", lbl))
  cat(sprintf("%-6s %12s %12s %12s %10s %7s\n","N","std_bias","mean_rank","KS p","delta","n_rank"))
  for (n in names(D)) {
    d <- D[[n]]
    del <- if (lbl == "UNCORRECTED") 0 else
           if (lbl == "pooled delta") mean(deltas) else mean(deltas[names(deltas) != n])
    z <- shifted(d, del); s <- stats(z$b, z$r)
    cat(sprintf("%-6s %+12.3f %12.3f %12.2e %10.5f %7d%s\n", n, s["std_bias"],
                s["mean_rank"], s["ks_p"], del, s["n_rank"],
                if (abs(s["std_bias"]) > 0.5) "  FAILS" else ""))
  }
  cat("\n")
}

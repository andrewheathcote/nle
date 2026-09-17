# Can we PREDICT SBC bias from the (cheap) score bias?
#
# A score bias g (per trial, expected over the true data distribution) displaces
# the posterior mode by, to first order,
#
#   delta = (N * I + Sigma0^-1)^-1 * (N * g)
#
# with I the per-trial Fisher information at the truth and Sigma0 the prior
# covariance. The posterior sd is sqrt(diag of the same inverse), so
#
#   standardised bias_j = delta_j / sqrt(V_jj),   V = (N*I + Sigma0^-1)^-1
#
# This matters because g costs a minute and SBC costs hours. It also explains
# why g_a alone does NOT predict the SBC bias on a: V is not diagonal, and a
# couples strongly to Z, so the displacement in a is driven by the whole g
# vector. In our two matched cases the two even have opposite signs.
#
# Validation: predict the 7 cells we have measured (c4single at N=100/200/400/
# 800, ens5 at N=100/200/400) and compare against the observed SBC bias.
#
#   Rscript sbc/predict_sbc_from_g.R
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(WienR))

# sampled-scale parameter order used throughout (s is a constant, excluded)
P <- c("v","a","t0","Z","sv","SZ","st0")
# SBC prior (sbc/nci/sbc_specs.R)
pmean <- c(v=1, a=log(.8), t0=log(.3), Z=qnorm(.5), sv=log(.1), SZ=qnorm(.2), st0=log(.05))
psd   <- c(v=.15, a=.15, t0=.1, Z=.05, sv=.15, SZ=.15, st0=.15)
pmean <- pmean[P]; psd <- psd[P]

# analytic log density; th is the 7-vector on the sampled scale, s fixed at 1
an_lpdf <- function(th, rt, R) {
  w <- pnorm(th[["Z"]])
  WienR::dWDM(rt, response = ifelse(R == 1, "lower", "upper"),
              a = exp(th[["a"]]), v = th[["v"]], w = w, t0 = exp(th[["t0"]]),
              sv = exp(th[["sv"]]), sw = 2 * pnorm(th[["SZ"]]) * min(w, 1 - w),
              st0 = exp(th[["st0"]]), precision = 1e-8)$logvalue
}

fisher_per_trial <- function(th, h = 0.02) {
  t0n <- exp(th[["t0"]])
  rt  <- t0n + exp(seq(log(1e-3), log(8), length.out = 400))
  dr  <- c(diff(rt)[1], (head(diff(rt), -1) + tail(diff(rt), -1)) / 2, tail(diff(rt), 1))
  I <- matrix(0, length(P), length(P), dimnames = list(P, P))
  for (R in 1:2) {
    lp <- an_lpdf(th, rt, R)
    wgt <- exp(lp) * dr                       # true defective density x width
    # score per rt, central differences on the sampled scale
    S <- sapply(P, function(p) {
      tp <- th; tm <- th; tp[[p]] <- tp[[p]] + h; tm[[p]] <- tm[[p]] - h
      (an_lpdf(tp, rt, R) - an_lpdf(tm, rt, R)) / (2 * h)
    })
    ok <- is.finite(wgt) & wgt > 0 & apply(is.finite(S), 1, all)
    I <- I + t(S[ok, , drop = FALSE] * wgt[ok]) %*% S[ok, , drop = FALSE]
  }
  I
}

cat("computing per-trial Fisher information at the SBC prior mean...\n")
I <- fisher_per_trial(pmean)
ev <- eigen(I, symmetric = TRUE)$values
cat(sprintf("  eigenvalues %.3g .. %.3g   (positive definite: %s)\n",
            min(ev), max(ev), all(ev > 0)))
cat(sprintf("  implied per-trial sd if orthogonal: %s\n",
            paste(sprintf("%s=%.3f", P, 1/sqrt(diag(I))), collapse=" ")))
cat("\n  correlation implied by I^-1 (posterior coupling):\n")
V1 <- solve(I); cc <- V1 / sqrt(outer(diag(V1), diag(V1)))
print(round(cc, 2))

predict_bias <- function(g, N) {
  V <- solve(N * I + diag(1 / psd^2))
  delta <- V %*% (N * g[P])
  as.numeric(delta) / sqrt(diag(V))
}
saveRDS(list(I = I, pmean = pmean, psd = psd, predict = predict_bias),
        "sbc/fisher_ddm.rds")
cat("\nsaved sbc/fisher_ddm.rds\n")

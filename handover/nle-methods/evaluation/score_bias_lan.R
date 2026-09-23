# Score bias for a LIKELIHOOD APPROXIMATION NETWORK (LAN), using our estimator.
#
# A LAN (Fengler et al., lnccbrown/LANfactory) is an MLP mapping
# (params, rt, choice) -> log-likelihood, distributed as ONNX and consumed by
# HSSM. Architecturally it is the same shape as our direct-regression baseline,
# so the SAME estimator applies with no change of method -- which is the point:
# comparing two models with one estimator, rather than two estimators.
#
# Estimator is sbc/score_bias_draws.R's, unchanged in form:
#   g = d/dtheta E_true[log f_hat] at the truth, central difference, expectation
#   over a log-spaced rt grid weighted by the TRUE defective density times the
#   trapezoid widths. Validated by running it on the ANALYTIC likelihood, which
#   must return ~0.
#
# Model mapping, determined empirically against WienR rather than assumed
# (agreement 0.003-0.10 log units at the check points):
#   a_wienr = 2 * a_lan   (LAN boundaries are at +-a)
#   w = z, t0 = t - st, st0 = 2 * st, sv = 0, sw = 0
# The LAN's non-decision time is Uniform(t - st, t + st), so `st` is the kernel
# HALF-width; WienR's st0 is the full width.
#
#   Rscript sbc/score_bias_lan.R [n_draws]
# Env: LAN_H (step, default 0.01), LAN_CORES (default 3 -- keep light if an SBC
#      is running)
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(jsonlite); library(WienR); library(parallel)})

ND <- as.integer(commandArgs(trailingOnly = TRUE)[1]); if (is.na(ND)) ND <- 100L
H <- as.numeric(Sys.getenv("LAN_H", "0.01"))
CORES <- as.integer(Sys.getenv("LAN_CORES", "3"))
PARS <- c("v", "a", "z", "st")        # t handled separately: moving support

L <- fromJSON("port/export/lan_ddm_uniform_st.json", simplifyVector = TRUE)
Ws <- lapply(seq_len(nrow(L$layers)), function(i) list(W = L$layers$W[[i]], b = L$layers$b[[i]]))
lan <- function(p, rt, choice) {          # p named c(v,a,z,t,st); choice +1/-1
  n <- length(rt)
  x <- cbind(matrix(rep(p[c("v","a","z","t","st")], each = n), nrow = n), rt, choice)
  for (i in 1:3) x <- tanh(sweep(x %*% Ws[[i]]$W, 2, Ws[[i]]$b, "+"))
  as.numeric(sweep(x %*% Ws[[4]]$W, 2, Ws[[4]]$b, "+"))
}
an <- function(p, rt, choice) WienR::dWDM(rt,
  response = if (choice > 0) "upper" else "lower",
  a = 2 * p[["a"]], v = p[["v"]], w = p[["z"]],
  t0 = p[["t"]] - p[["st"]], sv = 0, sw = 0, st0 = 2 * p[["st"]],
  precision = 1e-8)$logvalue

Lbar <- function(p, grid, f) { num <- 0; den <- 0
  for (ch in c(1, -1)) { lp <- f(p, grid$rt, ch); w <- grid$w[[if (ch > 0) 1 else 2]]
    ok <- is.finite(lp) & is.finite(w) & w > 0
    if (any(ok)) { num <- num + sum(w[ok] * lp[ok]); den <- den + sum(w[ok]) } }
  if (den <= 0) NA_real_ else num / den }
gslope <- function(p0, par, grid, f) { pp <- p0; pm <- p0
  pp[[par]] <- pp[[par]] + H; pm[[par]] <- pm[[par]] - H
  (Lbar(pp, grid, f) - Lbar(pm, grid, f)) / (2 * H) }

# draws from the network's OWN training box, held well inside it so nothing is
# extrapolation, and st kept above the 0.01 floor the model card recommends
set.seed(20260923)
draw <- function() c(v = runif(1, -2, 2), a = runif(1, 0.5, 2.0), z = runif(1, 0.35, 0.65),
                     t = runif(1, 0.3, 1.0), st = runif(1, 0.02, 0.20))
cat(sprintf("LAN ddm_uniform_st   draws %d   H = %g   cores %d\n", ND, H, CORES))
cat("box sampled well inside the training box; st above the 0.01 floor\n\n")

one <- function(i) {
  set.seed(20260923 + i); p <- draw()
  t0n <- p[["t"]] - p[["st"]]
  # Grid to 60 s, not 8: with a up to 2.0 (a_wienr = 4.0) and v near zero the
  # RT distribution has a long tail, and an 8 s grid loses up to 2.7% of the
  # mass -- which breaks the score identity and leaves the ANALYTIC control at
  # -0.021 on `a` no matter how small H gets. Truncation in the GRID and
  # truncation in H are different failures with the same symptom.
  rt <- t0n + exp(seq(log(1e-3), log(60), length.out = 240))
  dr <- c(diff(rt)[1], (head(diff(rt), -1) + tail(diff(rt), -1)) / 2, tail(diff(rt), 1))
  grid <- list(rt = rt, w = lapply(c(1, -1), function(ch) exp(an(p, rt, ch)) * dr))
  gl <- sapply(PARS, function(q) gslope(p, q, grid, lan))
  ga <- sapply(PARS, function(q) gslope(p, q, grid, an))
  # pointwise error where the mass is
  e <- unlist(lapply(c(1, -1), function(ch) { d <- lan(p, rt, ch) - an(p, rt, ch)
    ok <- is.finite(d) & exp(an(p, rt, ch)) * dr > 1e-6; d[ok] }))
  c(gl, setNames(ga, paste0(PARS, ".an")),
    med_err = median(abs(e)), max_err = max(abs(e)))
}
R <- do.call(rbind, mclapply(seq_len(ND), one, mc.cores = CORES))
ok <- apply(is.finite(R), 1, all); R <- R[ok, , drop = FALSE]
cat(sprintf("usable draws: %d / %d\n\n", sum(ok), ND))

cat("VALIDATION - the same estimator on the ANALYTIC likelihood (must be ~0)\n  ")
cat(paste(sprintf("%s=%+.5f", PARS, colMeans(R[, paste0(PARS, ".an"), drop = FALSE])),
          collapse = "  "), "\n\n")
cat("SCORE BIAS of the LAN\n")
cat(sprintf("%-10s %s\n", "", paste(sprintf("%12s", PARS), collapse = "")))
cat(sprintf("%-10s %s\n", "mean g",
    paste(sprintf("%+12.5f", colMeans(R[, PARS, drop = FALSE])), collapse = "")))
cat(sprintf("%-10s %s\n", "se",
    paste(sprintf("%12.5f", apply(R[, PARS, drop = FALSE], 2, sd) / sqrt(nrow(R))),
          collapse = "")))
cat(sprintf("\nPOINTWISE |log f_LAN - log f_analytic| where the mass is:\n"))
cat(sprintf("  median %.4f   mean of per-draw max %.4f\n",
            median(R[, "med_err"]), mean(R[, "max_err"])))
saveRDS(R, "sbc/score_bias_lan_results.rds")

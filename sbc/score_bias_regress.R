# Score bias and normalisation for the DIRECT-REGRESSION baseline.
#
# Uses exactly the estimator of sbc/score_bias_draws.R — same grid, same
# trapezoid weights, same central difference, same analytic control — so the
# comparison against a flow is a comparison of MODELS, not of estimators. Only
# the model evaluation differs: here a plain MLP maps
# (standardised theta, standardised log rt, R) to the JOINT log-density, with no
# classifier and no normalisation guarantee.
#
# It also reports what a flow gets for free and this does not:
#   TOTAL MASS  int sum_R exp(f_hat(rt, R | theta)) drt, which a flow makes
#   exactly 1 by construction. A CONSTANT departure from 1 is harmless for
#   inference (it shifts every log-likelihood equally), so what matters is how
#   much it VARIES with theta: the effective likelihood is f_hat - log Z(theta)
#   and the posterior is displaced by d/dtheta log Z. That displacement is
#   already inside the score bias below, but reporting Z separately says how
#   much of the bias is normalisation drift rather than shape error.
#
#   Rscript sbc/score_bias_regress.R <n_draws> <tag1> [tag2 ...]
# Tags are run tags, e.g. reg_b1k -> results/model_states/reg_b1k_budget/regress.json
# Env: SBG_CORES (default 4), SBG_H (step, default 0.01)

if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(jsonlite); library(parallel); library(WienR)})

args <- commandArgs(trailingOnly = TRUE)
n_draws <- as.integer(args[1]); tags <- args[-1]
CORES <- as.integer(Sys.getenv("SBG_CORES", "4"))
H <- as.numeric(Sys.getenv("SBG_H", "0.01"))
FI <- c(v=1L,a=2L,t0=3L,s=4L,Z=5L,SZ=6L,sv=7L,st0=8L)
PARS <- c("v","a","Z","sv","SZ","st0")     # t0 excluded: moving support

nets <- lapply(tags, function(t)
  fromJSON(sprintf("results/model_states/%s_budget/regress.json", t),
           simplifyVector = TRUE))
names(nets) <- tags

.gelu <- function(x) 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

# joint log-density from the regression net. theta is the 8-vector on the
# sampled scale, in the order the net was trained on.
reg_logL <- function(net, th, rt, R) {
  n <- length(rt)
  raw <- cbind(matrix(th[1:8], nrow = n, ncol = 8, byrow = TRUE), log(rt))
  z <- sweep(sweep(raw, 2, net$scaler_mean, "-"), 2, net$scaler_scale, "/")
  x <- cbind(z, rep(as.numeric(R), n))
  nl <- length(net$layers$W)
  for (i in seq_len(nl)) {
    x <- x %*% net$layers$W[[i]]
    x <- sweep(x, 2, net$layers$b[[i]], "+")
    if (i < nl) x <- .gelu(x)
  }
  as.numeric(x)
}

an_lpdf <- function(th, rt, R) { w <- pnorm(th[FI[["Z"]]])
  WienR::dWDM(rt, response = ifelse(R == 1, "lower", "upper"), a = exp(th[FI[["a"]]]),
    v = th[FI[["v"]]], w = w, t0 = exp(th[FI[["t0"]]]), sv = exp(th[FI[["sv"]]]),
    sw = 2 * pnorm(th[FI[["SZ"]]]) * min(w, 1 - w), st0 = exp(th[FI[["st0"]]]),
    precision = 1e-8)$logvalue }

Lbar <- function(th, grid, f) { num <- 0; den <- 0
  for (R in 1:2) { lp <- f(th, grid$rt, R); w <- grid$w[[R]]
    ok <- is.finite(lp) & is.finite(w) & w > 0
    if (any(ok)) { num <- num + sum(w[ok] * lp[ok]); den <- den + sum(w[ok]) } }
  if (den <= 0) NA_real_ else num / den }
gslope <- function(th0, par, grid, f) { j <- FI[[par]]
  tp <- th0; tp[j] <- tp[j] + H; tm <- th0; tm[j] <- tm[j] - H
  (Lbar(tp, grid, f) - Lbar(tm, grid, f)) / (2 * H) }

x <- readRDS("sbc/nci/results_ens5/DDMnn_N400_R500.rds")
truth <- as.matrix(x$SBC$med$alpha) - as.matrix(x$SBC$bias$alpha)
n_draws <- min(n_draws, nrow(truth))
cat(sprintf("draws: %d   tags: %s   cores: %d   H: %g\n\n", n_draws,
            paste(tags, collapse = ", "), CORES, H))

one <- function(i) {
  tr <- truth[i, ]
  th0 <- c(tr[["v"]], tr[["a"]], tr[["t0"]], 0, tr[["Z"]], tr[["SZ"]], tr[["sv"]], tr[["st0"]])
  t0n <- exp(tr[["t0"]])
  rt <- t0n + exp(seq(log(1e-3), log(8), length.out = 160))
  dr <- c(diff(rt)[1], (head(diff(rt), -1) + tail(diff(rt), -1)) / 2, tail(diff(rt), 1))
  grid <- list(rt = rt, w = lapply(1:2, function(R) exp(an_lpdf(th0, rt, R)) * dr))
  g <- unlist(lapply(PARS, function(p) {
    v <- c(sapply(tags, function(tg)
             gslope(th0, p, grid, function(th, rt, R) reg_logL(nets[[tg]], th, rt, R))),
           analytic = gslope(th0, p, grid, an_lpdf))
    stats::setNames(v, paste0(p, ".", names(v)))
  }))
  # total mass, on a grid wide enough to hold essentially all of it
  rtm <- t0n + exp(seq(log(1e-4), log(30), length.out = 600))
  drm <- c(diff(rtm)[1], (head(diff(rtm), -1) + tail(diff(rtm), -1)) / 2, tail(diff(rtm), 1))
  mass <- sapply(tags, function(tg)
    sum(sapply(1:2, function(R) sum(exp(reg_logL(nets[[tg]], th0, rtm, R)) * drm))))
  c(g, stats::setNames(mass, paste0("mass.", tags)))
}

res <- do.call(rbind, mclapply(seq_len(n_draws), one, mc.cores = CORES))
ok <- apply(is.finite(res), 1, all)
cat(sprintf("usable draws: %d / %d\n\n", sum(ok), n_draws))
res <- res[ok, , drop = FALSE]

cat("VALIDATION  estimator on the ANALYTIC likelihood (must be ~0):\n  ")
cat(paste(sprintf("%s=%+.5f", PARS,
                  sapply(PARS, function(p) mean(res[, paste0(p, ".analytic")]))),
          collapse = "  "), "\n\n")

cat("SCORE BIAS\n")
cat(sprintf("%-14s %s\n", "tag", paste(sprintf("%11s", PARS), collapse = " ")))
for (tg in tags)
  cat(sprintf("%-14s %s\n", tg,
      paste(sprintf("%+11.5f", sapply(PARS, function(p) mean(res[, paste0(p, ".", tg)]))),
            collapse = " ")))

cat("\nTOTAL MASS  (a flow is exactly 1 by construction)\n")
for (tg in tags) {
  m <- res[, paste0("mass.", tg)]
  cat(sprintf("  %-14s mean %.5f   sd across theta %.5f   range [%.4f, %.4f]\n",
              tg, mean(m), sd(m), min(m), max(m)))
  cat(sprintf("  %-14s  -> sd of log Z = %.5f; a CONSTANT offset is harmless,\n",
              "", sd(log(m))))
  cat(sprintf("  %-14s     it is the variation with theta that displaces the posterior\n", ""))
}
saveRDS(res, "sbc/score_bias_regress_results.rds")

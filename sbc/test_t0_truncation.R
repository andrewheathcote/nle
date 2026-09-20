# Does enforcing the true support -- ll = -Inf when t0 > rt -- fix the t0
# displacement?
#
# The score-bias estimator integrates over a grid starting AT t0, so it never
# probes below the boundary and cannot see this defect. The place it bites is
# inference: when the sampler proposes t0 above an observed rt, the analytic
# penalty is infinite while the flow's is ~10 log units, so t0 drifts up.
#
# Cheap decisive test: simulate a dataset at a known truth, then profile the
# log-likelihood in t0 holding everything else at the truth, for
#   (a) the analytic, (b) the flow as shipped, (c) the flow truncated.
# Report where each profile peaks relative to the true t0.
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(WienR)})
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")
TAG <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(TAG)) TAG <- "cap256w_c4"
ST0Z <- grepl("^z0_", TAG)
fl <- ddm_load(sprintf("port/export/ddm_%s_weights.json", TAG))
N <- as.integer(Sys.getenv("NTRIAL", "400"))

x <- readRDS("sbc/nci/results_ens5/DDMnn_N400_R500.rds")
truth <- as.matrix(x$SBC$med$alpha) - as.matrix(x$SBC$bias$alpha)

an_l <- function(th, rt, R) { w <- pnorm(th[5])
  WienR::dWDM(rt, response = ifelse(R == 1, "lower", "upper"), a = exp(th[2]),
    v = th[1], w = w, t0 = exp(th[3]), sv = exp(th[7]),
    sw = 2 * pnorm(th[6]) * min(w, 1 - w),
    st0 = if (ST0Z) 0 else exp(th[8]), precision = 1e-8)$logvalue }

cat(sprintf("flow %s   N = %d trials/dataset\n", TAG, N))
cat(sprintf("%-5s %11s %11s %11s %11s\n", "draw", "true t0",
            "analytic", "flow", "flow+trunc"))
res <- matrix(NA_real_, 0, 3)
for (i in 1:8) {
  tr <- truth[i, ]
  th <- if (ST0Z) c(tr[["v"]],tr[["a"]],tr[["t0"]],0,tr[["Z"]],tr[["SZ"]],tr[["sv"]])
        else c(tr[["v"]],tr[["a"]],tr[["t0"]],0,tr[["Z"]],tr[["SZ"]],tr[["sv"]],tr[["st0"]])
  t0n <- exp(tr[["t0"]])
  # simulate a dataset from the ANALYTIC at the truth (inverse-cdf by grid)
  set.seed(1000 + i)
  grid <- t0n + exp(seq(log(1e-4), log(20), length.out = 4000))
  dens <- lapply(1:2, function(R) exp(an_l(th, grid, R)))
  dr <- c(diff(grid), tail(diff(grid), 1))
  pR <- sapply(dens, function(d) sum(d * dr)); pR <- pR / sum(pR)
  R <- sample(1:2, N, TRUE, prob = pR)
  rt <- sapply(R, function(r) {
    cw <- cumsum(dens[[r]] * dr); cw <- cw / tail(cw, 1)
    grid[which.max(cw >= runif(1))] })

  # profile in t0, everything else at the truth
  lt <- seq(log(t0n) - 0.25, log(t0n) + 0.25, length.out = 121)
  prof <- function(f, trunc) sapply(lt, function(v) {
    th2 <- th; th2[3] <- v
    ll <- sapply(1:2, function(r) { k <- R == r
      if (!any(k)) return(0)
      l <- f(th2, rt[k], r)
      if (trunc) l[rt[k] <= exp(v)] <- -Inf
      sum(l) })
    sum(ll) })
  pa <- prof(an_l, TRUE)                                   # analytic: already 0 below t0
  pf <- prof(function(t, r, rr) ddm_eval(fl, t, r, rr)$log_pdf, FALSE)
  pt <- prof(function(t, r, rr) ddm_eval(fl, t, r, rr)$log_pdf, TRUE)
  pk <- function(p) exp(lt[which.max(replace(p, !is.finite(p), -Inf))]) - t0n
  v <- c(pk(pa), pk(pf), pk(pt)) * 1000
  res <- rbind(res, v)
  cat(sprintf("%-5d %11.4f %+11.2f %+11.2f %+11.2f\n", i, t0n, v[1], v[2], v[3]))
}
cat(sprintf("\n%-5s %11s %+11.2f %+11.2f %+11.2f   (ms from truth)\n", "MEAN", "",
            mean(res[,1]), mean(res[,2]), mean(res[,3])))
cat(sprintf("%-5s %11s %11.2f %11.2f %11.2f   (sd)\n", "SD", "",
            sd(res[,1]), sd(res[,2]), sd(res[,3])))

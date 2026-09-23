# Our DDM flow against a LIKELIHOOD APPROXIMATION NETWORK (LAN), matched.
#
# `sbc/score_bias_lan.R` scored the LAN on ITS parameterisation over ITS box.
# That is not comparable with our numbers, for two reasons this script removes:
#
#   SCALE. Our score bias differentiates on the SAMPLED scale (log a, probit w,
#   log st0); that script perturbed the LAN's natural a. Since d/dlog a =
#   a * d/da, the two differ by a factor of a -- roughly 1.25 over that box.
#   RE-EXPRESSED, the LAN's a-score is ~8x larger than the raw number suggests.
#
#   REGION. The two models were scored over different parameter boxes.
#
# What is NOT removed, because it cannot be: the models are not nested. The LAN
# has sv = 0 and sw = 0 BY CONSTRUCTION; our flow's training box floors them at
# sv = 0.01 and SZ = 0.05. So each model is scored against ITS OWN exact truth,
# on the same four parameters, on the same scales, over the same region of
# (v, a, w, t0, st0). That is the fairest available comparison, and the residual
# difference in the generating model is stated rather than hidden.
#
#   Rscript sbc/compare_lan.R [n_draws]
# Env: CL_CORES (default 3), CL_H (default 0.01)
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(jsonlite); library(WienR); library(parallel)
                                library(EMC2)})
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")

ND <- as.integer(commandArgs(trailingOnly = TRUE)[1]); if (is.na(ND)) ND <- 80L
H <- as.numeric(Sys.getenv("CL_H", "0.01"))
CORES <- as.integer(Sys.getenv("CL_CORES", "3"))

L <- fromJSON("port/export/lan_ddm_uniform_st.json", simplifyVector = TRUE)
Ws <- lapply(seq_len(nrow(L$layers)), function(i) list(W = L$layers$W[[i]], b = L$layers$b[[i]]))
lan_lpdf <- function(v, a_w, w, t0, st0, rt, ch) {      # our convention in, LAN inside
  n <- length(rt)
  x <- cbind(matrix(rep(c(v, a_w / 2, w, t0 + st0 / 2, st0 / 2), each = n), nrow = n),
             rt, ch)
  for (i in 1:3) x <- tanh(sweep(x %*% Ws[[i]]$W, 2, Ws[[i]]$b, "+"))
  as.numeric(sweep(x %*% Ws[[4]]$W, 2, Ws[[4]]$b, "+"))
}
fl <- ddm_load("port/export/ddm_cap256w_c4_weights.json")
SV_MIN <- 0.01; SZ_MIN <- 0.05        # our training-box floors
flow_lpdf <- function(v, a_w, w, t0, st0, rt, ch) {
  th <- c(v = v, a = log(a_w), t0 = log(t0), s = 0, Z = qnorm(w),
          SZ = qnorm(SZ_MIN), sv = log(SV_MIN), st0 = log(st0))
  ddm_eval(fl, th, rt, if (ch > 0) 2L else 1L)$log_pdf
}
an_lpdf <- function(v, a_w, w, t0, st0, rt, ch, sv = 0, sw = 0)
  WienR::dWDM(rt, response = if (ch > 0) "upper" else "lower", a = a_w, v = v, w = w,
              t0 = t0, sv = sv, sw = sw, st0 = st0, precision = 1e-8)$logvalue

# the four parameters both models have, on OUR sampled scales
PARS <- c("v", "log_a", "probit_w", "log_st0")
perturb <- function(p, par, d) { q <- p
  q[[switch(par, v = "v", log_a = "a_w", probit_w = "w", log_st0 = "st0")]] <-
    switch(par,
      v        = p$v + d,
      log_a    = exp(log(p$a_w) + d),
      probit_w = pnorm(qnorm(p$w) + d),
      log_st0  = exp(log(p$st0) + d))
  q }

# expectation of log f_hat under the model's own truth, trapezoid-weighted, on a
# grid to 60 s (an 8 s grid loses up to 2.7% of the mass in this box and pins the
# analytic control at -0.02 on a, whatever H is -- see score_bias_lan.R)
Lbar <- function(p, f, grid) { num <- 0; den <- 0
  for (ch in c(1, -1)) { lp <- f(p$v, p$a_w, p$w, p$t0, p$st0, grid$rt, ch)
    wt <- grid$w[[if (ch > 0) 1 else 2]]
    ok <- is.finite(lp) & is.finite(wt) & wt > 0
    if (any(ok)) { num <- num + sum(wt[ok] * lp[ok]); den <- den + sum(wt[ok]) } }
  if (den <= 0) NA_real_ else num / den }
gvec <- function(p, f, grid) sapply(PARS, function(q)
  (Lbar(perturb(p, q, H), f, grid) - Lbar(perturb(p, q, -H), f, grid)) / (2 * H))

mkgrid <- function(p, sv, sw) { rt <- p$t0 + exp(seq(log(1e-3), log(60), length.out = 240))
  dr <- c(diff(rt)[1], (head(diff(rt), -1) + tail(diff(rt), -1)) / 2, tail(diff(rt), 1))
  list(rt = rt, w = lapply(c(1, -1), function(ch)
         exp(an_lpdf(p$v, p$a_w, p$w, p$t0, p$st0, rt, ch, sv, sw)) * dr)) }

# region: inside BOTH training boxes with margin (see the box table below)
set.seed(20260924)
one <- function(i) {
  set.seed(20260924 + i)
  p <- list(v = runif(1, -2, 2), a_w = runif(1, 0.8, 3.0), w = runif(1, 0.35, 0.65),
            t0 = runif(1, 0.15, 0.8), st0 = runif(1, 0.02, 0.30))
  sw_f <- 2 * SZ_MIN * min(p$w, 1 - p$w)
  g_lan  <- gvec(p, lan_lpdf,  mkgrid(p, 0, 0))
  a_lan  <- gvec(p, function(...) an_lpdf(..., sv = 0, sw = 0), mkgrid(p, 0, 0))
  g_flow <- gvec(p, flow_lpdf, mkgrid(p, SV_MIN, sw_f))
  a_flow <- gvec(p, function(...) an_lpdf(..., sv = SV_MIN, sw = sw_f),
                 mkgrid(p, SV_MIN, sw_f))
  c(setNames(g_lan, paste0("lan.", PARS)), setNames(a_lan, paste0("lanA.", PARS)),
    setNames(g_flow, paste0("flow.", PARS)), setNames(a_flow, paste0("flowA.", PARS)))
}
R <- do.call(rbind, mclapply(seq_len(ND), one, mc.cores = CORES))
R <- R[apply(is.finite(R), 1, all), , drop = FALSE]

cat(sprintf("\nMATCHED SCORE BIAS   %d draws, H = %g, our sampled scales\n", nrow(R), H))
cat(sprintf("%-12s %11s %11s %11s %11s\n", "", PARS[1], PARS[2], PARS[3], PARS[4]))
row <- function(lab, pre) { m <- colMeans(R[, paste0(pre, ".", PARS), drop = FALSE])
  s <- apply(R[, paste0(pre, ".", PARS), drop = FALSE], 2, sd) / sqrt(nrow(R))
  cat(sprintf("%-12s %s\n", lab, paste(sprintf("%+7.4f(%.4f)", m, s), collapse = " "))) }
row("LAN", "lan"); row("  control", "lanA")
row("our flow", "flow"); row("  control", "flowA")

## ---- speed, single core, per 1000 trial evaluations ------------------------
bench <- function(f, n = 1000, reps = 20) {
  rt <- runif(n, 0.4, 1.5); ch <- sample(c(1, -1), n, TRUE)
  t <- system.time(for (k in seq_len(reps)) f(rt, ch))[3] / reps
  unname(t * 1000) }
p <- list(v = 1, a_w = 1.6, w = .5, t0 = .3, st0 = .1)
ms <- c(
  `LAN (R, this script)`      = bench(function(rt, ch) lan_lpdf(p$v,p$a_w,p$w,p$t0,p$st0, rt, 1)),
  `our flow (R port)`         = bench(function(rt, ch) flow_lpdf(p$v,p$a_w,p$w,p$t0,p$st0, rt, 1)),
  `our flow (compiled port)`  = bench(function(rt, ch) {
     pars <- cbind(v=p$v, a=p$a_w, t0=p$t0, s=1, Z=p$w, SZ=SZ_MIN, sv=SV_MIN, st0=p$st0)[rep(1,length(rt)), ]
     EMC2:::dDDMnn(rt, rep(2L, length(rt)), pars) }),
  `analytic exact 1e-8`       = bench(function(rt, ch) an_lpdf(p$v,p$a_w,p$w,p$t0,p$st0, rt, 1, .01, .05)),
  `analytic default 5e-3`     = bench(function(rt, ch)
     WienR::dWDM(rt, response="upper", a=p$a_w, v=p$v, w=p$w, t0=p$t0, sv=.01,
                 sw=.05, st0=p$st0, precision=5e-3)$logvalue))
cat("\nSPEED, ms per 1000 trials, one core (R-level for both networks)\n")
for (n in names(ms)) cat(sprintf("  %-24s %8.2f ms\n", n, ms[[n]]))

saveRDS(list(scores = R, speed_ms = ms), "sbc/compare_lan_results.rds")
cat("\nwrote sbc/compare_lan_results.rds\n")

# Speed of the flow against WienR, for likelihood ALONE and likelihood+CDF.
#
# The flow returns pdf and cdf from ONE evaluation -- the cdf is Phi(z), already
# computed on the way to the density -- so its pdf+cdf cost equals its pdf cost.
# WienR needs two calls. That asymmetry is the whole point of the comparison and
# is why it must be measured rather than assumed.
#
#   Rscript sbc/bench_pdf_cdf.R
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(WienR)})
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")

N <- as.integer(Sys.getenv("NBENCH", "10000"))
REPS <- as.integer(Sys.getenv("REPS", "3"))
set.seed(1)
rt <- 0.3 + rexp(N, 2); Ri <- sample(1:2, N, TRUE)
Rc <- ifelse(Ri == 1, "lower", "upper")
us <- function(sec) 1e6 * sec / N * 1000          # microseconds per 1000 trials
best <- function(f) { b <- Inf
  for (i in seq_len(REPS)) b <- min(b, system.time(f())[["elapsed"]]); b }

fl <- ddm_load("port/export/ddm_cap256w_c4_weights.json")
ptr <- EMC2:::ddm_build(fl)
th <- c(1, log(.8), log(.3), 0, qnorm(.5), qnorm(.2), log(.1), log(.05))
Th <- matrix(th, nrow = N, ncol = 8, byrow = TRUE)
tf <- best(function() EMC2:::ddm_eval_trials_cpp(ptr, Th, rt, as.integer(Ri)))
cat(sprintf("FLOW cap256w_c4 (one call -> pdf AND cdf): %.0f us/1000\n\n", us(tf)))
flush.console()

W <- function(st0, prec, cdf) {
  f <- if (cdf) WienR::pWDM else WienR::dWDM
  best(function() f(rt, response = Rc, a = .8, v = 1, w = .5, t0 = .3,
                    sv = .1, sw = .2, st0 = st0, precision = prec))
}
cfgs <- list(list("st0 estimated (0.05), 1e-8", .05, 1e-8),
             list("st0 estimated (0.05), 5e-3", .05, 5e-3),
             list("st0 = 0 (fixed),      1e-8", 0,   1e-8),
             list("st0 = 0 (fixed),      5e-3", 0,   5e-3))
cat(sprintf("%-30s %10s %10s %11s %9s %9s\n", "WienR", "pdf", "cdf",
            "pdf+cdf", "flow pdf", "flow both"))
for (k in cfgs) {
  d <- us(W(k[[2]], k[[3]], FALSE)); p <- us(W(k[[2]], k[[3]], TRUE))
  cat(sprintf("%-30s %10.0f %10.0f %11.0f %8.1fx %8.1fx\n",
              k[[1]], d, p, d + p, d / us(tf), (d + p) / us(tf)))
  flush.console()
}

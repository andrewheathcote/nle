# Plumbing test for the zero-box encoding, runnable BEFORE any zero-box flow
# exists. It takes the shipped cap256w_c4 export, relabels it as zerobox with
# the plan's constants and network-scale bounds, and checks that the R port and
# the compiled port agree on everything the encoding touches. The numbers are
# not a model of anything; the agreement is the point.
#
#   Rscript port/tests/validate_zerobox_plumbing.R
if (dir.exists("../../.Rlibs")) .libPaths(c(normalizePath("../../.Rlibs"), .libPaths()))
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(jsonlite))
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")
Rcpp::sourceCpp("port/src/flow_ddm.cpp", verbose = FALSE)
ok <- function(cond, lab, ...) { cat(sprintf("%s  %s\n", if (cond) "PASS" else "FAIL", sprintf(lab, ...))); cond }

fl <- fromJSON("port/export/ddm_cap256w_c4_weights.json", simplifyDataFrame = FALSE)
ZC <- list(sv = 0.01, SZ = 0.05, st0 = 0.002)
fl$context_encoding <- "zerobox"; fl$zerobox_c <- ZC
cn <- fl$context_names
zb <- function(x, p) asinh(x / ZC[[p]])
lo <- fl$bounds_sampled$lower; hi <- fl$bounds_sampled$upper
for (p in names(ZC)) { j <- match(p, cn); lo[j] <- 0
  hi[j] <- zb(c(sv = 4, SZ = 0.99, st0 = 0.49)[[p]], p) }
fl$bounds_sampled <- list(lower = lo, upper = hi)
tmp <- tempfile(fileext = ".json"); write_json(fl, tmp, auto_unbox = TRUE, digits = NA)
flR <- ddm_load(tmp)
ptr <- ddm_build(fromJSON(tmp, simplifyDataFrame = FALSE))

set.seed(7); all_ok <- TRUE
draw <- function(regime) {
  th <- c(v = rnorm(1, 1, .5), a = log(runif(1, .5, 2)), t0 = log(runif(1, .2, .6)), s = 0,
          Z = qnorm(runif(1, .35, .65)), SZ = qnorm(runif(1, .1, .5)),
          sv = log(runif(1, .05, 1)), st0 = log(runif(1, .01, .3)))[cn]
  if (regime %in% c("st0_0", "all_0")) th[["st0"]] <- -Inf
  if (regime %in% c("sv_0", "all_0")) th[["sv"]] <- -Inf
  if (regime %in% c("SZ_0", "all_0")) th[["SZ"]] <- -Inf
  th }
worst <- 0; box_agree <- TRUE; rule_ok <- TRUE
for (regime in c("full", "st0_0", "sv_0", "SZ_0", "all_0")) for (k in 1:20) {
  th <- draw(regime); t0n <- exp(th[["t0"]])
  rt <- c(t0n * 0.9, t0n, t0n + 1e-4, t0n + runif(5, .01, 1.5)); R <- rep(1:2, length.out = length(rt))
  r <- ddm_eval(flR, th, rt, R)
  cc <- ddm_eval_trials_cpp(ptr, matrix(th, length(rt), length(th), byrow = TRUE), rt, as.integer(R))
  box_agree <- box_agree && isTRUE(r$in_box)
  fin <- is.finite(r$log_pdf) & is.finite(cc$log_pdf)
  worst <- max(worst, max(abs(r$log_pdf[fin] - cc$log_pdf[fin])))
  # -Inf must appear in the same places, and only below/at t0 when st0 = 0
  same_inf <- identical(is.infinite(r$log_pdf), is.infinite(cc$log_pdf))
  expect_inf <- if (is.infinite(th[["st0"]])) rt <= t0n else rep(FALSE, length(rt))
  rule_ok <- rule_ok && same_inf && identical(unname(is.infinite(r$log_pdf)), unname(expect_inf))
}
all_ok <- ok(box_agree, "every draw with exact zeros is inside the network-scale box") && all_ok
all_ok <- ok(worst < 1e-10, "R port vs compiled port, 100 draws x 8 rt over 5 regimes: max |diff| %.1e", worst) && all_ok
all_ok <- ok(rule_ok, "st0 = 0 support rule: -Inf exactly at rt <= t0, identical in both ports, absent otherwise") && all_ok
# the encoding itself: exact zero -> exactly 0, and a value above the old floor is log-like
ctx <- .emc2_to_zerobox(flR, draw("all_0"))
all_ok <- ok(all(ctx[c("sv", "SZ", "st0")] == 0), "exact zeros map to context exactly 0") && all_ok
all_ok <- ok(abs(zb(1, "sv") - log(2 / 0.01)) < 1e-3, "asinh is log-like above c: asinh(1/0.01) vs log(2/0.01), diff %.1e", abs(zb(1, "sv") - log(2 / 0.01))) && all_ok
cat(if (all_ok) "\nALL PASS\n" else "\nFAILURES ABOVE\n"); quit(status = if (all_ok) 0 else 1)

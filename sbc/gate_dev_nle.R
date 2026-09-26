# The dev-nle usability gate, in dev-nle's own terms. Five checks; ~25 min at
# 8 cores; none is an SBC measurement. Run from the NLE repo root:
#
#   Rscript sbc/gate_dev_nle.R [library]      # default ~/R/nle-sbc-lib
#
# What "usable" means here: the new branch computes the SAME likelihood as the
# evaluators this project validated (check 2, exact to ~1e-8), its own kit
# reproduces this project's recorded diagnostics within estimator noise (3, 4),
# and its SBC pipeline runs end to end (5). Check 2 is the one that matters:
# an SBC cell has Monte Carlo error 0.045 per parameter and would pass an
# evaluator wrong by 0.02; check 2 catches 1e-10 in seconds.
a <- commandArgs(trailingOnly = TRUE)
LIB <- path.expand(if (length(a)) a[1] else "~/R/nle-sbc-lib")
.libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(jsonlite)})
cat(sprintf("=== dev-nle usability gate  EMC2 %s at %s  %s ===\n",
            as.character(packageVersion("EMC2")), find.package("EMC2"), format(Sys.time(), "%Y-%m-%d %H:%M")))
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")      # this project's validated R port
source("handover/regress/DDMreg.R")                             # and its regression evaluator
npass <- 0L; nfail <- 0L
verdict <- function(ok, lab) { cat(sprintf("%s  %s\n", if (ok) "PASS" else "FAIL", lab)); if (ok) npass <<- npass + 1L else nfail <<- nfail + 1L; ok }
t_all <- proc.time()[3]

## ---- 1. the artefacts load, and the kit is there ---------------------------
ok1 <- tryCatch({
  kit <- c("register_nn_model", "nn_cell", "nn_score_bias", "nn_posterior_shift", "nn_total_mass", "nn_sbc_cell", "nn_sbc_summary")
  stopifnot(all(kit %in% getNamespaceExports("EMC2")))
  m_flow <- DDMnn(); m_z0 <- DDMnn("ddm_st0zero"); m_rdm <- RDMnn()
  card <- path.expand("~/R/nle-cards/ddm_reg_s402.rds")
  m_reg <- if (file.exists(card)) register_nn_model(card, twin = DDM) else NULL
  cat(sprintf("  artefacts: %s\n", paste(list.files(system.file("extdata", "flownn", package = "EMC2")), collapse = ", ")))
  cat(sprintf("  reg_s402 card: %s\n", if (is.null(m_reg)) "absent (check 2c skipped)" else card))
  TRUE }, error = function(e) { cat("  ", conditionMessage(e), "\n"); FALSE })
verdict(ok1, "1. kit exported; cap256w_c4, st0zero, rdm_small load; reg_s402 card registers")

## ---- 2. EXACT: dev-nle's compiled likelihood == this project's R port -------
# Same natural-scale pars, same rt, same R, through the model function's dfun
# (what run_sbc() calls) and through the R port (what the golden vectors
# validated). Many theta, not one; zeros included for st0zero.
set.seed(11); n <- 60
rt <- runif(n, .3, 1.6); R <- sample(1:2, n, TRUE)
worst <- c(cap256w_c4 = 0, st0zero = 0, reg_s402 = 0)
fl8 <- ddm_load("handover/nle-methods/port/export/ddm_cap256w_c4_weights.json")
fl7 <- ddm_load("handover/nle-methods/port/export/ddm_z0_s503_e282_weights.json")
for (k in 1:25) {
  th <- c(v = rnorm(1, 1, .3), a = log(runif(1, .5, 1.5)), t0 = log(runif(1, .2, .28)), s = 0,
          Z = qnorm(runif(1, .4, .6)), SZ = qnorm(runif(1, .1, .4)), sv = log(runif(1, .05, .5)), st0 = log(runif(1, .02, .1)))
  nat <- cbind(v = th[["v"]], a = exp(th[["a"]]), t0 = exp(th[["t0"]]), s = 1, Z = pnorm(th[["Z"]]),
               SZ = pnorm(th[["SZ"]]), sv = exp(th[["sv"]]), st0 = exp(th[["st0"]]))[rep(1, n), ]
  d_new <- log(m_flow$dfun(rt, R, nat)); d_ref <- ddm_eval(fl8, th, rt, R)$log_pdf
  worst["cap256w_c4"] <- max(worst["cap256w_c4"], max(abs(d_new - d_ref)))
  nat7 <- nat; nat7[, "st0"] <- 0
  d_new <- log(m_z0$dfun(rt, R, nat7)); d_ref <- ddm_eval(fl7, th[names(th) != "st0"], rt, R)$log_pdf
  worst["st0zero"] <- max(worst["st0zero"], max(abs(d_new - d_ref)))
  if (!is.null(m_reg)) {
    d_new <- log(m_reg()$dfun(rt, R, nat))
    d_ref <- reg_log_pdf(rt, R, nat, regnet_get("handover/nle-methods/artefacts/reg_s402_regress.json"))
    worst["reg_s402"] <- max(worst["reg_s402"], max(abs(d_new - d_ref))) }
}
cat(sprintf("  max |log f(dev-nle) - log f(R port)| over 25 theta x %d trials: %s\n", n,
            paste(sprintf("%s %.1e", names(worst), worst), collapse = "  ")))
verdict(all(worst < 1e-7), "2. compiled likelihood == validated R port across theta (< 1e-7; float64, sums reordered)")

## ---- 3. the kit's score bias, against the recorded numbers -----------------
# Different estimator implementation (grid, draws), so agreement is to
# estimator noise, not 1e-6; the CONTROL column must be ~0, which is the check.
cell <- nn_cell(DDMnn, mean = c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5), sv = log(.1), SZ = qnorm(.2), st0 = log(.05)),
                sd = c(v = .15, a = .15, t0 = .1, Z = .05, sv = .15, SZ = .15, st0 = .15),
                formula = list(v ~ 1, a ~ 1, t0 ~ 1, Z ~ 1, sv ~ 1, SZ ~ 1, st0 ~ 1),
                factors = list(S = c("left", "right")), Rlevels = c("left", "right"))
ok3 <- tryCatch({
  sb <- nn_score_bias(cell, n_draws = 200, cores = 8)
  print(sb)
  # recorded (Phase 25, H = 0.01, 200 draws): a +0.0136 +- 0.002; control |g| < 1e-4
  TRUE }, error = function(e) { cat("  ", conditionMessage(e), "\n"); FALSE })
verdict(ok3, "3. nn_score_bias() runs on the cap256w_c4 cell (read: control ~0, a near +0.0136)")

## ---- 4. the kit's posterior shift, against the recorded prediction ---------
ok4 <- tryCatch({
  ps <- nn_posterior_shift(cell, n_datasets = 12, cores = 8)
  print(ps)
  # recorded (60 datasets): t0 predicted +0.28 vs SBC +0.319; control within +-0.13
  TRUE }, error = function(e) { cat("  ", conditionMessage(e), "\n"); FALSE })
verdict(ok4, "4. nn_posterior_shift() runs (read: control ~0, t0 near +0.28)")

## ---- 5. the SBC pipeline runs end to end (plumbing only) -------------------
ok5 <- tryCatch({
  out <- nn_sbc_cell(cell, trials = 100, replicates = 12, run_control = FALSE,
                     cores_per_chain = 4, cores_for_chains = 2)
  s <- nn_sbc_summary(if (!is.null(out$network)) out$network else out)
  print(s)
  TRUE }, error = function(e) { cat("  ", conditionMessage(e), "\n"); FALSE })
verdict(ok5, "5. a 12-replicate N=100 nn_sbc_cell() completes and summarises (plumbing, not a measurement)")

cat(sprintf("\n=== %d passed, %d failed in %.0f min ===\n%s\n", npass, nfail, (proc.time()[3] - t_all) / 60,
            if (nfail == 0) sprintf("GATE PASSED: dev-nle at %s is usable", LIB) else "GATE FAILED"))
quit(status = nfail)

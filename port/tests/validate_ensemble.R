# Validate the C++ ensemble path (ddm_build_ensemble / ddm_ens_eval_trials_cpp)
# now shipped in EMC2 as the DDMnn likelihood (ddm_ens5.rds):
#   1. mixture identity: ensemble pdf/cdf == mean of member pdf/cdf (1e-12)
#   2. EMC2 DDMnn log-likelihood == R-side 5-member averaging (gate-study impl)
#   3. box rejection through the ensemble path
#   4. speed of the ensemble likelihood in a full calc_ll_R call
# Run from the nle repo root:  Rscript port/tests/validate_ensemble.R
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(EMC2))
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")

fails <- 0L
check <- function(label, diff, tol) {
  ok <- is.finite(diff) && diff < tol
  cat(sprintf("%-52s max|diff| = %10.3e  tol = %g  %s\n",
              label, diff, tol, if (ok) "PASS" else "FAIL"))
  if (!ok) fails <<- fails + 1L
}

tags <- c("arch64w", "arch64w_s101", "arch64w_s102", "arch64w_s103", "arch64w_s104")
members <- lapply(tags, function(tg) ddm_load(sprintf("port/export/ddm_%s_weights.json", tg)))
ens_ptr <- EMC2:::ddm_build_ensemble(members)
mem_ptrs <- lapply(members, EMC2:::ddm_build)

# --- 1. mixture identity on a broad grid ---
set.seed(7)
lo <- members[[1]]$bounds_sampled$lower; hi <- members[[1]]$bounds_sampled$upper
n <- 400L
Theta <- t(replicate(n, runif(length(lo), lo, hi)))
rt <- exp(rnorm(n, -0.3, 0.6)); R <- sample(1:2, n, TRUE)
ev <- EMC2:::ddm_ens_eval_trials_cpp(ens_ptr, Theta, rt, R)
mean_pdf <- Reduce(`+`, lapply(mem_ptrs, function(p)
  EMC2:::ddm_eval_trials_cpp(p, Theta, rt, R)$pdf)) / length(mem_ptrs)
mean_cdf <- Reduce(`+`, lapply(mem_ptrs, function(p)
  EMC2:::ddm_eval_trials_cpp(p, Theta, rt, R)$cdf)) / length(mem_ptrs)
check("ensemble pdf == mean member pdf", max(abs(ev$pdf - mean_pdf)), 1e-12)
check("ensemble cdf == mean member cdf", max(abs(ev$cdf - mean_cdf)), 1e-12)

# --- 2. EMC2 DDMnn (ens5) vs the gate study's R-side averaging ---
des <- design(factors = list(subjects = 1, S = c("left", "right")),
              Rlevels = c("left", "right"),
              formula = list(v ~ 1, a ~ 1, t0 ~ 1, s ~ 1, Z ~ 1, sv ~ 1,
                             SZ ~ 1, st0 ~ 1),
              constants = c(s = log(1)), model = DDMnn, verbose = FALSE)
p_true <- c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5),
            sv = log(.1), SZ = qnorm(.2), st0 = log(.05))
set.seed(11); dat <- make_data(p_true, des, n_trials = 500)
dadm <- EMC2:::design_model(dat, des, verbose = FALSE)
m_pkg <- attr(dadm, "model")()
m_ref <- m_pkg
m_ref$dfun <- function(rt, R, pars) {
  th <- EMC2:::.theta_DDMnn(pars); Ri <- as.integer(R)
  Reduce(`+`, lapply(mem_ptrs, function(p)
    EMC2:::ddm_eval_trials_cpp(p, th, rt, Ri)$pdf)) / length(mem_ptrs)
}
ll_pkg <- EMC2:::calc_ll_R(p_true, m_pkg, dadm)
ll_ref <- EMC2:::calc_ll_R(p_true, m_ref, dadm)
check("EMC2 DDMnn ll == R-side 5-member average ll", abs(ll_pkg - ll_ref), 1e-8)

# --- 3. box rejection through the ensemble path ---
p_out <- p_true; p_out["a"] <- log(6)
ll_out <- EMC2:::calc_ll_R(p_out, m_pkg, dadm)
floor_ll <- length(attr(dadm, "expand")) * log(1e-10)
box_ok <- isTRUE(all.equal(ll_out, floor_ll))
cat(sprintf("%-52s %s\n", "box rejection (a = 6 -> min_ll floor)",
            if (box_ok) "PASS" else "FAIL"))
if (!box_ok) fails <- fails + 1L

# --- 4. speed ---
t_ens <- system.time(for (i in 1:50) EMC2:::calc_ll_R(p_true, m_pkg, dadm))[3] / 50
cat(sprintf("\ncalc_ll_R, N=500 trials, ens5 likelihood: %.2f ms per call\n", t_ens * 1e3))

cat(sprintf("\n%s: %d check group(s) failed\n",
            if (fails == 0L) "ALL PASSED" else "FAILURE", fails))
if (fails > 0L) quit(status = 1L)

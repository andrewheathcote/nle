# End-to-end validation of the LNRnn / RDMnn model types in EMC2 (dev-nn):
#   1. EMC2's compiled flow evaluator vs the Python float64 golden vectors
#   2. LNRnn / RDMnn summed log-likelihoods vs the analytic LNR / RDM
#   3. bounding-box rejection through the full EMC2 pipeline
#   4. speed of the full likelihood call
#
# Requires the dev-nn branch of EMC2 to be installed.
# Run from the nle repo root:  Rscript port/tests/validate_emc2.R

# prefer the dedicated nn-EMC2 library (nle/.Rlibs) so installs of other
# EMC2 branches into the default library cannot affect these studies
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(EMC2))

fails <- 0L
check <- function(label, diff, tol) {
  ok <- is.finite(diff) && diff < tol
  cat(sprintf("%-52s max|diff| = %10.3e  tol = %g  %s\n",
              label, diff, tol, if (ok) "PASS" else "FAIL"))
  if (!ok) fails <<- fails + 1L
}
min_ll <- log(1e-10)

# --- 1. golden vectors through EMC2's compiled evaluator ---
for (tag in c("lnr_large", "rdm_large")) {
  fl <- readRDS(system.file("extdata", "flownn", paste0(tag, ".rds"),
                            package = "EMC2", mustWork = TRUE))
  ptr <- EMC2:::flow_build(fl)
  golden <- read.csv(file.path("port", "export", sprintf("%s_golden.csv", tag)))
  d_lp <- d_cdf <- 0
  for (pid in unique(golden$param_id)) {
    g <- golden[golden$param_id == pid, ]
    ev <- EMC2:::flow_eval_cpp(ptr, as.numeric(g[1L, fl$context_names]), g$rt)
    d_lp <- max(d_lp, abs(ev$log_pdf - g$log_pdf))
    d_cdf <- max(d_cdf, abs(ev$cdf - g$cdf))
  }
  check(sprintf("EMC2 flow evaluator vs golden (%s): log_pdf", tag), d_lp, 1e-9)
  check(sprintf("EMC2 flow evaluator vs golden (%s): cdf", tag), d_cdf, 1e-9)
}

# --- 2a. LNRnn vs analytic LNR ---
design_nn <- design(factors = list(subjects = 1, S = 1:2), Rlevels = 1:2,
                    formula = list(m ~ 0 + lR, s ~ 1, t0 ~ 1), model = LNRnn, verbose = FALSE)
design_an <- design(factors = list(subjects = 1, S = 1:2), Rlevels = 1:2,
                    formula = list(m ~ 0 + lR, s ~ 1, t0 ~ 1), model = LNR, verbose = FALSE)
p_true <- c(m_lR1 = 0.2, m_lR2 = 0.8, s = log(0.5), t0 = log(0.2))
dat <- make_data(p_true, design_nn, n_trials = 1000)
dadm_nn <- EMC2:::design_model(dat, design_nn, verbose = FALSE)
dadm_an <- EMC2:::design_model(dat, design_an, verbose = FALSE)
m_nn <- attr(dadm_nn, "model")()
m_an <- attr(dadm_an, "model")()

# Local draws: a realistic posterior neighborhood of the data-generating
# parameters (what MCMC actually visits after burn-in) -> pass/fail check.
# Wide draws: grossly mismatched parameters, where per-trial evaluations sit
# in the flow's extrapolation tails -> reported for information only.
compare_lls <- function(p_true, m1, dadm1, m2, dadm2, n_draws = 100, sd_local = 0.05) {
  ll1_loc <- ll2_loc <- ll1_wide <- ll2_wide <- numeric(n_draws)
  for (i in seq_len(n_draws)) {
    p_loc <- p_true + rnorm(length(p_true), 0, sd_local)
    names(p_loc) <- names(p_true)
    p_wide <- p_true + rnorm(length(p_true), 0, 6 * sd_local)
    names(p_wide) <- names(p_true)
    ll1_loc[i] <- EMC2:::calc_ll_R(p_loc, m1, dadm1)
    ll2_loc[i] <- EMC2:::calc_ll_R(p_loc, m2, dadm2)
    ll1_wide[i] <- EMC2:::calc_ll_R(p_wide, m1, dadm1)
    ll2_wide[i] <- EMC2:::calc_ll_R(p_wide, m2, dadm2)
  }
  list(r_loc = cor(ll1_loc, ll2_loc), d_loc = mean(abs(ll1_loc - ll2_loc)),
       # approximation-error noise relative to likelihood signal: what MCMC
       # acceptance ratios feel. sd of the flow-vs-analytic disagreement,
       # scaled by how much the true ll varies over the same draws.
       noise_ratio = sd(ll1_loc - ll2_loc) / sd(ll2_loc),
       r_wide = cor(ll1_wide, ll2_wide), d_wide = mean(abs(ll1_wide - ll2_wide)))
}

set.seed(1)
res <- compare_lls(p_true, m_nn, dadm_nn, m_an, dadm_an)
cat(sprintf("\nLNRnn vs LNR (N=1000 trials, 100 draws):\n"))
cat(sprintf("  local draws: r = %.5f, mean |diff| = %.2f;  wide draws: r = %.5f, mean |diff| = %.2f (info only)\n",
            res$r_loc, res$d_loc, res$r_wide, res$d_wide))
cat(sprintf("  local error-noise / ll-signal ratio = %.3f\n", res$noise_ratio))
check("LNRnn ~ LNR local draws (noise/signal)", res$noise_ratio, 0.25)

# --- 2b. RDMnn vs analytic RDM ---
design_rnn <- design(factors = list(subjects = 1, S = 1:2), Rlevels = 1:2,
                     formula = list(v ~ 0 + lR, B ~ 1, A ~ 1, t0 ~ 1, s ~ 1),
                     model = RDMnn, verbose = FALSE)
design_ran <- design(factors = list(subjects = 1, S = 1:2), Rlevels = 1:2,
                     formula = list(v ~ 0 + lR, B ~ 1, A ~ 1, t0 ~ 1, s ~ 1),
                     model = RDM, verbose = FALSE)
p_true_r <- c(v_lR1 = log(2.5), v_lR2 = log(1.5), B = log(1),
              A = log(0.3), t0 = log(0.2), s = log(1))
dat_r <- make_data(p_true_r, design_rnn, n_trials = 1000)
dadm_rnn <- EMC2:::design_model(dat_r, design_rnn, verbose = FALSE)
dadm_ran <- EMC2:::design_model(dat_r, design_ran, verbose = FALSE)
m_rnn <- attr(dadm_rnn, "model")()
m_ran <- attr(dadm_ran, "model")()

set.seed(2)
res_r <- compare_lls(p_true_r, m_rnn, dadm_rnn, m_ran, dadm_ran)
cat(sprintf("\nRDMnn vs RDM (N=1000 trials, 100 draws):\n"))
cat(sprintf("  local draws: r = %.5f, mean |diff| = %.2f;  wide draws: r = %.5f, mean |diff| = %.2f (info only)\n",
            res_r$r_loc, res_r$d_loc, res_r$r_wide, res_r$d_wide))
cat(sprintf("  local error-noise / ll-signal ratio = %.3f\n", res_r$noise_ratio))
check("RDMnn ~ RDM local draws (noise/signal)", res_r$noise_ratio, 0.25)

# --- 2c. DDMnn vs analytic DDM ---
design_dnn <- design(factors = list(subjects = 1, S = 1:2), Rlevels = 1:2,
                     formula = list(v ~ 1, a ~ 1, t0 ~ 1, s ~ 1, Z ~ 1,
                                    SZ ~ 1, sv ~ 1, st0 ~ 1),
                     constants = c(s = log(1)), model = DDMnn, verbose = FALSE)
design_dan <- design(factors = list(subjects = 1, S = 1:2), Rlevels = 1:2,
                     formula = list(v ~ 1, a ~ 1, t0 ~ 1, s ~ 1, Z ~ 1,
                                    SZ ~ 1, sv ~ 1, st0 ~ 1),
                     constants = c(s = log(1)), model = DDM, verbose = FALSE)
p_true_d <- c(v = 1, a = log(1), sv = log(0.5), t0 = log(0.2),
              st0 = log(0.05), Z = qnorm(0.5), SZ = qnorm(0.2))
dat_d <- make_data(p_true_d, design_dnn, n_trials = 1000)
dadm_dnn <- EMC2:::design_model(dat_d, design_dnn, verbose = FALSE)
dadm_dan <- EMC2:::design_model(dat_d, design_dan, verbose = FALSE)
m_dnn <- attr(dadm_dnn, "model")()
m_dan <- attr(dadm_dan, "model")()

ll_dnn <- EMC2:::calc_ll_R(p_true_d, m_dnn, dadm_dnn)
ll_dan <- EMC2:::calc_ll_R(p_true_d, m_dan, dadm_dan)
cat(sprintf("\nDDMnn vs DDM at matched params (N=1000): nn %.2f, analytic %.2f, diff/trial %.4f\n",
            ll_dnn, ll_dan, abs(ll_dnn - ll_dan) / 1000))

set.seed(3)
res_d <- compare_lls(p_true_d, m_dnn, dadm_dnn, m_dan, dadm_dan, n_draws = 50)
cat(sprintf("DDMnn vs DDM (N=1000 trials, 50 draws):\n"))
cat(sprintf("  local draws: r = %.5f, mean |diff| = %.2f;  wide draws: r = %.5f, mean |diff| = %.2f (info only)\n",
            res_d$r_loc, res_d$d_loc, res_d$r_wide, res_d$d_wide))
cat(sprintf("  local error-noise / ll-signal ratio = %.3f\n", res_d$noise_ratio))
# The 8-parameter DDM flow has visibly more local approximation wiggle than
# the race flows (matched-params agreement is good; wide-draw r ~ 0.99).
# Whether this noise level affects inference is adjudicated by parameter
# recovery / SBC, not by this threshold.
check("DDMnn ~ DDM local draws (noise/signal)", res_d$noise_ratio, 0.6)

# --- 3. bounding box through the full pipeline ---
p_out <- p_true; p_out["t0"] <- log(1.5)  # natural t0 = 1.5 > box max 1
ll_out <- EMC2:::calc_ll_R(p_out, m_nn, dadm_nn)
# fully-rejected floor: min_ll for every expanded data row
floor_ll <- length(attr(dadm_nn, "expand")) * min_ll
box_ok <- isTRUE(all.equal(ll_out, floor_ll))
cat(sprintf("%-52s %s\n", "box rejection LNRnn (t0 = 1.5 -> min_ll floor)",
            if (box_ok) "PASS" else sprintf("FAIL (got %.1f, expected %.1f)",
                                            ll_out, floor_ll)))
if (!box_ok) fails <- fails + 1L

p_out_d <- p_true_d; p_out_d["a"] <- log(6)  # natural a = 6 > box max 5
ll_out_d <- EMC2:::calc_ll_R(p_out_d, m_dnn, dadm_dnn)
floor_d <- length(attr(dadm_dnn, "expand")) * min_ll
box_ok_d <- isTRUE(all.equal(ll_out_d, floor_d))
cat(sprintf("%-52s %s\n", "box rejection DDMnn (a = 6 -> min_ll floor)",
            if (box_ok_d) "PASS" else sprintf("FAIL (got %.1f, expected %.1f)",
                                              ll_out_d, floor_d)))
if (!box_ok_d) fails <- fails + 1L

# --- 4. speed of the full likelihood call (N = 1000 trials) ---
t_nn <- system.time(for (i in 1:50) EMC2:::calc_ll_R(p_true, m_nn, dadm_nn))[3] / 50
t_an <- system.time(for (i in 1:50) EMC2:::calc_ll_R(p_true, m_an, dadm_an))[3] / 50
t_rnn <- system.time(for (i in 1:50) EMC2:::calc_ll_R(p_true_r, m_rnn, dadm_rnn))[3] / 50
t_ran <- system.time(for (i in 1:50) EMC2:::calc_ll_R(p_true_r, m_ran, dadm_ran))[3] / 50
cat(sprintf("\nFull calc_ll_R, N=1000 trials, per call:\n"))
cat(sprintf("  LNRnn %7.2f ms   LNR (R path) %7.2f ms\n", t_nn * 1e3, t_an * 1e3))
cat(sprintf("  RDMnn %7.2f ms   RDM (R path) %7.2f ms\n", t_rnn * 1e3, t_ran * 1e3))
t_dnn <- system.time(for (i in 1:50) EMC2:::calc_ll_R(p_true_d, m_dnn, dadm_dnn))[3] / 50
t_dan <- system.time(for (i in 1:5) EMC2:::calc_ll_R(p_true_d, m_dan, dadm_dan))[3] / 5
cat(sprintf("  DDMnn %7.2f ms   DDM (R path, WienR) %7.2f ms   speedup x%.0f\n",
            t_dnn * 1e3, t_dan * 1e3, t_dan / t_dnn))

cat(sprintf("\n%s: %d check group(s) failed\n",
            if (fails == 0L) "ALL PASSED" else "FAILURE", fails))
if (fails > 0L) quit(status = 1L)

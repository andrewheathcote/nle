# Is DDMnn's high local error-noise/signal ratio (~0.5) partly an artifact
# of EMC2's default WienR precision (5e-3, an absolute density tolerance)?
#
# Design: same DDMnn-vs-DDM local-draw comparison as validate_emc2.R, but
# the analytic reference is computed at several precisions. If the 5e-3
# reference itself wobbles across nearby parameter vectors, that wobble
# inflates sd(ll_nn - ll_an) and gets misattributed to the flow.
#
# Reports:
#   1. cost of the analytic likelihood vs precision (per 1000-trial call)
#   2. reference error: sd over draws of (ll_an[prec] - ll_an[finest])
#   3. flow noise/signal recomputed against each reference
#
# Run from the nle repo root:  Rscript port/tests/ddm_precision_check.R
# Uses up to 9 cores (parallel::mclapply).

suppressPackageStartupMessages(library(EMC2))
N_CORES <- 9L
N_DRAWS <- 50L
PRECISIONS <- c(5e-3, 1e-4, 1e-6, 1e-8)   # finest = reference standard

design_dnn <- design(factors = list(subjects = 1, S = 1:2), Rlevels = 1:2,
                     formula = list(v ~ 1, a ~ 1, t0 ~ 1, s ~ 1, Z ~ 1,
                                    SZ ~ 1, sv ~ 1, st0 ~ 1),
                     constants = c(s = log(1)), model = DDMnn, verbose = FALSE)
design_dan <- design(factors = list(subjects = 1, S = 1:2), Rlevels = 1:2,
                     formula = list(v ~ 1, a ~ 1, t0 ~ 1, s ~ 1, Z ~ 1,
                                    SZ ~ 1, sv ~ 1, st0 ~ 1),
                     constants = c(s = log(1)), model = DDM, verbose = FALSE)
p_true <- c(v = 1, a = log(1), sv = log(0.5), t0 = log(0.2),
            st0 = log(0.05), Z = qnorm(0.5), SZ = qnorm(0.2))
set.seed(101)
dat <- make_data(p_true, design_dnn, n_trials = 1000)
dadm_nn <- EMC2:::design_model(dat, design_dnn, verbose = FALSE)
dadm_an <- EMC2:::design_model(dat, design_dan, verbose = FALSE)
m_nn <- attr(dadm_nn, "model")()
m_an <- attr(dadm_an, "model")()

# analytic model with dfun at a chosen precision
make_m_prec <- function(prec) {
  m <- m_an
  m$dfun <- function(rt, R, pars) EMC2:::dDDM(rt, R, pars, precision = prec)
  m
}

# --- 1. timing vs precision ---
cat("Analytic DDM likelihood cost (per 1000-trial call):\n")
for (prec in PRECISIONS) {
  m_p <- make_m_prec(prec)
  n_rep <- if (prec >= 1e-4) 10 else 3
  t_p <- system.time(for (i in seq_len(n_rep))
    EMC2:::calc_ll_R(p_true, m_p, dadm_an))[3] / n_rep
  cat(sprintf("  precision %g: %8.1f ms\n", prec, t_p * 1e3))
}

# --- 2 & 3. local draws, all precisions + flow, in parallel ---
set.seed(3)
draws <- lapply(seq_len(N_DRAWS), function(i) {
  p <- p_true + rnorm(length(p_true), 0, 0.05)
  names(p) <- names(p_true)
  p
})

one_draw <- function(p) {
  out <- c(nn = EMC2:::calc_ll_R(p, m_nn, dadm_nn))
  for (prec in PRECISIONS)
    out[sprintf("an_%g", prec)] <- EMC2:::calc_ll_R(p, make_m_prec(prec), dadm_an)
  out
}
res <- do.call(rbind, parallel::mclapply(draws, one_draw, mc.cores = N_CORES))

finest <- sprintf("an_%g", min(PRECISIONS))
ll_ref <- res[, finest]
cat(sprintf("\nSignal: sd of reference ll over %d local draws = %.2f\n",
            N_DRAWS, sd(ll_ref)))

cat("\nReference error vs finest precision:\n")
for (prec in setdiff(PRECISIONS, min(PRECISIONS))) {
  d <- res[, sprintf("an_%g", prec)] - ll_ref
  cat(sprintf("  precision %g: sd = %8.4f  max|diff| = %8.4f  (noise/signal %.3f)\n",
              prec, sd(d), max(abs(d)), sd(d) / sd(ll_ref)))
}

cat("\nFlow (DDMnn) noise/signal against each reference:\n")
for (prec in PRECISIONS) {
  ll_an <- res[, sprintf("an_%g", prec)]
  d <- res[, "nn"] - ll_an
  cat(sprintf("  vs precision %g: sd(diff) = %8.2f  noise/signal = %.3f  r = %.5f\n",
              prec, sd(d), sd(d) / sd(ll_an), cor(res[, "nn"], ll_an)))
}

cat(sprintf("\nMatched-params check: nn %.2f vs analytic(finest) %.2f, diff/trial %.4f\n",
            EMC2:::calc_ll_R(p_true, m_nn, dadm_nn),
            EMC2:::calc_ll_R(p_true, make_m_prec(min(PRECISIONS)), dadm_an),
            abs(EMC2:::calc_ll_R(p_true, m_nn, dadm_nn) -
                EMC2:::calc_ll_R(p_true, make_m_prec(min(PRECISIONS)), dadm_an)) / 1000))

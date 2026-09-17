# Simulate DDM trials AND the true per-trial score, d log p(rt, R | theta)/d theta.
#
# For the exact-score-matching test we need a POINTWISE score target. The
# targetless penalty (E[score] -> 0) failed because it is satisfiable by
# flattening log f in theta, and since I = -E[grad^2 log f] that destroys
# curvature (kappa 1.027 -> 0.46). A pointwise target pins the function's
# theta-dependence, so it should not have that degeneracy. Using the EXACT
# score here isolates "does targeted score matching work" from "how good is a
# classifier's estimate of the score" - if it fails with the exact target, a
# classifier (a noisier version of the same signal) cannot rescue it.
#
# Central differences on the SAMPLED scale, reusing one dadm so only the
# parameters change:  s_j = [ll(theta + h e_j) - ll(theta - h e_j)] / (2h).
# The stored score is of the JOINT defective density (as `ll` itself is), so
# training must subtract d log P(R | theta)/d theta to get the conditional.
#
# Self-contained rather than wrapping simulate_trials(), whose return value
# drops the `subjects` column that design_model needs.

source("simulation/sim_functions.R")

simulate_trials_score <- function(p_vec, design, param_id, n_trials = 2048,
                                  rt_resolution = 1e-12, h = 0.05,
                                  min_ll = log(1e-10)) {
  dat <- EMC2::make_data(p_vec, design, n_trials = n_trials)
  dadm <- EMC2:::design_model(dat, design, rt_resolution = rt_resolution,
                              verbose = FALSE)
  model <- attr(dadm, "model")()
  .gpm <- if (exists("get_pars_matrix", asNamespace("EMC2")))
    EMC2:::get_pars_matrix else EMC2:::get_pars_matrix_oo

  ll_at <- function(p)
    log_likelihood_ddm_per_trial(.gpm(p, dadm, model = model), dadm, model,
                                 min_ll = min_ll)
  ll <- ll_at(p_vec)

  scores <- vapply(names(p_vec), function(nm) {
    pp <- p_vec; pp[nm] <- pp[nm] + h
    pm <- p_vec; pm[nm] <- pm[nm] - h
    (ll_at(pp) - ll_at(pm)) / (2 * h)
  }, numeric(nrow(dat)))
  colnames(scores) <- paste0("score_", names(p_vec))

  # same S convention as simulate_trials(): correct response set by drift sign
  dat$S <- if (p_vec["v"] == 0) sample(1:2, 1) else if (p_vec["v"] < 0) 1 else 2
  C <- as.numeric(dat$S) == as.numeric(dat$R)

  par_mat <- matrix(rep(unname(p_vec), times = nrow(dat)),
                    nrow = nrow(dat), byrow = TRUE)
  colnames(par_mat) <- names(p_vec)

  cbind(param_id = unname(param_id),
        data.frame(S = dat$S, R = dat$R, rt = dat$rt, C = C, ll = ll),
        as.data.frame(par_mat), as.data.frame(scores))
}

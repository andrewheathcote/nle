##############################
###### KDE estimation ########
##############################
# KDE building blocks shared by kde/KDE.R and kde/KDE_timing.R.
# fit_kde(): one density per response category, scaled by the choice proportion
#            p(R) so the returned approxfun integrates to the *joint* p(rt, R).
# eval_kde(): look up the right per-(param_id, R) density at a trial's rt and
#            return its log, floored at min_ll.
# `...` forwards to stats::density (e.g. bw = ...); the bandwidth grid is set by
# the caller (KDE.R). Behaviour is intentionally fixed.

# KDE fitting function
fit_kde <- function(data, ...) {
  p_correct   <- mean(data$R == 1) # p(RT, R) = p(RT | R) * p(R)
  p_incorrect <- 1 - p_correct
  num_correct <- sum(data$R == 1)
  num_incorrect <- sum(data$R == 2)


  res <- list()

  if (num_correct > 1) {
    dens_correct   <- density(data$rt[data$R == 1], ...)
    res[[1]] <- approxfun(dens_correct$x,   p_correct   * dens_correct$y, yleft = 0, yright = 0)
  }

  if (num_incorrect > 1) {
    dens_incorrect <- density(data$rt[data$R == 2], ...)
    # NOTE: density is 0 outside the observed rt range (yleft/yright = 0); the
    # eval_kde floor then maps that to min_ll. Truncation could set an explicit
    # support range instead, but is intentionally not used here.
    res[[2]] <- approxfun(dens_incorrect$x, p_incorrect * dens_incorrect$y, yleft = 0, yright = 0)
  }
  return(res)
}

# function for evaluating the correct KDE function.
# NOTE: evaluates one trial (row) at a time via apply(test_data, 1, ...); the
# per-parameter-set lookup is kdes[[param_id]][[R]]. Vectorising over parameter
# sets is intentionally left to the caller's apply loop.
eval_kde <- function(data, kdes, min_ll = log(1e-10)) {
  data_num <- as.numeric(data)
  names(data_num) <- names(data)
  est <- tryCatch(
    kdes[[data_num["param_id"]]][[data_num["R"]]](data_num["rt"]),
    error = function(e) NA
  ) # Catches cases where density could not be fit because of too little responses in that category

  if (est != 0 && !is.na(est)) {
    est <- log(est)
    est <- max(est, min_ll)
  } else {
    est <- min_ll
  }

  return(est)
}

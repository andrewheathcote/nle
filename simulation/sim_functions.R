require(renv)
renv::restore()
library(EMC2)

param_matrix <- function(n_draws, param_names, lower, upper,
                         transforms = NULL, seed = NULL, Sobol = TRUE) {

  if (length(param_names) != length(lower) || length(param_names) != length(upper))
    stop("Dimensions of parameter names and bounds must correspond.")

  if (!is.null(seed))
    set.seed(seed)

  dim <- length(param_names)
  if (Sobol) {
    # Sobol points in [0,1]^dim
    u <- randtoolbox::sobol(
      n        = n_draws,
      dim      = dim,
      normal   = FALSE
    )

    # Map to natural scale: theta = lower + u * (upper - lower)
    theta_nat <- sweep(u, 2, upper - lower, `*`)
    theta_nat <- sweep(theta_nat, 2, lower, `+`)

  } else {
    # independent uniform sampling
    theta_nat <- mapply(runif, min = lower, max = upper, MoreArgs = list(n = n_draws))
  }

  if (n_draws == 1)
    theta_nat <- matrix(theta_nat, nrow = 1)

  # Apply per-parameter transforms (inverse of model()$transform) to reach sampled scale.
  if (is.null(transforms))
    transforms <- rep("log", dim) # For RDM

  theta_t <- theta_nat
  for (i in seq_along(transforms)) {
    theta_t[, i] <- switch(transforms[i],
                           "log"      = log(theta_nat[, i]),
                           "probit"    = qnorm(theta_nat[, i]),
                           "identity" = theta_nat[, i],
                           stop(paste("Unknown transform:", transforms[i]))
    )
  }

  colnames(theta_t) <- param_names
  return(theta_t)
}

#### Likelihood functions

# EMC2 trial-wise ll race function
log_likelihood_race_per_trial <- function(pars,dadm,model,min_ll=log(1e-10))
{
  if (any(names(dadm)=="RACE")){# Some accumulators not present
    pars[as.numeric(dadm$lR)>as.numeric(as.character(dadm$RACE)),] <- NA
  }

  if (is.null(attr(pars,"ok"))){
    ok <- !logical(dim(pars)[1])
  } else ok <- attr(pars,"ok")
  lds <- numeric(dim(dadm)[1]) # log pdf (winner) or survivor (losers)
  lds[dadm$winner] <- log(model$dfun(rt=dadm$rt[dadm$winner],
                                     pars=pars[dadm$winner,])) # PDF for winner
  n_acc <- length(levels(dadm$R)) # number of accumulators
  if (n_acc>1) lds[!dadm$winner] <- log(1-model$pfun(rt=dadm$rt[!dadm$winner],pars=pars[!dadm$winner,])) # CDF for losers
  lds[is.na(lds) | !ok] <- min_ll # replace NAs
  if (n_acc>1) {
    ll <- lds[dadm$winner]
    if (n_acc==2) {
      ll <- ll + lds[!dadm$winner] # sum is the overall likelihood
    } else {
      ll <- ll + apply(matrix(lds[!dadm$winner],nrow=n_acc-1),2,sum)
    }
    ll[is.na(ll)] <- min_ll
    return(pmax(min_ll,ll[attr(dadm,"expand")])) # removed sum() operation
  } else return(pmax(min_ll,lds[attr(dadm,"expand")])) # removed sum() operation
}

log_likelihood_ddm_per_trial <- function(pars,dadm,model,min_ll=log(1e-10))
  # DDM summed log likelihood, with protection against numerical issues
{
  like <- numeric(dim(dadm)[1])
  if (any(attr(pars,"ok")))
    like[attr(pars,"ok")] <- EMC2:::dDDM(dadm$rt[attr(pars,"ok")],dadm$R[attr(pars,"ok")],
                                         pars[attr(pars,"ok"),,drop=FALSE], precision = 5e-10)
  like[attr(pars,"ok")][is.na(like[attr(pars,"ok")])] <- 0
  pmax(min_ll,log(like[attr(dadm,"expand")]))
}


### Simulation functions
simulate_trials <- function(p_vec, design, param_id, n_trials = 2048, # 2048 trials per condition
                            rt_resolution = 1e-12) {

  # Simulate trials
  dat <- EMC2::make_data(p_vec, design, n_trials = n_trials)

  # Compute likelihood per trial
  dadm  <- EMC2:::design_model(dat, design, rt_resolution = rt_resolution, verbose = FALSE) # bins rts and determines winner
  model <- attr(dadm, "model")() # extract model
  pars  <- EMC2:::get_pars_matrix(p_vec, dadm, model = model) # This applies transforms and bounds the parameters

  if (model$c_name == "RDM") {
    ll <- log_likelihood_race_per_trial(pars, dadm, model, min_ll = log(1e-10))

    # Sensibly set S as the one with the bigger (log) rate
    S <- which.max(p_vec[c("v_lR1", "v_lR2")])
    dat$S <- S

  } else if (model$c_name == "DDM") {
    ll <- log_likelihood_ddm_per_trial(pars, dadm, model, min_ll = log(1e-10))

    # if the drift rate is exactly 0, the correct response/stimulus is just randomly determined
    dat$S <- if (p_vec["v"] == 0) sample(1:2, 1) else if (p_vec["v"] < 0) 1 else 2

  } else if (model$c_name == "LNR") {
    ll <- log_likelihood_race_per_trial(pars, dadm, model, min_ll = log(1e-10)) # same function as for RDM

    # Sensibly set S as the one with the smaller mean
    S <- which.min(p_vec[c("m_lR1", "m_lR2")])
    dat$S <- S

  }

  # Correctness flag based on drift rate/mean
  C <- as.numeric(dat$S) == as.numeric(dat$R)

  # Repeat parameter vector per row
  par_mat <- matrix(rep(unname(p_vec), times = nrow(dat)),
                    nrow = nrow(dat), byrow = TRUE)
  colnames(par_mat) <- names(p_vec)

  res <- cbind(
    param_id = unname(param_id),
    data.frame(S = dat$S, R = dat$R, rt = dat$rt, C = C, ll = ll),
    as.data.frame(par_mat)
  )

  return(res)
}


simulate_trials_rdm_single <- function(p_vec, design = NULL, param_id, n_trials = 2048,
                                       min_ll = log(1e-10), rt_resolution = 1e-12) {
  n <- n_trials
  pars <- matrix(rep(p_vec[c("v","B","t0","A","s")], each = n),
                 nrow = n,
                 dimnames = list(NULL, c("v","B","t0","A","s")))

  pars_nat <- exp(pars) # transform to natural scale before sampling

  # Simulate RTs
  rt <- EMC2:::rWald(n,
                     B = pars_nat[,"B"] / pars_nat[,"s"],
                     v = pars_nat[,"v"] / pars_nat[,"s"],
                     A = pars_nat[,"A"] / pars_nat[,"s"]) + pars_nat[,"t0"]

  if (!is.null(rt_resolution))
    rt <- floor(rt / rt_resolution) * rt_resolution

  # ll per trial
  ll <- pmax(log(EMC2:::dRDM(rt, pars_nat)), min_ll)

  res <- cbind(
    param_id = unname(param_id),
    data.frame(S = NA, R = 1, rt = rt, C = NA, ll = ll),
    as.data.frame(pars)
  )

  return(res)
}

simulate_trials_lnr_single <- function(p_vec, design = NULL, param_id, n_trials = 2048,
                                       min_ll = log(1e-10), rt_resolution = 1e-12) {
  n <- n_trials
  pars <- matrix(rep(p_vec[c("m", "s", "t0")], each = n),
                 nrow = n,
                 dimnames = list(NULL, c("m", "s", "t0")))

  pars_nat <- pars # m doesn't need transform
  pars_nat[, c("s", "t0")] <- exp(pars[, c("s", "t0"), drop = FALSE])

  rt <- stats::rlnorm(n, meanlog = pars_nat[, "m"], sdlog = pars_nat[, "s"]) + pars_nat[, "t0"]

  if (!is.null(rt_resolution))
    rt <- floor(rt / rt_resolution) * rt_resolution

  ll <- pmax(log(EMC2:::dLNR(rt, pars_nat)), min_ll)

  res <- cbind(
    param_id = unname(param_id),
    data.frame(S = NA, R = 1, rt = rt, C = NA, ll = ll),
    as.data.frame(pars)
  )

  return(res)
}


# Vectorized variants for n_trials = 1 (one trial per parameter set).
# Process an entire batch of parameter sets in one vectorized call instead of
# dispatching one apply iteration per row.
simulate_trials_rdm_single_batch <- function(params_mat, param_ids, design = NULL,
                                             n_trials = 1, min_ll = log(1e-10),
                                             rt_resolution = 1e-12) {
  if (n_trials != 1)
    stop("simulate_trials_rdm_single_batch supports only n_trials = 1")

  n <- nrow(params_mat)
  pars_nat <- exp(params_mat)

  rt <- EMC2:::rWald(n,
                     B = pars_nat[, "B"] / pars_nat[, "s"],
                     v = pars_nat[, "v"] / pars_nat[, "s"],
                     A = pars_nat[, "A"] / pars_nat[, "s"]) + pars_nat[, "t0"]

  if (!is.null(rt_resolution))
    rt <- floor(rt / rt_resolution) * rt_resolution

  ll <- pmax(log(EMC2:::dRDM(rt, pars_nat)), min_ll)

  cbind(
    param_id = unname(param_ids),
    data.frame(S = NA, R = 1L, rt = rt, C = NA, ll = ll),
    as.data.frame(params_mat)
  )
}

simulate_trials_lnr_single_batch <- function(params_mat, param_ids, design = NULL,
                                             n_trials = 1, min_ll = log(1e-10),
                                             rt_resolution = 1e-12) {
  if (n_trials != 1)
    stop("simulate_trials_lnr_single_batch supports only n_trials = 1")

  n <- nrow(params_mat)
  m_vec  <- params_mat[, "m"]
  s_nat  <- exp(params_mat[, "s"])
  t0_nat <- exp(params_mat[, "t0"])

  rt <- stats::rlnorm(n, meanlog = m_vec, sdlog = s_nat) + t0_nat

  if (!is.null(rt_resolution))
    rt <- floor(rt / rt_resolution) * rt_resolution

  pars_nat <- cbind(m = m_vec, s = s_nat, t0 = t0_nat)
  ll <- pmax(log(EMC2:::dLNR(rt, pars_nat)), min_ll)

  cbind(
    param_id = unname(param_ids),
    data.frame(S = NA, R = 1L, rt = rt, C = NA, ll = ll),
    as.data.frame(params_mat)
  )
}


# Two-accumulator batched variants for the test/KDE-fit simulations.
# n_trials here is interpreted as trials per stimulus (matching simulate_trials,
# which receives n_trials per level of S). Output rows per call:
#   nrow(params_mat) * n_trials * 2.
# Row layout (for param p, stim s in {1,2}, trial t):
#   contiguous block of n_trials * 2 rows per param, with stim 1 first then
#   stim 2; trials within a stim are contiguous.
simulate_trials_rdm_batch <- function(params_mat, param_ids, design = NULL,
                                      n_trials = 1, min_ll = log(1e-10),
                                      rt_resolution = 1e-12) {
  n_params <- nrow(params_mat)
  rows_per_param <- n_trials * 2L
  N <- n_params * rows_per_param

  param_row <- rep(seq_len(n_params), each = rows_per_param)
  params_expanded <- params_mat[param_row, , drop = FALSE]

  v1 <- exp(params_expanded[, "v_lR1"]); v2 <- exp(params_expanded[, "v_lR2"])
  B1 <- exp(params_expanded[, "B_lR1"]); B2 <- exp(params_expanded[, "B_lR2"])
  A1 <- exp(params_expanded[, "A_lR1"]); A2 <- exp(params_expanded[, "A_lR2"])
  s1 <- exp(params_expanded[, "s_lR1"]); s2 <- exp(params_expanded[, "s_lR2"])
  t0_nat <- exp(params_expanded[, "t0"])

  ft1 <- EMC2:::rWald(N, B = B1 / s1, v = v1 / s1, A = A1 / s1)
  ft2 <- EMC2:::rWald(N, B = B2 / s2, v = v2 / s2, A = A2 / s2)
  ft <- cbind(ft1, ft2)

  # max.col with random tie-breaking: ties are measure-zero for continuous
  # draws, and random tie-break avoids the deterministic bias of which.min.
  R <- max.col(-ft, ties.method = "random")
  rt <- ft[cbind(seq_len(N), R)] + t0_nat

  if (!is.null(rt_resolution))
    rt <- floor(rt / rt_resolution) * rt_resolution
  
  # Note that the determination of S here is different from simulate_trials
  # This is however irrelevant because the flow only sees R
  S <- rep(rep(c(1L, 2L), each = n_trials), times = n_params)

  win1 <- R == 1L
  v_win <- ifelse(win1, v1, v2); v_los <- ifelse(win1, v2, v1)
  B_win <- ifelse(win1, B1, B2); B_los <- ifelse(win1, B2, B1)
  A_win <- ifelse(win1, A1, A2); A_los <- ifelse(win1, A2, A1)
  s_win <- ifelse(win1, s1, s2); s_los <- ifelse(win1, s2, s1)

  pars_win <- cbind(v = v_win, B = B_win, t0 = t0_nat, A = A_win, s = s_win)
  pars_los <- cbind(v = v_los, B = B_los, t0 = t0_nat, A = A_los, s = s_los)

  ll_win <- log(EMC2:::dRDM(rt, pars_win))
  ll_los <- log(1 - EMC2:::pRDM(rt, pars_los))
  ll <- ll_win + ll_los
  ll[is.na(ll)] <- min_ll
  ll <- pmax(min_ll, ll)

  C <- as.integer(S == R)

  cbind(
    param_id = unname(rep(param_ids, each = rows_per_param)),
    data.frame(S = S, R = R, rt = rt, C = C, ll = ll),
    as.data.frame(params_expanded)
  )
}

simulate_trials_lnr_batch <- function(params_mat, param_ids, design = NULL,
                                      n_trials = 1, min_ll = log(1e-10),
                                      rt_resolution = 1e-12) {
  n_params <- nrow(params_mat)
  rows_per_param <- n_trials * 2L
  N <- n_params * rows_per_param

  param_row <- rep(seq_len(n_params), each = rows_per_param)
  params_expanded <- params_mat[param_row, , drop = FALSE]

  m1 <- params_expanded[, "m_lR1"]; m2 <- params_expanded[, "m_lR2"]
  s1 <- exp(params_expanded[, "s_lR1"]); s2 <- exp(params_expanded[, "s_lR2"])
  t0_nat <- exp(params_expanded[, "t0"])

  ft1 <- stats::rlnorm(N, meanlog = m1, sdlog = s1)
  ft2 <- stats::rlnorm(N, meanlog = m2, sdlog = s2)
  ft <- cbind(ft1, ft2)

  R <- max.col(-ft, ties.method = "random")
  rt <- ft[cbind(seq_len(N), R)] + t0_nat

  if (!is.null(rt_resolution))
    rt <- floor(rt / rt_resolution) * rt_resolution
    
  # Note that the determination of S here is different from simulate_trials
  # This is however irrelevant because the flow only sees R
  S <- rep(rep(c(1L, 2L), each = n_trials), times = n_params)

  win1 <- R == 1L
  m_win <- ifelse(win1, m1, m2); m_los <- ifelse(win1, m2, m1)
  s_win <- ifelse(win1, s1, s2); s_los <- ifelse(win1, s2, s1)

  pars_win <- cbind(m = m_win, s = s_win, t0 = t0_nat)
  pars_los <- cbind(m = m_los, s = s_los, t0 = t0_nat)

  ll_win <- log(EMC2:::dLNR(rt, pars_win))
  ll_los <- log(1 - EMC2:::pLNR(rt, pars_los))
  ll <- ll_win + ll_los
  ll[is.na(ll)] <- min_ll
  ll <- pmax(min_ll, ll)

  C <- as.integer(S == R)

  cbind(
    param_id = unname(rep(param_ids, each = rows_per_param)),
    data.frame(S = S, R = R, rt = rt, C = C, ll = ll),
    as.data.frame(params_expanded)
  )
}


# Function to simulate one data batch
run_batch <- function(sim_fun, batch_params, param_ids, p, n_cores, seed, ...) {
  mat <- cbind(param_id = param_ids, batch_params)
  f   <- function(row, ...) {
    p()
    sim_fun(param_id = row["param_id"], p_vec = row[names(row) != "param_id"], ...)
  }
  if (n_cores > 1) {
    future.apply::future_apply(mat, 1, f, ..., future.seed = seed, 
                               future.chunk.size = max(1L, ceiling(nrow(mat) / 200)))
  } else {
    apply(mat, 1, f, ...)
  }
}

simulate_data <- function(param_names, lower, upper, design, n_params,
                          sim_fun = simulate_trials,
                          transforms = NULL, out_dir,
                          n_trials = 2048, chunk_size = 1e6,
                          rt_resolution = 1e-12, seed = NULL,
                          n_cores = parallelly::availableCores() - 1,
                          Sobol = TRUE, params = NULL, batched = FALSE) {
  if (!is.null(seed))
    set.seed(seed)

  # In batched mode, sim_fun is invoked once per chunk on the whole parameter
  # matrix (vectorized internally). Parallel apply would be pure overhead.
  if (batched)
    n_cores <- 1

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  # Sobol sample parameters
  if (is.null(params))
    params <- param_matrix(n_params, param_names, lower, upper, transforms, Sobol = Sobol)

  # Save all parameter sets upfront (param_id links to trial files)
  arrow::write_parquet(
    cbind(param_id = seq_len(n_params), as.data.frame(params)),
    file.path(out_dir, "parameters.parquet")
  )

  # Trial files go in a subdirectory
  trials_dir <- file.path(out_dir, "trials")
  dir.create(trials_dir, recursive = TRUE, showWarnings = FALSE)

  # Split params into batches so each batch produces ~chunk_size trial rows
  n_S              <- ifelse(!is.null(design$Ffactors$S), length(design$Ffactors$S), 1)
  rows_per_param   <- n_trials * n_S
  params_per_chunk <- max(1L, floor(chunk_size / rows_per_param))
  batches          <- split(seq_len(n_params),
                            ceiling(seq_len(n_params) / params_per_chunk))

  # Customized progress bar
  progressr::handlers(progressr::handler_progress(
    format = "[:bar] :percent | :current/:total | Elapsed: :elapsed | ETA: :eta",
    width = 60
  ))

  if (n_cores > 1)
    future::plan(future::multisession, workers = n_cores)

  progressr::with_progress({
    p <- progressr::progressor(steps = n_params)

    t_sim <- structure(rep(0, 5),
                       names = c("user.self","sys.self","elapsed","user.child","sys.child"),
                       class = "proc_time")

    for (i in seq_along(batches)) {
      batch_params <- params[batches[[i]], , drop = FALSE]

      t_sim <- t_sim + system.time({
        if (batched) {
          results <- sim_fun(params_mat = batch_params,
                             param_ids = batches[[i]],
                             design = design,
                             n_trials = n_trials,
                             rt_resolution = rt_resolution)
        } else {
          results <- run_batch(sim_fun, batch_params, batches[[i]], p, n_cores, seed,
                               design = design, n_trials = n_trials,
                               rt_resolution = rt_resolution)
        }
      })

      arrow::write_parquet(
        if (batched) results else dplyr::bind_rows(results),
        file.path(trials_dir, sprintf("trials_%04d.parquet", i))
      )

      if (batched)
        p(amount = nrow(batch_params))
    }
  })
  saveRDS(t_sim, file.path(out_dir, "sim_time.RDS"))

  if (n_cores > 1)
    future::plan(future::sequential) # reset to one core

  message(sprintf(
    "Saved %d parameter sets across %d trial file(s) to '%s'",
    n_params, length(batches), out_dir
  ))
}

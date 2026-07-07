library(arrow)
library(dplyr)
library(microbenchmark)
source("kde/kde_functions.R")

# Timing driver for the two KDE costs (plans/kde_timing.md):
#   - FIT  cost: fit_kde() -> density() + approxfun, scales with training budget
#   - EVAL cost: eval_kde() -> approxfun lookup, scales with number of trials N
# Mirrors benchmark/scripts_julian/RDM_timing.R: one row per parameter set,
# N_EVAL columns, times in nanoseconds.

# KL-optimal bandwidth selected by evaluate_bws (decision #2): closest KL to 0.
best_bw <- function(model, bud) {
  df <- readRDS(file.path("results", model, bud, "bw_eval.rds"))[[1]]
  df$bw[which.min(abs(df$KL))]
}

# Vectorized lower bound for eval (decision #5): direct approxfun calls over the
# rt vector, by response, instead of apply()-ing eval_kde row by row. Mirrors
# eval_kde's semantics (log, min_ll floor, 0/NA -> min_ll) without its overhead.
eval_kde_vec <- function(rt, R, kde, min_ll = log(1e-10)) {
  est <- rep(NA_real_, length(rt))
  for (r in seq_along(kde)) {
    f <- kde[[r]]
    if (!is.null(f)) {
      idx <- R == r
      est[idx] <- f(rt[idx])
    }
  }
  ifelse(is.na(est) | est == 0, min_ll, pmax(log(est), min_ll))
}

N_EVAL <- 10
models  <- c("DDM", "LNR", "RDM")
fit_budgets <- c("small", "medium", "large")   # training-sim budget -> FIT cost
eval_Ns     <- c("N=100", "N=1000", "N=10000") # evaluated trials    -> EVAL cost

for (model in models) {
  print(model)

  ## ---- FIT timing: vary training budget ----
  for (bud in fit_budgets) {
    print(bud)
    data <- open_dataset(file.path("data/kde_fit", bud, model, "trials")) |>
      select(all_of(c("param_id", "R", "rt", "ll"))) |>
      collect() |> as.data.frame()
    data_split <- split(data, data$param_id)

    bw <- best_bw(model, bud)

    fit_times <- matrix(nrow = length(data_split), ncol = N_EVAL)
    print("Fitting KDEs to training data...")
    for (j in seq_along(data_split)) {
      print(j)
      mb <- microbenchmark(fit_kde(data_split[[j]], bw = bw), times = N_EVAL)
      fit_times[j, ] <- mb$time # in nanoseconds
    }

    out_dir <- file.path("results", model, "timing", "kde_fit", bud)
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    write.csv(fit_times, file.path(out_dir, "times.csv"), row.names = FALSE)
  }

  ## ---- EVAL timing: benchmark data, fit at eval_params (Option A) ----
  # Fit one KDE per eval-param row from the Step-0 training sim, then time
  # evaluating each benchmark N-trial dataset under it (mirrors RDM_timing.R,
  # which evaluates the same dat under each of the 1000 eval params).
  eval_params <- read.csv(file.path("benchmark/scripts_julian", model, "eval_params.csv"))
  n_param     <- nrow(eval_params)

  # fit a KDE per eval param from the Step-0 training data (large-budget bw),
  # timing each refit (FIT cost at the eval params, KDE_TRAIN_N budget)
  bw_eval <- best_bw(model, "large")
  train <- open_dataset(file.path("benchmark/scripts_julian", model, "kde_train", "trials")) |>
    select(all_of(c("param_id", "R", "rt", "ll"))) |>
    collect() |> as.data.frame()
  train_split <- split(train, train$param_id)

  fit_eval_times <- matrix(nrow = n_param, ncol = N_EVAL)
  kdes_eval      <- vector("list", n_param)
  print("Fitting KDEs to evaluation data...")
  for (j in seq_len(n_param)) {
    print(j)
    mb <- microbenchmark(kdes_eval[[j]] <- fit_kde(train_split[[j]], bw = bw_eval), times = N_EVAL)
    fit_eval_times[j, ] <- mb$time # in nanoseconds
  }
  names(kdes_eval) <- names(train_split)

  out_dir <- file.path("results", model, "timing", "kde_fit", "eval_params")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(fit_eval_times, file.path(out_dir, "times.csv"), row.names = FALSE)

  for (Nname in eval_Ns) {
    print(Nname)
    dat <- open_dataset(file.path("benchmark/scripts_julian", model, Nname, "trials")) |>
      collect() |> as.data.frame() # one N-trial dataset (n_params = 1)
    N_trials <- nrow(dat)
    rt_vec   <- as.numeric(dat$rt)
    R_vec    <- as.integer(as.character(dat$R))

    eval_times     <- matrix(nrow = n_param, ncol = N_EVAL)  # apply() workflow cost
    eval_times_vec <- matrix(nrow = n_param, ncol = N_EVAL)  # vectorized lower bound (decision #5)
    lls            <- matrix(nrow = n_param, ncol = N_trials)
    
    print("Timing KDE evaluation...")
    for (i in seq_len(n_param)) {
      print(i)
      td <- transform(dat, param_id = 1L)  # eval under eval-param i's KDE
      kd <- kdes_eval[i]                   # 1-element list so eval_kde's [[param_id]] resolves
      kf <- kdes_eval[[i]]

      mb <- microbenchmark(
        lls[i, ] <- apply(td[, c("param_id", "R", "rt")], 1, eval_kde, kdes = kd),
        times = N_EVAL
      )
      eval_times[i, ] <- mb$time # in nanoseconds
      
      # vectorized
      mbv <- microbenchmark(eval_kde_vec(rt_vec, R_vec, kf), times = N_EVAL)
      eval_times_vec[i, ] <- mbv$time
    }

    out_dir <- file.path("results", model, "timing", "kde_eval", Nname)
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    write.csv(eval_times,     file.path(out_dir, "times.csv"),     row.names = FALSE)
    write.csv(eval_times_vec, file.path(out_dir, "times_vec.csv"), row.names = FALSE)
    write.csv(lls,            file.path(out_dir, "lls.csv"),       row.names = FALSE)
  }
}

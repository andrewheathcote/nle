source('simulation/sim_functions.R')
source('simulation/designs.R')

# Training simulation at the benchmark eval_params (Step 0 of plans/kde_timing.md).
# The benchmark dirs only hold the *evaluation* datasets, so to time KDE
# evaluation against analytic & flow we first fit one KDE per eval_params row.
# Here we generate a large training set at exactly those params; param_id in the
# output is aligned to the eval_params row order.

KDE_TRAIN_N <- 10000 # trials per eval param to fit the density (matches `large`)

path <- file.path('benchmark/scripts_julian')

# DDM
eval_paramsDDM <- as.matrix(read.csv(file.path(path, "DDM", "eval_params.csv")))
simulate_data(
  param_namesDDM,
  lowerDDM,
  upperDDM,
  designDDM,
  n_params = nrow(eval_paramsDDM),
  params = eval_paramsDDM, # fit at eval params
  transforms = transformsDDM,
  out_dir = file.path(path, "DDM", "kde_train"),
  chunk_size = 1000,
  seed = 1,
  n_trials = KDE_TRAIN_N / 2, # one trial per stimulus
  Sobol = FALSE
)

# RDM
eval_paramsRDM <- as.matrix(read.csv(file.path(path, "RDM", "eval_params.csv")))
simulate_data(
  param_namesRDM,
  lowerRDM,
  upperRDM,
  designRDM,
  n_params = nrow(eval_paramsRDM),
  params = eval_paramsRDM, # fit at eval params
  out_dir = file.path(path, "RDM", "kde_train"),
  chunk_size = 1000,
  seed = 1,
  n_trials = KDE_TRAIN_N / 2, # one trial per stimulus
  sim_fun = simulate_trials_rdm_batch,
  batched = TRUE,
  Sobol = FALSE
)

# LNR
eval_paramsLNR <- as.matrix(read.csv(file.path(path, "LNR", "eval_params.csv")))
simulate_data(
  param_namesLNR,
  lowerLNR,
  upperLNR,
  designLNR,
  n_params = nrow(eval_paramsLNR),
  params = eval_paramsLNR, # fit at eval params
  transforms = transformsLNR,
  out_dir = file.path(path, "LNR", "kde_train"),
  chunk_size = 1000,
  seed = 1,
  n_trials = KDE_TRAIN_N / 2, # one trial per stimulus
  sim_fun = simulate_trials_lnr_batch,
  batched = TRUE,
  Sobol = FALSE
)

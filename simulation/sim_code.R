source("simulation/sim_functions.R")
source("simulation/designs.R")

# Sample a test & validation set using uniform proposals
# Sample a training set using Sobol' sampling
# TODO Use parameters in the test set to simulate new trails for the KDE, to be able to evaluate at test values later

###############################################
################# Simulation ##################
###############################################

# Seeds
# randomly generated with sample(1:10000, 3)
TRAIN_SEED <- 3876
VAL_SEED <- 1656
TEST_SEED <- 2251

PRIORPRED_SIZE <- 1e3

# TODO do we want Sobol sampling here or not?
# Prior predictive samples
path <- file.path("data", 'prior')

# RDM
simulate_data(param_namesRDM, lowerRDM, upperRDM, designRDM,
              n_params = PRIORPRED_SIZE, transforms = NULL,
              out_dir = file.path(path, "RDM"), chunk_size = 1e6, seed = 1,
              Sobol = FALSE)

# DDM
simulate_data(param_namesDDM, lowerDDM, upperDDM, designDDM,
              n_params = PRIORPRED_SIZE, transforms = transformsDDM,
              out_dir = file.path(path, "DDM"),
              chunk_size = 1e6,
              seed = 1, Sobol = FALSE)

# LNR
simulate_data(param_namesLNR, lowerLNR, upperLNR, designLNR,
              n_params = PRIORPRED_SIZE, transforms = transformsLNR,
              chunk_size = 1e6, out_dir = file.path(path, "LNR"),
              seed = 1, Sobol = FALSE)


# Budgets
SMALL_TOTAL  <- 50000
MEDIUM_TOTAL <- 200000
LARGE_TOTAL  <- 1000000

# Training sets
BUDGETS <- c(
  "large" = LARGE_TOTAL,
  "medium" = MEDIUM_TOTAL,
  "small" = SMALL_TOTAL
)

for (i in seq_along(BUDGETS)) {
  budget_name <- names(BUDGETS)[i]

  path <- file.path("data", "train", budget_name)
  
  # DDM
  simulate_data(
    param_namesDDM, 
    lowerDDM, 
    upperDDM, 
    designDDM,
    n_params = BUDGETS[i] / 2, # one trial per stimulus so budget can be halved 
    transforms = transformsDDM,
    out_dir = file.path(path, 'DDM'),
    chunk_size = 1e5,
    seed = TRAIN_SEED, 
    n_trials = 1
  )

  # RDM
  # simulate_data(param_namesRDM, lowerRDM, upperRDM, designRDM,
  #               n_params = BUDGETS[i], transforms = NULL,
  #               out_dir = file.path(path, 'RDM'), chunk_size = 1e6, seed = 1)

  simulate_data(
    param_namesRDM_single,
    lowerRDM_single,
    upperRDM_single,
    n_params = BUDGETS[i],
    transforms = NULL,
    out_dir = file.path(path, "RDM_single"),
    chunk_size = 1e5,
    seed = TRAIN_SEED,
    design = designRDM_single,
    sim_fun = simulate_trials_rdm_single_batch,
    n_trials = 1,
    batched = TRUE
  )

  # LNR
  # simulate_data(param_namesLNR, lowerLNR, upperLNR, designLNR,
  #               n_params = BUDGETS[i], transforms = transformsLNR,
  #               chunk_size = 1e6, out_dir = file.path(path, 'LNR'),
  #               seed = 1)

  simulate_data(
    param_names = param_namesLNR_single,
    lower = lowerLNR_single,
    upper = upperLNR_single,
    design = designLNR_single,
    transforms = transformsLNR_single,
    n_params = BUDGETS[i] / 2, # LNR has smaller budget
    chunk_size = 1e5,
    out_dir = file.path(path, "LNR_single"),
    sim_fun = simulate_trials_lnr_single_batch,
    seed = TRAIN_SEED,
    n_trials = 1,
    batched = TRUE
  )
}


# Test & validation sets
# These sets are sampled from independent uniforms
# as they might otherwise have sequences similar to the Sobol' samples
# used during training.

VAL_SIZE <- 10000
TEST_SIZE <- 1000

for (set in c("val", "test")) {
  path <- file.path("data", set)

  if (set == "test") { # test set with two accumulators for RDM and LNR
    # RDM
    simulate_data(
      param_namesRDM,
      lowerRDM,
      upperRDM,
      designRDM,
      n_params = TEST_SIZE,
      transforms = NULL,
      out_dir = file.path(path, "RDM"),
      chunk_size = 1e6,
      seed = TEST_SEED,
      Sobol = FALSE,
      sim_fun = simulate_trials_rdm_batch,
      batched = TRUE,
      n_trials = 1000 # per stimulus, so 1000 * 2 = 2000 (same as for other models)
    )

    # LNR
    simulate_data(
      param_namesLNR,
      lowerLNR,
      upperLNR,
      designLNR,
      n_params = TEST_SIZE,
      transforms = transformsLNR,
      chunk_size = 1e6,
      out_dir = file.path(path, "LNR"),
      seed = TEST_SEED,
      Sobol = FALSE,
      sim_fun = simulate_trials_lnr_batch,
      batched = TRUE,
      n_trials = 1000 # per stimulus, so 1000 * 2 = 2000 (same as for other models)
    )
    
    # DDM
    simulate_data(
      param_namesDDM, 
      lowerDDM, 
      upperDDM, 
      designDDM,
      n_params = TEST_SIZE, 
      transforms = transformsDDM,
      out_dir = file.path(path, "DDM"),
      chunk_size = 1e6,
      seed = TEST_SEED, 
      Sobol = FALSE,
      n_trials = 1000 # per stimulus, so 1000 * 2 = 2000 (same as for other models)
    )
  } else {
    simulate_data(
      param_namesRDM_single, 
      lowerRDM_single, 
      upperRDM_single,
      n_params = VAL_SIZE, 
      transforms = NULL,
      out_dir = file.path(path, "RDM_single"), 
      chunk_size = 1e5, 
      seed = VAL_SEED,
      design = designRDM_single, 
      sim_fun = simulate_trials_rdm_single_batch,
      n_trials = 1,
      batched = TRUE,
      Sobol = FALSE
    )
    
    simulate_data(
      param_names = param_namesLNR_single,
      lower = lowerLNR_single,
      upper = upperLNR_single,
      design = designLNR_single,
      transforms = transformsLNR_single,
      n_params = VAL_SIZE,
      out_dir = file.path(path, "LNR_single"),
      sim_fun = simulate_trials_lnr_single_batch,
      n_trials = 1,
      chunk_size = 1e5,
      batched = TRUE,
      Sobol = FALSE,
      seed = VAL_SEED
    )
    
    # DDM
    simulate_data(
      param_namesDDM, 
      lowerDDM, 
      upperDDM, 
      designDDM,
      n_params = VAL_SIZE / 2, # one trial per stimulus so budget can be halved
      transforms = transformsDDM,
      out_dir = file.path(path, "DDM"),
      chunk_size = 1e5,
      seed = VAL_SEED, 
      Sobol = FALSE,
      n_trials = 1
    )
  }

  
}



# Simulation tests KDE
# Load parameters
library(dplyr)
library(arrow)

test_path <- "data/test"


params_test_LNR <- as.matrix(open_dataset(file.path(test_path, "LNR/parameters.parquet")) |>
  collect() |> select(!"param_id"))

params_test_RDM <- as.matrix(open_dataset(file.path(test_path, "RDM/parameters.parquet")) |>
  collect() |> select(!"param_id"))

params_test_DDM <- as.matrix(open_dataset(file.path(test_path, "DDM/parameters.parquet")) |>
  collect() |> select(!"param_id"))

# Seed
SEED_KDE <- 2231 # generated with sample(1:10000, 1)

# Budgets
SMALL_KDE_TRIALS  <- 1000
MEDIUM_KDE_TRIALS <- 10000 
LARGE_KDE_TRIALS  <- 100000

BUDGETS_KDE <- c(
  "small" = SMALL_KDE_TRIALS,
  "medium" = MEDIUM_KDE_TRIALS,
  "large" = LARGE_KDE_TRIALS
)

for (i in seq_along(BUDGETS_KDE)) {
  budget_name <- names(BUDGETS_KDE)[i]
  
  path <- file.path('data/kde_fit', budget_name)
  
  # LNR
  simulate_data(
    param_namesLNR,
    lowerLNR,
    upperLNR,
    designLNR,
    n_params = nrow(params_test_LNR),
    transforms = transformsLNR,
    chunk_size = 1e6,
    out_dir = file.path(path, "LNR"),
    seed = SEED_KDE,
    Sobol = FALSE,
    params = params_test_LNR,
    sim_fun = simulate_trials_lnr_batch,
    batched = TRUE,
    n_trials = BUDGETS_KDE[i] / 2 # two stimuli
  )

  # RDM
  simulate_data(
    param_namesRDM,
    lowerRDM,
    upperRDM,
    designRDM,
    n_params = nrow(params_test_RDM),
    transforms = NULL,
    out_dir = file.path(path, "RDM"),
    chunk_size = 1e6,
    seed = SEED_KDE,
    Sobol = FALSE,
    params = params_test_RDM,
    sim_fun = simulate_trials_rdm_batch,
    batched = TRUE,
    n_trials = BUDGETS_KDE[i] / 2 # two stimuli
  )
  
  # DDM
  simulate_data(
    param_namesDDM, 
    lowerDDM, 
    upperDDM, 
    designDDM,
    n_params = nrow(params_test_DDM), 
    transforms = transformsDDM,
    out_dir = file.path(path, "DDM"),
    chunk_size = 1e6,
    seed = SEED_KDE, 
    Sobol = FALSE,
    params = params_test_DDM,
    n_trials = BUDGETS_KDE[i] / 2 # two stimuli
  )
  
}

# validation data for determining bandwidth
# (writes data/kde_fit/val/<MODEL>/, the dir kde/KDE.R reads for bandwidth selection)
path <- file.path('data/kde_fit', 'val')

# LNR
simulate_data(
  param_namesLNR,
  lowerLNR,
  upperLNR,
  designLNR,
  n_params = nrow(params_test_LNR),
  transforms = transformsLNR,
  chunk_size = 1e6,
  out_dir = file.path(path, "LNR"),
  seed = 2026,
  Sobol = FALSE,
  params = params_test_LNR,
  sim_fun = simulate_trials_lnr_batch,
  batched = TRUE,
  n_trials = 1000 / 2 # two stimuli
)

# RDM
simulate_data(
  param_namesRDM,
  lowerRDM,
  upperRDM,
  designRDM,
  n_params = nrow(params_test_RDM),
  transforms = NULL,
  out_dir = file.path(path, "RDM"),
  chunk_size = 1e6,
  seed = 2026,
  Sobol = FALSE,
  params = params_test_RDM,
  sim_fun = simulate_trials_rdm_batch,
  batched = TRUE,
  n_trials = 1000 / 2 # two stimuli
)

# DDM
simulate_data(
  param_namesDDM, 
  lowerDDM, 
  upperDDM, 
  designDDM,
  n_params = nrow(params_test_DDM), 
  transforms = transformsDDM,
  out_dir = file.path(path, "DDM"),
  chunk_size = 1e6,
  seed = 2026, 
  Sobol = FALSE,
  params = params_test_DDM,
  n_trials = 1000 / 2 # two stimuli
)







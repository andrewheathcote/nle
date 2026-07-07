source('simulation/sim_functions.R')
source('simulation/designs.R')


budgets <- c(
  "N=100" = 100,
  "N=1000" = 1000,
  "N=10000" = 10000
)

path <- file.path('benchmark/scripts_julian')
for (i in seq_along(budgets)) {
  b_name <- names(budgets)[i]
  
  simulate_data(
    param_namesDDM,
    lowerDDM,
    upperDDM,
    designDDM,
    n_params = 1,
    transforms = transformsDDM,
    out_dir = file.path(path, 'DDM', b_name),
    chunk_size = 1000,
    seed = 1,
    n_trials = budgets[i] / 2, # one trial per stimulus
    Sobol = FALSE
  )
  
  simulate_data(
    param_namesRDM,
    lowerRDM,
    upperRDM,
    designRDM,
    n_params = 1,
    out_dir = file.path(path, "RDM", b_name),
    chunk_size = 1000,
    seed = 1,
    n_trials = budgets[i] / 2, # one trial per stimulus
    sim_fun = simulate_trials_rdm_batch,
    batched = TRUE,
    Sobol = FALSE
  )
  
  simulate_data(
    param_namesLNR,
    lowerLNR,
    upperLNR,
    designLNR,
    n_params = 1,
    transforms = transformsLNR,
    out_dir = file.path(path, "LNR", b_name),
    chunk_size = 1000,
    seed = 1,
    n_trials = budgets[i] / 2, # one trial per stimulus
    sim_fun = simulate_trials_lnr_batch,
    batched = TRUE,
    Sobol = FALSE
  )
  
}

# Get 1000 parameter samples per model for likelihood evaluation
# DDM
eval_paramsDDM <- param_matrix(
  1000, 
  param_namesDDM, 
  lowerDDM, 
  upperDDM, 
  transformsDDM, 
  seed = 2, 
  Sobol = FALSE
)

write.csv(eval_paramsDDM, "benchmark/scripts_julian/DDM/eval_params.csv", row.names = FALSE)

# RDM
eval_paramsRDM <- param_matrix(
  1000, 
  param_namesRDM, 
  lowerRDM, 
  upperRDM, 
  seed = 2, 
  Sobol = FALSE
)

write.csv(eval_paramsRDM, "benchmark/scripts_julian/RDM/eval_params.csv", row.names = FALSE)

# LNR
eval_paramsLNR <- param_matrix(
  1000, 
  param_namesLNR, 
  lowerLNR, 
  upperLNR, 
  transformsLNR, 
  seed = 2, 
  Sobol = FALSE
)

write.csv(eval_paramsLNR, "benchmark/scripts_julian/LNR/eval_params.csv", row.names = FALSE)


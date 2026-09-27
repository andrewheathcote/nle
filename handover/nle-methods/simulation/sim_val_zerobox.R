# Zero-box VALIDATION set: 25k draws x 40 rows, a different seed from training.
#
# Checkpoints are retained by validation loss, so a validation set drawn from a
# different parameterisation silently selects the wrong checkpoints (TRAPS.md).
# This one is drawn by the same stratified sampler as the training data, so the
# eight regimes are represented in the same proportions.
#
# Run from the package root:  Rscript simulation/sim_val_zerobox.R
.lib <- Sys.getenv("EMC2NN_LIB", "")
if (!nzchar(.lib)) for (cand in c(".Rlibs", "../../.Rlibs")) if (dir.exists(cand)) { .lib <- normalizePath(cand); break }
if (nzchar(.lib)) { .libPaths(c(.lib, .libPaths())); Sys.setenv(R_LIBS = paste(.libPaths(), collapse = .Platform$path.sep)) }
suppressPackageStartupMessages(library(EMC2))
cat("EMC2 at", find.package("EMC2"), "\n")
if (nzchar(.lib)) stopifnot(startsWith(normalizePath(find.package("EMC2")), .lib))
source("simulation/sim_functions.R"); source("simulation/designs.R"); source("simulation/zerobox.R")

N_PARAMS <- 25000L; N_TRIALS <- 20L; VAL_SEED <- 9124   # not the training seed
V <- ZB_VERSION_DEFAULT
out <- file.path("data", "val", paste0("zerobox", zb_data_suffix(V)), "DDM")
params <- param_matrix_zerobox(N_PARAMS, param_namesDDM, lowerDDM, upperDDM, transformsDDM, seed = VAL_SEED, version = V)
dir.create(out, recursive = TRUE, showWarnings = FALSE)
arrow::write_parquet(data.frame(param_id = seq_len(N_PARAMS), regime = attr(params, "regime")),
                     file.path(out, "regimes.parquet"))
t0 <- proc.time()[3]
simulate_data(param_namesDDM, lowerDDM, upperDDM, designDDM, n_params = N_PARAMS,
              transforms = transformsDDM, params = params, out_dir = out, chunk_size = 1e5,
              seed = VAL_SEED, n_trials = N_TRIALS, n_cores = as.integer(Sys.getenv("SIM_CORES", "8")))
d <- arrow::read_parquet(file.path(out, "trials", "trials_0001.parquet"))
floor_frac <- mean(d$ll <= log(1e-10) + 1e-9)
cat(sprintf("=== val done in %.1f min; ll at floor: %.2f%% ===\n", (proc.time()[3] - t0) / 60, 100 * floor_frac))
stopifnot(floor_frac < 0.05)

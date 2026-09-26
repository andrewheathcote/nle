# Zero-box DDM training data: 600k draws x 40 rows = 24e6 rows.
#
# Same pipeline as sim_ddm.R (simulate_data -> EMC2 make_data -> EMC2:::dDDM
# at precision 5e-10 for the stored ll), with the parameter matrix drawn by
# param_matrix_zerobox() so that sv, SZ and st0 can be exactly zero. Regime
# labels are written alongside so per-regime validation needs no re-derivation.
#
# Run from the package root:  Rscript simulation/sim_ddm_zerobox.R [n_params]
#
# The .libPaths line is not optional: without it a second EMC2 3.4.1 in the
# default library silently floors every stored ll at log(1e-10) (pilot run
# 2026-09-26: 80,000 of 80,000 rows at the floor). Asserted below.
# Resolve the dev-nn EMC2 library: EMC2NN_LIB, else a .Rlibs here or at the
# repo root. Export it as R_LIBS too, so the future::multisession workers that
# simulate_data spawns see the same build -- they do not inherit .libPaths().
.lib <- Sys.getenv("EMC2NN_LIB", "")
if (!nzchar(.lib)) for (cand in c(".Rlibs", "../../.Rlibs")) if (dir.exists(cand)) { .lib <- normalizePath(cand); break }
if (nzchar(.lib)) { .libPaths(c(.lib, .libPaths())); Sys.setenv(R_LIBS = paste(.libPaths(), collapse = .Platform$path.sep)) }
suppressPackageStartupMessages(library(EMC2))
cat("EMC2 at", find.package("EMC2"), "\n")
if (nzchar(.lib)) stopifnot(startsWith(normalizePath(find.package("EMC2")), .lib))
source("simulation/sim_functions.R")
source("simulation/designs.R")
source("simulation/zerobox.R")

a <- commandArgs(trailingOnly = TRUE)
N_PARAMS <- if (length(a)) as.integer(a[1]) else 600000L
N_TRIALS <- 20L                                   # per stimulus -> 40 rows per draw
TRAIN_SEED <- 3876
out <- file.path("data", "train", sprintf("zerobox%dk", N_PARAMS %/% 1000L), "DDM")

params <- param_matrix_zerobox(N_PARAMS, param_namesDDM, lowerDDM, upperDDM,
                               transformsDDM, seed = TRAIN_SEED)
dir.create(out, recursive = TRUE, showWarnings = FALSE)
arrow::write_parquet(data.frame(param_id = seq_len(N_PARAMS), regime = attr(params, "regime")),
                     file.path(out, "regimes.parquet"))
cat(sprintf("=== zero-box: %d draws x %d rows = %d rows -> %s ===\n",
            N_PARAMS, 2 * N_TRIALS, N_PARAMS * 2 * N_TRIALS, out))
print(table(attr(params, "regime")))
t0 <- proc.time()[3]
simulate_data(param_namesDDM, lowerDDM, upperDDM, designDDM,
              n_params = N_PARAMS, transforms = transformsDDM, params = params,
              out_dir = out, chunk_size = 1e5, seed = TRAIN_SEED, n_trials = N_TRIALS,
              n_cores = as.integer(Sys.getenv("SIM_CORES", "8")))   # respect the local core cap
cat(sprintf("=== done in %.1f min ===\n", (proc.time()[3] - t0) / 60))
d <- arrow::read_parquet(file.path(out, "trials", "trials_0001.parquet"))
floor_frac <- mean(d$ll <= log(1e-10) + 1e-9)
cat(sprintf("EMC2 at %s; ll at floor: %.2f%% of first chunk\n", find.package("EMC2"), 100 * floor_frac))
stopifnot(floor_frac < 0.05)

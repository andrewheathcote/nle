# RDM single-accumulator training data at the scale the DDM now uses.
#
# The race flow models ONE accumulator's first-passage density
# f(rt | v, B, t0, s, A); the race likelihood is then assembled at inference as
# winner density x surviving accumulators. So a "row" here is one accumulator,
# one trial, one parameter draw — n_trials = 1 is the only mode the batched
# simulator supports, which means the budget is pure theta-diversity.
#
# `rdm_large` (Julian's, the only RDM flow that exists) used 1e6 rows. The DDM
# is now on 16e6. Matching that is the first thing the RDM has never had.
#
# IMPORTANT: prefer .Rlibs. The system EMC2 is a different 3.4.1 build whose DDM
# likelihood silently returns 0; assume nothing about its race path either.
#
#   Rscript simulation/sim_rdm_16m.R
# Env: RDM_N (rows, default 16e6), RDM_OUT (default data/train/rdm16m)

if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
source("simulation/sim_functions.R")
source("simulation/designs.R")

N   <- as.numeric(Sys.getenv("RDM_N", "16e6"))
OUT <- Sys.getenv("RDM_OUT", "data/train/rdm16m")
SEED <- 3876

cat(sprintf("RDM single-accumulator: %s rows -> %s\n", format(N, big.mark=","), OUT))
cat(sprintf("context: %s\n", paste(param_namesRDM_single, collapse=", ")))
.t <- proc.time()[3]
simulate_data(
  param_namesRDM_single, lowerRDM_single, upperRDM_single,
  n_params = N, transforms = NULL,
  out_dir = file.path(OUT, "RDM"), chunk_size = 1e5, seed = SEED,
  design = designRDM_single, sim_fun = simulate_trials_rdm_single_batch,
  n_trials = 1, batched = TRUE
)
cat(sprintf("=== simulated in %.1f min ===\n", (proc.time()[3] - .t)/60))

# sanity: the stored ll must be finite and not pinned at the floor, which is how
# the EMC2 dual-install problem manifests (see sbc/SBC_PLAN.md)
suppressPackageStartupMessages(library(arrow))
f <- list.files(file.path(OUT, "RDM", "trials"), pattern="parquet$", full.names=TRUE)[1]
d <- as.data.frame(read_parquet(f))
cat(sprintf("check: %d rows  ll finite %.3f  at floor %.4f  ll range [%.2f, %.2f]\n",
            nrow(d), mean(is.finite(d$ll)), mean(d$ll <= log(1e-10) + 1e-9),
            min(d$ll), max(d$ll)))
cat(sprintf("       rt range [%.3f, %.3f]   cols: %s\n",
            min(d$rt), max(d$rt), paste(names(d), collapse=" ")))

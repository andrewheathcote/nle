# Scaled-DDM data (handover/SCALED_DDM_PLAN.md): training, validation and choice-probability sets.
#   Rscript simulation/sim_ddm_scaled.R train [n_params]     600k draws x 40 rows (seed 5101)
#   Rscript simulation/sim_ddm_scaled.R val   [n_params]     25k draws x 40 rows  (seed 5102)
#   Rscript simulation/sim_ddm_scaled.R choice [n_train n_val]   exact P(R = 2) targets
# Env: EMC2NN_LIB (the dev library; .Rlibs otherwise), SIM_CORES (default 8).
# Same pipeline as sim_ddm_zerobox.R (EMC2 make_data, the stored ll from dDDM at precision
# 5e-10), then each chunk is rewritten to the scaled columns: rt is u = rt - T0SIM.
.lib <- Sys.getenv("EMC2NN_LIB", "")
if (!nzchar(.lib)) for (cand in c(".Rlibs", "../../.Rlibs")) if (dir.exists(cand)) { .lib <- normalizePath(cand); break }
if (nzchar(.lib)) { .libPaths(c(.lib, .libPaths())); Sys.setenv(R_LIBS = paste(.libPaths(), collapse = .Platform$path.sep)) }
suppressPackageStartupMessages(library(EMC2))
cat("EMC2 at", find.package("EMC2"), "\n")
if (nzchar(.lib)) stopifnot(startsWith(normalizePath(find.package("EMC2")), .lib))
source("simulation/sim_functions.R"); source("simulation/designs.R"); source("simulation/scaled.R")
a <- commandArgs(trailingOnly = TRUE)
SPLIT <- a[1]; stopifnot(SPLIT %in% c("train", "val", "choice"))
CORES <- as.integer(Sys.getenv("SIM_CORES", "8"))

if (SPLIT == "choice") {
  suppressPackageStartupMessages({library(WienR); library(parallel)})
  N <- c(train = if (length(a) > 1) as.integer(a[2]) else 200000L, val = if (length(a) > 2) as.integer(a[3]) else 20000L)
  for (sp in c("val", "train")) {
    pm <- param_matrix_scaled(N[[sp]], seed = c(train = 5103L, val = 5104L)[[sp]])
    U <- pm$unit; t0 <- proc.time()[3]
    pv <- unlist(mclapply(seq_len(nrow(U)), function(i)
      pWDM(1e6, response = "upper", a = 1, v = U[i, "kappa"], w = U[i, "Z"], t0 = 0, sv = U[i, "eta"],
           sw = 2 * U[i, "SZ"] * min(U[i, "Z"], 1 - U[i, "Z"]), st0 = 0, precision = 1e-8)$value, mc.cores = CORES))
    stopifnot(length(pv) == nrow(U), all(is.finite(pv)), max(pv - 1, 0) < 1e-8, max(-pv, 0) < 1e-8)
    out <- data.frame(kappa = U[, "kappa"], eta = log(U[, "eta"]), Z = qnorm(U[, "Z"]), SZ = qnorm(U[, "SZ"]),
                      p_R2 = pmin(pmax(pv, 0), 1), regime = pm$regime)
    dir.create("data/choice_prob", recursive = TRUE, showWarnings = FALSE)
    f <- sprintf("data/choice_prob/%s_scaled.parquet", sp); arrow::write_parquet(out, f)
    cat(sprintf("  %s n=%d  P(R=2) %.4f-%.4f  %.1f min\n", f, nrow(out), min(pv), max(pv), (proc.time()[3] - t0) / 60))
  }
  quit(save = "no")
}

N_PARAMS <- if (length(a) > 1) as.integer(a[2]) else if (SPLIT == "train") 600000L else 25000L
SEED <- c(train = 5101L, val = 5102L)[[SPLIT]]
out <- file.path("data", SPLIT, sprintf("scaled%s", if (SPLIT == "train") sprintf("%dk", N_PARAMS %/% 1000L) else ""), "DDM")
pm <- param_matrix_scaled(N_PARAMS, SEED)
dir.create(out, recursive = TRUE, showWarnings = FALSE)
arrow::write_parquet(data.frame(param_id = seq_len(N_PARAMS), regime = pm$regime), file.path(out, "regimes.parquet"))
cat(sprintf("=== scaled DDM %s: %d draws x 40 rows -> %s ===\n", SPLIT, N_PARAMS, out)); print(table(pm$regime))
t0 <- proc.time()[3]
simulate_data(param_namesDDM, lowerDDM, upperDDM, designDDM, n_params = N_PARAMS, transforms = transformsDDM,
              params = pm$emc2, out_dir = out, chunk_size = 1e5, seed = SEED, n_trials = 20L, n_cores = CORES)
for (f in list.files(file.path(out, "trials"), "\\.parquet$", full.names = TRUE))
  arrow::write_parquet(scaled_columns(as.data.frame(arrow::read_parquet(f))), f)
d <- arrow::read_parquet(file.path(out, "trials", "trials_0001.parquet"))
floor_frac <- mean(d$ll <= log(1e-10) + 1e-9)
q <- quantile(log(d$rt), c(0, 1e-4, .001, .5, .999, 1e-4 + .9998, 1))
cat(sprintf("=== done in %.1f min; ll at floor %.2f%%; log u quantiles: %s ===\n", (proc.time()[3] - t0) / 60,
            100 * floor_frac, paste(sprintf("%.2f", q), collapse = " ")))
stopifnot(floor_frac < 0.05, all(d$rt > 0))

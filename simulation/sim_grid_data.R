# OFF-SAMPLE grid rows for the distillation "untapped lever".
#
# MLE can only learn where the simulator put data, so its gradient is weighted
# by the sampling density - which is exactly why the flow's errors concentrate
# at the leading edge and the 1-3 s shoulder, regions carrying little mass. But
# an analytic likelihood can be evaluated ANYWHERE. These rows place rt on a
# broad log-uniform grid over the spline support instead of drawing it from the
# model, and attach the analytic log-density at each point.
#
# They are NOT model samples, so they carry no valid NLL contribution: training
# must use them for the distillation (MSE) term only. They are marked with
# param_id offset by GRID_OFFSET so the loader can separate them.
#
# Run from the nle repo root:  Rscript simulation/sim_grid_data.R

source("simulation/sim_functions.R")
source("simulation/designs.R")

set.seed(4242)
N_PARAMS   <- as.integer(Sys.getenv("GRID_N_PARAMS", "25000"))      # same count as tpt40/score40
N_RT       <- 40L         # rt points per draw, matching 40 rows each
GRID_OFFSET <- 1e7        # marks these rows as off-sample
# rt is placed RELATIVE to each draw's own t0, spanning from just after the
# leading edge out past the shoulder. A fixed log-uniform range over the whole
# support wastes ~47% of points below the min_ll floor (measured), where the
# target is a floor value carrying no information.
OFF_LO     <- 1e-3        # seconds after t0
OFF_HI     <- 8.0
LL_FLOOR   <- log(1e-10)
FLOOR_MARGIN <- 1.0       # drop rows within this of the floor
out <- file.path("data", "train", "grid40", "DDM")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

design <- designDDM
lower <- lowerDDM; upper <- upperDDM
pm <- param_matrix(N_PARAMS, param_namesDDM, lower, upper,
                   transforms = transformsDDM, seed = 4242, Sobol = TRUE)

.gpm <- if (exists("get_pars_matrix", asNamespace("EMC2")))
  EMC2:::get_pars_matrix else EMC2:::get_pars_matrix_oo

rows <- vector("list", N_PARAMS)
t0 <- proc.time()[3]
for (i in seq_len(N_PARAMS)) {
  p_vec <- pm[i, param_namesDDM]
  # rt log-uniform over the support; response split evenly
  t0n <- exp(p_vec[["t0"]])
  rt <- t0n + exp(runif(N_RT, log(OFF_LO), log(OFF_HI)))
  R  <- rep(1:2, length.out = N_RT)
  dat <- data.frame(subjects = factor(1), S = factor(rep(1, N_RT), levels = 1:2),
                    R = factor(R, levels = 1:2), rt = rt)
  dadm <- EMC2:::design_model(dat, design, rt_resolution = 1e-12, verbose = FALSE)
  model <- attr(dadm, "model")()
  pars <- .gpm(p_vec, dadm, model = model)
  ll <- log_likelihood_ddm_per_trial(pars, dadm, model, min_ll = log(1e-10))
  keep <- ll > LL_FLOOR + FLOOR_MARGIN
  if (!any(keep)) next
  d <- as.data.frame(as.list(p_vec))
  rows[[i]] <- cbind(param_id = GRID_OFFSET + i, S = 1, R = R[keep],
                     rt = rt[keep], C = TRUE, ll = ll[keep],
                     d[rep(1, sum(keep)), , drop = FALSE])
  if (i %% 2500 == 0)
    cat(sprintf("  %d/%d (%.0f s)\n", i, N_PARAMS, proc.time()[3] - t0))
}
rows <- rows[!vapply(rows, is.null, logical(1))]
res <- do.call(rbind, rows)
cat(sprintf("retained %d of %d candidate rows (%.1f%%) above the min_ll floor\n",
            nrow(res), N_PARAMS * N_RT, 100 * nrow(res) / (N_PARAMS * N_RT)))
arrow::write_dataset(res, out, format = "parquet")
cat(sprintf("=== %d off-sample rows -> %s in %.1f min ===\n",
            nrow(res), out, (proc.time()[3] - t0) / 60))

# Scaled-DDM training data for nleflow (the NLE project's simulation/sim_ddm_scaled.R
# and simulation/scaled.R, driven by the spec).
#
# The network models the unit-scale decision time u of a diffusion with a = s = 1,
# so the simulator runs EMC2's DDM at a = s = 1, st0 = 0 and a fixed T0SIM and stores
# u = rt - T0SIM. One row is one trial; trials_per_draw rows share a parameter draw.
# Columns: param_id, R (1 lower, 2 upper), rt (= u), ll (dDDM at precision 5e-10,
# floored at log 1e-10), and the context on the stored scale: kappa, log eta (-Inf at
# 0), probit Z, probit SZ (-Inf at 0; SZ is EMC2's start-point proportion).
#
# Draws: each input follows the spec's `sampling` strata (uniform, log_uniform or
# asinh_uniform, each over its range or the box), then `regimes` set inputs to exactly
# 0 jointly. The project drew Sobol points (randtoolbox, with a seeded shift); this
# simulator draws pseudo-random uniforms, which keeps it to EMC2, WienR and jsonlite.
#
# The choice sets (NLEFLOW_CHOICE_TRAIN / _VAL): fresh draws from the same sampler with
# the exact P(R = 2) from WienR::pWDM at a = s = 1, the classifier's targets.
#
# Run by `nleflow simulate <spec>` (see nleflow/simulate.py for the variables). Also:
#   NLEFLOW_RLIB      the library holding EMC2 (asserted when given)
#   NLEFLOW_FORMAT    parquet (default when arrow is installed) or csv
#   NLEFLOW_SIM_SEED  the data seed (default 5101, the project's; validation 5102,
#                     choice sets 5103 and 5104). training.seed is the network's.
LIB <- Sys.getenv("NLEFLOW_RLIB", "")
if (nzchar(LIB)) .libPaths(c(normalizePath(path.expand(LIB)), .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(WienR); library(jsonlite); library(parallel)})
if (nzchar(LIB))
  stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(path.expand(LIB))))
have_arrow <- requireNamespace("arrow", quietly = TRUE)
FORMAT <- Sys.getenv("NLEFLOW_FORMAT", if (have_arrow) "parquet" else "csv")
if (FORMAT == "parquet" && !have_arrow) stop("NLEFLOW_FORMAT=parquet needs the arrow package")
`%||%` <- function(a, b) if (is.null(a)) b else a

spec <- fromJSON(Sys.getenv("NLEFLOW_SPEC"), simplifyVector = FALSE)
CTX <- unlist(spec$context)
stopifnot(identical(CTX, c("kappa", "eta", "Z", "SZ")), spec$response == "choice",
          spec$scaling$kind == "ddm_unit")
box <- lapply(spec$box_natural, unlist)
ZC  <- lapply(spec$zerobox_c, unlist)
TPD <- as.integer(spec$training$trials_per_draw %||% 40L)
T0SIM <- 0.05
MIN_LL <- log(1e-10)
CORES <- as.integer(Sys.getenv("NLEFLOW_CORES", "4"))
SEED  <- as.integer(Sys.getenv("NLEFLOW_SIM_SEED", "5101"))
DRAWS_PER_FILE <- 500L
env_n <- function(v, default) { x <- Sys.getenv(v, ""); if (nzchar(x)) as.numeric(x) else default }

# --- the sampler: strata per input, then the regimes ------------------------------------
draw_input <- function(name, n) {
  strata <- spec$sampling[[name]]
  if (is.null(strata)) strata <- list(list(kind = "uniform", share = 1))
  shares <- vapply(strata, function(s) s$share, 0)
  which_s <- sample.int(length(strata), n, replace = TRUE, prob = shares)
  x <- numeric(n)
  for (k in seq_along(strata)) {
    i <- which_s == k; m <- sum(i); if (!m) next
    st <- strata[[k]]; r <- if (is.null(st$range)) box[[name]] else unlist(st$range)
    x[i] <- switch(st$kind,
      uniform       = runif(m, r[1], r[2]),
      log_uniform   = exp(runif(m, log(r[1]), log(r[2]))),
      asinh_uniform = { c0 <- ZC[[name]]; sinh(runif(m, asinh(r[1] / c0), asinh(r[2] / c0))) * c0 },
      zero          = 0,
      stop("unknown stratum kind ", st$kind))
  }
  x
}
draw_params <- function(n) {
  U <- sapply(CTX, draw_input, n = n)
  if (!is.null(spec$regimes)) {
    shares <- vapply(spec$regimes, function(r) r$share, 0)
    reg <- sample.int(length(spec$regimes), n, replace = TRUE, prob = shares)
    for (k in seq_along(spec$regimes)) for (z in unlist(spec$regimes[[k]]$zero)) U[reg == k, z] <- 0
  }
  U
}
stored <- function(U)                      # natural (kappa, eta, Z, SZ) -> the storage scale
  cbind(kappa = U[, "kappa"], eta = log(U[, "eta"]), Z = qnorm(U[, "Z"]), SZ = qnorm(U[, "SZ"]))
emc2_pars <- function(U) {                 # natural DDM parameters at a = s = 1 (SZ as the width)
  cbind(v = U[, "kappa"], a = 1, t0 = T0SIM, Z = U[, "Z"],
        SZ = 2 * U[, "SZ"] * pmin(U[, "Z"], 1 - U[, "Z"]), sv = U[, "eta"], st0 = 0, s = 1)
}
write_rows <- function(d, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  if (FORMAT == "parquet") arrow::write_parquet(d, path) else write.csv(d, path, row.names = FALSE)
}

# --- trials -----------------------------------------------------------------------------
simulate_set <- function(rows, OUT, seed) {
  n_draws <- as.integer(ceiling(rows / TPD))
  n_files <- as.integer(ceiling(n_draws / DRAWS_PER_FILE))
  cat(sprintf("scaled DDM: %s draws x %d trials -> %s (%d %s files), seed %d, EMC2 %s at %s\n",
              format(n_draws, big.mark = ","), TPD, OUT, n_files, FORMAT, seed,
              as.character(packageVersion("EMC2")), find.package("EMC2")))
  one_file <- function(k) {
    set.seed(seed + k)
    first <- (k - 1L) * DRAWS_PER_FILE + 1L
    m <- min(DRAWS_PER_FILE, n_draws - first + 1L)
    U <- draw_params(m)
    P <- emc2_pars(U)[rep(seq_len(m), each = TPD), , drop = FALSE]
    Rf <- factor(rep(1L, nrow(P)), levels = 1:2)
    sim <- EMC2:::rDDM(Rf, P)                         # make_data's sampler and precision
    ll <- pmax(log(EMC2:::dDDM(sim$rt, sim$R, P, precision = 5e-10)), MIN_LL)
    S <- stored(U)[rep(seq_len(m), each = TPD), , drop = FALSE]
    d <- data.frame(param_id = rep(first - 1L + seq_len(m), each = TPD), R = as.integer(sim$R),
                    rt = sim$rt - T0SIM, ll = ll, kappa = S[, "kappa"], eta = S[, "eta"],
                    Z = S[, "Z"], SZ = S[, "SZ"])
    write_rows(d, file.path(OUT, sprintf("trials_%04d.%s", k, FORMAT)))
    c(n = nrow(d), eta0 = sum(U[, "eta"] == 0), SZ0 = sum(U[, "SZ"] == 0),
      bad = sum(!is.finite(d$rt) | d$rt <= 0 | !is.finite(ll)), floor = sum(ll <= MIN_LL + 1e-9))
  }
  .t <- proc.time()[3]
  res <- do.call(rbind, mclapply(seq_len(n_files), one_file, mc.cores = CORES))
  stopifnot(is.matrix(res), nrow(res) == n_files, sum(res[, "bad"]) == 0)
  cat(sprintf("=== %s rows in %.1f min; eta = 0 in %.3f and SZ = 0 in %.3f of draws; ll at floor %.2f%% ===\n",
              format(sum(res[, "n"]), big.mark = ","), (proc.time()[3] - .t) / 60,
              sum(res[, "eta0"]) / n_draws, sum(res[, "SZ0"]) / n_draws, 100 * sum(res[, "floor"]) / sum(res[, "n"])))
  saveRDS(list(rows = rows, draws = n_draws, trials_per_draw = TPD, seed = seed, t0sim = T0SIM,
               format = FORMAT, emc2 = as.character(packageVersion("EMC2"))), file.path(OUT, "sim_config.rds"))
}

# --- the choice sets: exact P(R = 2) ----------------------------------------------------
choice_set <- function(n, path, seed) {
  set.seed(seed)
  U <- draw_params(n)
  .t <- proc.time()[3]
  pv <- unlist(mclapply(seq_len(n), function(i)
    pWDM(1e6, response = "upper", a = 1, v = U[i, "kappa"], w = U[i, "Z"], t0 = 0, sv = U[i, "eta"],
         sw = 2 * U[i, "SZ"] * min(U[i, "Z"], 1 - U[i, "Z"]), st0 = 0, precision = 1e-8)$value,
    mc.cores = CORES))
  stopifnot(length(pv) == n, all(is.finite(pv)), max(pv - 1, 0) < 1e-8, max(-pv, 0) < 1e-8)
  S <- stored(U)
  write_rows(data.frame(kappa = S[, "kappa"], eta = S[, "eta"], Z = S[, "Z"], SZ = S[, "SZ"],
                        p_R2 = pmin(pmax(pv, 0), 1)), path)
  cat(sprintf("choice set %s: %d draws, P(R = 2) in [%.4f, %.4f], %.1f min\n", path, n, min(pv), max(pv),
              (proc.time()[3] - .t) / 60))
}

simulate_set(env_n("NLEFLOW_ROWS", 24e6), Sys.getenv("NLEFLOW_TRAIN"), SEED)
simulate_set(env_n("NLEFLOW_VAL_ROWS", 1e6), Sys.getenv("NLEFLOW_VAL"), SEED + 1L)
choice_set(as.integer(env_n("NLEFLOW_CHOICE_ROWS", 2e5)), Sys.getenv("NLEFLOW_CHOICE_TRAIN"), SEED + 2L)
choice_set(as.integer(env_n("NLEFLOW_CHOICE_VAL_ROWS", 2e4)), Sys.getenv("NLEFLOW_CHOICE_VAL"), SEED + 3L)

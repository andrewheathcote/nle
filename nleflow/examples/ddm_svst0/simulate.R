# Training data for the DDM with drift and non-decision-time variability
# (examples/ddm_svst0): EMC2's DDM at the spec's `fixed` values (s = 1, SZ = 0), the
# context drawn from the spec's box, strata and regimes.
#
# One row is one trial; trials_per_draw rows share a parameter draw. Columns:
# param_id, R (1 lower, 2 upper), rt (seconds), ll (EMC2's dDDM, i.e. WienR's dWDM,
# at precision 1e-6, floored at log 1e-10; diagnostics only), and the context on
# EMC2's sampled scale: v, log a, log t0, probit Z, log sv and log st0 (-Inf at 0).
#
# Draws: each input follows the spec's `sampling` strata (uniform, log_uniform or
# asinh_uniform, each over its range or the box; an input without strata is uniform
# over its box), then `regimes` set inputs to exactly 0 jointly.
#
# The choice sets (NLEFLOW_CHOICE_TRAIN / _VAL): fresh draws from the same sampler with
# the exact P(R = 2) from WienR::pWDM (it does not depend on t0 or st0), the
# classifier's targets.
#
# Run by `nleflow simulate <spec>` (see nleflow/simulate.py for the variables). Also:
#   NLEFLOW_RLIB      the library holding EMC2, or a list a:b as R_LIBS (asserted when given)
#   NLEFLOW_FORMAT    parquet (default when arrow is installed) or csv
#   NLEFLOW_SIM_SEED  the data seed (default 6101; validation 6102, choice sets 6103
#                     and 6104). training.seed is the network's.
#   NLEFLOW_LL        0 to skip the ll column of the training set (it is always
#                     written for the validation set)
LIB <- Sys.getenv("NLEFLOW_RLIB", "")
LIBS <- if (nzchar(LIB)) normalizePath(path.expand(strsplit(LIB, ":", fixed = TRUE)[[1]])) else character(0)
if (nzchar(LIB)) .libPaths(c(LIBS, .libPaths()))   # one library or a list, as R_LIBS
suppressPackageStartupMessages({library(EMC2); library(WienR); library(jsonlite); library(parallel)})
if (nzchar(LIB))
  stopifnot(any(startsWith(normalizePath(find.package("EMC2")), LIBS)))
have_arrow <- requireNamespace("arrow", quietly = TRUE)
FORMAT <- Sys.getenv("NLEFLOW_FORMAT", if (have_arrow) "parquet" else "csv")
if (FORMAT == "parquet" && !have_arrow) stop("NLEFLOW_FORMAT=parquet needs the arrow package")
`%||%` <- function(a, b) if (is.null(a)) b else a

spec <- fromJSON(Sys.getenv("NLEFLOW_SPEC"), simplifyVector = FALSE)
CTX <- unlist(spec$context)
FIXED <- unlist(spec$fixed)
stopifnot(spec$response == "choice", is.null(spec$scaling),
          setequal(c(CTX, names(FIXED)), c("v", "a", "t0", "Z", "sv", "st0", "s", "SZ")))
TR  <- unlist(spec$transforms)
box <- lapply(spec$box_natural, unlist)
ZC  <- lapply(spec$zerobox_c, unlist)
STORED <- unlist(spec$stored)
TPD <- as.integer(spec$training$trials_per_draw %||% 1L)
MIN_LL <- log(1e-10)
CORES <- as.integer(Sys.getenv("NLEFLOW_CORES", "4"))
SEED  <- as.integer(Sys.getenv("NLEFLOW_SIM_SEED", "6101"))
WITH_LL <- Sys.getenv("NLEFLOW_LL", "1") != "0"
DRAWS_PER_FILE <- max(1L, as.integer(20000L / TPD))
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
  if (!is.matrix(U)) U <- matrix(U, nrow = n, dimnames = list(NULL, CTX))
  if (!is.null(spec$regimes)) {
    shares <- vapply(spec$regimes, function(r) r$share, 0)
    reg <- sample.int(length(spec$regimes), n, replace = TRUE, prob = shares)
    for (k in seq_along(spec$regimes)) for (z in unlist(spec$regimes[[k]]$zero)) U[reg == k, z] <- 0
  }
  U
}
stored <- function(U) {                    # natural -> EMC2's sampled scale (the spec's transforms)
  S <- U
  for (c in CTX) {
    t <- if (TR[[c]] == "zerobox") STORED[[c]] else TR[[c]]
    S[, c] <- switch(t, identity = U[, c], log = log(U[, c]), probit = qnorm(U[, c]))
  }
  S
}
emc2_pars <- function(U) {                 # natural DDM parameters as dDDM / rDDM take them
  P <- cbind(U, matrix(FIXED, nrow(U), length(FIXED), byrow = TRUE, dimnames = list(NULL, names(FIXED))))
  P[, "SZ"] <- 2 * P[, "SZ"] * pmin(P[, "Z"], 1 - P[, "Z"])    # the proportion as the width dDDM takes
  P[, c("v", "a", "t0", "Z", "SZ", "sv", "st0", "s")]
}
write_rows <- function(d, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  if (FORMAT == "parquet") arrow::write_parquet(d, path) else write.csv(d, path, row.names = FALSE)
}

# --- trials -----------------------------------------------------------------------------
simulate_set <- function(rows, OUT, seed, with_ll) {
  n_draws <- as.integer(ceiling(rows / TPD))
  n_files <- as.integer(ceiling(n_draws / DRAWS_PER_FILE))
  cat(sprintf("DDM (sv, st0): %s draws x %d trials -> %s (%d %s files), seed %d, ll %s, EMC2 %s at %s\n",
              format(n_draws, big.mark = ","), TPD, OUT, n_files, FORMAT, seed, with_ll,
              as.character(packageVersion("EMC2")), find.package("EMC2")))
  one_file <- function(k) {
    set.seed(seed + k)
    first <- (k - 1L) * DRAWS_PER_FILE + 1L
    m <- min(DRAWS_PER_FILE, n_draws - first + 1L)
    U <- draw_params(m)
    P <- emc2_pars(U)[rep(seq_len(m), each = TPD), , drop = FALSE]
    Rf <- factor(rep(1L, nrow(P)), levels = 1:2)
    sim <- EMC2:::rDDM(Rf, P)                         # make_data's sampler and precision
    ll <- if (with_ll) pmax(log(EMC2:::dDDM(sim$rt, sim$R, P, precision = 1e-6)), MIN_LL) else NA_real_
    S <- stored(U)[rep(seq_len(m), each = TPD), , drop = FALSE]
    d <- data.frame(param_id = rep(first - 1L + seq_len(m), each = TPD), R = as.integer(sim$R),
                    rt = sim$rt, ll = ll, S, check.names = FALSE)
    write_rows(d, file.path(OUT, sprintf("trials_%04d.%s", k, FORMAT)))
    lr <- log(d$rt)
    c(n = nrow(d), sv0 = sum(U[, "sv"] == 0), st00 = sum(U[, "st0"] == 0),
      bad = sum(!is.finite(d$rt) | d$rt <= 0 | (with_ll & !is.finite(d$ll))),
      floor = if (with_ll) sum(ll <= MIN_LL + 1e-9) else 0,
      lo = sum(lr < spec$spline$range[[1]]), hi = sum(lr > spec$spline$range[[2]]),
      upper = sum(d$R == 2))
  }
  .t <- proc.time()[3]
  res <- do.call(rbind, mclapply(seq_len(n_files), one_file, mc.cores = CORES))
  stopifnot(is.matrix(res), nrow(res) == n_files, sum(res[, "bad"]) == 0)
  N <- sum(res[, "n"])
  cat(sprintf(paste0("=== %s rows in %.1f min; sv = 0 in %.3f and st0 = 0 in %.3f of draws; R = 2 in %.3f; ",
                     "ll at floor %.3f%%; log rt below / above the spline range %.1e / %.1e ===\n"),
              format(N, big.mark = ","), (proc.time()[3] - .t) / 60, sum(res[, "sv0"]) / n_draws,
              sum(res[, "st00"]) / n_draws, sum(res[, "upper"]) / N, 100 * sum(res[, "floor"]) / N,
              sum(res[, "lo"]) / N, sum(res[, "hi"]) / N))
  saveRDS(list(rows = rows, draws = n_draws, trials_per_draw = TPD, seed = seed, format = FORMAT,
               emc2 = as.character(packageVersion("EMC2"))), file.path(OUT, "sim_config.rds"))
}

# --- the choice sets: exact P(R = 2) ----------------------------------------------------
choice_set <- function(n, path, seed) {
  path <- sub("\\.(parquet|csv)$", paste0(".", FORMAT), path, ignore.case = TRUE)  # the name states the format written
  set.seed(seed)
  U <- draw_params(n)
  .t <- proc.time()[3]
  P <- emc2_pars(U)
  pv <- unlist(mclapply(seq_len(n), function(i)
    pWDM(1e6, response = "upper", a = P[i, "a"] / P[i, "s"], v = P[i, "v"] / P[i, "s"], w = P[i, "Z"], t0 = 0,
         sv = P[i, "sv"] / P[i, "s"], sw = P[i, "SZ"], st0 = 0, precision = 1e-8)$value, mc.cores = CORES))
  stopifnot(length(pv) == n, all(is.finite(pv)), max(pv - 1, 0) < 1e-8, max(-pv, 0) < 1e-8)
  write_rows(data.frame(stored(U), p_R2 = pmin(pmax(pv, 0), 1), check.names = FALSE), path)
  cat(sprintf("choice set %s: %d draws, P(R = 2) in [%.4f, %.4f], %.1f min\n", path, n, min(pv), max(pv),
              (proc.time()[3] - .t) / 60))
}

simulate_set(env_n("NLEFLOW_ROWS", 24e6), Sys.getenv("NLEFLOW_TRAIN"), SEED, WITH_LL)
simulate_set(env_n("NLEFLOW_VAL_ROWS", 1e6), Sys.getenv("NLEFLOW_VAL"), SEED + 1L, TRUE)
choice_set(as.integer(env_n("NLEFLOW_CHOICE_ROWS", 2e5)), Sys.getenv("NLEFLOW_CHOICE_TRAIN"), SEED + 2L)
choice_set(as.integer(env_n("NLEFLOW_CHOICE_VAL_ROWS", 2e4)), Sys.getenv("NLEFLOW_CHOICE_VAL"), SEED + 3L)

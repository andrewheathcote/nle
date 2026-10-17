# RDM single-accumulator training data whose box reaches A = 0, for nleflow
# (the NLE project's simulation/sim_rdm_a0.R, driven by the spec).
#
# One row is one accumulator, one trial, one parameter draw. Columns: param_id,
# S, R, rt, C, ll, and the context (v, B, t0, A, s) on EMC2's sampled (log)
# scale, A stored as log A with -Inf at exactly zero. v, B, t0, s are uniform
# on the natural scale over the spec's box; A follows the spec's strata:
#   uniform               uniform on [0, A_max], natural scale      (stored finite)
#   asinh_uniform         uniform in asinh(A / c_A) on [0, A_max]    (stored finite)
#   zero                  A = 0 exactly                              (stored -Inf)
# Strata are assigned by row at random, so every file holds the mixture.
#
# Run by `nleflow simulate <spec>`, which sets NLEFLOW_SPEC (the resolved spec,
# JSON), NLEFLOW_TRAIN, NLEFLOW_VAL, NLEFLOW_ROWS, NLEFLOW_VAL_ROWS and
# NLEFLOW_CORES (see nleflow/simulate.py). Also:
#   NLEFLOW_RLIB    the library holding EMC2 (asserted when given)
#   NLEFLOW_FORMAT  parquet (default when the arrow package is installed) or csv
#   NLEFLOW_SIM_SEED  the data seed (default 4876, the project's; the
#                   validation set uses seed + 1). training.seed is the
#                   network's seed, not the data's.
# Needs EMC2 (rWald, dRDM) and jsonlite; arrow for parquet.

LIB <- Sys.getenv("NLEFLOW_RLIB", "")
if (nzchar(LIB)) .libPaths(c(normalizePath(path.expand(LIB)), .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(jsonlite); library(parallel)})
if (nzchar(LIB))
  stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(path.expand(LIB))))
have_arrow <- requireNamespace("arrow", quietly = TRUE)
FORMAT <- Sys.getenv("NLEFLOW_FORMAT", if (have_arrow) "parquet" else "csv")
if (FORMAT == "parquet" && !have_arrow) stop("NLEFLOW_FORMAT=parquet needs the arrow package")
stopifnot(FORMAT %in% c("parquet", "csv"))

`%||%` <- function(a, b) if (is.null(a)) b else a   # R < 4.4 has no %||%
spec <- fromJSON(Sys.getenv("NLEFLOW_SPEC"))
stopifnot(identical(spec$context, c("v", "B", "t0", "s", "A")), spec$transforms$A == "zerobox")
box   <- spec$box_natural
C_A   <- spec$zerobox_c$A
SH    <- spec$sampling$A
SHARES <- c(natural = SH$uniform %||% 0, asinh = SH$asinh_uniform %||% 0, zero = SH$zero %||% 0)
stopifnot(abs(sum(SHARES) - 1) < 1e-9)
LOWER <- c(v = box$v[1], B = box$B[1], t0 = box$t0[1], s = box$s[1])
UPPER <- c(v = box$v[2], B = box$B[2], t0 = box$t0[2], s = box$s[2])
A_MAX <- box$A[2]
MIN_LL <- log(1e-10)
CORES <- as.integer(Sys.getenv("NLEFLOW_CORES", "4"))
CHUNK <- as.numeric(Sys.getenv("NLEFLOW_CHUNK", "1e5"))
SEED  <- as.integer(Sys.getenv("NLEFLOW_SIM_SEED", "4876"))

env_n <- function(v, default) { x <- Sys.getenv(v, ""); if (nzchar(x)) as.numeric(x) else default }

simulate_set <- function(N, OUT, seed) {
  dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
  n_files <- ceiling(N / CHUNK)
  cat(sprintf("RDM A0 box: %s rows -> %s (%d %s files), seed %d, EMC2 %s at %s\n",
              format(N, big.mark = ","), OUT, n_files, FORMAT, seed,
              as.character(packageVersion("EMC2")), find.package("EMC2")))
  one_file <- function(k) {
    set.seed(seed + k)                          # one stream per file, reproducible at any CORES
    n <- if (k < n_files) CHUNK else N - (n_files - 1) * CHUNK
    nat <- sapply(names(LOWER), function(p) runif(n, LOWER[[p]], UPPER[[p]]))
    stratum <- sample(names(SHARES), n, replace = TRUE, prob = SHARES)
    A <- numeric(n)
    i <- stratum == "natural"; A[i] <- runif(sum(i), 0, A_MAX)
    i <- stratum == "asinh";   A[i] <- C_A * sinh(runif(sum(i), 0, asinh(A_MAX / C_A)))
    rt <- EMC2:::rWald(n, B = nat[, "B"] / nat[, "s"], v = nat[, "v"] / nat[, "s"],
                       A = A / nat[, "s"]) + nat[, "t0"]
    rt <- floor(rt / 1e-12) * 1e-12
    pars <- cbind(v = nat[, "v"], B = nat[, "B"], A = A, t0 = nat[, "t0"], s = nat[, "s"])
    ll <- pmax(log(EMC2:::dRDM(rt, pars)), MIN_LL)
    d <- data.frame(param_id = as.integer((k - 1) * CHUNK + seq_len(n)), S = NA, R = 1L,
                    rt = rt, C = NA, ll = ll,
                    v = log(nat[, "v"]), B = log(nat[, "B"]), t0 = log(nat[, "t0"]),
                    A = log(A),                 # -Inf at exactly zero
                    s = log(nat[, "s"]))
    if (FORMAT == "parquet") arrow::write_parquet(d, file.path(OUT, sprintf("trials_%04d.parquet", k)))
    else write.csv(d, file.path(OUT, sprintf("trials_%04d.csv", k)), row.names = FALSE)
    c(n = n, zero = sum(A == 0), bad = sum(!is.finite(rt) | !is.finite(ll) | rt <= nat[, "t0"]))
  }
  .t <- proc.time()[3]
  res <- do.call(rbind, mclapply(seq_len(n_files), one_file, mc.cores = CORES))
  stopifnot(is.matrix(res), nrow(res) == n_files, sum(res[, "bad"]) == 0)
  cat(sprintf("=== simulated %s rows in %.1f min; A = 0 exactly in %.4f ===\n",
              format(sum(res[, "n"]), big.mark = ","), (proc.time()[3] - .t) / 60,
              sum(res[, "zero"]) / sum(res[, "n"])))
  saveRDS(list(N = N, seed = seed, c_A = C_A, shares = SHARES, lower = LOWER, upper = UPPER,
               A_max = A_MAX, format = FORMAT, emc2 = as.character(packageVersion("EMC2"))),
          file.path(OUT, "sim_config.rds"))
}

simulate_set(env_n("NLEFLOW_ROWS", 24e6), Sys.getenv("NLEFLOW_TRAIN"), SEED)
simulate_set(env_n("NLEFLOW_VAL_ROWS", 1e6), Sys.getenv("NLEFLOW_VAL"), SEED + 1L)

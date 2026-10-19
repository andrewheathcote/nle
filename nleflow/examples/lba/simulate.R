# LBA single-accumulator training data for nleflow: one accumulator of EMC2's
# LBA (posdrift: the rate is N(v, sv) truncated to be positive), one trial per
# parameter draw, the box reaching A = 0 exactly.
#
# One row is one accumulator, one trial, one parameter draw. Columns: param_id,
# R, rt, ll (EMC2's dLBA, floored at log 1e-10), and the context (v, sv, B, A,
# t0) on EMC2's sampled scale: v as it is (identity), sv, B, t0 as logs, A as
# log A with -Inf at exactly zero. Each input is drawn from the spec's strata
# (`sampling`), uniform on the natural box when it has none:
#   uniform               uniform on the natural scale over its range (default the box)
#   log_uniform           uniform in log x over its range
#   asinh_uniform         uniform in asinh(x / c) over its range (zero-box inputs)
#   zero                  exactly 0 (zero-box inputs; stored -Inf)
# Strata are assigned by row at random, so every file holds the mixture.
#
# The finishing time is (b - A U) / rate with b = B + A, U uniform and the rate
# drawn by inversion from its upper tail, rate = v + sv qnorm(U' pnorm(v / sv),
# lower.tail = FALSE), which stays accurate where pnorm(v / sv) is small.
#
# Run by `nleflow simulate <spec>`, which sets NLEFLOW_SPEC (the resolved spec,
# JSON), NLEFLOW_TRAIN, NLEFLOW_VAL, NLEFLOW_ROWS, NLEFLOW_VAL_ROWS and
# NLEFLOW_CORES. Also:
#   NLEFLOW_RLIB      the library holding EMC2, or a list a:b as R_LIBS (asserted when given)
#   NLEFLOW_FORMAT    parquet (default when the arrow package is installed) or csv
#   NLEFLOW_SIM_SEED  the data seed (default 9017; the validation set uses
#                     seed + 1). training.seed is the network's seed.
# Needs EMC2 (dLBA) and jsonlite; arrow for parquet.

LIB <- Sys.getenv("NLEFLOW_RLIB", "")
LIBS <- if (nzchar(LIB)) normalizePath(path.expand(strsplit(LIB, ":", fixed = TRUE)[[1]])) else character(0)
if (nzchar(LIB)) .libPaths(c(LIBS, .libPaths()))   # one library or a list, as R_LIBS
suppressPackageStartupMessages({library(EMC2); library(jsonlite); library(parallel)})
if (nzchar(LIB))
  stopifnot(any(startsWith(normalizePath(find.package("EMC2")), LIBS)))
have_arrow <- requireNamespace("arrow", quietly = TRUE)
FORMAT <- Sys.getenv("NLEFLOW_FORMAT", if (have_arrow) "parquet" else "csv")
if (FORMAT == "parquet" && !have_arrow) stop("NLEFLOW_FORMAT=parquet needs the arrow package")
stopifnot(FORMAT %in% c("parquet", "csv"))

spec <- fromJSON(Sys.getenv("NLEFLOW_SPEC"), simplifyVector = FALSE)
CTX <- unlist(spec$context)
stopifnot(identical(CTX, c("v", "sv", "B", "A", "t0")), spec$model == "LBA")
TR  <- unlist(spec$transforms)
BOX <- lapply(spec$box_natural, unlist)
ZC  <- unlist(spec$zerobox_c)
STRATA <- lapply(CTX, function(p) {
  s <- spec$sampling[[p]]
  if (is.null(s)) return(list(list(kind = "uniform", share = 1, range = BOX[[p]])))
  lapply(s, function(r) list(kind = r$kind, share = r$share,
                             range = if (is.null(r$range)) BOX[[p]] else unlist(r$range)))
})
names(STRATA) <- CTX
MIN_LL <- log(1e-10)
CORES <- as.integer(Sys.getenv("NLEFLOW_CORES", "4"))
CHUNK <- as.numeric(Sys.getenv("NLEFLOW_CHUNK", "1e5"))
SEED  <- as.integer(Sys.getenv("NLEFLOW_SIM_SEED", "9017"))

env_n <- function(v, default) { x <- Sys.getenv(v, ""); if (nzchar(x)) as.numeric(x) else default }

# n natural values of input p from its strata
draw <- function(p, n) {
  st <- STRATA[[p]]
  k <- sample.int(length(st), n, replace = TRUE, prob = vapply(st, `[[`, 0, "share"))
  x <- numeric(n)
  for (j in seq_along(st)) {
    i <- k == j; m <- sum(i); r <- st[[j]]$range
    x[i] <- switch(st[[j]]$kind,
      uniform       = runif(m, r[1], r[2]),
      log_uniform   = exp(runif(m, log(r[1]), log(r[2]))),
      asinh_uniform = ZC[[p]] * sinh(runif(m, asinh(r[1] / ZC[[p]]), asinh(r[2] / ZC[[p]]))),
      zero          = rep(0, m))
  }
  x
}

stored <- function(p, x) switch(TR[[p]], identity = x, log = log(x), zerobox = log(x))   # log 0 = -Inf

simulate_set <- function(N, OUT, seed) {
  dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
  n_files <- ceiling(N / CHUNK)
  cat(sprintf("LBA box: %s rows -> %s (%d %s files), seed %d, EMC2 %s at %s\n",
              format(N, big.mark = ","), OUT, n_files, FORMAT, seed,
              as.character(packageVersion("EMC2")), find.package("EMC2")))
  one_file <- function(k) {
    set.seed(seed + k)                          # one stream per file, reproducible at any CORES
    n <- if (k < n_files) CHUNK else N - (n_files - 1) * CHUNK
    nat <- sapply(CTX, draw, n = n)
    b <- nat[, "B"] + nat[, "A"]
    rate <- nat[, "v"] + nat[, "sv"] * qnorm(runif(n) * pnorm(nat[, "v"] / nat[, "sv"]), lower.tail = FALSE)
    rt <- (b - nat[, "A"] * runif(n)) / rate + nat[, "t0"]
    rt <- floor(rt / 1e-12) * 1e-12
    pars <- cbind(v = nat[, "v"], sv = nat[, "sv"], b = b, A = nat[, "A"], t0 = nat[, "t0"])
    ll <- pmax(log(EMC2:::dLBA(rt, pars)), MIN_LL)
    d <- data.frame(param_id = as.integer((k - 1) * CHUNK + seq_len(n)), R = 1L, rt = rt, ll = ll)
    for (p in CTX) d[[p]] <- stored(p, nat[, p])
    if (FORMAT == "parquet") arrow::write_parquet(d, file.path(OUT, sprintf("trials_%04d.parquet", k)))
    else write.csv(d, file.path(OUT, sprintf("trials_%04d.csv", k)), row.names = FALSE)
    c(n = n, zero = sum(nat[, "A"] == 0), bad = sum(!is.finite(rt) | !is.finite(ll) | rt <= nat[, "t0"]),
      floored = sum(ll <= MIN_LL), tail = sum(log(rt) > spec$spline$range[[2]]))
  }
  .t <- proc.time()[3]
  res <- do.call(rbind, mclapply(seq_len(n_files), one_file, mc.cores = CORES))
  stopifnot(is.matrix(res), nrow(res) == n_files, sum(res[, "bad"]) == 0)
  tot <- colSums(res)
  cat(sprintf("=== simulated %s rows in %.1f min; A = 0 exactly in %.4f; ll floored in %.5f; log rt beyond the spline range in %.6f ===\n",
              format(tot[["n"]], big.mark = ","), (proc.time()[3] - .t) / 60,
              tot[["zero"]] / tot[["n"]], tot[["floored"]] / tot[["n"]], tot[["tail"]] / tot[["n"]]))
  saveRDS(list(N = N, seed = seed, strata = STRATA, box = BOX, format = FORMAT,
               emc2 = as.character(packageVersion("EMC2"))),
          file.path(OUT, "sim_config.rds"))
}

simulate_set(env_n("NLEFLOW_ROWS", 24e6), Sys.getenv("NLEFLOW_TRAIN"), SEED)
simulate_set(env_n("NLEFLOW_VAL_ROWS", 1e6), Sys.getenv("NLEFLOW_VAL"), SEED + 1L)

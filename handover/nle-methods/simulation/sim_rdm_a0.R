# RDM single-accumulator training data whose box reaches A = 0
# (handover/RDM_A0_BOX_PLAN.md, section 2).
#
# One row is one accumulator, one trial, one parameter draw, as in
# simulation/sim_rdm_16m.R, and the columns are the same: param_id, S, R, rt,
# C, ll, v, B, t0, A, s, the parameters on EMC2's sampled (log) scale. The one
# difference is A:
#
#   stratum                                   share   stored log A
#   uniform on [0, 2], natural scale           70%    finite
#   uniform in asinh(A / C_A) on [0, 2]        15%    finite
#   A = 0 exactly                              15%    -Inf
#
# v, B, t0, s are uniform on the natural scale over the box of
# simulation/designs.R (lowerRDM_single / upperRDM_single), as before. Draws
# are pseudo-random (the 16m set used Sobol points); strata are assigned by
# row at random, so every file holds the mixture.
#
# Self-contained on purpose: it needs EMC2 and arrow only, so it runs on tux26
# against the library the cells use.
#
#   Rscript simulation/sim_rdm_a0.R
# Env: A0_N (rows, default 24e6)   A0_OUT (default data/train/rdm24m_a0)
#      A0_SEED (default 4876; use another for the validation set)
#      A0_CORES (default 8)        A0_CHUNK (rows per file, default 1e5)
#      A0_LIB  library holding EMC2 (default ./.Rlibs when present). Asserted.

LIB <- Sys.getenv("A0_LIB", "")
if (nzchar(LIB)) {
  .libPaths(c(normalizePath(path.expand(LIB)), .libPaths()))
} else if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(arrow); library(parallel)})
if (nzchar(LIB))
  stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(path.expand(LIB))))

N     <- as.numeric(Sys.getenv("A0_N", "24e6"))
OUT   <- Sys.getenv("A0_OUT", "data/train/rdm24m_a0")
SEED  <- as.integer(Sys.getenv("A0_SEED", "4876"))
CORES <- as.integer(Sys.getenv("A0_CORES", "8"))
CHUNK <- as.numeric(Sys.getenv("A0_CHUNK", "1e5"))

C_A    <- 0.05                                  # asinh scale; the card's zerobox_c
SHARES <- c(natural = .70, asinh = .15, zero = .15)
LOWER  <- c(v = 1e-3, B = 0.1, t0 = 0.05, s = 0.1)
UPPER  <- c(v = 5,    B = 3,   t0 = 1,    s = 2)
A_MAX  <- 2
MIN_LL <- log(1e-10)

dir_trials <- file.path(OUT, "RDM", "trials")
dir.create(dir_trials, recursive = TRUE, showWarnings = FALSE)
n_files <- ceiling(N / CHUNK)
cat(sprintf("RDM A0 box: %s rows -> %s (%d files), seed %d, EMC2 %s at %s\n",
            format(N, big.mark = ","), OUT, n_files, SEED,
            as.character(packageVersion("EMC2")), find.package("EMC2")))
cat(sprintf("A strata: %s; c_A = %g\n",
            paste(sprintf("%s %.0f%%", names(SHARES), 100 * SHARES), collapse = ", "), C_A))

one_file <- function(k) {
  set.seed(SEED + k)                            # one stream per file, reproducible at any CORES
  n <- if (k < n_files) CHUNK else N - (n_files - 1) * CHUNK
  nat <- sapply(names(LOWER), function(p) runif(n, LOWER[[p]], UPPER[[p]]))
  stratum <- sample(names(SHARES), n, replace = TRUE, prob = SHARES)
  A <- numeric(n)
  i <- stratum == "natural"; A[i] <- runif(sum(i), 0, A_MAX)
  i <- stratum == "asinh";   A[i] <- C_A * sinh(runif(sum(i), 0, asinh(A_MAX / C_A)))
  # stratum "zero" stays exactly 0
  rt <- EMC2:::rWald(n, B = nat[, "B"] / nat[, "s"], v = nat[, "v"] / nat[, "s"],
                     A = A / nat[, "s"]) + nat[, "t0"]
  rt <- floor(rt / 1e-12) * 1e-12
  pars <- cbind(v = nat[, "v"], B = nat[, "B"], A = A, t0 = nat[, "t0"], s = nat[, "s"])
  ll <- pmax(log(EMC2:::dRDM(rt, pars)), MIN_LL)
  d <- data.frame(param_id = as.integer((k - 1) * CHUNK + seq_len(n)), S = NA, R = 1L,
                  rt = rt, C = NA, ll = ll,
                  v = log(nat[, "v"]), B = log(nat[, "B"]), t0 = log(nat[, "t0"]),
                  A = log(A),                   # -Inf at exactly zero
                  s = log(nat[, "s"]))
  write_parquet(d, file.path(dir_trials, sprintf("trials_%04d.parquet", k)))
  c(n = n, zero = sum(A == 0), bad = sum(!is.finite(rt) | !is.finite(ll) | rt <= nat[, "t0"]))
}

.t <- proc.time()[3]
res <- do.call(rbind, mclapply(seq_len(n_files), one_file, mc.cores = CORES))
stopifnot(is.matrix(res), nrow(res) == n_files)
cat(sprintf("=== simulated %s rows in %.1f min; A = 0 exactly in %.4f; bad rows %d ===\n",
            format(sum(res[, "n"]), big.mark = ","), (proc.time()[3] - .t) / 60,
            sum(res[, "zero"]) / sum(res[, "n"]), sum(res[, "bad"])))
saveRDS(list(N = N, seed = SEED, c_A = C_A, shares = SHARES, lower = LOWER, upper = UPPER,
             A_max = A_MAX, emc2 = as.character(packageVersion("EMC2"))),
        file.path(OUT, "RDM", "sim_config.rds"))

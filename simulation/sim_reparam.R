# DDM training data in the (SZ0, SZ, SZ1) parameterisation.
#
#   SZ0 = 0 -> lower edge of the start-point distribution
#   SZ  = its width
#   SZ1 = upper edge -> a
#   a = SZ0 + SZ + SZ1,  w = (SZ0 + SZ/2)/a,  sw = SZ/a
#
# Every positive triple is valid, so the constraint that forces EMC2's
#     Ttransform: sw = 2 * SZ_prop * min(Z, 1-Z)
# (non-differentiable at Z = 0.5, and coupling SZ to Z and a) disappears. The
# flow is trained on the triple; parameter count stays 8, architecture
# unchanged.
#
# Simulation goes through EMC2's existing simulate_trials rather than WienR's
# rWDM: rWDM fails internally on parts of this box (emitting "convex2?" and
# never returning), whereas the EMC2 path is the proven one - it produced the
# original 1M-row sets in ~1.6 min. We simply INVERT the Ttransform to find the
# EMC2 parameters that reproduce the sw we want:
#     Z_emc2  = qnorm(w)
#     SZ_emc2 = qnorm( sw / (2 * min(w, 1-w)) )
# which is <= 1 by construction (SZ0, SZ1 > 0), clipped just below 1 for qnorm.
# So the SIMULATED data are identical in distribution to what the triple means,
# while the stored CONTEXT columns are the new parameterisation.
#
#   Rscript simulation/sim_reparam.R

# The project-local .Rlibs holds the EMC2 build this pipeline was validated
# against. The system library has a DIFFERENT build of the same version number
# (3.4.1) whose DDM likelihood returns 0 for these parameters, pinning every ll
# at the min_ll floor while make_data still returns sensible rt - a silent
# corruption that only shows up in the ll column. Always prefer .Rlibs.
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))

source("simulation/sim_functions.R")
source("simulation/designs.R")
suppressPackageStartupMessages(library(arrow))

N_PARAMS <- as.integer(Sys.getenv("RP_N_PARAMS", "400000"))
N_TRIALS <- as.integer(Sys.getenv("RP_N_TRIALS", "20"))   # per stimulus -> 40 rows
SEED <- 3876; SZ_LO <- 0.002; SZ_HI <- 5
out <- file.path("data", "train", "reparam40", "DDM")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

set.seed(SEED)
lu <- function(n, lo, hi) exp(runif(n, log(lo), log(hi)))
P <- NULL
while (is.null(P) || nrow(P) < N_PARAMS) {
  m <- 3L * N_PARAMS
  d <- data.frame(v = runif(m, -6, 6), SZ0 = lu(m, SZ_LO, SZ_HI),
                  SZ = lu(m, SZ_LO, SZ_HI), SZ1 = lu(m, SZ_LO, SZ_HI),
                  t0 = lu(m, 0.05, 1), s = lu(m, 0.1, 2),
                  sv = lu(m, 0.01, 4), st0 = lu(m, 0.01, 0.49))
  d$a <- d$SZ0 + d$SZ + d$SZ1
  # Match the OLD box on both a and the relative start point. Unbounded, the
  # triple gives w down to 0.001 - essentially starting on the lower boundary -
  # where EMC2's make_data returns a degenerate result. Bounding w to the old
  # Z-range keeps the two parameterisations comparable, which is the point:
  # the reparameterisation should be the only difference.
  # Reject against EMC2's own DDM bounds, which apply AFTER the Ttransform.
  # The binding one is SZ in [0.01, 0.99] applied to sw itself - so the
  # constraint is a MODEL bound, not an artefact of the old parameterisation,
  # and the triple does not escape it. What the reparameterisation still buys
  # is that the network never sees the min(Z, 1-Z) mapping: it is fed SZ0, SZ,
  # SZ1 directly, so the non-differentiable function is gone from its input
  # even though the trainable region is the same.
  d$w_chk  <- (d$SZ0 + d$SZ / 2) / d$a
  d$sw_chk <- d$SZ / d$a
  d <- d[d$a >= 0.05 & d$a <= 5 &
         d$w_chk >= 0.1 & d$w_chk <= 0.9 &
         d$sw_chk >= 0.01 & d$sw_chk <= 0.99, , drop = FALSE]
  d$w_chk <- NULL; d$sw_chk <- NULL
  P <- if (is.null(P)) d else rbind(P, d)
}
P <- P[seq_len(N_PARAMS), , drop = FALSE]
P$w  <- (P$SZ0 + P$SZ / 2) / P$a
P$sw <- P$SZ / P$a
P$SZprop <- pmin(P$sw / (2 * pmin(P$w, 1 - P$w)), 0.999)
cat(sprintf("%d draws: a [%.3f, %.3f]  w [%.3f, %.3f]  SZprop [%.3f, %.3f]\n",
            nrow(P), min(P$a), max(P$a), min(P$w), max(P$w),
            min(P$SZprop), max(P$SZprop)))

# EMC2 sampled scale: v identity, a/t0/s/sv/st0 log, Z/SZ probit
# simulate_data() adds param_id itself and indexes columns by name, so pass a
# bare matrix in param_namesDDM order.
params <- cbind(v = P$v, a = log(P$a), t0 = log(P$t0), s = log(P$s),
                Z = qnorm(P$w), SZ = qnorm(P$SZprop),
                sv = log(P$sv), st0 = log(P$st0))
stopifnot(identical(colnames(params), param_namesDDM))
.tstart <- proc.time()[3]  # NOT 't0': that is a DDM parameter name and
                          # the design formulae are evaluated in the global env,
                          # which silently clobbers the non-decision time in the
                          # likelihood path (every ll pinned at the min_ll floor).
simulate_data(param_namesDDM, lowerDDM, upperDDM, designDDM,
              n_params = N_PARAMS, transforms = transformsDDM,
              out_dir = out, chunk_size = 1e5, seed = SEED,
              n_trials = N_TRIALS, params = params,
              n_cores = as.integer(Sys.getenv("RP_CORES", "4")))
cat(sprintf("=== simulated in %.1f min; now rewriting context columns ===\n",
            (proc.time()[3] - .tstart) / 60))

# replace the EMC2 (a, Z, SZ) context columns with the triple the flow trains on
files <- list.files(out, pattern = "\\.parquet$", full.names = TRUE, recursive = TRUE)
for (f in files) {
  d <- as.data.frame(read_parquet(f))
  i <- d$param_id
  d$a <- log(P$SZ0[i]); d$Z <- log(P$SZ[i]); d$SZ <- log(P$SZ1[i])
  names(d)[names(d) == "a"] <- "SZ0"; names(d)[names(d) == "Z"] <- "SZw"
  names(d)[names(d) == "SZ"] <- "SZ1"
  write_parquet(d, f)
}
cat(sprintf("=== done in %.1f min -> %s ===\n", (proc.time()[3] - .tstart) / 60, out))

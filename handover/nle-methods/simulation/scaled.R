# The scaled DDM (handover/SCALED_DDM_PLAN.md): a network over the unit-scale decision time
# u = (rt - t0) (s/a)^2 of a diffusion with a = s = 1, conditioned on kappa = v a / s^2,
# eta = sv a / s^2, Z and SZ. t0, st0, a and s are applied exactly at evaluation
# (port/R/ddm_scaled.R), so the simulator runs EMC2's DDM at a = s = 1, st0 = 0 and a fixed
# T0SIM, and stores u = rt - T0SIM.
#
# Storage scale (the parquet), as for the zero box: kappa identity, eta log (-Inf at an exact
# zero), Z probit, SZ probit (-Inf at zero). The network scale is the same except eta and SZ are
# asinh(x / c) (flow/src/flow/data.py, encoding "zerobox", constants below).
#
# Box (AH 2026-10-13): kappa in [-10, 10]; eta in {0} U [0.01, 8]; Z in [0.1, 0.9];
# SZ in {0} U [0.05, 0.99]. Regimes: full 55%, eta = 0 15%, SZ = 0 15%, both 15%.
SC_C     <- c(eta = 0.01, SZ = 0.05)
SC_KAPPA <- c(-10, 10); SC_KAPPA_CORE <- c(-3, 3)        # half the draws in the core
SC_ETA   <- c(0.01, 8);  SC_ETA_LOGMIX <- 0.75            # free eta: 75% log-uniform, 25% asinh-uniform on [0.01, 8]
# Free eta starts at 0.01 because the simulator is EMC2's DDM at a = s = 1, where eta is sv and
# EMC2 refuses sv in (0, 0.01). eta in (0, 0.01) is reached (sv = 0.01, a < 1) but its density is
# indistinguishable from eta = 0; the asinh encoding interpolates there, as the zero box did for sv.
SC_Z     <- c(0.1, 0.9)
SC_SZ    <- c(0.05, 0.99)
SC_T0SIM <- 0.05
SC_REGIMES <- data.frame(regime = c("full", "eta_0", "SZ_0", "eta_SZ_0"),
                         eta0 = c(FALSE, TRUE, FALSE, TRUE), SZ0 = c(FALSE, FALSE, TRUE, TRUE),
                         share = c(0.55, 0.15, 0.15, 0.15), stringsAsFactors = FALSE)
SC_INPUTS <- c("kappa", "eta", "Z", "SZ")

# n draws -> list(unit = natural (kappa, eta, Z, SZ), emc2 = EMC2-scale DDM rows for make_data,
# regime). Sobol-stratified; shuffled so chunk files are not regime-blocked.
param_matrix_scaled <- function(n, seed) {
  set.seed(seed)
  counts <- round(SC_REGIMES$share * n); counts[1] <- counts[1] + n - sum(counts)
  regime <- rep(SC_REGIMES$regime, counts)
  # randtoolbox's scrambling is disabled, so the Sobol points do not depend on the seed and the
  # validation draws would repeat the first training draws: a seeded random shift (Cranley-
  # Patterson rotation) keeps the low discrepancy and makes the sets differ.
  u <- suppressWarnings(randtoolbox::sobol(n = n, dim = 5, normal = FALSE))
  u <- (u + matrix(runif(5), n, 5, byrow = TRUE)) %% 1
  core <- seq_len(n) %% 2L == 0L
  kappa <- ifelse(core, SC_KAPPA_CORE[1] + u[, 1] * diff(SC_KAPPA_CORE), SC_KAPPA[1] + u[, 1] * diff(SC_KAPPA))
  logmix <- (seq_len(n) %% round(1 / (1 - SC_ETA_LOGMIX))) != 0L
  eta <- ifelse(logmix, exp(log(SC_ETA[1]) + u[, 2] * diff(log(SC_ETA))),
                sinh(asinh(SC_ETA[1] / SC_C[["eta"]]) + u[, 2] * diff(asinh(SC_ETA / SC_C[["eta"]]))) * SC_C[["eta"]])
  Z <- SC_Z[1] + u[, 3] * diff(SC_Z)
  lo <- asinh(SC_SZ[1] / SC_C[["SZ"]]); hi <- asinh(SC_SZ[2] / SC_C[["SZ"]])
  SZ <- sinh(lo + u[, 4] * (hi - lo)) * SC_C[["SZ"]]
  eta[regime %in% c("eta_0", "eta_SZ_0")] <- 0
  SZ[regime %in% c("SZ_0", "eta_SZ_0")] <- 0
  o <- sample.int(n)
  unit <- cbind(kappa = kappa, eta = eta, Z = Z, SZ = SZ)[o, ]
  emc2 <- cbind(v = unit[, "kappa"], a = 0, t0 = log(SC_T0SIM), s = 0, Z = qnorm(unit[, "Z"]),
                SZ = qnorm(unit[, "SZ"]), sv = log(unit[, "eta"]), st0 = -Inf)
  list(unit = unit, emc2 = emc2, regime = regime[o])
}

# A simulated trial chunk (simulate_data's columns) -> the scaled network's columns.
scaled_columns <- function(d) {
  stopifnot(all(d$a == 0), all(d$s == 0), all(abs(d$t0 - log(SC_T0SIM)) < 1e-12), all(is.infinite(d$st0)))
  data.frame(param_id = d$param_id, R = d$R, rt = d$rt - SC_T0SIM, ll = d$ll,
             kappa = d$v, eta = d$sv, Z = d$Z, SZ = d$SZ)
}

# The zero-box DDM: sv, SZ and st0 may be EXACTLY zero.
#
# Sampler and context transform for a single flow that covers the full DDM and
# every reduced model with one or more variability parameters fixed at zero.
# See handover/ZERO_BOX_RETRAIN_PLAN.md.
#
# Two scales, kept deliberately separate:
#
#   EMC2 scale    what the simulator and the stored `ll` use: log a, log t0,
#                 log s, probit Z, probit SZ, log sv, log st0 -- with -Inf at
#                 an exact zero. EMC2:::dDDM handles that exactly (verified to
#                 <= 2e-8 against WienR 1e-8 in every regime).
#   network scale what the flow is conditioned on: the same, except that the
#                 three variability parameters use asinh(x / c), which is 0 at
#                 0, linear below c and log-like above it.
#
# The admissible set mirrors EMC2's DDM()$bound: sv in {0} U [0.01, 4],
# SZ in {0} U [0.05, 0.99], st0 in [0, 0.49]. EMC2 refuses sv or SZ strictly
# between 0 and the floor, so there is nothing to learn there; st0 is admitted
# continuously and that gap is where the density moves fastest, so it is
# sampled densely.

ZB_C <- c(sv = 0.01, SZ = 0.05, st0 = 0.002)      # asinh scale constants
ZB_FLOOR <- c(sv = 0.01, SZ = 0.05, st0 = 0)      # lower end of the FREE range
ZB_MAX   <- c(sv = 4.00, SZ = 0.99, st0 = 0.49)

# The eight regimes and their shares of the draws (plan, section 1.3).
ZB_REGIMES <- data.frame(
  regime = c("full", "st0_0", "sv_0", "SZ_0", "st0_sv_0", "st0_SZ_0", "sv_SZ_0", "all_0"),
  sv0  = c(FALSE, FALSE, TRUE,  FALSE, TRUE,  FALSE, TRUE,  TRUE),
  SZ0  = c(FALSE, FALSE, FALSE, TRUE,  FALSE, TRUE,  TRUE,  TRUE),
  st00 = c(FALSE, TRUE,  FALSE, FALSE, TRUE,  TRUE,  FALSE, TRUE),
  share = c(0.40, 0.14, 0.08, 0.08, 0.07, 0.07, 0.06, 0.10),
  stringsAsFactors = FALSE)
stopifnot(abs(sum(ZB_REGIMES$share) - 1) < 1e-12)

zb_asinh <- function(x, par) asinh(x / ZB_C[[par]])
zb_sinh  <- function(u, par) sinh(u) * ZB_C[[par]]

# EMC2-scale parameter matrix -> the network's context (same columns, same
# order; only sv, SZ, st0 change). Exact zeros arrive as -Inf and leave as 0.
zerobox_context <- function(params_emc2) {
  ctx <- params_emc2
  nat <- list(sv = exp(params_emc2[, "sv"]), st0 = exp(params_emc2[, "st0"]),
              SZ = pnorm(params_emc2[, "SZ"]))
  for (p in names(nat)) ctx[, p] <- zb_asinh(nat[[p]], p)
  ctx
}

# Draw n parameter sets on the EMC2 scale, stratified over the eight regimes.
# The five unchanged parameters are Sobol-uniform on the natural scale inside
# lower/upper (as param_matrix does); each free variability parameter is
# Sobol-uniform in its asinh coordinate over [floor, max].
param_matrix_zerobox <- function(n, param_names, lower, upper, transforms, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  stopifnot(identical(param_names, c("v", "a", "t0", "s", "Z", "SZ", "sv", "st0")))
  names(lower) <- names(upper) <- names(transforms) <- param_names
  counts <- round(ZB_REGIMES$share * n)
  counts[1] <- counts[1] + (n - sum(counts))      # absorb rounding in "full"
  regime <- rep(ZB_REGIMES$regime, counts)
  u <- randtoolbox::sobol(n = n, dim = 8, normal = FALSE, scrambling = 1, seed = if (is.null(seed)) 4711 else seed)
  colnames(u) <- param_names
  out <- matrix(NA_real_, n, 8, dimnames = list(NULL, param_names))
  for (p in c("v", "a", "t0", "s", "Z")) {
    nat <- lower[[p]] + u[, p] * (upper[[p]] - lower[[p]])
    out[, p] <- switch(transforms[[p]], identity = nat, log = log(nat), probit = qnorm(nat))
  }
  zero_of <- list(sv = ZB_REGIMES$sv0, SZ = ZB_REGIMES$SZ0, st0 = ZB_REGIMES$st00)
  for (p in c("sv", "SZ", "st0")) {
    is0 <- rep(zero_of[[p]], counts)
    lo <- zb_asinh(ZB_FLOOR[[p]], p); hi <- zb_asinh(ZB_MAX[[p]], p)
    nat <- zb_sinh(lo + u[, p] * (hi - lo), p)
    nat[is0] <- 0
    out[, p] <- if (p == "SZ") qnorm(nat) else log(nat)   # exact zero -> -Inf
  }
  # shuffle so chunks are not regime-blocked (chunk files feed shards)
  o <- sample.int(n)
  structure(out[o, ], regime = regime[o])
}

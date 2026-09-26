# DDMnn0: the EMC2 model function for the ZERO-BOX DDM flow, in which sv, SZ
# and st0 may be exactly zero.
#
# Defined outside EMC2 so it can be sourced beside an unmodified dev-nn install
# (the DDMnn7 pattern). Three things differ from EMC2::DDMnn, and each is a
# trap this project has already paid for:
#
#  1. The theta builder applies the zero-box encoding ITSELF -- asinh(x / c)
#     for sv, SZ, st0 with c read from the bundle -- and hands EMC2's compiled
#     evaluator a context already on the network scale. EMC2 ships its own
#     compiled copy of the port, which does not know 'zerobox'; giving it
#     network-scale theta with a bundle whose bounds are on that scale makes
#     the copy correct without modification. (The bundle's context_encoding is
#     rewritten to 'emc2' on load for exactly that reason.)
#  2. p_types defaults for sv, SZ, st0 are -Inf (= 0), as the analytic DDM's
#     are, and NOT DDMnn's 0.05 / etc. Omitting one from a formula then fixes it
#     at exactly zero in BOTH simulation and evaluation. (Phase 37.)
#  3. Bound exceptions at 0 for all three, so an exact zero is not flagged out
#     of bounds and every replicate fails. (Phase 37, the second half.)
#
# Plus the st0 = 0 support rule -- density 0 for rt <= t0 -- applied here in R,
# because EMC2's compiled copy will not apply it.
suppressPackageStartupMessages(library(EMC2))

.zb_cache <- new.env(parent = emptyenv())

ddmnn0_get <- function(name = "ddm_zerobox") {
  ptr <- .zb_cache[[name]]
  if (is.null(ptr) || !EMC2:::ddm_ptr_valid(ptr)) {
    path <- system.file("extdata", "flownn", paste0(name, ".rds"), package = "EMC2", mustWork = TRUE)
    b <- readRDS(path)
    members <- if (is.null(b$members)) list(b) else b$members
    for (m in members) {
      stopifnot(identical(m$context_names, c("v", "a", "t0", "s", "Z", "SZ", "sv", "st0")),
                identical(m$context_encoding, "zerobox"), !is.null(m$zerobox_c))
      stopifnot(all(m$bounds_sampled$lower[match(c("sv", "SZ", "st0"), m$context_names)] == 0))
    }
    .zb_cache[[paste0(name, "_c")]] <- members[[1]]$zerobox_c
    # the compiled copy standardises what it is given; we give it the network scale
    members <- lapply(members, function(m) { m$context_encoding <- "emc2"; m })
    ptr <- EMC2:::ddm_build_ensemble(members)
    .zb_cache[[name]] <- ptr
  }
  ptr
}

.theta_DDMnn0 <- function(pars, cc = .zb_cache[["ddm_zerobox_c"]]) {
  if (is.null(cc)) { ddmnn0_get(); cc <- .zb_cache[["ddm_zerobox_c"]] }
  cbind(pars[, "v"], log(pars[, "a"]), log(pars[, "t0"]), log(pars[, "s"]),
        qnorm(pars[, "Z"]),
        asinh(pars[, "SZ"] / cc$SZ),        # exact 0 -> exactly 0
        asinh(pars[, "sv"] / cc$sv),
        asinh(pars[, "st0"] / cc$st0))
}

dDDMnn0 <- function(rt, R, pars) {
  d <- EMC2:::ddm_ens_eval_trials_cpp(ddmnn0_get(), .theta_DDMnn0(pars), rt, as.integer(R))$pdf
  hard <- pars[, "st0"] == 0 & rt <= pars[, "t0"]       # exact support boundary
  d[hard] <- 0
  d
}
pDDMnn0 <- function(rt, R, pars) {
  p <- EMC2:::ddm_ens_eval_trials_cpp(ddmnn0_get(), .theta_DDMnn0(pars), rt, as.integer(R))$cdf
  p[pars[, "st0"] == 0 & rt <= pars[, "t0"]] <- 0
  p
}

DDMnn0 <- function() {
  m <- EMC2::DDMnn()
  m$p_types[c("sv", "SZ", "st0")] <- -Inf                 # exact zero when omitted
  ex <- m$bound$exception
  for (p in c("sv", "SZ", "st0")) { if (is.null(ex)) ex <- c(); ex[p] <- 0 }
  m$bound$exception <- ex
  stopifnot(all(is.infinite(m$p_types[c("sv", "SZ", "st0")])),
            all(m$bound$exception[c("sv", "SZ", "st0")] == 0))
  m$dfun <- function(rt, R, pars) dDDMnn0(rt, R, pars)
  m$pfun <- function(rt, R, pars) pDDMnn0(rt, R, pars)
  m
}

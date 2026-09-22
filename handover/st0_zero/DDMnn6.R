# DDMnn6: EMC2 model function for the decision-time (rt - t0) DDM flow.
#
# The flow is conditioned on SIX parameters and models d = rt - t0. The COMPILED
# port is dimension-generic but knows nothing about the shift, so the shift is
# applied here: strip t0 from the parameter matrix, subtract it from rt, and
# hand the compiled evaluator the six-column context and d. The Jacobian of
# rt -> rt - t0 is one, so no density correction is needed.
#
# rt at or below t0 is outside the support and gets zero density -- a true
# property of the model that the flow itself cannot represent, since a spline
# over log d has positive density everywhere.
suppressPackageStartupMessages(library(EMC2))

.flownn6_cache <- new.env(parent = emptyenv())

ddmnn6_get <- function(name = "ddm_dectime") {
  ptr <- .flownn6_cache[[name]]
  if (is.null(ptr) || !EMC2:::ddm_ptr_valid(ptr)) {
    fl <- readRDS(system.file("extdata", "flownn", paste0(name, ".rds"),
                              package = "EMC2", mustWork = TRUE))
    members <- if (is.null(fl$members)) list(fl) else fl$members
    for (m in members)
      stopifnot(identical(m$context_names, c("v", "a", "s", "Z", "SZ", "sv")),
                identical(m$context_encoding, "dectime"))
    ptr <- EMC2:::ddm_build_ensemble(members)
    .flownn6_cache[[name]] <- ptr
  }
  ptr
}

.theta_DDMnn6 <- function(pars)
  cbind(pars[, "v"], log(pars[, "a"]), log(pars[, "s"]),
        qnorm(pars[, "Z"]), qnorm(pars[, "SZ"]), log(pars[, "sv"]))

.eval6 <- function(rt, R, pars, what) {
  d <- rt - pars[, "t0"]
  ok <- is.finite(d) & d > 0
  out <- numeric(length(rt))
  if (any(ok)) {
    ev <- EMC2:::ddm_ens_eval_trials_cpp(ddmnn6_get(),
            .theta_DDMnn6(pars)[ok, , drop = FALSE], d[ok], as.integer(R)[ok])
    out[ok] <- ev[[what]]
  }
  out            # zero below the support, for both pdf and cdf
}

dDDMnn6 <- function(rt, R, pars) .eval6(rt, R, pars, "pdf")
pDDMnn6 <- function(rt, R, pars) .eval6(rt, R, pars, "cdf")

# EMC2's DDMnn is specified for the EIGHT-parameter training box, and two of its
# entries are wrong for an st0 = 0 build. Both were found on 2026-09-23, after
# they had silently invalidated the Phase 32/35 SBC cells:
#
#   p_types["st0"] = -2.996  (= 0.05 s), where DDM's is -Inf (= 0). Omitting st0
#     from a formula therefore SIMULATES at st0 = 0.05 while the 7-input flow is
#     EVALUATED at st0 = 0 -- a model mismatch, not a flow defect.
#   bound$minmax["st0"] = [0.01, 0.49] with no exception at 0, where DDM carries
#     exception = c(sv = 0, SZ = 0, st0 = 0). Without the exception, st0 = 0 is
#     flagged out of bounds and every replicate fails.
#
# Both are fixed here so the design behaves like DDM's for st0, and asserted.
.fix_st0_zero <- function(m) {
  m$p_types["st0"] <- -Inf
  ex <- m$bound$exception
  m$bound$exception <- if (is.null(ex)) c(st0 = 0) else { ex["st0"] <- 0; ex }
  stopifnot(is.infinite(m$p_types[["st0"]]), m$p_types[["st0"]] < 0,
            m$bound$exception[["st0"]] == 0)
  m
}

DDMnn6 <- function() {
  m <- .fix_st0_zero(EMC2::DDMnn())
  m$dfun <- function(rt, R, pars) dDDMnn6(rt, R, pars)
  m$pfun <- function(rt, R, pars) pDDMnn6(rt, R, pars)
  m
}

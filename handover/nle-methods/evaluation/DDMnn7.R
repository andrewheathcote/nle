# DDMnn7: the EMC2 model function for the st0 = 0 (7-parameter) DDM flow.
#
# Defined here rather than patched into EMC2, so it can be sourced alongside an
# unmodified dev-nn install.
#
# Why a separate model is needed at all. EMC2:::.theta_DDMnn hardcodes the
# 8-column context and ignores the export's context_names:
#
#   cbind(v, log(a), log(t0), log(s), qnorm(Z), qnorm(SZ), log(sv), log(st0))
#
# With st0 omitted from a design formula EMC2 sets it to its default of -Inf on
# the sampled scale, so exp() gives st0 = 0 exactly -- which is what we want for
# SIMULATION, but log(0) = -Inf is then fed to the flow as an 8th input. So the
# 8-column builder cannot be reused; only the theta builder and the bundle name
# change. Everything else is copied from EMC2::DDMnn verbatim.
#
# The compiled port itself is dimension-generic: ddm_build / ddm_eval_trials_cpp
# accept a 7-column bundle (verified). No C++ change is required.
suppressPackageStartupMessages(library(EMC2))

.flownn7_cache <- new.env(parent = emptyenv())

ddmnn7_get <- function(name = "ddm_st0zero") {
  ptr <- .flownn7_cache[[name]]
  if (is.null(ptr) || !EMC2:::ddm_ptr_valid(ptr)) {
    path <- system.file("extdata", "flownn", paste0(name, ".rds"),
                        package = "EMC2", mustWork = TRUE)
    fl <- readRDS(path)
    members <- if (is.null(fl$members)) list(fl) else fl$members
    # fail loudly on a context mismatch rather than silently mis-wiring
    for (m in members)
      stopifnot(identical(m$context_names,
                          c("v", "a", "t0", "s", "Z", "SZ", "sv")))
    ptr <- EMC2:::ddm_build_ensemble(members)
    .flownn7_cache[[name]] <- ptr
  }
  ptr
}

.theta_DDMnn7 <- function(pars)
  cbind(pars[, "v"], log(pars[, "a"]), log(pars[, "t0"]), log(pars[, "s"]),
        qnorm(pars[, "Z"]), qnorm(pars[, "SZ"]), log(pars[, "sv"]))

dDDMnn7 <- function(rt, R, pars)
  EMC2:::ddm_ens_eval_trials_cpp(ddmnn7_get(), .theta_DDMnn7(pars), rt,
                                 as.integer(R))$pdf
pDDMnn7 <- function(rt, R, pars)
  EMC2:::ddm_ens_eval_trials_cpp(ddmnn7_get(), .theta_DDMnn7(pars), rt,
                                 as.integer(R))$cdf

DDMnn7 <- function() {
  m <- EMC2::DDMnn()
  m$dfun <- function(rt, R, pars) dDDMnn7(rt, R, pars)
  m$pfun <- function(rt, R, pars) pDDMnn7(rt, R, pars)
  m
}

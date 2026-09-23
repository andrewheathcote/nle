# DDMreg: the EMC2 model function for the DIRECT-REGRESSION baseline.
#
# A plain MLP maps (standardised theta, standardised log rt, R) to the JOINT
# log-density. No flow, no classifier, and -- unlike a flow -- no normalisation
# guarantee: total mass is ~1.000 but not exactly, and what matters for
# inference is how it VARIES with theta (measured sd(log Z) ~ 0.002, an order of
# magnitude below the score biases being compared).
#
# The evaluation is identical to `evaluation/score_bias_regress.R`'s `reg_logL`
# -- same scaler, same GELU, same column order -- and is asserted against it in
# `sbc/nci/SBCddmreg.R` before any cell runs.
#
# theta is the 8-vector on the SAMPLED scale, in the order the net was trained
# on: (v, log a, log t0, log s, qnorm Z, qnorm SZ, log sv, log st0), with log rt
# appended and the raw response (1 = lower, 2 = upper) appended AFTER scaling.
suppressPackageStartupMessages(library(EMC2))

.regnet_cache <- new.env(parent = emptyenv())

regnet_get <- function(path = Sys.getenv("REG_JSON",
                         "results/model_states/reg_s402_budget/regress.json")) {
  net <- .regnet_cache[[path]]
  if (is.null(net)) {
    stopifnot(file.exists(path))
    net <- jsonlite::fromJSON(path, simplifyVector = TRUE)
    stopifnot(length(net$scaler_mean) == 9L, length(net$scaler_scale) == 9L,
              length(net$layers$W) == length(net$layers$b))
    .regnet_cache[[path]] <- net
  }
  net
}

.gelu_reg <- function(x) 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

.theta_DDMreg <- function(pars)
  cbind(pars[, "v"], log(pars[, "a"]), log(pars[, "t0"]), log(pars[, "s"]),
        qnorm(pars[, "Z"]), qnorm(pars[, "SZ"]), log(pars[, "sv"]),
        log(pars[, "st0"]))

# log joint density, one row per trial (pars may vary by row)
reg_log_pdf <- function(rt, R, pars, net = regnet_get()) {
  raw <- cbind(.theta_DDMreg(pars), log(rt))
  z <- sweep(sweep(raw, 2, net$scaler_mean, "-"), 2, net$scaler_scale, "/")
  x <- cbind(z, as.numeric(R))
  nl <- length(net$layers$W)
  for (i in seq_len(nl)) {
    x <- sweep(x %*% net$layers$W[[i]], 2, net$layers$b[[i]], "+")
    if (i < nl) x <- .gelu_reg(x)
  }
  as.numeric(x)
}

dDDMreg <- function(rt, R, pars) exp(reg_log_pdf(rt, R, pars))

DDMreg <- function() {
  m <- EMC2::DDMnn()          # 8-parameter box: p_types, bounds and rfun as-is
  m$dfun <- function(rt, R, pars) dDDMreg(rt, R, pars)
  # A regression net gives no CDF. Fail loudly rather than silently falling
  # through to whatever flow happens to be installed.
  m$pfun <- function(rt, R, pars)
    stop("DDMreg has no CDF: the regression baseline models the density only")
  m
}

# Does EMC2's likelihood path, as SBC ran it, agree with the R port AS A
# FUNCTION OF THETA?
#
# Phase 37: importance-reweighting through the R port predicts SBC std bias of
# ~0.2 on t0 for the st0 = 0 flow, while the EMC2 SBC gave +3.144 (coverage 0).
# The shipped flow, which runs through EMC2's built-in DDMnn, agrees with its
# SBC. Phase 32b compared the two paths only at matched theta via dDDMnn7; this
# compares the full chain the sampler uses -- design, p_vector mapping,
# transforms, dadm, log_likelihood with its min_ll floor -- against the port,
# over many theta around the truth, on EMC2-simulated data.
#
#   Rscript sbc/test_emc2_vs_port.R [n_theta]
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(WienR)})
source("handover/st0_zero/DDMnn7.R")
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")
a <- commandArgs(trailingOnly = TRUE)
NT <- if (length(a)) as.integer(a[1]) else 40L

# Read the bundle of record from the package rather than the EMC2 install: the
# laptop install was found (2026-09-22) holding a stale triage flow, z0_s501.
BUNDLE <- "handover/nle-methods/artefacts/ddm_st0zero.rds"
ddmnn7_get <- function(name = "ddm_st0zero") {
  ptr <- .flownn7_cache[[name]]
  if (is.null(ptr) || !EMC2:::ddm_ptr_valid(ptr)) {
    b <- readRDS(BUNDLE); ms <- if (is.null(b$members)) list(b) else b$members
    ptr <- EMC2:::ddm_build_ensemble(ms); .flownn7_cache[[name]] <- ptr }
  ptr }
b <- readRDS(BUNDLE); m <- if (is.null(b$members)) b else b$members[[1]]
cat(sprintf("bundle %s: budget=%s step=%s\n", BUNDLE, m$budget,
            as.character(m$checkpoint_step_flow)))
fl <- ddm_load("port/export/ddm_z0_s503_e282_weights.json")
cat(sprintf("port export z0_s503_e282: budget=%s step=%s\n", fl$budget,
            as.character(fl$checkpoint_step_flow)))

mk <- function(model) design(factors = list(subjects = 1, S = c("left", "right")),
  Rlevels = c("left", "right"),
  formula = list(v ~ 1, a ~ 1, t0 ~ 1, s ~ 1, Z ~ 1, SZ ~ 1, sv ~ 1),
  model = model, use_matchfun = FALSE, constants = c(s = log(1)), verbose = FALSE)
des_f <- mk(DDMnn7); des_a <- mk(DDM)
sp <- names(sampled_pars(des_f))
PM <- c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5), sv = log(.1), SZ = qnorm(.2))[sp]
PS <- c(v = .15, a = .15, t0 = .1, Z = .05, sv = .15, SZ = .15)[sp]

port_ll <- function(x, rt, R, f) {
  th <- c(v = x[["v"]], a = x[["a"]], t0 = x[["t0"]], s = 0, Z = x[["Z"]],
          SZ = x[["SZ"]], sv = x[["sv"]])
  sum(sapply(1:2, function(r) { k <- R == r; if (!any(k)) return(0)
    sum(f(th, rt[k], r)) })) }
fl_l <- function(th, rt, R) ddm_eval(fl, th, rt, R)$log_pdf
an_l <- function(th, rt, R) { w <- pnorm(th[["Z"]])
  WienR::dWDM(rt, response = ifelse(R == 1, "lower", "upper"), a = exp(th[["a"]]),
    v = th[["v"]], w = w, t0 = exp(th[["t0"]]), sv = exp(th[["sv"]]),
    sw = 2 * pnorm(th[["SZ"]]) * min(w, 1 - w), precision = 1e-8)$logvalue }

emc_ll <- function(des, dadm, x) {
  mod <- attr(dadm, "model"); if (is.function(mod)) mod <- mod()
  EMC2:::calc_ll_R(x[sp], mod, dadm) }

set.seed(11)
out <- NULL
for (d in 1:3) {
  truth <- rnorm(length(sp), PM, PS); names(truth) <- sp
  dat <- make_data(truth, des_f, n_trials = 200)
  rt <- dat$rt; R <- as.integer(dat$R)
  dadm_f <- EMC2:::design_model(dat, des_f, compress = FALSE, rt_resolution = 1e-12, verbose = FALSE)
  dadm_a <- EMC2:::design_model(dat, des_a, compress = FALSE, rt_resolution = 1e-12, verbose = FALSE)
  for (i in 0:NT) {
    x <- if (i == 0) truth else truth + PS * rnorm(length(sp)) *
                                  c(v = 1, a = 1, t0 = .3, Z = 1, sv = 1, SZ = 1)[sp]
    out <- rbind(out, data.frame(d = d, i = i, t(x),
      emc_f = emc_ll(des_f, dadm_f, x), port_f = port_ll(x, rt, R, fl_l),
      emc_a = emc_ll(des_a, dadm_a, x), port_a = port_ll(x, rt, R, an_l)))
  }
}
out$df <- out$emc_f - out$port_f; out$da <- out$emc_a - out$port_a
fin <- is.finite(out$df) & is.finite(out$da)
cat(sprintf("\nfinite rows %d / %d\n", sum(fin), nrow(out)))
for (d in 1:3) { o <- out[fin & out$d == d, ]
  cat(sprintf("dataset %d  FLOW emc-port: mean %+.4f  sd %.2e  range %.2e | ANALYTIC: mean %+.4f sd %.2e\n",
      d, mean(o$df), sd(o$df), diff(range(o$df)), mean(o$da), sd(o$da))) }
cat("\ncorrelation of the flow discrepancy with each parameter:\n")
print(round(sapply(sp, function(p) cor(out[fin, p], out$df[fin])), 3))
saveRDS(out, "sbc/emc2_vs_port.rds")

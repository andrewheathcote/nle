# Acceptance test for the zero-box sampler and its labels. One line per check.
#
#   Rscript simulation/test_zerobox.R
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(WienR)})
source("simulation/sim_functions.R"); source("simulation/designs.R"); source("simulation/zerobox.R")
ok <- function(cond, lab, ...) { cat(sprintf("%s  %s\n", if (cond) "PASS" else "FAIL", sprintf(lab, ...))); invisible(cond) }

n <- 20000L
V <- ZB_VERSION_DEFAULT; REG <- zb_regimes(V)
cat(sprintf("sampler version %d (ZB_VERSION)\n", V))
P <- param_matrix_zerobox(n, param_namesDDM, lowerDDM, upperDDM, transformsDDM, seed = 1, version = V)
reg <- attr(P, "regime")
all_ok <- TRUE

# 1. strata shares
tab <- table(reg)[REG$regime] / n
all_ok <- ok(max(abs(tab - REG$share)) < 2e-3, "strata shares within 0.2%%: worst |diff| %.4f", max(abs(tab - REG$share))) && all_ok

# 2. zeros exactly where the regime says, and nowhere else
sv0 <- is.infinite(P[, "sv"]); SZ0 <- is.infinite(P[, "SZ"]); st00 <- is.infinite(P[, "st0"])
want <- REG[match(reg, REG$regime), ]
all_ok <- ok(all(sv0 == want$sv0) && all(SZ0 == want$SZ0) && all(st00 == want$st00),
             "exact zeros (-Inf) match regime labels on all %d draws", n) && all_ok

# 3. free values inside the admissible set
sv <- exp(P[!sv0, "sv"]); SZ <- pnorm(P[!SZ0, "SZ"]); st0 <- exp(P[!st00, "st0"])
all_ok <- ok(min(sv) >= 0.01 && max(sv) <= 4 && min(SZ) >= 0.05 && max(SZ) <= 0.99 && min(st0) >= 0 && max(st0) <= 0.49,
             "free ranges: sv [%.3f, %.2f]  SZ [%.3f, %.2f]  st0 [%.4f, %.2f]", min(sv), max(sv), min(SZ), max(SZ), min(st0), max(st0)) && all_ok
all_ok <- ok(mean(st0 < 0.01) > if (V == 1L) 0.15 else 0.05, "st0 gap (0, 0.01) is sampled: %.1f%% of free st0", 100 * mean(st0 < 0.01)) && all_ok
if (V >= 2L) {  # the point of version 2: density where the SBC prior lives, as a share of ALL rows
  in_region <- mean(!st00 & exp(P[, "st0"]) >= 0.03 & exp(P[, "st0"]) <= 0.10)
  all_ok <- ok(in_region > 0.18, "st0 in [0.03, 0.10]: %.1f%% of all rows (v1 12%%, shipped flow 31%%)", 100 * in_region) && all_ok
}

# 4. the network context: finite everywhere, exactly 0 at zeros, monotone in x
ctx <- zerobox_context(P)
all_ok <- ok(all(is.finite(ctx)), "context finite on all draws (no -Inf survives)") && all_ok
all_ok <- ok(all(ctx[sv0, "sv"] == 0) && all(ctx[SZ0, "SZ"] == 0) && all(ctx[st00, "st0"] == 0), "context is exactly 0 at exact zeros") && all_ok
all_ok <- ok(max(abs(zb_sinh(zb_asinh(c(0, 1e-4, .01, .3), "st0"), "st0") - c(0, 1e-4, .01, .3))) < 1e-12, "asinh/sinh round-trip exact") && all_ok

# 5. labels: the pipeline's stored ll against WienR 1e-8, one draw per regime
get_pm <- if (exists("get_pars_matrix", asNamespace("EMC2"))) EMC2:::get_pars_matrix else EMC2:::get_pars_matrix_oo
worst <- 0
for (r in ZB_REGIMES$regime) {
  i <- which(reg == r)[1]; pv <- P[i, ]
  set.seed(i); dat <- EMC2::make_data(pv, designDDM, n_trials = 100)
  dadm <- EMC2:::design_model(dat, designDDM, rt_resolution = 1e-12, verbose = FALSE)
  pars <- get_pm(pv, dadm, model = attr(dadm, "model")())
  ll <- log(EMC2:::dDDM(dadm$rt, dadm$R, pars, precision = 5e-10))
  # the DDM depends on (v, a, sv) only through v/s, a/s, sv/s -- the s-scaling
  # trap of Phase 7; the reference must apply it or every draw with s != 1 fails
  w <- pnorm(pv[["Z"]]); SZn <- pnorm(pv[["SZ"]]); SZn[!is.finite(pv[["SZ"]])] <- 0
  sc <- exp(pv[["s"]])
  ref <- WienR::dWDM(dadm$rt, response = as.integer(dadm$R), a = exp(pv[["a"]]) / sc, v = pv[["v"]] / sc,
                     w = w, t0 = exp(pv[["t0"]]), sv = exp(pv[["sv"]]) / sc, sw = 2 * SZn * min(w, 1 - w),
                     st0 = exp(pv[["st0"]]), precision = 1e-8)$logvalue
  worst <- max(worst, max(abs(ll - ref)[is.finite(ll) & is.finite(ref)]))
}
all_ok <- ok(worst < 1e-6, "stored ll vs WienR 1e-8, one draw per regime: worst |diff| %.1e", worst) && all_ok

cat(if (all_ok) "\nALL PASS\n" else "\nFAILURES ABOVE\n")
quit(status = if (all_ok) 0 else 1)

# Zero-box registration acceptance on dev-nle, from the NLE side: dev-nle's
# dfun on the fixture card == this project's R port, in every zero regime,
# including the st0 = 0 support boundary; then p_types / exceptions / a tiny
# SBC cell with exact zeros as defaults and constants.
#   Rscript sbc/gate_zerobox_dev_nle.R [library] [card]
a <- commandArgs(trailingOnly = TRUE)
LIB <- path.expand(if (length(a)) a[1] else "~/R/nle-sbc-lib")
CARD <- path.expand(if (length(a) > 1) a[2] else "~/R/nle-cards/ddm_zerobox_fixture.json")
.libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(jsonlite)})
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")
npass <- 0L; nfail <- 0L
verdict <- function(ok, lab) { cat(sprintf("%s  %s\n", if (ok) "PASS" else "FAIL", lab)); if (ok) npass <<- npass + 1L else nfail <<- nfail + 1L; ok }
cat(sprintf("=== zero-box acceptance  EMC2 %s  card %s ===\n", as.character(packageVersion("EMC2")), basename(CARD)))

## 1. registers, with the right defaults and exceptions
mk <- register_nn_model(CARD, twin = DDM); m <- mk()
cat("  p_types:", paste(names(m$p_types), signif(m$p_types, 3), collapse = " "), "\n")
cat("  exceptions:", paste(names(m$bound$exception), m$bound$exception, collapse = " "), "\n")
verdict(all(is.infinite(m$p_types[c("sv", "SZ", "st0")])) && all(m$bound$exception[c("sv", "SZ", "st0")] == 0),
        "1. registers; p_types -Inf and bound exceptions 0 for sv, SZ, st0")

## 2. exact: dfun == R port in every regime, -Inf where the port has -Inf
fl <- ddm_load(CARD); set.seed(3); n <- 60
regimes <- list(full = c(TRUE, TRUE, TRUE), sv0 = c(FALSE, TRUE, TRUE), SZ0 = c(TRUE, FALSE, TRUE),
                st00 = c(TRUE, TRUE, FALSE), all0 = c(FALSE, FALSE, FALSE))
worst <- setNames(numeric(length(regimes)), names(regimes)); infok <- TRUE; ninf <- 0L
for (rg in names(regimes)) for (k in 1:15) {
  pos <- regimes[[rg]]
  t0 <- runif(1, .2, .4)
  rt <- c(runif(n / 2, t0 - .05, t0 + .05), runif(n / 2, t0 + .05, 1.6)); R <- sample(1:2, n, TRUE)  # both sides of t0
  th <- c(v = rnorm(1, 1, .3), a = log(runif(1, .5, 1.5)), t0 = log(t0), s = 0, Z = qnorm(runif(1, .4, .6)),
          SZ = if (pos[2]) qnorm(runif(1, .1, .4)) else -Inf, sv = if (pos[1]) log(runif(1, .05, .5)) else -Inf,
          st0 = if (pos[3]) log(runif(1, .02, .1)) else -Inf)
  nat <- cbind(v = th[["v"]], a = exp(th[["a"]]), t0 = exp(th[["t0"]]), s = 1, Z = pnorm(th[["Z"]]),
               SZ = if (pos[2]) pnorm(th[["SZ"]]) else 0, sv = exp(th[["sv"]]), st0 = exp(th[["st0"]]))[rep(1, n), ]
  d_new <- log(m$dfun(rt, R, nat)); d_ref <- ddm_eval(fl, th, rt, R)$log_pdf
  fin <- is.finite(d_ref)
  ninf <- ninf + sum(!fin)
  infok <- infok && all(is.infinite(d_new[!fin]) & d_new[!fin] < 0) && all(is.finite(d_new[fin]))
  worst[rg] <- max(worst[rg], if (any(fin)) max(abs(d_new[fin] - d_ref[fin])) else 0)
}
cat(sprintf("  max |diff| on finite: %s; -Inf agreed at %d support-boundary points\n",
            paste(sprintf("%s %.1e", names(worst), worst), collapse = "  "), ninf))
verdict(all(worst < 1e-7) && infok && ninf > 0, "2. dfun == R port in full/sv0/SZ0/st00/all0 regimes; -Inf matches for rt <= t0 at st0 = 0")

## 3. plumbing: exact zeros as omitted defaults and as a named constant survive design/make_data/SBC
ok3 <- tryCatch({
  cell <- nn_cell(mk, mean = c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5), SZ = qnorm(.2)),
                  sd = c(v = .15, a = .15, t0 = .1, Z = .05, SZ = .15),
                  formula = list(v ~ 1, a ~ 1, t0 ~ 1, Z ~ 1, SZ ~ 1), constants = c(sv = -Inf),
                  factors = list(S = c("left", "right")), Rlevels = c("left", "right"))
  out <- nn_sbc_cell(cell, trials = 100, replicates = 2, run_control = FALSE, cores_per_chain = 4, cores_for_chains = 2)
  print(out$summary)
  mp <- mapped_pars(cell$design, cell$mean)              # natural scale, per design cell
  cat("  mapped_pars sv/st0:", unique(mp[, "sv"]), unique(mp[, "st0"]), "\n")
  dat <- make_data(cell$mean, cell$design, n_trials = 500)  # the simulator uses the exact zeros too:
  cat("  simulated min rt - t0:", signif(min(dat$rt) - exp(cell$mean[["t0"]]), 3), "(no st0 smear below t0)\n")
  all(mp[, "sv"] == 0) && all(mp[, "st0"] == 0) && min(dat$rt) > exp(cell$mean[["t0"]])
}, error = function(e) { cat("  ", conditionMessage(e), "\n"); FALSE })
verdict(ok3, "3. nn_sbc_cell() with st0 omitted and sv = -Inf constant runs; mapped_pars exact zeros")
cat(sprintf("\n=== %d passed, %d failed ===\n", npass, nfail)); quit(status = nfail)

# Fit ONE simulated dataset with the flow and with the analytic, same data,
# same prior, same sampler -- and compare the t0 posteriors.
#
# Phase 33 left +8 ms of the st0=0 SBC bias unexplained: neither a profile nor a
# joint MLE (even multi-start) reproduces +12.6 ms. This asks the sampler
# directly, which is the object SBC actually measures.
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(EMC2))
source("handover/st0_zero/DDMnn7.R")
CORES <- as.integer(Sys.getenv("SAMP_CORES", "9"))
N <- as.integer(Sys.getenv("NTRIAL", "400"))
SEED <- as.integer(Sys.getenv("SEED", "11"))

mk <- function(model) design(
  factors = list(subjects = 1, S = c("left", "right")),
  Rlevels = c("left", "right"),
  formula = list(v ~ 1, a ~ 1, t0 ~ 1, s ~ 1, Z ~ 1, SZ ~ 1, sv ~ 1),
  model = model, use_matchfun = FALSE,
  constants = c(s = log(1)), verbose = FALSE)

PM <- c(v=1, a=log(.8), t0=log(.3), Z=qnorm(.5), sv=log(.1), SZ=qnorm(.2))
PS <- c(v=.15, a=.15, t0=.1, Z=.05, sv=.15, SZ=.15)

des_an <- mk(DDM); des_fl <- mk(DDMnn7)
sp <- names(sampled_pars(des_an)); pmean <- PM[sp]; psd <- PS[sp]
stopifnot(!any(is.na(pmean)))
cat("sampled:", paste(sp, collapse=", "), "\n")

set.seed(SEED)
true_p <- rnorm(length(sp), pmean, psd); names(true_p) <- sp
cat("truth  :", paste(sprintf("%s=%.4f", sp, true_p), collapse=" "), "\n")
cat(sprintf("true t0 = %.4f s\n\n", exp(true_p[["t0"]])))

# ONE dataset, simulated from the ANALYTIC, used by both fits
dat <- make_data(true_p, design = des_an, n_trials = N / 2)
cat(sprintf("data: %d trials, rt range [%.3f, %.3f], min rt = %.4f\n\n",
            nrow(dat), min(dat$rt), max(dat$rt), min(dat$rt)))

run <- function(des, lab) {
  pr <- prior(des, type = "single", pmean = pmean, psd = psd)
  emc <- make_emc(dat, des, type = "single", prior_list = pr, n_chains = 3)
  t0 <- proc.time()[3]
  emc <- fit(emc, cores_per_chain = max(1, CORES %/% 3), verbose = FALSE)
  cat(sprintf("%s fitted in %.1f min\n", lab, (proc.time()[3] - t0) / 60))
  emc
}
e_an <- run(des_an, "analytic")
e_fl <- run(des_fl, "flow    ")

summ <- function(emc, lab) {
  # get_pars returns a NAMED LIST of parameters, each holding one mcmc object
  # (iterations x 1). Flatten to an iterations x parameters matrix.
  g <- get_pars(emc, selection = "alpha", merge_chains = TRUE)
  s <- sapply(g[sp], function(p) as.numeric(if (is.list(p)) p[[1]] else p))
  colnames(s) <- sp
  stopifnot(all(sp %in% colnames(s)))
  cat(sprintf("\n%s\n", lab))
  cat(sprintf("  %-5s %10s %10s %10s %10s\n", "par", "truth", "post mean", "post sd", "(mean-t)/sd"))
  for (p in sp) {
    v <- s[, p]
    cat(sprintf("  %-5s %10.4f %10.4f %10.4f %10.2f\n", p, true_p[[p]],
                mean(v), sd(v), (mean(v) - true_p[[p]]) / sd(v)))
  }
  s
}
sa <- summ(e_an, "ANALYTIC"); sf <- summ(e_fl, "FLOW")
cat(sprintf("\nt0 in ms at the truth: analytic %+.2f   flow %+.2f   excess %+.2f\n",
            (mean(sa[,"t0"]) - true_p[["t0"]]) * exp(true_p[["t0"]]) * 1000,
            (mean(sf[,"t0"]) - true_p[["t0"]]) * exp(true_p[["t0"]]) * 1000,
            (mean(sf[,"t0"]) - mean(sa[,"t0"])) * exp(true_p[["t0"]]) * 1000))
saveRDS(list(truth = true_p, analytic = sa, flow = sf, data = dat, emc_an = e_an, emc_fl = e_fl),
        "sbc/sampler_posterior_check.rds")

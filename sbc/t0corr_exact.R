# Exact test of the cap256w_c4 t0 correction (t0_hat * exp(-0.00276)) in the eight-case review's
# case 8 cell (full regime), on the final sampler (2026-10-13).
#
# run_sbc() keeps ranks, medians and bias but not posterior draws, so every corrected rank so far
# came from a normal-posterior approximation, and the two approximations tried disagree in sign.
# This driver repeats run_SBC_subject() step for step (same make_data / make_emc / fit calls, the
# same defaults SBC_single() adds, the same ESS-thinned rank) and additionally saves each
# replicate's draws, so the corrected rank is the rank of the truth among (draws - delta) on the
# sampled (log t0) scale: exact, no approximation. The EMC2 build is untouched.
#
# Built-in check: the uncorrected ranks must reproduce run_sbc's failure in this cell
# (results_8case_N800/full: t0 KS p 1e-5, mean rank ~0.43).
# Pass rule, declared before running: corrected t0 ranks inside the 95% envelope AND KS p > .05.
#
#   Rscript sbc/t0corr_exact.R <trials> [replicates]       (run; resumable)
#   Rscript sbc/t0corr_exact.R <trials> analyse            (ranks from the saved draws)
# Env: DELTA (default 0.00276), LIB (dev-nle library), CPC (replicates in parallel, default 2), CFC (cores for chains, default 3),
#      OUT (default sbc/nci/results_t0corr)
a <- commandArgs(trailingOnly = TRUE)
TRIALS <- as.integer(a[1]); MODE <- if (length(a) > 1 && a[2] == "analyse") "analyse" else "run"
REPS <- if (MODE == "run" && length(a) > 1) as.integer(a[2]) else 500L
LIB <- path.expand(Sys.getenv("LIB", "~/R/nle-lib"))
.libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages(library(EMC2))
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
CPC <- as.integer(Sys.getenv("CPC", "2")); CFC <- as.integer(Sys.getenv("CFC", "3"))
OUT <- file.path(Sys.getenv("OUT", "sbc/nci/results_t0corr"), paste0("full_N", TRIALS))
DRAWS <- file.path(OUT, "draws"); dir.create(DRAWS, recursive = TRUE, showWarnings = FALSE)
DELTA <- as.numeric(Sys.getenv("DELTA", "0.00276"))   # fixed in advance, never fitted in the run it tests
# 0.00276: METHODS.md 3.5 (failed at N=800, 2026-10-13). 0.0019: the rank-centring delta from that N=800
# run, tested out of sample at N=1600 (fresh prior draws, seed 20261013 + trials).

if (MODE == "run") {
  cat(sprintf("=== t0 correction, exact: case 8 full c4  N=%d reps=%d  cores %dx%d  EMC2 %s  %s ===\n", TRIALS, REPS,
              CPC, CFC, as.character(packageVersion("EMC2")), format(Sys.time(), "%Y-%m-%d %H:%M")))
  # Cell exactly as zerobox_s4_cell_8case.R builds the full regime.
  PM <- c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5), sv = log(.1), SZ = qnorm(.2), st0 = log(.05))
  PS <- c(v = .15, a = .15, t0 = .1, Z = .05, sv = .15, SZ = .15, st0 = .15)
  mk <- register_nn_model("port/export/ddm_cap256w_c4_weights.json", twin = DDM)
  cell <- nn_cell(mk, mean = PM, sd = PS, formula = lapply(names(PM), function(p) as.formula(paste(p, "~ 1"))),
                  factors = list(S = c("left", "right")), Rlevels = c("left", "right"))
  mp <- mapped_pars(cell$design, cell$mean); stopifnot(all(mp[, c("sv", "SZ", "st0")] > 0))
  EMC2:::nn_sbc_box_check(cell$design, cell$prior)
  design_in <- cell$design; prior_in <- cell$prior
  # The defaults SBC_single() adds, and the cell's own arguments.
  dots <- EMC2:::add_defaults(list(cores_per_chain = CPC, cores_for_chains = CFC),
                              max_tries = 50, compress = FALSE, rt_resolution = 1e-12,
                              stop_criteria = list(min_es = 100, max_gd = 1.1, selection = c("alpha", "mu", "Sigma")),
                              cores_per_chain = 1)
  dots[["verbose"]] <- FALSE; dots[["verboseProgress"]] <- FALSE
  pf <- file.path(OUT, "prior_alpha.rds")
  if (!file.exists(pf)) { set.seed(20261013 + TRIALS); saveRDS(parameters(prior_in, N = REPS, selection = "alpha"), pf) }
  prior_alpha <- readRDS(pf)
  todo <- Filter(function(i) !file.exists(file.path(DRAWS, sprintf("rep_%03d.rds", i))), seq_len(nrow(prior_alpha)))
  cat(sprintf("%d of %d replicates to run\n", length(todo), nrow(prior_alpha)))
  one <- function(rep) {
    p_vector <- prior_alpha[rep, ]
    data <- tryCatch(do.call(make_data, c(list(parameters = p_vector, design = design_in, n_trials = TRIALS),
                                          EMC2:::fix_dots(dots, make_data))), error = function(e) e)
    if (!is.data.frame(data)) return(message("rep ", rep, ": make_data failed"))
    emc <- suppressMessages(do.call(make_emc, c(list(data = data, design = design_in, prior_list = prior_in, type = "single"),
                                                EMC2:::fix_dots(dots, make_emc))))
    emc <- tryCatch(do.call(fit, c(list(emc = emc), EMC2:::fix_dots(dots, fit))),
                    warning = function(w) { message("rep ", rep, " warning: ", conditionMessage(w)); NULL },
                    error = function(e) { message("rep ", rep, " error: ", conditionMessage(e)); NULL })
    if (is.null(emc)) return(invisible(NULL))   # dropped, as run_sbc drops it
    ESS <- ess_summary(emc, stat = NULL)[[1]]
    alpha_rec <- get_pars(emc, selection = "alpha", return_mcmc = FALSE, merge_chains = TRUE, flatten = TRUE)[, 1, ]
    CI <- credint(emc)[[1]]
    saveRDS(list(truth = p_vector, ESS = ESS, draws = alpha_rec, CI = CI),
            file.path(DRAWS, sprintf("rep_%03d.rds", rep)))
    message("done rep ", rep, "  ", format(Sys.time(), "%H:%M"))
  }
  invisible(EMC2:::auto_mclapply(todo, one, mc.cores = CPC))
  cat(sprintf("=== run done %s ===\n", format(Sys.time(), "%Y-%m-%d %H:%M")))
}

# Analysis: ranks exactly as get_ranks_ESS(), uncorrected and with the draws shifted by -delta.
fs <- list.files(DRAWS, "^rep_.*\\.rds$", full.names = TRUE)
R <- lapply(fs, readRDS)
rk <- function(x, p, d = 0) EMC2:::get_ranks_ESS(x$draws[p, ] - d, x$ESS[[p]], x$truth[[p]])
ks <- function(r) suppressWarnings(ks.test(r, "punif")$p.value)
env <- function(r) { lims <- EMC2:::get_lims(length(r), 500, EMC2:::get_gamma(length(r), 500))
  y <- ecdf(r)(lims$z) - lims$z; all(y >= lims$lower - 1e-12 & y <= lims$upper + 1e-12) }
pars <- rownames(R[[1]]$draws)
tab <- do.call(rbind, lapply(pars, function(p) {
  b <- vapply(R, function(x) median(x$draws[p, ]) - x$truth[[p]], 0)
  dp <- if (p == "t0") DELTA else 0
  r0 <- vapply(R, rk, 0, p = p); r1 <- vapply(R, rk, 0, p = p, d = dp)
  data.frame(parameter = p, n = length(R), mean_bias = mean(b), std_bias = mean(b) / sd(b),
             mean_rank = mean(r0), ks_p = ks(r0), in_envelope = env(r0),
             delta = dp, corr_mean_rank = mean(r1), corr_ks_p = ks(r1), corr_in_envelope = env(r1))
}))
cat(sprintf("\n=== N=%d: %d replicates with draws ===\n", TRIALS, length(R)))
print(tab, digits = 3, row.names = FALSE)
# The delta that would centre t0 here, from the draws (for a re-estimate if the fixed one fails).
cat(sprintf("t0: mean (median - truth) %+.5f; fixed delta %.5f\n", tab$mean_bias[tab$parameter == "t0"], DELTA))
write.csv(tab, file.path(OUT, "summary.csv"), row.names = FALSE)
pass <- with(tab[tab$parameter == "t0", ], corr_in_envelope && corr_ks_p > .05)
# Descriptive only (not the test): t0 rank summaries over a delta grid.
for (d in c(0, .001, .0015, .0019, .002, .0025, .00276)) { r <- vapply(R, rk, 0, p = "t0", d = d)
  cat(sprintf("  delta %.5f: mean rank %.3f  KS p %.3g  envelope %s\n", d, mean(r), ks(r), env(r))) }
cat(sprintf("PASS RULE (corrected t0 in envelope and KS p > .05): %s\n", if (pass) "PASS" else "FAIL"))

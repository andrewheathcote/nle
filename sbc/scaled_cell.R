# SBC cell for the scaled DDM network (handover/SCALED_DDM_PLAN.md): one of the eight-case review's
# cells with the likelihood replaced by DDMsc (handover/nle-methods/port/R/ddm_scaled.R): a network
# over the unit-scale decision time given (kappa, eta, Z, SZ), with t0, st0, a and s exact.
#
# The cell (prior, design, pins, SBC settings) is built exactly as sbc/zerobox_s4_cell_8case.R
# builds it, from a registered zero-box card (v1) that only supplies nn_cell's scaffolding; the
# design's prepared model is then wrapped so that its likelihood is DDMsc's and its bounds are
# DDM()'s (as sbc/st0conv_cell.R did for S2). Network-only (run_control = FALSE): the matched
# analytic controls exist in sbc/nci/results_8case/<regime> (same prior, design and N).
#
#   Rscript sbc/scaled_cell.R <regime> <card> [replicates] [trials]
#     regime: sv0 (case 2) | SZ0 (3) | st00 (4) | st0only (5) | SZonly (7) | full (8) | all0
# Env: LIB (dev-nle library, default ~/R/nle-lib), CPC (default 2), CFC (default 3), ORACLE=1,
#      OUT (default sbc/nci/results_scaled), BENCH=1 (checks and speed only), PKG (the package
#      root holding port/R/ddm_scaled.R, default handover/nle-methods)
a <- if (exists("ARGS")) ARGS else commandArgs(trailingOnly = TRUE)
REGIME <- a[1]; CARD <- normalizePath(a[2], mustWork = TRUE)
REPS <- if (length(a) > 2) as.integer(a[3]) else 500L
TRIALS <- if (length(a) > 3) as.integer(a[4]) else 400L
LIB <- path.expand(Sys.getenv("LIB", "~/R/nle-lib"))
.libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages(library(EMC2))
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
source(file.path(Sys.getenv("PKG", "handover/nle-methods"), "port/R/ddm_scaled.R"))
CPC <- as.integer(Sys.getenv("CPC", "2")); CFC <- as.integer(Sys.getenv("CFC", "3"))
OUT <- file.path(Sys.getenv("OUT", "sbc/nci/results_scaled"), sub("_weights\\.json$", "", basename(CARD)), REGIME)
SCAFFOLD <- Sys.getenv("SCAFFOLD", "port/export/ddm_zb_s903_e271_weights.json")
if (!file.exists(SCAFFOLD)) SCAFFOLD <- file.path(Sys.getenv("PKG", "handover/nle-methods"), SCAFFOLD)
PIN <- switch(REGIME, full = c(), sv0 = c(sv = -Inf), SZ0 = c(SZ = -Inf), st00 = c(st0 = -Inf),
              all0 = c(sv = -Inf, SZ = -Inf, st0 = -Inf),
              st0only = c(sv = -Inf, SZ = -Inf), SZonly = c(sv = -Inf, st0 = -Inf), stop("unknown regime ", REGIME))
cat(sprintf("=== scaled-DDM cell %s  card %s  N=%d reps=%d  cores %dx%d  EMC2 %s  %s ===\n", REGIME, basename(CARD),
            TRIALS, REPS, CPC, CFC, as.character(packageVersion("EMC2")), format(Sys.time(), "%Y-%m-%d %H:%M")))

PM <- c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5), sv = log(.1), SZ = qnorm(.2), st0 = log(.05))
PS <- c(v = .15, a = .15, t0 = .1, Z = .05, sv = .15, SZ = .15, st0 = .15)
FREE <- setdiff(names(PM), names(PIN))
# M0=1 (EMC2 dev-nle-m0, NLEFLOW_PLAN.md section 11.1): the card registers directly as a flow_scaled
# model, with no scaffold; its R path is nn_eval_scaled (the port's ddm_sc_density with SZ as the
# proportion) and its compiled path the patched evaluator. The scaffold route below stays for the
# AWS/laptop builds that predate it.
M0 <- Sys.getenv("M0") == "1"
mk <- if (M0) register_nn_model(CARD, twin = DDM) else register_nn_model(SCAFFOLD, twin = DDM)
cell <- nn_cell(mk, mean = PM[FREE], sd = PS[FREE],
                formula = lapply(FREE, function(p) as.formula(paste(p, "~ 1"))),
                constants = if (length(PIN)) PIN else NULL,
                factors = list(S = c("left", "right")), Rlevels = c("left", "right"))

# ---- the scaled model: the prepared design model with DDMsc's likelihood and DDM's bounds ------
# ORACLE=1 swaps the network for the analytic unit-scale DDM (port/R/ddm_scaled.R): the cell's
# likelihood must then equal the analytic control's, which checks the wiring through EMC2.
ev <- if (Sys.getenv("ORACLE") == "1") ddm_sc_oracle else ddm_sc_card_evaluator(CARD)
orig_model <- cell$design$model
sc_model <- function() {
  m <- orig_model()
  m$c_name <- NULL                                    # R likelihood path: log_likelihood_ddm -> dfun
  m$bound <- DDM()$bound
  # The registered model's Ttransform leaves SZ as EMC2's proportion; DDMsc takes DDM()'s
  # post-Ttransform convention, the absolute relative width sw = 2 SZ min(Z, 1 - Z).
  m$dfun <- function(rt, R, pars) {
    pars[, "SZ"] <- 2 * pars[, "SZ"] * pmin(pars[, "Z"], 1 - pars[, "Z"])
    ddm_sc_density(rt, as.integer(R), pars, ev) }
  m
}
# NATIVE=1: the same model on EMC2's compiled path (EMC2 branch dev-nle-scaled; ddm_sc_native_reg).
# The registration is built once: EMC2 calls the model function on every likelihood batch.
# Under M0 the registered model's own compiled path is used, and no registration is rebuilt.
sc_native_reg <- if (Sys.getenv("NATIVE") == "1" && !M0) ddm_sc_native_reg(orig_model()$nn, CARD)
sc_native <- if (M0) orig_model else function() {
  m <- sc_model()
  m$c_name <- "NN"
  m$nn <- sc_native_reg
  m
}
cell$design$model <- if (Sys.getenv("NATIVE") == "1") sc_native else sc_model
mp <- mapped_pars(cell$design, cell$mean)
for (p in names(PIN)) stopifnot(all(mp[, p] == 0))
for (p in setdiff(c("sv", "SZ", "st0"), names(PIN))) stopifnot(all(mp[, p] > 0))

# ---- checks: against the analytic DDM on one data set, 200 prior draws --------------------------
set.seed(7)
dat <- suppressMessages(make_data(cell$mean, cell$control_design, n_trials = TRIALS / 2))
X <- EMC2:::nn_cell_draws(cell, if (Sys.getenv("ORACLE") == "1") 40 else 200)
dc <- EMC2:::nn_dadm(dat, cell$control_design); dn <- EMC2:::nn_dadm(dat, cell$design)
tm <- function(dadm, model) { invisible(EMC2:::calc_ll_manager(X[1:5, , drop = FALSE], dadm, model))
  t <- system.time(ll <- as.numeric(EMC2:::calc_ll_manager(X, dadm, model)))[["elapsed"]]; list(ll = ll, ms = 1000 * t / nrow(X)) }
A <- tm(dc, cell$control_design$model); S <- tm(dn, sc_model)
cat(sprintf("per-likelihood ms (N=%d, single thread): analytic %.2f | scaled network (R level) %.2f\n", nrow(dat), A$ms, S$ms))
cat(sprintf("LL vs analytic over %d prior draws: cor %.4f; mean diff %+.2f; sd diff %.2f\n",
            nrow(X), cor(A$ll, S$ll), mean(S$ll - A$ll), sd(S$ll - A$ll)))
if (Sys.getenv("NATIVE") == "1") {
  Nt <- tm(dn, sc_native)
  cat(sprintf("per-likelihood ms: scaled network (compiled) %.3f | speed-up vs analytic %.2fx (R level %.2fx)\n",
              Nt$ms, A$ms / Nt$ms, A$ms / S$ms))
  cat(sprintf("compiled vs R level over %d prior draws: max |diff| %.2e\n", nrow(X), max(abs(Nt$ll - S$ll))))
  saveRDS(list(regime = REGIME, card = basename(CARD), analytic = A$ms, r_level = S$ms, compiled = Nt$ms,
               maxdiff = max(abs(Nt$ll - S$ll))), Sys.getenv("BENCH_RDS", tempfile()))
}
if (Sys.getenv("BENCH") == "1") quit(save = "no")

out <- nn_sbc_cell(cell, trials = TRIALS, replicates = REPS, run_control = FALSE, archive_dir = OUT,
                   cores_per_chain = CPC, cores_for_chains = CFC,
                   info = list(regime = REGIME, card = basename(CARD), pinned = PIN,
                               why = "scaled DDM: unit-scale network, t0/st0/a/s exact (port/R/ddm_scaled.R)"))
cat(sprintf("\n=== cell %s done %s ===\n", REGIME, format(Sys.time(), "%Y-%m-%d %H:%M")))
print(out$summary, digits = 3)
saveRDS(out$summary, file.path(OUT, "summary.rds"))

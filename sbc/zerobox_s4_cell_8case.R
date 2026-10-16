# Zero-box S4: one SBC cell on dev-nle, with its matched analytic control.
#
# The cell's network is the selected zero-box card; the control is the
# registered twin (the analytic DDM) in the SAME design, so the two runs differ
# in the likelihood only (nn_cell builds both; nn_sbc_cell runs both). Zero
# parameters are named constants at -Inf on the sampled scale, never
# defaults, and mapped_pars() is asserted before anything runs -- the Phase 37
# lesson, applied prospectively.
#
#   Rscript sbc/zerobox_s4_cell.R <regime> [card] [replicates] [trials]
#     regime: full | sv0 | SZ0 | st00 | all0 | st0only | SZonly
# COPY for the eight-case DDM review (2026-10-12): adds st0only (case 5: sv, SZ pinned) and SZonly
# (case 7: sv, st0 pinned), and env RUN_CONTROL=0 for a network-only pass; nothing else changed.
# Env: LIB (dev-nle library), CPC (cores per chain, default 12), CFC (cores for
#      chains, default 3), OUT (archive root, default sbc/nci/results_zerobox)
# Resumable: nn_sbc_cell() reloads a finished run from the archive.
a <- commandArgs(trailingOnly = TRUE)
if (length(a) < 1) stop("usage: zerobox_s4_cell.R <regime> [card] [replicates] [trials]")
REGIME <- a[1]
CARD <- path.expand(if (length(a) > 1) a[2] else "handover/nle-methods/port/export/ddm_zb_s903_e271_weights.json")
REPS <- if (length(a) > 2) as.integer(a[3]) else 500L
TRIALS <- if (length(a) > 3) as.integer(a[4]) else 400L
LIB <- path.expand(Sys.getenv("LIB", "~/emc2nn/nle/.Rlibs-nle"))
.libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages(library(EMC2))
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
CPC <- as.integer(Sys.getenv("CPC", "12")); CFC <- as.integer(Sys.getenv("CFC", "3"))
OUT <- file.path(Sys.getenv("OUT", "sbc/nci/results_zerobox"), REGIME)
PIN <- switch(REGIME, full = c(), sv0 = c(sv = -Inf), SZ0 = c(SZ = -Inf), st00 = c(st0 = -Inf),
              all0 = c(sv = -Inf, SZ = -Inf, st0 = -Inf),
              st0only = c(sv = -Inf, SZ = -Inf), SZonly = c(sv = -Inf, st0 = -Inf), stop("unknown regime ", REGIME))
cat(sprintf("=== zero-box S4 cell %s  card %s  N=%d reps=%d  cores %dx%d  EMC2 %s  %s ===\n", REGIME,
            basename(CARD), TRIALS, REPS, CPC, CFC, as.character(packageVersion("EMC2")), format(Sys.time(), "%Y-%m-%d %H:%M")))

# The SBC prior of every DDM cell in this project (sbc_specs.R), minus the pinned parameters.
PM <- c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5), sv = log(.1), SZ = qnorm(.2), st0 = log(.05))
PS <- c(v = .15, a = .15, t0 = .1, Z = .05, sv = .15, SZ = .15, st0 = .15)
FREE <- setdiff(names(PM), names(PIN))
mk <- register_nn_model(CARD, twin = DDM)
cell <- nn_cell(mk, mean = PM[FREE], sd = PS[FREE],
                formula = lapply(FREE, function(p) as.formula(paste(p, "~ 1"))),
                constants = if (length(PIN)) PIN else NULL,
                factors = list(S = c("left", "right")), Rlevels = c("left", "right"))
print(cell)

# Assert the design simulates and evaluates the model the cell claims, for the
# network AND the control, before a replicate runs.
for (d in list(cell$design, cell$control_design)) {
  mp <- mapped_pars(d, cell$mean)
  for (p in names(PIN)) stopifnot(all(mp[, p] == 0))
  for (p in setdiff(c("sv", "SZ", "st0"), names(PIN))) stopifnot(all(mp[, p] > 0))
}
cat("mapped_pars asserted: pinned exactly 0, free positive, in both designs\n")
dat <- make_data(cell$mean, cell$design, n_trials = 200)
if ("st0" %in% names(PIN)) stopifnot(min(dat$rt) > exp(PM[["t0"]]))
cat("simulator respects the pins\n")

out <- nn_sbc_cell(cell, trials = TRIALS, replicates = REPS, run_control = Sys.getenv("RUN_CONTROL", "1") != "0", archive_dir = OUT,
                   cores_per_chain = CPC, cores_for_chains = CFC,
                   info = list(regime = REGIME, card = basename(CARD), pinned = PIN))
cat(sprintf("\n=== cell %s done %s ===\n", REGIME, format(Sys.time(), "%Y-%m-%d %H:%M")))
print(out$summary, digits = 3)
saveRDS(out$summary, file.path(OUT, "summary.rds"))

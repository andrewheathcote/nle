# Phase 4: SBC across the identification 2x2 for sv and SZ.
#
#   Rscript SBCid.R <cell>
# where <cell> is one of P1 P2 P3 P4 C3 C4 (see the table below).
# Results land in results_id/. One cell per PBS job so they run concurrently.

rm(list = ls())
library(EMC2)
source("sbc_specs_id.R")

CELLS <- list(
  # nn cells: the production flow, across the identification 2x2
  P1 = list(model = "DDMnn", sv = 0.1, SZ = 0.2, N = 400L,  reps = 500L),
  P2 = list(model = "DDMnn", sv = 0.1, SZ = 0.2, N = 3200L, reps = 500L),
  P3 = list(model = "DDMnn", sv = 1.5, SZ = 0.4, N = 400L,  reps = 500L),
  P4 = list(model = "DDMnn", sv = 1.5, SZ = 0.4, N = 3200L, reps = 500L),
  # analytic controls: validate the machinery at the NEW generating point
  C3 = list(model = "DDM",   sv = 1.5, SZ = 0.4, N = 400L,  reps = 500L),
  C4 = list(model = "DDM",   sv = 1.5, SZ = 0.4, N = 3200L, reps = 200L),

  # --- race models: the same identification audit -------------------------
  # RDM's A (start-point variability) is the analogue of the DDM's sv: at the
  # Phase 1 cell (N=100) it carries N*I/prior = 0.10, so RDMnn's "PASS" on A
  # was close to content-free. Unlike sv this is cheap to fix - A reaches
  # parity at N ~ 1040 with no change to the generating point.
  R1 = list(model = "RDMnn", N = 1000L, reps = 500L),   # A at parity (0.96)
  R2 = list(model = "RDM",   N = 1000L, reps = 500L),   # analytic control
  # LNR is well identified at N=100 (m 18.9, m_lMTRUE 13.2, s 2.35 x prior),
  # so its Phase 1 result was a real test - and it found a real anomaly in s
  # (mean rank .43, KS p 1.5e-5), which the measured score bias predicts
  # (g_s = +0.032, 18x its MC error, correct sign). These cells ask whether
  # that anomaly WORSENS with N, as the DDM's st0 did (.94 -> .84 -> .67).
  L1 = list(model = "LNRnn", N = 400L,  reps = 500L),
  L2 = list(model = "LNRnn", N = 1000L, reps = 500L),
  L3 = list(model = "LNR",   N = 1000L, reps = 500L)    # analytic control
)

args <- commandArgs(trailingOnly = TRUE)
if (!length(args) || !args[1] %in% names(CELLS))
  stop("usage: Rscript SBCid.R <", paste(names(CELLS), collapse = "|"), ">")
cl <- CELLS[[args[1]]]

# Build guards, per model family. The race cells (LNRnn/RDMnn) need only the
# race path, which has been in dev-nn since bd70c709 and is unchanged since,
# so they run on the Phase 3 Gadi install with NO reinstall. The DDM cells
# additionally need whichever DDM entry point the shipped flow uses.
if (grepl("^(LNR|RDM)nn$", cl$model))
  stopifnot("flow_eval_trials_cpp" %in% ls(getNamespace("EMC2")))
if (cl$model == "DDMnn")
  stopifnot("ddm_eval_trials_cpp" %in% ls(getNamespace("EMC2")))

if (is.null(cl$sv)) {
  # race cells: no sv/SZ override, so the standard spec applies
  cat(sprintf("cell %s: %s  N=%d  reps=%d\n", args[1], cl$model, cl$N, cl$reps))
  run_sbc_cell(cl$model, cl$N, replicates = cl$reps, cores = 48,
               out_dir = "results_id")
} else {
  cat(sprintf("cell %s: %s  sv=%.1f  SZ=%.2f  N=%d  reps=%d\n",
              args[1], cl$model, cl$sv, cl$SZ, cl$N, cl$reps))
  run_sbc_cell_id(cl$model, cl$N, replicates = cl$reps, cores = 48,
                  out_dir = "results_id", sv = cl$sv, SZ = cl$SZ)
}
cat("\ncell complete.\n")

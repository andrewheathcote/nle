# Pre-compute the 5-member ensemble's CONDITIONAL log-density on the training
# set, so a single flow can be distilled against it.
#
# The ensemble costs 4.3x a single flow (3880 vs 900 us/1000) and is the
# largest component of the 6525 us production cost - which currently sits at
# parity with EMC2's analytic dDDM (6300-11400 us), an awkward place to be when
# speed is the port's whole justification. If one flow can reproduce the
# mixture, cost falls to ~1400 us.
#
# The argument it should work is the one that motivated score matching: a
# pointwise density target pins the function's derivatives, so a student that
# matches the mixture should inherit its reduced score bias. The risk is that a
# single [256,128] flow cannot represent a 5-member mixture - which is exactly
# what the test measures.
#
# Stored target is the CONDITIONAL log f(rt | theta, R) = log_pdf_joint - log p_R,
# matching what the flow models (loss_fn_distill adds log P(R) back).
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")
suppressPackageStartupMessages(library(EMC2)); suppressPackageStartupMessages(library(arrow))

MEM <- c("cap256w_c3","cap256w_s101_c3","cap256w_s102_c3","cap256w_s103_c3","cap256w_s104_c3")
ptr <- EMC2:::ddm_build_ensemble(lapply(MEM, function(m)
  ddm_load(sprintf("port/export/ddm_%s_weights.json", m))))
sc <- ddm_load(sprintf("port/export/ddm_%s_weights.json", MEM[1]))$scaler
CTX <- c("v","a","t0","s","Z","SZ","sv","st0")
src <- "data/train/tpt40p400k/DDM/trials"; out <- "data/train/ensdistil2/DDM"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
files <- list.files(src, pattern = "\\.parquet$", full.names = TRUE, recursive = TRUE)
cat(sprintf("=== %d files -> %s ===\n", length(files), out))
t0 <- proc.time()[3]
for (i in seq_along(files)) {
  d <- as.data.frame(read_parquet(files[i]))
  Th <- as.matrix(d[, CTX])
  ev <- EMC2:::ddm_ens_eval_trials_cpp(ptr, Th, d$rt, as.integer(d$R))
  d$ll <- ev$log_pdf - log(pmax(ev$p_R, 1e-12))   # conditional target
  write_parquet(d, file.path(out, sprintf("part-%02d.parquet", i)))
  cat(sprintf("  %d/%d  (%.0f s)\n", i, length(files), proc.time()[3] - t0))
}
cat(sprintf("=== done in %.1f min ===\n", (proc.time()[3] - t0) / 60))

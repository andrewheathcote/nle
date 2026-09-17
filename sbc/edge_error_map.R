# Step 0 diagnostic: where does the DDM flow's likelihood error live, and
# does it move with training budget? Compares flow joint log-density against
# WienR (precision 1e-8) across the rt axis at the precision-study
# generating point, for the small / medium / large budget checkpoints.
#
# Run from the nle repo root:  Rscript sbc/edge_error_map.R

# prefer the dedicated nn-EMC2 library (nle/.Rlibs) so installs of other
# EMC2 branches into the default library cannot affect these studies
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
source("port/R/flow_race.R")
source("port/R/flow_ddm.R")
suppressPackageStartupMessages(library(EMC2))   # for compiled ddm_build/eval

# generating point (natural scale): v=1, a=.8, t0=.3, s=1, Z=.5, sv=.1,
# SZ=.2, st0=.05 -> sampled scale in flow order [v,a,t0,s,Z,SZ,sv,st0]
theta <- c(1, log(.8), log(.3), log(1), qnorm(.5), qnorm(.2), log(.1), log(.05))
t0n <- 0.3

rt <- c(seq(t0n + .005, t0n + .15, by = .005),
        seq(t0n + .16, 1, by = .02), seq(1.05, 3, by = .05))
regions <- cut(rt - t0n, c(0, .1, .7, Inf),
               labels = c("edge (rt-t0<=.1)", "bulk", "tail"))

an <- function(R) WienR::dWDM(rt, response = ifelse(R == 1, "lower", "upper"),
                              a = .8, v = 1, w = .5, t0 = t0n, sv = .1,
                              sw = 2 * .2 * .5, st0 = .05,
                              precision = 1e-8)$logvalue

cat(sprintf("%-8s %-6s %-18s %12s %12s\n",
            "budget", "R", "region", "mean(D)", "mean|D|"))
tags <- commandArgs(trailingOnly = TRUE)
if (length(tags) == 0) tags <- c("small", "medium", "large")
for (b in tags) {
  fl <- ddm_load(sprintf("port/export/ddm_%s_weights.json", b))
  ptr <- EMC2:::ddm_build(fl)
  for (R in 1:2) {
    d <- EMC2:::ddm_eval_cpp(ptr, theta, rt, rep(R, length(rt)))$log_pdf - an(R)
    for (reg in levels(regions)) {
      sel <- regions == reg
      cat(sprintf("%-8s %-6d %-18s %+12.4f %12.4f\n",
                  b, R, reg, mean(d[sel]), mean(abs(d[sel]))))
    }
  }
}

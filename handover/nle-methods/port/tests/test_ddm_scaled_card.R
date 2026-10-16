# Card test of port/R/ddm_scaled.R (handover/SCALED_DDM_PLAN.md section 3, test 2): a scaled card's
# golden vectors (float64 port of the exporter) through DDMsc's evaluator, i.e. through EMC2
# dev-nle's compiled flow evaluator with the inputs DDMsc builds.
#   Rscript port/tests/test_ddm_scaled_card.R port/export/ddm_<tag>_weights.json
.libPaths(c(path.expand(Sys.getenv("LIB", "~/R/nle-lib")), .libPaths()))
suppressPackageStartupMessages(library(EMC2))
source("port/R/ddm_scaled.R")
card <- commandArgs(trailingOnly = TRUE)[1]
g <- read.csv(sub("_weights\\.json$", "_golden.csv", card))
U <- cbind(kappa = g$kappa, eta = exp(g$eta), Z = pnorm(g$Z), SZ = pnorm(g$SZ))
ev <- ddm_sc_card_evaluator(card)
o <- ev(U, g$rt, g$R)
lp <- log(o$pdf); ok <- is.finite(g$log_pdf_joint) & g$log_pdf_joint > log(1e-10)
d_lp <- max(abs(lp[ok] - g$log_pdf_joint[ok])); d_cdf <- max(abs(o$cdf - g$cdf_defective))
cat(sprintf("%d golden rows (%d above the floor), %d with eta = 0, %d with SZ = 0\n", nrow(g), sum(ok),
            sum(U[, "eta"] == 0) / 1, sum(U[, "SZ"] == 0)))
cat(sprintf("max |log pdf - golden| %.2e ; max |cdf - golden| %.2e\n", d_lp, d_cdf))
cat("CARD TEST:", if (d_lp < 1e-8 && d_cdf < 1e-10) "PASS" else "FAIL", "\n")

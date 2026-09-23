# SBC for the DIRECT-REGRESSION baseline (Phase 31), the one route that beats
# the flow on the centre of the score-bias distribution and had never been
# through SBC. Phase 37 is the reason it now is: score bias does not settle a
# calibration question, and a cheap number in hand is not a verdict.
#
#   Rscript sbc/nci/SBCddmreg.R <out_dir> <trials> <reps>
# Env: SBC_CORES (per chain; 3 chains, so total = 3x), REG_JSON (the net)
#
# SELECTION, declared: `reg_s402` of the four seeds, chosen on |g_a| + |g_t0|
# = 0.0072 (against 0.0139, 0.0397, 0.0330 for b1k, s401, s403), measured on
# draws independent of this cell.
rm(list = ls())
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(jsonlite)})
source("handover/regress/DDMreg.R")
setwd("sbc/nci"); source("sbc_specs.R"); setwd("../..")

a <- commandArgs(trailingOnly = TRUE)
out_dir <- a[1]; trials <- as.integer(a[2]); reps <- as.integer(a[3])
cores <- as.integer(Sys.getenv("SBC_CORES", "8"))

net <- regnet_get()
cat(sprintf("regression net: %s   layers %s\n",
            Sys.getenv("REG_JSON", "results/model_states/reg_s402_budget/regress.json"),
            paste(sapply(net$layers$W, function(w) paste(dim(w), collapse = "x")),
                  collapse = " -> ")))

# The wrapper must agree with the estimator that measured this net. Same theta,
# same rt, same R, evaluated both ways.
.chk <- local({
  th <- c(v = 1, a = log(.8), t0 = log(.3), s = 0, Z = qnorm(.5),
          SZ = qnorm(.2), sv = log(.1), st0 = log(.05))
  rt <- c(.45, .6, .9); R <- c(1L, 2L, 1L)
  pars <- cbind(v = th[["v"]], a = exp(th[["a"]]), t0 = exp(th[["t0"]]), s = 1,
                Z = pnorm(th[["Z"]]), SZ = pnorm(th[["SZ"]]),
                sv = exp(th[["sv"]]), st0 = exp(th[["st0"]]))[rep(1, 3), ]
  mine <- reg_log_pdf(rt, R, pars)
  raw <- cbind(matrix(th, nrow = 3, ncol = 8, byrow = TRUE), log(rt))
  z <- sweep(sweep(raw, 2, net$scaler_mean, "-"), 2, net$scaler_scale, "/")
  x <- cbind(z, as.numeric(R)); nl <- length(net$layers$W)
  for (i in seq_len(nl)) {
    x <- sweep(x %*% net$layers$W[[i]], 2, net$layers$b[[i]], "+")
    if (i < nl) x <- .gelu_reg(x)
  }
  max(abs(mine - as.numeric(x)))
})
cat(sprintf("wrapper vs score_bias_regress evaluation: max |diff| = %.2e\n", .chk))
stopifnot(.chk < 1e-12)

cat(sprintf("DDMreg SBC  trials=%d reps=%d cores/chain=%d -> %s\n",
            trials, reps, cores, out_dir))
run_sbc_cell("DDMreg", trials, replicates = reps, cores = cores, out_dir = out_dir)

x <- readRDS(file.path(out_dir, sprintf("DDMreg_N%d_R%d.rds", trials, reps)))$SBC
b <- x$bias$alpha
cat("\nparameter  STD-BIAS (gate |.|<=0.5)   KS p\n")
for (p in colnames(b)) {
  v <- b[, p][is.finite(b[, p])]
  r <- x$rank$alpha[, p]; r <- r[is.finite(r)]
  ks <- suppressWarnings(stats::ks.test(r / max(r), "punif")$p.value)
  cat(sprintf("%-9s  %+.3f%s   %.3g\n", p, mean(v) / sd(v),
              if (abs(mean(v) / sd(v)) > 0.5) "  <- FAILS" else "", ks))
}
cat("\nDDMreg SBC complete.\n")

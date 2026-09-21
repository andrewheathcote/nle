# SBC of the DDM flow with st0 held CONSTANT rather than estimated.
#
# Purpose. With st0 estimated, the flow's t0 is biased (+0.319 at N=400, rank
# KS p 1.8e-07) and the error leaks into a (r = -0.28), SZ (+0.33) and Z (+0.19).
# The proposed mechanism is a t0/st0 leading-edge trade-off: t0 and st0 jointly
# determine where the density turns on, so a flow that smooths a sharp onset
# must trade one against the other. Remove st0 from the estimation problem and
# the trade-off has no partner, so the bias should vanish.
#
# Two cells, differing only in the fixed value:
#   A  st0 = 0.05  the prior mean -- "st0 known", tests the ESTIMATION claim
#   B  st0 = 0.01  the training-box FLOOR -- the nearest the artefact can get
#                  to a hard edge, and the direct DDM analogue of the RDM
#
# NOTE st0 = 0 is NOT representable: the flow's box is st0 in [0.01, 0.49]
# (log scale [-4.61, -0.71]), and log(0) = -Inf. A genuinely st0-free network
# has to be trained as a 7-parameter model. Cell B is the closest approximation
# the existing artefact permits, and 0.01 is the exact box edge -- if B fails
# where A passes, re-run at st0 = 0.02 to separate "near-hard edge" from "box
# boundary" before concluding anything.
#
#   Rscript sbc/nci/SBCddm_st0fixed.R <out_dir> <trials> <reps> <st0_value>
# Env: SBC_CORES (cores PER CHAIN; EMC2 runs 3 chains, so total = 3x)
rm(list = ls())
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
library(EMC2)
setwd("sbc/nci"); source("sbc_specs.R"); setwd("../..")

a <- commandArgs(trailingOnly = TRUE)
out_dir <- a[1]; trials <- as.integer(a[2]); reps <- as.integer(a[3])
st0_val <- as.numeric(a[4])
cores <- as.integer(Sys.getenv("SBC_CORES", "8"))
stopifnot(st0_val >= 0.01, st0_val <= 0.49)   # the flow's training box

# Same design and priors as make_sbc_cell("DDMnn"), with st0 moved from the
# sampled set into constants. Everything else is held identical so the result
# is comparable to the estimated-st0 cells.
des <- design(factors = list(subjects = 1, S = c("left", "right")),
              Rlevels = c("left", "right"),
              formula = list(v ~ 1, a ~ 1, t0 ~ 1, s ~ 1, Z ~ 1, sv ~ 1,
                             SZ ~ 1, st0 ~ 1),
              model = DDMnn, use_matchfun = FALSE,
              constants = c(s = log(1), st0 = log(st0_val)), verbose = FALSE)
pmean <- c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5),
           sv = log(.1), SZ = qnorm(.2))
psd   <- c(v = .15, a = .15, t0 = .1, Z = .05, sv = .15, SZ = .15)
sp <- names(sampled_pars(des))
cat("sampled parameters:", paste(sp, collapse = ", "), "\n")
# alignment matters: prior() takes pmean positionally, so a mismatch between
# the sampled order and the pmean order would silently mis-specify the prior
stopifnot(identical(sp, names(pmean)))
prior_in <- prior(des, type = "single", pmean = pmean, psd = psd)

b <- readRDS(system.file("extdata", "flownn", "ddm_ens5.rds",
                         package = "EMC2", mustWork = TRUE))
f <- if (is.null(b$members)) b else b$members[[1]]
cat(sprintf("installed DDM flow: budget=%s flow_step=%s clf=%s bins=%d  members=%d\n",
            f$budget, as.character(f$checkpoint_step_flow),
            as.character(f$checkpoint_step_classifier), f$spline$num_bins,
            length(if (is.null(b$members)) list(b) else b$members)))
cat(sprintf("DDMnn SBC  st0 FIXED at %.3f  trials=%d  reps=%d  cores/chain=%d -> %s\n",
            st0_val, trials, reps, cores, out_dir))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
tag <- sprintf("DDMnn_st0fix%03d_N%d_R%d", round(st0_val * 1000), trials, reps)
t0 <- proc.time()[3]
SBC <- run_sbc(des, prior_in, replicates = reps, trials = trials,
               cores_per_chain = cores, verbose = TRUE,
               fileName = file.path(out_dir, paste0(tag, "_tmp.RData")))
mins <- (proc.time()[3] - t0) / 60
saveRDS(list(SBC = SBC, model = "DDMnn", trials = trials, replicates = reps,
             pmean = pmean, psd = psd, st0_fixed = st0_val, mins = mins),
        file.path(out_dir, paste0(tag, ".rds")))
cat(sprintf("===== %s done in %.1f min =====\n", tag, mins))
cov <- apply(SBC$coverage$alpha, 2, mean, na.rm = TRUE)
bias <- apply(SBC$bias$alpha, 2, mean, na.rm = TRUE)
sdb <- apply(SBC$bias$alpha, 2, sd, na.rm = TRUE)
cat("parameter  coverage  mean-bias  STD-BIAS (gate |.|<=0.5)\n")
for (i in seq_along(cov))
  cat(sprintf("%-9s  %.3f     %+.4f    %+.3f%s\n", names(cov)[i], cov[i],
              bias[i], bias[i]/sdb[i],
              if (abs(bias[i]/sdb[i]) > 0.5) "  <- FAILS" else ""))
cat("\nDDM st0-fixed SBC complete.\n")

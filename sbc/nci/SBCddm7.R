# SBC for the st0 = 0 (7-parameter) DDM flow.
#
#   Rscript sbc/nci/SBCddm7.R <out_dir> <trials> <reps>
# Env: SBC_CORES (per chain; EMC2 runs 3 chains, so total = 3x)
#
# st0 is absent from the formula, so EMC2 sets it to its default -Inf on the
# sampled scale and exp() gives st0 = 0 exactly. The flow is DDMnn7, defined in
# handover/st0_zero/DDMnn7.R, which differs from DDMnn only in the theta builder
# (7 columns) and the bundle name.
rm(list = ls())
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
library(EMC2)
source("handover/st0_zero/DDMnn7.R")
setwd("sbc/nci"); source("sbc_specs.R"); setwd("../..")

a <- commandArgs(trailingOnly = TRUE)
out_dir <- a[1]; trials <- as.integer(a[2]); reps <- as.integer(a[3])
cores <- as.integer(Sys.getenv("SBC_CORES", "8"))

des <- design(factors = list(subjects = 1, S = c("left", "right")),
              Rlevels = c("left", "right"),
              formula = list(v ~ 1, a ~ 1, t0 ~ 1, s ~ 1, Z ~ 1, SZ ~ 1, sv ~ 1),
              model = DDMnn7, use_matchfun = FALSE,
              constants = c(s = log(1)), verbose = FALSE)
# prior() takes pmean POSITIONALLY, and sampled_pars() order follows the
# formula, not the canonical DDM order -- writing the formula as (Z, SZ, sv)
# rather than (Z, sv, SZ) silently transposes two priors. So look the values up
# BY NAME in whatever order sampled_pars returns, and assert nothing is missing.
PM <- c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5), sv = log(.1), SZ = qnorm(.2))
PS <- c(v = .15, a = .15, t0 = .1, Z = .05, sv = .15, SZ = .15)
sp <- names(sampled_pars(des))
cat("sampled parameters:", paste(sp, collapse = ", "), "\n")
stopifnot(all(sp %in% names(PM)), length(sp) == length(PM))
pmean <- PM[sp]; psd <- PS[sp]
cat("priors aligned to that order:",
    paste(sprintf("%s=%.3f", names(pmean), pmean), collapse = " "), "\n")
prior_in <- prior(des, type = "single", pmean = pmean, psd = psd)

b <- readRDS(system.file("extdata", "flownn", "ddm_st0zero.rds",
                         package = "EMC2", mustWork = TRUE))
m <- if (is.null(b$members)) b else b$members[[1]]
cat(sprintf("installed flow: budget=%s step=%s bins=%d context=%s members=%d\n",
            m$budget, as.character(m$checkpoint_step_flow), m$spline$num_bins,
            paste(m$context_names, collapse = ","),
            length(if (is.null(b$members)) list(b) else b$members)))
cat(sprintf("DDMnn7 SBC  trials=%d reps=%d cores/chain=%d -> %s\n",
            trials, reps, cores, out_dir))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
tag <- sprintf("DDMnn7_N%d_R%d", trials, reps)
t0 <- proc.time()[3]
SBC <- run_sbc(des, prior_in, replicates = reps, trials = trials,
               cores_per_chain = cores, verbose = TRUE,
               fileName = file.path(out_dir, paste0(tag, "_tmp.RData")))
mins <- (proc.time()[3] - t0) / 60
saveRDS(list(SBC = SBC, model = "DDMnn7", trials = trials, replicates = reps,
             pmean = pmean, psd = psd, mins = mins),
        file.path(out_dir, paste0(tag, ".rds")))
cat(sprintf("===== %s done in %.1f min =====\n", tag, mins))
cov <- apply(SBC$coverage$alpha, 2, mean, na.rm = TRUE)
bias <- apply(SBC$bias$alpha, 2, mean, na.rm = TRUE)
sdb <- apply(SBC$bias$alpha, 2, sd, na.rm = TRUE)
cat("parameter  coverage  mean-bias  STD-BIAS (gate |.|<=0.5)\n")
for (i in seq_along(cov))
  cat(sprintf("%-9s  %.3f     %+.4f    %+.3f%s\n", names(cov)[i], cov[i], bias[i],
              bias[i]/sdb[i], if (abs(bias[i]/sdb[i]) > 0.5) "  <- FAILS" else ""))
cat("\nDDMnn7 SBC complete.\n")

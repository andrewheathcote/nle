# Phase 0: compression arm. Does EMC2's default rt compression (flooring
# rts to a 1/60 s grid) shift t0/a estimates, and does it interact with the
# DDMnn flow's leading-edge distortion? See sbc/SBC_PLAN.md.
#
# Reuses the main-study datasets and cache (same seeds): the uncompressed
# DDM / DDMnn cells at N = 100 load from cache; new cells fit both models
# with compress = TRUE at rt_resolution 1/60, 1/1000 (and 1/100, nn only).
#
# Run from the nle repo root:  Rscript sbc/run_compression_arm_ddm.R

source("sbc/precision_component.R")

N_CORES <- 9L
N_REPS <- 4L
LOG <- "sbc/precision_ddm_progress.log"
OUT <- "sbc/compression_arm_results.rds"

design_maker <- function(model) {
  design(factors = list(subjects = 1, S = c("left", "right")),
         Rlevels = c("left", "right"),
         formula = list(v ~ 1, a ~ 1, t0 ~ 1, s ~ 1, Z ~ 1, sv ~ 1,
                        SZ ~ 1, st0 ~ 1),
         constants = c(s = log(1)), model = model, verbose = FALSE)
}
PRIOR_MEAN <- c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5),
                sv = log(.1), SZ = qnorm(.2), st0 = log(.05))
PRIOR_SD <- c(v = .15, a = .15, t0 = .1, Z = .05,
              sv = .15, SZ = .15, st0 = .15)
prior_maker <- function(des) {
  pnames <- names(sampled_pars(des))
  prior(des, type = "single", pmean = PRIOR_MEAN[pnames], psd = PRIOR_SD[pnames])
}

cmp <- function(res) list(compress = TRUE, rt_resolution = res)
specs <- list(
  list(label = "DDM", model = DDM),                                   # cached
  list(label = "DDMnn", model = DDMnn),                               # cached
  list(label = "DDM_r60", model = DDM, make_emc_args = cmp(1/60)),
  list(label = "DDMnn_r60", model = DDMnn, make_emc_args = cmp(1/60)),
  list(label = "DDM_r1000", model = DDM, make_emc_args = cmp(1/1000)),
  list(label = "DDMnn_r1000", model = DDMnn, make_emc_args = cmp(1/1000)),
  list(label = "DDMnn_r100", model = DDMnn, make_emc_args = cmp(1/100))
)

cat(sprintf("Compression arm: %d cells (8 cached) on %d cores; progress -> %s\n",
            length(specs) * N_REPS, N_CORES, LOG))
t0 <- proc.time()[3]
res <- run_precision_study(specs, design_maker, prior_maker,
                           p_true = PRIOR_MEAN, n_grid = 100L,
                           n_reps = N_REPS, cores = N_CORES, log_file = LOG,
                           cache_dir = "sbc/cache_precision_ddm")
saveRDS(res, OUT)
cat(sprintf("\nDone in %.1f min; results -> %s\n", (proc.time()[3] - t0) / 60, OUT))

# ---------------- analysis ----------------
res_ok <- res[!res$failed, ]
base <- function(lab) sub("_r[0-9]+$", "", lab)
res_ok$resolution <- ifelse(grepl("_r60$", res_ok$label), "1/60",
                     ifelse(grepl("_r100$", res_ok$label), "1/100",
                     ifelse(grepl("_r1000$", res_ok$label), "1/1000", "none")))

cat("\nPosterior-median shift vs the uncompressed cell of the same model\n")
cat("(paired by replicate dataset; mean +- se over 4 reps; sampled scale):\n")
for (p in c("t0", "a", "v", "Z")) {
  cat(sprintf("%-3s", p))
  for (lab in c("DDM_r60", "DDM_r1000", "DDMnn_r60", "DDMnn_r100", "DDMnn_r1000")) {
    m0 <- res_ok[res_ok$label == base(lab) & res_ok$par == p, ]
    m1 <- res_ok[res_ok$label == lab & res_ok$par == p, ]
    m0 <- m0[order(m0$rep), ]; m1 <- m1[order(m1$rep), ]
    d <- m1$post_med - m0$post_med
    cat(sprintf("  %s %+.4f+-%.4f", lab, mean(d), sd(d) / sqrt(length(d))))
  }
  cat("\n")
}

cat("\nBias vs truth (mean over reps, standardized by posterior sd):\n")
bt <- aggregate(cbind(bias, post_sd) ~ label + par, data = res_ok, FUN = mean)
bt$std <- round(bt$bias / bt$post_sd, 2)
for (p in c("t0", "a")) {
  sub <- bt[bt$par == p, c("label", "bias", "std")]
  cat(sprintf("%-3s %s\n", p,
              paste(sprintf("%s %+.3f(%.1fsd)", sub$label, sub$bias, sub$std),
                    collapse = "  ")))
}

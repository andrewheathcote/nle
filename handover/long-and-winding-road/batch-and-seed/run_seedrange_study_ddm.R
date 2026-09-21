# Seed-replication + spline-range gate study:
#   arch64w_s101 — identical config to arch64w, training seed 101 (vs 100):
#                  calibrates how much variant-to-variant difference is
#                  training noise rather than real effect;
#   arch64wR3    — arch64w config with spline support narrowed to log-rt
#                  [-3, 3] (0.05-20 s, the data range) from [-5, 5]:
#                  concentrates all knots where data occur.
# References (analytic DDM, DDMnn large, DDMnn_arch64w) from cache.
#
# Run from the nle repo root:  Rscript sbc/run_seedrange_study_ddm.R

source("sbc/precision_component.R")
source("port/R/flow_race.R")
source("port/R/flow_ddm.R")

N_CORES <- 9L
N_GRID <- c(100L, 200L, 400L, 800L)
N_REPS <- 4L
LOG <- "sbc/precision_ddm_progress.log"
OUT <- "sbc/seedrange_study_results.rds"

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

# DDMnn variant whose dfun/pfun evaluate a flow built from a local export
# (bounds/priors/transforms identical across budgets — designs.R is shared)
make_DDMnn_budget <- function(budget) {
  fl <- ddm_load(sprintf("port/export/ddm_%s_weights.json", budget))
  ptr <- EMC2:::ddm_build(fl)
  force(ptr)
  function() {
    m <- DDMnn()
    m$dfun <- function(rt, R, pars)
      EMC2:::ddm_eval_trials_cpp(ptr, EMC2:::.theta_DDMnn(pars),
                                 rt, as.integer(R))$pdf
    m$pfun <- function(rt, R, pars)
      EMC2:::ddm_eval_trials_cpp(ptr, EMC2:::.theta_DDMnn(pars),
                                 rt, as.integer(R))$cdf
    m
  }
}

specs <- list(
  list(label = "DDM", model = DDM, n_grid = c(100L, 200L)),   # cached
  list(label = "DDMnn", model = DDMnn),                       # cached (large)
  list(label = "DDMnn_arch64w", model = make_DDMnn_budget("arch64w")),   # cached
  list(label = "DDMnn_arch64w_s101", model = make_DDMnn_budget("arch64w_s101")),
  list(label = "DDMnn_arch64wR3", model = make_DDMnn_budget("arch64wR3"))
)

n_fits <- nrow(make_precision_cells(specs, N_GRID, N_REPS))
cat(sprintf("Seed/range study: %d cells (24 cached) on %d cores; progress -> %s\n",
            n_fits, N_CORES, LOG))
t0 <- proc.time()[3]
res <- run_precision_study(specs, design_maker, prior_maker,
                           p_true = PRIOR_MEAN, n_grid = N_GRID,
                           n_reps = N_REPS, cores = N_CORES, log_file = LOG,
                           cache_dir = "sbc/cache_precision_ddm")
saveRDS(res, OUT)
cat(sprintf("\nDone in %.1f min; results -> %s\n", (proc.time()[3] - t0) / 60, OUT))

# ---------------- analysis ----------------
res_ok <- res[!res$failed, ]
budget_of <- c(DDMnn = 1e6, DDMnn_arch64w = 1e6, DDMnn_arch64w_s101 = 1e6, DDMnn_arch64wR3 = 1e6)

cat("\nStandardized bias (mean bias / mean posterior sd) by budget and N:\n")
bt <- aggregate(cbind(bias, post_sd) ~ label + par + n_trials, data = res_ok,
                FUN = mean)
for (p in c("t0", "a", "sv", "st0", "v", "Z", "SZ")) {
  cat(sprintf("%-4s", p))
  for (lab in c("DDMnn", "DDMnn_arch64w", "DDMnn_arch64w_s101", "DDMnn_arch64wR3")) {
    s2 <- bt[bt$label == lab & bt$par == p, ]
    s2 <- s2[order(s2$n_trials), ]
    cat(sprintf("  %s: %s", sub("DDMnn_?", "", lab),
                paste(sprintf("%+.1f", s2$bias / s2$post_sd), collapse = "/")))
  }
  cat("   (N=100/200/400/800)\n")
}

cat("\nG1 gate (|std bias| <= 0.5 at N=800), worst parameter per budget:\n")
for (lab in c("DDMnn", "DDMnn_arch64w", "DDMnn_arch64w_s101", "DDMnn_arch64wR3")) {
  s2 <- bt[bt$label == lab & bt$n_trials == 800, ]
  s2$std <- abs(s2$bias / s2$post_sd)
  worst <- s2[which.max(s2$std), ]
  cat(sprintf("  %-14s worst %s %.1f sd  %s\n", lab,
              worst$par, worst$std, if (worst$std <= 0.5) "PASS" else "FAIL"))
}

cat("\n|bias| by variant (t0, a; absolute, N-pooled):\n")
for (p in c("t0", "a")) {
  b_abs <- sapply(c("DDMnn", "DDMnn_arch64w", "DDMnn_arch64w_s101", "DDMnn_arch64wR3"), function(lab)
    mean(abs(bt$bias[bt$label == lab & bt$par == p])))
  cat(sprintf("  %-3s |bias| = %s  (large/arch64w/s101/R3)\n",
              p, paste(sprintf("%.4f", b_abs), collapse = " / ")))
}

cat("\nInformation ratio kappa vs analytic (pooled over params v,a,t0,Z):\n")
for (lab in c("DDMnn", "DDMnn_arch64w", "DDMnn_arch64w_s101", "DDMnn_arch64wR3")) {
  ei <- effective_info(res_ok, ref = "DDM", target = lab, prior_sd = PRIOR_SD)
  ei <- ei[ei$par %in% c("v", "a", "t0", "Z"), ]
  cat(sprintf("  %-14s median kappa %.2f\n", lab, median(ei$kappa)))
}

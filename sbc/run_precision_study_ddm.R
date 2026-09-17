# Small laptop study: does DDMnn's likelihood-approximation noise behave
# like a loss of information (fewer effective trials), and if so how many
# extra trials recover analytic-DDM precision at the SBC base of 100?
#
# Paired design: identical datasets fitted with analytic DDM and DDMnn at
# N in {100, 200, 400, 800}, 4 replicate datasets, SBC-matched fit
# settings. Reports per-parameter posterior sd / CI width, the information
# ratio kappa (nn information per trial relative to analytic), the implied
# crossover trial count 100 / kappa, and bias diagnostics.
#
# Design/prior follow the EMC2 SBC vignette's single-participant DDM,
# EXCEPT the SZ prior mean: the vignette's qnorm(.05) sits exactly on the
# DDMnn training-box boundary, so both models use qnorm(.2) here.
#
# Run from the nle repo root (writes sbc/precision_ddm_results.rds):
#   Rscript sbc/run_precision_study_ddm.R

source("sbc/precision_component.R")

N_CORES <- 9L
N_GRID <- c(100L, 200L, 400L, 800L)
N_REPS <- 4L
LOG <- "sbc/precision_ddm_progress.log"
OUT <- "sbc/precision_ddm_results.rds"

design_maker <- function(model) {
  design(factors = list(subjects = 1, S = c("left", "right")),
         Rlevels = c("left", "right"),
         formula = list(v ~ 1, a ~ 1, t0 ~ 1, s ~ 1, Z ~ 1, sv ~ 1,
                        SZ ~ 1, st0 ~ 1),
         constants = c(s = log(1)), model = model, verbose = FALSE)
}

# vignette prior with SZ mean moved inside the DDMnn training box
PRIOR_MEAN <- c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5),
                sv = log(.1), SZ = qnorm(.2), st0 = log(.05))
PRIOR_SD <- c(v = .15, a = .15, t0 = .1, Z = .05,
              sv = .15, SZ = .15, st0 = .15)

prior_maker <- function(des) {
  pnames <- names(sampled_pars(des))
  prior(des, type = "single", pmean = PRIOR_MEAN[pnames], psd = PRIOR_SD[pnames])
}

# analytic fits are ~12x slower per fit (25 min at N=100), so the analytic
# reference runs only at the two lowest N; the information model bridges.
specs <- list(list(label = "DDM", model = DDM, n_grid = c(100L, 200L)),
              list(label = "DDMnn", model = DDMnn))

n_fits <- nrow(make_precision_cells(specs, N_GRID, N_REPS))
cat(sprintf("Running %d fits on %d cores; progress -> %s\n",
            n_fits, N_CORES, LOG))
cat("", file = LOG)
t0 <- proc.time()[3]
res <- run_precision_study(specs, design_maker, prior_maker,
                           p_true = PRIOR_MEAN, n_grid = N_GRID,
                           n_reps = N_REPS, cores = N_CORES, log_file = LOG,
                           cache_dir = "sbc/cache_precision_ddm")
saveRDS(res, OUT)
cat(sprintf("\nDone in %.1f min; results -> %s\n", (proc.time()[3] - t0) / 60, OUT))

# ---------------- analysis ----------------
res_ok <- res[!res$failed, ]
n_fail <- sum(res$failed) / length(unique(res$par[!is.na(res$par)]))
if (n_fail > 0) cat(sprintf("NOTE: %g fit(s) failed\n", n_fail))

cat("\nMean posterior sd (across reps), per parameter and N:\n")
agg <- aggregate(post_sd ~ label + n_trials + par, data = res_ok,
                 FUN = function(x) sqrt(mean(x^2)))
for (p in unique(agg$par)) {
  sub <- agg[agg$par == p, ]
  cat(sprintf("  %-4s", p))
  for (n in N_GRID) {
    an <- sub$post_sd[sub$label == "DDM" & sub$n_trials == n]
    nn <- sub$post_sd[sub$label == "DDMnn" & sub$n_trials == n]
    if (length(nn))
      cat(sprintf("  N=%d: nn %.4f%s", n, nn,
                  if (length(an)) sprintf(" an %.4f (x%.2f)", an, nn / an) else ""))
  }
  cat("\n")
}

cat("\nInformation ratio kappa (nn / analytic, per trial) and crossover:\n")
ei <- effective_info(res_ok, ref = "DDM", target = "DDMnn", prior_sd = PRIOR_SD)
for (p in unique(ei$par)) {
  k <- ei$kappa[ei$par == p]
  cat(sprintf("  %-4s kappa by N: %s   median %.2f -> crossover ~%d trials\n",
              p, paste(sprintf("%.2f", k), collapse = " "),
              median(k), round(100 / max(median(k), 0.05))))
}
k_all <- median(ei$kappa)
cat(sprintf("\nOverall median kappa = %.2f -> DDMnn needs ~%.1fx trials",
            k_all, 1 / k_all))
cat(sprintf(" (crossover ~%d at base 100)\n", round(100 / k_all)))

cat("\nBias check (mean posterior-median bias / posterior sd, worst parameter):\n")
bias_agg <- aggregate(cbind(bias, post_sd) ~ label + par, data = res_ok, FUN = mean)
bias_agg$std_bias <- bias_agg$bias / bias_agg$post_sd
for (l in unique(bias_agg$label)) {
  sub <- bias_agg[bias_agg$label == l, ]
  worst <- sub[which.max(abs(sub$std_bias)), ]
  cat(sprintf("  %-6s worst |bias|/sd: %s %.2f\n", l, worst$par, worst$std_bias))
}

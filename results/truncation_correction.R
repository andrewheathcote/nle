# Truncation correction for DDM likelihood-approximation accuracy
# -----------------------------------------------------------------------------
# The DDM data are simulated with a hard RT ceiling (EMC2 rDDM default
# max_rt = 20 s; unfinished draws are *discarded*, not censored -- verified:
# no atom at the cap). The KDE/flow are fit on that truncated data, so they
# estimate the conditional density f(rt, R | rt <= cap, theta), whereas the
# stored analytic `ll` is the untruncated f(rt, R | theta). On a given draw the
# two differ by the constant offset -log P(RT <= cap | theta).
#
# This helper computes that per-parameter normalizer and adds a truncation-
# consistent reference `ll_trunc = ll - log P(RT <= cap | theta)` so the
# accuracy comparison puts all three methods on the same (truncated)
# distribution. It removes the normalization offset (effect 1); any residual
# gap -- including boundary error near the cap (effect 2) -- is genuine
# approximation behavior and is left intact.
#
# It does NOT modify the simulation/KDE/flow pipeline and writes only NEW files
# under results/truncation/. Run:
#   Rscript results/truncation_correction.R
# or source() it to use the functions directly.

suppressMessages({
  library(dplyr)
  library(readr)
  library(arrow)
  library(WienR)
})

# --- Parameter transform: sampled scale -> WienR natural scale ----------------
# Mirrors transform_DDM_params() in results/ll analysis.R, including EMC2's
# start-point reparameterization SZ -> 2*SZ*min(Z, 1-Z) (see EMC2:::DDM).
transform_DDM_params <- function(dat) {
  dat <- as.data.frame(dat)
  dat[, c("a", "t0", "sv", "st0", "s")] <- exp(dat[, c("a", "t0", "sv", "st0", "s")])
  dat[, c("Z", "SZ")] <- apply(dat[, c("Z", "SZ")], 2, pnorm)
  dat[, "SZ"] <- 2 * dat[, "SZ"] * pmin(dat[, "Z"], 1 - dat[, "Z"])
  dat
}

# --- P(RT <= t | theta) via the two defective CDFs ----------------------------
# WienR uses the s = 1 (diffusion-coefficient-free) parameterization, so the
# rate/threshold/variability are divided by s exactly as EMC2:::pDDM does.
# Returns both the probability `value` and the solver's own absolute-error
# estimate `err` (summed over the two boundary responses; worst-case bound).
prob_below <- function(nat, t, precision = 1e-20, n.evals = 50000) {
  args_at <- function(resp) WienR::pWDM(
    t   = rep(t, nrow(nat)), response = resp,
    a   = nat$a  / nat$s,  v  = nat$v / nat$s, w = nat$Z,
    t0  = nat$t0,          sv = nat$sv / nat$s, sw = nat$SZ,
    st0 = nat$st0, precision = precision, n.evals = n.evals
  )[c("value", "err")]
  lo <- args_at("lower"); up <- args_at("upper")
  list(value = lo$value + up$value, err = lo$err + up$err)
}

# --- Per-param normalizer with convergence diagnostics ------------------------
# Returns one row per param_id with:
#   P_below_cap : analytic mass captured below the cap
#   logZ        : log P(RT <= cap | theta)   (the shift to subtract from `ll`)
#   logZ_err    : error budget on logZ = -log P, from WienR's own absolute error
#                 estimate. Since logZ is a log, the relevant quantity is the
#                 RELATIVE error err_P / P (absolute error propagates through the
#                 log as a relative one). This bounds how many nats logZ carries.
#   converged   : TRUE iff logZ_err < tol (logZ trustworthy to `tol` nats), with
#                 a cheap far-horizon mass ~ 1 as a secondary guard against the
#                 solver's err estimate being optimistic in extreme regimes.
compute_logZ <- function(params_sampled, cap = 20,
                         precision = 1e-20, n.evals = 50000, tol = 1e-3) {
  stopifnot("param_id" %in% names(params_sampled))
  pid <- params_sampled$param_id
  nat <- transform_DDM_params(params_sampled[, c("v", "a", "t0", "s",
                                                 "Z", "SZ", "sv", "st0")])

  cap_res <- prob_below(nat, cap, precision, n.evals)
  far_res <- prob_below(nat, 1e4, precision, n.evals)            # sanity ~ 1
  p_cap   <- cap_res$value

  p_cap_c  <- pmin(pmax(p_cap, .Machine$double.eps), 1)          # clamp for log
  logZ_err <- cap_res$err / p_cap_c                              # rel. err = err on logZ
  converged <- logZ_err < tol & abs(far_res$value - 1) < tol

  data.frame(
    param_id    = pid,
    P_below_cap = p_cap,
    P_err       = cap_res$err,
    P_far       = far_res$value,
    logZ        = log(p_cap_c),
    ll_bias     = -log(p_cap_c),
    logZ_err    = logZ_err,
    converged   = converged,
    a_over_s    = nat$a / nat$s
  )
}

# --- Attach the truncation-consistent reference to a prediction table ---------
# pred_df must have columns `param_id` and `ll`. Adds `ll_trunc` and a
# `logZ_converged` flag (so non-converged normalizers can be excluded
# downstream rather than silently corrupting the corrected metric).
add_ll_trunc <- function(pred_df, logZ_df) {
  pred_df %>%
    left_join(logZ_df %>% select(param_id, logZ, converged),
              by = "param_id") %>%
    mutate(ll = ll - logZ) %>%
    rename(logZ_converged = converged)
}

# --- Script entry point -------------------------------------------------------
if (sys.nframe() == 0) {
  CAP        <- 20
  MODEL      <- "DDM"
  BUDGETS    <- c("small", "medium", "large")
  PARAMS_SRC <- file.path("data", "test", MODEL, "parameters.parquet")
  OUT_DIR    <- file.path("results", "truncation", MODEL)
  dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

  message(sprintf("Computing log P(RT <= %g) per param from %s ...", CAP, PARAMS_SRC))
  params <- as.data.frame(read_parquet(PARAMS_SRC))
  logZ   <- compute_logZ(params, cap = CAP)

  write_csv(logZ, file.path(OUT_DIR, sprintf("logZ_by_param_cap%g.csv", CAP)))
  message(sprintf("  %d params | %d flagged non-converged | median bias %.2e, max %.3f nats",
                  nrow(logZ), sum(!logZ$converged),
                  median(logZ$ll_bias), max(logZ$ll_bias)))
  if (any(!logZ$converged))
    message("  non-converged param_ids: ",
            paste(logZ$param_id[!logZ$converged], collapse = ", "))

  # Augment each prediction table into a NEW file (originals untouched).
  for (method in c("flow", "kde")) {
    for (b in BUDGETS) {
      src <- file.path("results", MODEL, b, sprintf("%s_ll_pred.csv", method))
      if (!file.exists(src)) { message("  skip (missing): ", src); next }
      out <- add_ll_trunc(read_csv(src, show_col_types = FALSE), logZ)
      dst_dir <- file.path(OUT_DIR, b)
      dir.create(dst_dir, recursive = TRUE, showWarnings = FALSE)
      dst <- file.path(dst_dir, sprintf("%s_ll_pred.csv", method))
      write_csv(out, dst)
      message(sprintf("  wrote %s (%d rows)", dst, nrow(out)))
    }
  }
  message("Done. Compare *_ll_pred against ll_trunc; drop rows with !logZ_converged.")
}

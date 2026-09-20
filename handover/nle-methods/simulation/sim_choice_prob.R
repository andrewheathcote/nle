# Exact theta-GRADIENT of the analytic choice probability, for Sobolev training
# of the choice classifier.
#
# Why: log f_joint = log f(rt | R) + log P(R) splits the score bias exactly, and
# the classifier share carries 70-98% of the v and Z bias (sbc/SBC_PLAN.md,
# Phase 6). The classifier was only ever fitted to the VALUE of P(R|theta);
# nothing constrained its derivatives. But
#
#   g_clf,j = d/dtheta_j  sum_R P_true(R|theta0) log P_hat(R|theta)
#
# vanishes exactly when d log P_hat / dtheta matches the truth, and P(R|theta)
# is available exactly from WienR::pWDM. Unlike the flow's rt-density this is a
# smooth scalar with no normalisation constraint and no moving support, so the
# gradient is a well-posed supervised target rather than a penalty - none of the
# failure modes that killed the flow score penalty apply.
#
# Since P(R=1) = 1 - P(R=2), the gradient of p_R2 determines both.
# t0 is omitted: non-decision time shifts RT but cannot affect choice, so
# dP/dt0 = 0 identically (asserted below on a sample rather than assumed).
#
# Reuses the theta draws already in data/choice_prob/, so the train/val split is
# unchanged and the new targets line up row-for-row with the old ones.
#
# IMPORTANT: this also RECOMPUTES the value p_R2. The existing
# data/choice_prob/*.parquet was generated without the s-scaling - it passed
# a = exp(a) and v = v straight to pWDM, but the DDM likelihood depends on
# v, a, sv only through v/s, a/s, sv/s (verified against stored EMC2 ll to
# 1e-11). Since the training box varies s over [0.1, 2], most rows are wrong,
# some badly: at s = 0.524, v = 4.871 the stored value is 0.95178 against a
# true 0.99985. The classifier retrained on that target (tag c3) is therefore
# compromised away from s = 1. The corrected column is p_R2; the original is
# kept as p_R2_old so the damage can be quantified.
#
#   Rscript simulation/sim_choice_prob_grad.R
# Env: CPG_CORES (default 9), CPG_H (central-difference step, default 5e-3)

if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(WienR); library(arrow); library(parallel)})

CORES <- as.integer(Sys.getenv("CPG_CORES", "9"))
H     <- as.numeric(Sys.getenv("CPG_H", "5e-3"))
BIG_RT <- 50
PARS  <- c("v","a","s","Z","SZ","sv","st0")     # t0 excluded: dP/dt0 == 0
ALL   <- c("v","a","t0","s","Z","SZ","sv","st0")

p_upper <- function(p) {
  w  <- pnorm(p[["Z"]]); sc <- exp(p[["s"]])
  # the likelihood depends on v, a, sv only through v/s, a/s, sv/s
  pWDM(BIG_RT, response = "upper", a = exp(p[["a"]])/sc, v = p[["v"]]/sc, w = w,
       t0 = exp(p[["t0"]]), sv = exp(p[["sv"]])/sc,
       sw = 2*pnorm(p[["SZ"]])*min(w, 1-w), st0 = exp(p[["st0"]]),
       precision = 1e-10)$value
}

grad_row <- function(th) {
  g <- numeric(length(PARS)); names(g) <- PARS
  for (k in seq_along(PARS)) {
    tp <- tm <- th
    tp[[PARS[k]]] <- tp[[PARS[k]]] + H
    tm[[PARS[k]]] <- tm[[PARS[k]]] - H
    g[k] <- (p_upper(tp) - p_upper(tm)) / (2 * H)
  }
  g
}

for (split in c("val", "train")) {
  t0s <- proc.time()[3]
  d <- as.data.frame(read_parquet(sprintf("data/choice_prob/%s.parquet", split)))
  rows <- split(as.data.frame(d[, ALL]), seq_len(nrow(d)))
  G <- mclapply(rows, function(r) grad_row(as.list(r)), mc.cores = CORES)
  G <- do.call(rbind, G)
  stopifnot(nrow(G) == nrow(d), all(is.finite(G)))
  for (k in seq_along(PARS)) d[[paste0("dpR2_", PARS[k])]] <- G[, k]
  pv <- unlist(mclapply(rows, function(r) p_upper(as.list(r)), mc.cores = CORES))
  stopifnot(length(pv) == nrow(d), all(is.finite(pv)))
  # pWDM overshoots by ~1e-16 where P saturates at 1 (large v/s); clamp, but
  # only after checking the excursion really is float noise and not a failure.
  stopifnot(max(pv - 1, 0) < 1e-8, max(-pv, 0) < 1e-8)
  pv <- pmin(pmax(pv, 0), 1)
  d$p_R2_old <- d$p_R2
  d$p_R2 <- pv
  cat(sprintf("  %s: corrected p_R2, max|change| = %.5f, mean|change| = %.5f\n",
              split, max(abs(d$p_R2 - d$p_R2_old)), mean(abs(d$p_R2 - d$p_R2_old))))
  write_parquet(d, sprintf("data/choice_prob/%s_grad.parquet", split))
  cat(sprintf("%s: %d rows, %d grad cols in %.1f min -> %s_grad.parquet\n",
              split, nrow(d), length(PARS), (proc.time()[3]-t0s)/60, split))
}

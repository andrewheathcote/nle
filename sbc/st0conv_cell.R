# S2 of the st0-convolution test (2026-10-13; rating-work/rtcon2/SUGGEST_DDM_ST0_CONVOLUTION.md):
# one SBC cell in which st0 is removed from the network's job. The network v1 (zb_s903_e271) is
# evaluated at st0 = 0, and the uniform st0 window [t0, t0 + st0] (WienR's lower-edge convention)
# is integrated EXACTLY by 16-node Gauss-Legendre quadrature in rt, over the support only:
#     L(rt) = (1/st0) * integral_0^min(st0, rt - t0) f_net(rt - u; theta, st0 = 0) du
# The network's context keeps the true t0 (the "shift:" form of sbc/st0conv_screen.R; S1 sum|g|
# about .010 in SZ0 against v2's .037).
#
# The smear lives in an R-level wrapper model: the registered v1 model with c_name = NULL, so
# EMC2 takes its R likelihood path, and dfun replaced by the quadrature. The final-sampler build is
# untouched; the sampler, prior, design and SBC machinery are those of zerobox_s4_cell_8case.R.
# The matched analytic control for this regime already exists (sbc/nci/results_8case/<regime>,
# same prior, design and N), so this is a network-only pass (run_control = FALSE).
#
# Prediction, declared before running (case 3, SZ0, N = 400): st0 drops from v2's +0.50
# (KS 2e-20) to about case 4's level (|std bias| <= 0.2); a stays at about v1's st0 = 0 level (case 4
# a -0.10); t0 is not predicted (S1 cannot see it).
#
#   Rscript sbc/st0conv_cell.R <regime> [replicates] [trials]      regime: SZ0 | sv0 | full | st0only
# Env: LIB, CPC (default 2), CFC (default 3), OUT (default sbc/nci/results_st0conv), BENCH=1 (speed only)
a <- if (exists("ARGS")) ARGS else commandArgs(trailingOnly = TRUE)
REGIME <- a[1]; REPS <- if (length(a) > 1) as.integer(a[2]) else 500L
TRIALS <- if (length(a) > 2) as.integer(a[3]) else 400L
LIB <- path.expand(Sys.getenv("LIB", "~/R/nle-lib"))
.libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages(library(EMC2))
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
CPC <- as.integer(Sys.getenv("CPC", "2")); CFC <- as.integer(Sys.getenv("CFC", "3"))
OUT <- file.path(Sys.getenv("OUT", "sbc/nci/results_st0conv"), REGIME)
CARD <- "port/export/ddm_zb_s903_e271_weights.json"
PIN <- switch(REGIME, full = c(), sv0 = c(sv = -Inf), SZ0 = c(SZ = -Inf), st0only = c(sv = -Inf, SZ = -Inf),
              stop("regime must have st0 free"))
cat(sprintf("=== st0-convolution S2 cell %s  v1 + exact smear  N=%d reps=%d  cores %dx%d  EMC2 %s  %s ===\n", REGIME,
            TRIALS, REPS, CPC, CFC, as.character(packageVersion("EMC2")), format(Sys.time(), "%Y-%m-%d %H:%M")))

PM <- c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5), sv = log(.1), SZ = qnorm(.2), st0 = log(.05))
PS <- c(v = .15, a = .15, t0 = .1, Z = .05, sv = .15, SZ = .15, st0 = .15)
FREE <- setdiff(names(PM), names(PIN))
mk <- register_nn_model(CARD, twin = DDM)
cell <- nn_cell(mk, mean = PM[FREE], sd = PS[FREE],
                formula = lapply(FREE, function(p) as.formula(paste(p, "~ 1"))),
                constants = if (length(PIN)) PIN else NULL,
                factors = list(S = c("left", "right")), Rlevels = c("left", "right"))

# ---- the wrapper model ------------------------------------------------------
GL <- local({ n <- 16; b <- seq_len(n - 1) / sqrt(4 * seq_len(n - 1)^2 - 1)
  J <- matrix(0, n, n); J[cbind(1:(n - 1), 2:n)] <- b; J[cbind(2:n, 1:(n - 1))] <- b
  e <- eigen(J, symmetric = TRUE); list(x = (e$values + 1) / 2, w = e$vectors[1, ]^2) })
orig_model <- cell$design$model           # the design's prepared model closure (not mk itself)
conv_model <- function() {
  m <- orig_model()
  d0 <- m$dfun
  reg <- environment(d0)$reg
  stopifnot(is.null(reg$pre), !reg$kind %in% EMC2:::nn_mlp_kinds)
  m$c_name <- NULL                        # R likelihood path (log_likelihood_ddm calls model$dfun)
  m$dfun <- function(rt, R, pars) {
    st0 <- pars[, "st0"]; P0 <- pars; P0[, "st0"] <- 0
    out <- numeric(length(rt))
    z <- st0 <= 0
    if (any(z)) out[z] <- d0(rt[z], R[z], P0[z, , drop = FALSE])
    if (any(!z)) { i <- !z
      hi <- pmax(pmin(st0[i], rt[i] - pars[i, "t0"]), 0); n <- sum(i); K <- length(GL$x)
      # All K nodes in ONE evaluator call (node-major stacking). This is nn_eval_joint's "pdf" path
      # (reg$pre is NULL, so the network time is rt); the context is built once per trial, not per node.
      tn <- rep(rt[i], K) - rep(GL$x, each = n) * rep(hi, K)
      th <- EMC2:::nn_context(P0[i, , drop = FALSE], reg)[rep(seq_len(n), K), , drop = FALSE]
      f <- numeric(n * K); ok <- tn > 0
      f[ok] <- EMC2:::ddm_ens_eval_trials_cpp(EMC2:::nn_ptr(reg), th[ok, , drop = FALSE], tn[ok],
                                              rep(as.integer(R[i]), K)[ok])[["pdf"]]
      out[i] <- colSums(GL$w * matrix(f, K, n, byrow = TRUE)) * hi / st0[i] }
    out }
  m
}
cell$design$model <- conv_model
mp <- mapped_pars(cell$design, cell$mean)
for (p in names(PIN)) stopifnot(all(mp[, p] == 0))
stopifnot(all(mp[, "st0"] > 0))

# ---- checks: the wrapper against the analytic DDM on one data set ------------
set.seed(7)
dat <- suppressMessages(make_data(cell$mean, cell$control_design, n_trials = TRIALS / 2))
X <- EMC2:::nn_cell_draws(cell, 200)
dc <- EMC2:::nn_dadm(dat, cell$control_design)
dn <- EMC2:::nn_dadm(dat, cell$design)
dv <- EMC2:::nn_dadm(dat, { d <- cell$design; d$model <- orig_model; d })
tm <- function(dadm, model) { invisible(EMC2:::calc_ll_manager(X[1:5, , drop = FALSE], dadm, model))
  t <- system.time(ll <- as.numeric(EMC2:::calc_ll_manager(X, dadm, model)))[["elapsed"]]; list(ll = ll, ms = 1000 * t / nrow(X)) }
A <- tm(dc, cell$control_design$model); C <- tm(dn, conv_model); V <- tm(dv, orig_model)
cat(sprintf("per-likelihood ms (N=%d, single thread): analytic %.2f | v1 native %.2f | v1 + smear (R level) %.2f\n",
            nrow(dat), A$ms, V$ms, C$ms))
cat(sprintf("LL vs analytic over %d prior draws: cor native %.4f, smear %.4f; mean diff native %+.2f, smear %+.2f\n",
            nrow(X), cor(A$ll, V$ll), cor(A$ll, C$ll), mean(V$ll - A$ll), mean(C$ll - A$ll)))
if (Sys.getenv("BENCH") == "1") quit(save = "no")

out <- nn_sbc_cell(cell, trials = TRIALS, replicates = REPS, run_control = FALSE, archive_dir = OUT,
                   cores_per_chain = CPC, cores_for_chains = CFC,
                   info = list(regime = REGIME, card = basename(CARD), pinned = PIN,
                               why = "S2: v1 at st0 = 0 with an exact quadrature smear over st0 (R-level wrapper model)"))
cat(sprintf("\n=== cell %s done %s ===\n", REGIME, format(Sys.time(), "%Y-%m-%d %H:%M")))
print(out$summary, digits = 3)
saveRDS(out$summary, file.path(OUT, "summary.rds"))

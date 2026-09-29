# Score bias for a RACE model (RDM), averaged over SBC prior draws.
#
# The DDM version factorises as P(R|theta) x f(rt|theta,R) and needs a choice
# classifier. A race has no classifier: the response probability emerges from
#
#     L(rt, R | theta) = f_R(rt) * prod_{j != R} S_j(rt)
#
# so the flow models ONE accumulator's density and the survivors come from the
# same network. port/R/flow_race.R returns log_pdf and log_sf together, which is
# what makes this cheap.
#
# Estimator, as in sbc/score_bias_draws.R:
#   g = d/dtheta E_true[log L_hat] at the truth, central difference, with the
#   expectation over a log-spaced rt grid weighted by the TRUE race density
#   times the trapezoid widths. Omitting the widths is not an expectation and
#   biases g badly.
# Validation: the same estimator on the ANALYTIC race likelihood must give ~0.
#
# t0 is handled separately and needs care. The RDM has NO start-time variability
# (no st0), so the support boundary at t0 is a hard jump — worse than the DDM,
# where st0 smeared it. Raising t0 puts zero analytic density where the truth
# has mass, so E_true[log L] = -Inf and the derivative does not exist. As in
# sbc/score_bias_t0.R we restrict to A = [c, Inf) with c above the largest
# perturbed onset, where the target is not zero but d/dt0 log P(A | theta),
# computable from the race CDF. Reported as g_flow(A) - g_analytic(A).
#
#   Rscript sbc/score_bias_race.R <n_draws> <tag1> [tag2 ...]
# Tags are export basenames, e.g. rdm_large -> port/export/rdm_large_weights.json
# Env: SBR_CORES (default 4)
#      SBR_CELL  "std" (default): the standard RDM cell, A sampled.
#                "A0": the comparison cell of MALTE_CRDM_PLAN.md section 3.1,
#                the standard cell minus A. The truth and the analytic control
#                have A = 0 exactly (the Wald); a flow that conditions on A is
#                evaluated at A_FLOOR = 1e-4, the edge of its box; a flow
#                without A (Malte's, threshold b = B) needs nothing.
#      SBR_LIB   library to prepend, holding the EMC2 that supplies the analytic
#                density (default: ./.Rlibs when present). Asserted.
#      SBR_OUT   results file (default sbc/score_bias_race_results.rds)
#      SBR_CSV   if set, also write the score-bias table there, long format:
#                cell, tag, par, g, se, n   (t0 relative to the analytic value)
#      SBR_EXPORT  directory of the cards (default port/export)
#      SBR_SKIP  drop the first k prior draws (default 0: the screen's draws)
#
# Card options understood (all evaluated by port/R/flow_race.R):
#   pre = "t0"                 the flow takes rt - t0, and t0 is not an input
#   context_scale = "natural"  inputs on the natural scale (Malte's cards);
#                              otherwise log, as every card of ours
#   an input named b           the threshold, b = B + A
#   context_transforms "zerobox"  the input is asinh(x / zerobox_c); such a
#                              card's box reaches 0, so in the A0 cell it is
#                              evaluated at A = 0 exactly, not at A_FLOOR

SBR_LIB <- Sys.getenv("SBR_LIB", "")
if (nzchar(SBR_LIB)) {
  .libPaths(c(normalizePath(path.expand(SBR_LIB)), .libPaths()))
} else if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
source("port/R/flow_race.R")
suppressPackageStartupMessages({library(EMC2); library(parallel)})
if (nzchar(SBR_LIB))
  stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(path.expand(SBR_LIB))))
CELL <- Sys.getenv("SBR_CELL", "std")
stopifnot(CELL %in% c("std", "A0"))
A_FLOOR <- 1e-4
EXPORT <- Sys.getenv("SBR_EXPORT", "port/export")

args <- commandArgs(trailingOnly = TRUE)
n_draws <- as.integer(args[1]); tags <- args[-1]
CORES <- as.integer(Sys.getenv("SBR_CORES", "4"))
# H = 0.01, not the DDM script's 0.05. The central difference carries O(H^2)
# truncation error, and the RDM's log-likelihood is curvier in these parameters:
# the analytic control for v is -0.00316 at H=0.05, -0.00051 at 0.02, -0.00013
# at 0.01 and -0.00003 at 0.005 — textbook quadratic convergence. At 0.05 the
# control was 25-40% of the flow signal, i.e. the estimator's own bias was a
# quarter of what it was meant to measure. At 0.01 it is ~1%.
H <- as.numeric(Sys.getenv("SBR_H", "0.01"))
PARS <- c("v", "v_lMTRUE", "B", "A")     # t0 done separately, see below
if (CELL == "A0") PARS <- setdiff(PARS, "A")

fls <- lapply(tags, function(t) flow_load(file.path(EXPORT, sprintf("%s_weights.json", t))))
names(fls) <- tags

# theta on the SBC sampled scale -> the two accumulators' natural parameters.
# v ~ lM with treatment coding: the mismatching accumulator gets v, the matching
# one v + v_lMTRUE. B, A, t0 are shared; s is a constant 1.
acc_pars <- function(th) {
  v1 <- exp(th[["v"]])                      # lM = FALSE
  v2 <- exp(th[["v"]] + th[["v_lMTRUE"]])   # lM = TRUE
  base <- c(B = exp(th[["B"]]), A = if (CELL == "A0") 0 else exp(th[["A"]]),
            t0 = exp(th[["t0"]]), s = 1)
  list(cbind(v = v1, t(base))[rep(1, 1), , drop = FALSE],
       cbind(v = v2, t(base))[rep(1, 1), , drop = FALSE])
}

# analytic single-accumulator density / CDF (EMC2 rescales by s internally)
an_d <- function(p, rt) EMC2:::dRDM(rt, p[rep(1, length(rt)), , drop = FALSE])
an_p <- function(p, rt) EMC2:::pRDM(rt, p[rep(1, length(rt)), , drop = FALSE])

# log race likelihood under the ANALYTIC model
an_logL <- function(th, rt, R) {
  P <- acc_pars(th)
  d <- an_d(P[[R]], rt)
  s <- 1 - an_p(P[[3 - R]], rt)
  log(pmax(d, 0)) + log(pmax(s, 0))
}

# log race likelihood under a FLOW. theta for the flow is the single-accumulator
# context on the SAMPLED scale in the export's order.
fl_logL <- function(fl, th, rt, R) {
  P <- acc_pars(th)
  cn <- unlist(fl$context_names)
  natural <- identical(fl$context_scale, "natural")
  bnd <- if (natural) fl$bounds_natural else fl$bounds_sampled
  zb <- names(fl$zerobox_c)
  mk <- function(p) sapply(cn, function(nm) {
    if (nm %in% zb) return(asinh(p[1, nm] / fl$zerobox_c[[nm]]))   # exact at 0
    x <- if (nm == "b") p[1, "B"] + p[1, "A"]          # threshold
         else if (nm == "A") max(p[1, "A"], A_FLOOR)    # binds in the A0 cell only
         else p[1, nm]
    x <- if (natural) x else log(x)                     # ours: all exp-transformed
    # A_FLOOR is the box edge, and log(1e-4) in R is 2e-15 below the card's
    # bound (numpy's log): without this every A0 evaluation is out of the box
    if (nm == "A" && CELL == "A0") x <- max(x, bnd$lower[match("A", cn)])
    x
  })
  # a decision-time flow takes rt - t0; t0 is shared by the accumulators
  t_arg <- if (identical(fl$pre, "t0")) rt - P[[1]][1, "t0"] else rt
  ew <- flow_eval(fl, mk(P[[R]]), t_arg)
  el <- flow_eval(fl, mk(P[[3 - R]]), t_arg)
  ew$log_pdf + el$log_sf
}

Lbar <- function(th, grid, f) {
  num <- 0; den <- 0
  for (R in 1:2) {
    lp <- f(th, grid$rt, R); w <- grid$w[[R]]
    ok <- is.finite(lp) & is.finite(w) & w > 0
    if (any(ok)) { num <- num + sum(w[ok] * lp[ok]); den <- den + sum(w[ok]) }
  }
  if (den <= 0) NA_real_ else num / den
}
gslope <- function(th0, par, grid, f) {
  tp <- th0; tm <- th0
  tp[[par]] <- tp[[par]] + H; tm[[par]] <- tm[[par]] - H
  (Lbar(tp, grid, f) - Lbar(tm, grid, f)) / (2 * H)
}

x <- readRDS("sbc/nci/results/RDMnn_N100_R500.rds")
truth <- as.matrix(x$SBC$med$alpha) - as.matrix(x$SBC$bias$alpha)
# A0: the same prior draws for the remaining parameters
if (CELL == "A0") truth <- truth[, colnames(truth) != "A", drop = FALSE]
# SBR_SKIP = k drops the first k draws, so a report can use draws the screen never saw
SKIP <- as.integer(Sys.getenv("SBR_SKIP", "0"))
if (SKIP > 0) truth <- truth[-seq_len(SKIP), , drop = FALSE]
n_draws <- min(n_draws, nrow(truth))
cat(sprintf("cell: %s   draws: %d   tags: %s   cores: %d\n\n", CELL, n_draws,
            if (length(tags) > 12) sprintf("%d tags", length(tags)) else paste(tags, collapse = ", "),
            CORES))

one <- function(i) {
  th0 <- as.list(truth[i, ])
  t0n <- exp(th0[["t0"]])
  # full-support grid for the ordinary parameters
  rt <- t0n + exp(seq(log(1e-4), log(100), length.out = 400))
  dr <- c(diff(rt)[1], (head(diff(rt), -1) + tail(diff(rt), -1)) / 2, tail(diff(rt), 1))
  grid <- list(rt = rt, w = lapply(1:2, function(R) exp(an_logL(th0, rt, R)) * dr))
  # unlist(lapply(...)), NOT c(sapply(...)): sapply returns a matrix here and
  # c() on a matrix discards the dimnames, leaving unnamed columns.
  out <- unlist(lapply(PARS, function(p) {
    v <- c(sapply(tags, function(tg)
             gslope(th0, p, grid, function(th, rt, R) fl_logL(fls[[tg]], th, rt, R))),
           analytic = gslope(th0, p, grid, an_logL))
    stats::setNames(v, paste0(p, ".", names(v)))
  }))
  # t0: restricted region A = [c, Inf), target d/dt0 log P(A)
  cut <- t0n * exp(H) * 1.05
  rt2 <- cut + exp(seq(log(1e-4), log(100), length.out = 400))
  dr2 <- c(diff(rt2)[1], (head(diff(rt2), -1) + tail(diff(rt2), -1)) / 2, tail(diff(rt2), 1))
  g2 <- list(rt = rt2, w = lapply(1:2, function(R) exp(an_logL(th0, rt2, R)) * dr2))
  t0v <- c(sapply(tags, function(tg)
             gslope(th0, "t0", g2, function(th, rt, R) fl_logL(fls[[tg]], th, rt, R))),
           analytic = gslope(th0, "t0", g2, an_logL))
  t0v <- stats::setNames(t0v, paste0("t0.", names(t0v)))
  c(out, t0v)
}

res <- do.call(rbind, mclapply(seq_len(n_draws), one, mc.cores = CORES))
ok <- apply(is.finite(res), 1, all)
cat(sprintf("usable draws: %d / %d\n\n", sum(ok), n_draws))
res <- res[ok, , drop = FALSE]

cat("VALIDATION — the estimator on the ANALYTIC likelihood must be ~0\n")
for (p in PARS) {
  v <- res[, paste0(p, ".analytic")]
  cat(sprintf("  %-10s %+.5f (se %.5f)\n", p, mean(v), sd(v) / sqrt(length(v))))
}
cat("  (t0 is on a restricted region, so its analytic value is NOT 0 by design;\n")
cat("   the flow is reported relative to it below)\n\n")

cat("SCORE BIAS\n")
cat(sprintf("%-16s %s\n", "tag",
            paste(sprintf("%11s", c(PARS, "t0")), collapse = " ")))
for (tg in tags) {
  vals <- c(sapply(PARS, function(p) mean(res[, paste0(p, ".", tg)])),
            mean(res[, paste0("t0.", tg)] - res[, "t0.analytic"]))
  cat(sprintf("%-16s %s\n", tg, paste(sprintf("%+11.5f", vals), collapse = " ")))
}
saveRDS(res, Sys.getenv("SBR_OUT", "sbc/score_bias_race_results.rds"))

CONTROL <- max(abs(sapply(PARS, function(p) mean(res[, paste0(p, ".analytic")]))))
cat(sprintf("\nCONTROL max |g(analytic)| over %s: %.5f\n", paste(PARS, collapse = ", "), CONTROL))
if (nzchar(Sys.getenv("SBR_CSV"))) {
  se <- function(v) sd(v) / sqrt(length(v))
  tab <- do.call(rbind, lapply(tags, function(tg) {
    g <- lapply(PARS, function(p) res[, paste0(p, ".", tg)])
    g[["t0"]] <- res[, paste0("t0.", tg)] - res[, "t0.analytic"]
    data.frame(cell = CELL, tag = tg, par = c(PARS, "t0"), g = sapply(g, mean),
               se = sapply(g, se), n = nrow(res), control = CONTROL, row.names = NULL)
  }))
  write.csv(tab, Sys.getenv("SBR_CSV"), row.names = FALSE)
  cat(sprintf("wrote %s (%d rows)\n", Sys.getenv("SBR_CSV"), nrow(tab)))
}

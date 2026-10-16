# S1 of the st0-convolution test (2026-10-13; rating-work/rtcon2/SUGGEST_DDM_ST0_CONVOLUTION.md).
# COPY of sbc/zerobox_screen.R. Its estimator, truths and report are unchanged. The additions are:
#   * tag "conv:<tag>": the network evaluated at st0 = 0 and smeared EXACTLY over the uniform st0
#     window by 16-node Gauss-Legendre quadrature over t0 in [t0, t0 + st0]. t0 is the lower edge,
#     as in WienR; this is checked below. With st0 = 0 the tag reduces to the plain network.
#   * tag "conv:analytic": the same quadrature applied to the analytic DDM. It must reproduce the
#     analytic density (this checks the window convention and the quadrature), and its g is a
#     second control.
#   * tag "shift:<tag>": as conv:, but the network's context stays at the true t0 and the quadrature
#     shifts rt instead (f0(rt - u; t0)). This is the form an SBC wrapper model can evaluate cheaply,
#     and the form a decision-time network would take (added 2026-10-13 for S2).
#   * regime st0only (sv and SZ pinned at 0). Truths are capped at n_draws.
# Prediction, declared before running: for conv:zb_s903_e271 in the full, sv0, SZ0 and st0only
# regimes, g_st0 falls well below v2's (zb2_s901_e274), toward the control. g_a improves on v2 in
# sv0 and SZ0. The other parameters stay at about v1's st0 = 0 level (st00 regime).
# t0 is not differenced (moving support), so this screen says nothing about t0.
#
# Zero-box S3 screening: score bias per checkpoint, PER REGIME, through
# dev-nle's own evaluator (the one the SBC cells will use).
#
# sbc/score_bias_draws.R differences every variability parameter at the truth;
# at an exact zero that is meaningless (the parameter is a named constant in
# the SBC cell, not sampled) and the compiled port it uses predates the
# zero-box encoding. This script keeps that validated estimator -- central
# difference of E_true[log f] over a density-weighted rt grid, H = 0.01, with
# the analytic likelihood as the control that must return ~0 -- but:
#   * takes a REGIME: which of sv, SZ, st0 are exactly zero (or, for Q3, st0
#     pinned in the gap below the old floor), and differences only the
#     parameters the SBC cell of that regime samples;
#   * evaluates each network through register_nn_model(card, twin = DDM)$dfun
#     on the dev-nle install, so the screen and the cell share one evaluator
#     (gate: dfun == R port to 1e-13, zero regimes included).
# Truths are the SBC prior draws (the 500 stored with the N = 400 cell), with
# the regime's zeros imposed, so g is averaged over the region SBC will test.
#
# Selection rule, declared before any number is read (RECIPE.md §3; within an
# architecture g_a ranks checkpoints at rho = -0.989 with SBC): rank on
# mean |g_a| across the five zero regimes (full, sv0, SZ0, st00, all0), ties on
# the sum of |g| over the sampled parameters; the top three go to the
# posterior-shift screen per regime (sbc/zerobox_postshift.R) before any SBC.
#
#   Rscript sbc/zerobox_screen.R <regime> <n_draws> <tag1> [tag2 ...]
#     regime: full | sv0 | SZ0 | st00 | all0 | gap1 | gap3  (gapK: st0 = K ms)
#     tag: cap256w_c4 (the shipped DDMnn), or an export name such as
#          zb_s901_e272 -> port/export/ddm_<tag>_weights.json
# Env: LIB (default ~/R/nle-sbc-lib), SB_CORES (default 8), SB_H (0.01),
#      SB_OUT (default sbc/zerobox_screen/<regime>.rds; merged if it exists)
a <- commandArgs(trailingOnly = TRUE)
if (length(a) < 3) stop("usage: zerobox_screen.R <regime> <n_draws> <tag> ...")
REGIME <- a[1]; n_draws <- as.integer(a[2]); tags <- a[-(1:2)]
LIB <- path.expand(Sys.getenv("LIB", "~/R/nle-lib"))
.libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(jsonlite); library(parallel); library(WienR)})
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
CORES <- as.integer(Sys.getenv("SB_CORES", "8"))
H <- as.numeric(Sys.getenv("SB_H", "0.01"))

## ---- the regime: which parameters are pinned, and at what -------------------
ZERO <- switch(REGIME, full = character(0), sv0 = "sv", SZ0 = "SZ", st00 = "st0", st0only = c("sv", "SZ"),
               all0 = c("sv", "SZ", "st0"), gap1 = character(0), gap3 = character(0),
               stop("unknown regime ", REGIME))
PIN <- c(sv = 0, SZ = 0, st0 = 0)[ZERO]                       # natural-scale constants
if (REGIME == "gap1") PIN <- c(st0 = 0.001)
if (REGIME == "gap3") PIN <- c(st0 = 0.003)
PARS <- setdiff(c("v", "a", "Z", "sv", "SZ", "st0"), names(PIN))  # t0 excluded: moving support
cat(sprintf("regime %s: pinned %s; differencing %s; H = %g; %d draws; cores %d\n", REGIME,
            if (length(PIN)) paste(names(PIN), PIN, sep = " = ", collapse = ", ") else "nothing",
            paste(PARS, collapse = " "), H, n_draws, CORES))

## ---- the networks, through dev-nle -----------------------------------------
build <- function(tg) {
  if (grepl("^(conv|shift):", tg)) { b <- sub("^(conv|shift):", "", tg); return(if (b == "analytic") NULL else build(b)) }
  f <- sprintf("port/export/ddm_%s_weights.json", tg)
  if (!file.exists(f)) f <- sprintf("handover/nle-methods/port/export/ddm_%s_weights.json", tg)
  if (!file.exists(f)) stop("no export for tag ", tg)
  register_nn_model(f, twin = DDM)()
}
base <- unique(setdiff(sub("^(conv|shift):", "", tags), "analytic"))
nets <- lapply(base, build); names(nets) <- base
# Gap regimes (st0 below the old 0.01 floor) are outside the shipped flow's box:
# dfun floors them at min_ll there, which shows up as an absurd g, so the gap
# rows are read only for zero-box cards (bounds_sampled lower = 0 for st0).

## ---- likelihoods on the NATURAL scale (dfun's scale) -----------------------
# th: named natural-scale vector v a t0 s Z SZ sv st0 (SZ = the fraction, as EMC2 has it)
nat_row <- function(th, n) matrix(th, n, length(th), byrow = TRUE, dimnames = list(NULL, names(th)))
an_lpdf <- function(th, rt, R) { w <- th[["Z"]]
  WienR::dWDM(rt, response = ifelse(R == 1, "lower", "upper"), a = th[["a"]], v = th[["v"]], w = w,
              t0 = th[["t0"]], sv = th[["sv"]], sw = 2 * th[["SZ"]] * min(w, 1 - w), st0 = th[["st0"]],
              precision = 1e-8)$logvalue }
nn_lpdf <- function(m, th, rt, R) log(m$dfun(rt, rep(as.integer(R), length(rt)), nat_row(th, length(rt))))
# Gauss-Legendre on [0, 1] (Golub-Welsch), 16 nodes.
GL <- local({ n <- 16; b <- seq_len(n - 1) / sqrt(4 * seq_len(n - 1)^2 - 1)
  J <- matrix(0, n, n); J[cbind(1:(n - 1), 2:n)] <- b; J[cbind(2:n, 1:(n - 1))] <- b
  e <- eigen(J, symmetric = TRUE); list(x = (e$values + 1) / 2, w = e$vectors[1, ]^2) })
# Exact uniform st0 smear of the st0 = 0 density f0: (1/st0) * integral_0^st0 f0(rt; t0 + u) du.
# f0 is zero for u > rt - t0, so each trial integrates over [0, min(st0, rt - t0)] (no kink inside).
# f0v(th, t0vec, rt, R): st0 = 0 density with a per-trial t0.
an_f0v <- function(th, t0v, rt, R) { th[["t0"]] <- 0; an_lpdf(th, rt - t0v, R) }   # t0 is a pure shift
nn_f0v <- function(m) function(th, t0v, rt, R) { P <- nat_row(th, length(rt)); P[, "t0"] <- t0v
  log(m$dfun(rt, rep(as.integer(R), length(rt)), P)) }
conv_lpdf <- function(f0v) function(th, rt, R) {
  st0 <- th[["st0"]]; th0 <- th; th0[["st0"]] <- 0
  if (st0 <= 0) return(f0v(th0, rep(th[["t0"]], length(rt)), rt, R))
  hi <- pmax(pmin(st0, rt - th[["t0"]]), 0); d <- 0
  for (k in seq_along(GL$x)) { lp <- f0v(th0, th[["t0"]] + hi * GL$x[k], rt, R)
    d <- d + GL$w[k] * ifelse(is.finite(lp), exp(lp), 0) }
  log(d * hi / st0) }
lpdf_of <- function(tg) {
  if (tg == "conv:analytic") return(conv_lpdf(an_f0v))
  m <- nets[[sub("^(conv|shift):", "", tg)]]
  if (startsWith(tg, "conv:")) return(conv_lpdf(nn_f0v(m)))
  if (startsWith(tg, "shift:")) return(conv_lpdf(function(th, t0v, rt, R) nn_lpdf(m, th, rt - (t0v - th[["t0"]]), R)))
  function(th, rt, R) nn_lpdf(m, th, rt, R) }
# Check, before anything runs: the analytic convolution matches dWDM with st0 > 0 (lower-edge window).
local({ th <- c(v = 1, a = .8, t0 = .3, s = 1, Z = .5, SZ = .2, sv = .1, st0 = .08)
  rt <- th[["t0"]] + c(.002, .01, .05, .079, .081, .2, .5, 1, 2)
  ex <- an_lpdf(th, rt, 1); keep <- ex > log(1e-6)   # log error is meaningless at ~1e-13 density
  e <- max(abs(conv_lpdf(an_f0v)(th, rt, 1)[keep] - ex[keep]))
  cat(sprintf("window check: max |log conv - log dWDM| = %.2e at st0 = 0.08 (lower-edge convention)\n", e))
  if (e > 1e-2) stop("st0 window convention or quadrature wrong") })

# Differencing is on EMC2's SAMPLED scale (log for a, sv, st0; probit for Z, SZ;
# identity for v), as score_bias_draws.R does, so g is comparable with the record.
to_nat <- function(s) c(v = s[["v"]], a = exp(s[["a"]]), t0 = exp(s[["t0"]]), s = 1, Z = pnorm(s[["Z"]]),
                        SZ = pnorm(s[["SZ"]]), sv = exp(s[["sv"]]), st0 = exp(s[["st0"]]))
Lbar <- function(th_s, grid, f) { num <- 0; den <- 0; th <- to_nat(th_s)
  for (R in 1:2) { lp <- f(th, grid$rt, R); w <- grid$w[[R]]
    ok <- is.finite(lp) & is.finite(w) & w > 0
    if (any(ok)) { num <- num + sum(w[ok] * lp[ok]); den <- den + sum(w[ok]) } }
  if (den <= 0) NA_real_ else num / den }
gslope <- function(th_s, par, grid, f) {
  tp <- th_s; tp[[par]] <- tp[[par]] + H; tm <- th_s; tm[[par]] <- tm[[par]] - H
  (Lbar(tp, grid, f) - Lbar(tm, grid, f)) / (2 * H) }

## ---- truths: the SBC prior draws, with the regime imposed ------------------
x <- readRDS("sbc/nci/results_ens5/DDMnn_N400_R500.rds")
truth <- as.matrix(x$SBC$med$alpha) - as.matrix(x$SBC$bias$alpha)   # sampled scale, v a t0 Z sv SZ st0
n_draws <- min(n_draws, nrow(truth))
pin_s <- c(sv = if ("sv" %in% names(PIN)) log(PIN[["sv"]]) else NA,
           SZ = if ("SZ" %in% names(PIN)) qnorm(PIN[["SZ"]]) else NA,
           st0 = if ("st0" %in% names(PIN)) log(PIN[["st0"]]) else NA)   # log(0) = -Inf, qnorm(0) = -Inf

one <- function(i) {
  tr <- truth[i, ]; th_s <- c(tr[c("v", "a", "t0", "Z", "SZ", "sv", "st0")])
  for (p in names(PIN)) th_s[[p]] <- pin_s[[p]]
  th <- to_nat(th_s); stopifnot(all(abs(th[names(PIN)] - PIN) <= 1e-12 * pmax(1, PIN)))  # exact zeros; the gap to fp
  rt <- th[["t0"]] + exp(seq(log(1e-3), log(8), length.out = 120))
  dr <- c(diff(rt)[1], (head(diff(rt), -1) + tail(diff(rt), -1)) / 2, tail(diff(rt), 1))
  grid <- list(rt = rt, w = lapply(1:2, function(R) exp(an_lpdf(th, rt, R)) * dr))
  g <- sapply(tags, function(tg) sapply(PARS, function(p)
    gslope(th_s, p, grid, lpdf_of(tg))))
  ctl <- sapply(PARS, function(p) gslope(th_s, p, grid, an_lpdf))
  list(g = matrix(g, length(PARS), length(tags), dimnames = list(PARS, tags)), ctl = ctl)
}
t_start <- proc.time()[3]
res <- mclapply(seq_len(n_draws), one, mc.cores = CORES)
bad <- sapply(res, inherits, "try-error") | sapply(res, is.null)
if (any(bad)) stop(sum(bad), " draws failed: ", conditionMessage(attr(res[[which(bad)[1]]], "condition")))
G <- array(NA_real_, c(n_draws, length(PARS), length(tags)), dimnames = list(NULL, PARS, tags))
A <- matrix(NA_real_, n_draws, length(PARS), dimnames = list(NULL, PARS))
for (i in seq_len(n_draws)) { G[i, , ] <- res[[i]]$g; A[i, ] <- res[[i]]$ctl }
cat(sprintf("done in %.1f min (%.1f s per tag-draw)\n", (proc.time()[3] - t_start) / 60, (proc.time()[3] - t_start) / (n_draws * length(tags))))

## ---- report: one line per tag; the control first -----------------------------
cat(sprintf("\n%-16s %s\n", "CONTROL (~0)", paste(sprintf(" %+8.5f        ", colMeans(A, na.rm = TRUE)), collapse = "")))
cat(sprintf("%-16s %s\n", "tag", paste(sprintf("%17s", PARS), collapse = "")))
slice <- function(tg) matrix(G[, , tg], ncol = length(PARS), dimnames = list(NULL, PARS))
for (tg in tags) {
  v <- slice(tg)
  cat(sprintf("%-16s %s   |g_a| %.5f  sum|g| %.5f\n", tg,
              paste(sprintf(" %+8.5f(%.4f)", colMeans(v, na.rm = TRUE), apply(v, 2, sd, na.rm = TRUE) / sqrt(n_draws)), collapse = ""),
              abs(mean(v[, "a"])), sum(abs(colMeans(v)))))
}
ctl_bad <- max(abs(colMeans(A, na.rm = TRUE)))
cat(sprintf("\nCONTROL max |g| = %.2e  (%s)\n", ctl_bad, if (ctl_bad < 5e-4) "PASS" else "FAIL: estimator invalid in this regime"))

## ---- save, merging with an earlier run of the same regime -------------------
OUT <- Sys.getenv("SB_OUT", sprintf("sbc/st0conv_screen/%s.rds", REGIME))
dir.create(dirname(OUT), showWarnings = FALSE, recursive = TRUE)
M  <- t(sapply(tags, function(tg) colMeans(slice(tg)))); colnames(M) <- PARS
SE <- t(sapply(tags, function(tg) apply(slice(tg), 2, sd) / sqrt(n_draws))); colnames(SE) <- paste0("se_", PARS)
tab <- data.frame(regime = REGIME, tag = tags, n = n_draws, M, SE, ctl_max = ctl_bad, row.names = NULL)
tab$abs_g_a <- abs(tab$a); tab$sum_abs_g <- rowSums(abs(tab[, PARS, drop = FALSE]))
if (file.exists(OUT)) { old <- readRDS(OUT)$table; old <- old[!old$tag %in% tags, ]
  tab <- rbind(old[, intersect(names(old), names(tab))], tab[, intersect(names(old), names(tab))]) }
saveRDS(list(regime = REGIME, pars = PARS, H = H, table = tab, G = G, A = A), OUT)
cat(sprintf("saved %s (%d tags)\n", OUT, nrow(tab)))

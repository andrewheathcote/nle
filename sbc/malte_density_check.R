# Density error of Malte Lueken's trained flows, done properly (stage M4, part B).
#
#   Rscript sbc/malte_density_check.R [cores] [part]      # from the repo root; part = 1, 2 or both
#
# Part 1  His own table reproduced: his 45 Wald and 540 CRDM parameter sets, his grid
#         (2000 log-spaced points on [1e-3, 4]), his formulas (KL with censored ends, W1, KS;
#         scripts/compare_neural_densities.py), but our evaluator (port/R/flow_race.R on the
#         converted cards) and our references (closed-form Wald; port/src/crdm.cpp, which
#         matched his solver to 1e-13). Compared set by set with density_comparison.nc.
#         Then the same with a Volterra reference at dt 1.25e-4, which says how much of the
#         dt 5e-4 numbers is the reference's own error.
# Part 2  The EMC2 session's first look (first_look.R) on the SAME parameter draws (same
#         seed, same order of runif calls), its crude number beside the proper one.
#         Proper: grid between the reference's 1e-10 and 1 - 1e-10 quantiles (Wald) or on
#         [1e-3, t_max] with t_max 6 (CRDM), trapezoid in t on a log grid, censored ends.
#
# The flow's argument is decision time (pre = "t0"), so t0 does not enter.
a <- commandArgs(trailingOnly = TRUE)
CORES <- if (length(a)) as.integer(a[1]) else 6L
PART <- if (length(a) > 1) a[2] else "both"
stopifnot(CORES <= 8L)
suppressPackageStartupMessages({library(Rcpp); library(parallel)})
source("port/R/flow_race.R")
sourceCpp("port/src/crdm.cpp")
EXP <- "port/export"
NC <- Sys.getenv("MALTE_NC_CSV", "sbc/malte_m4/malte_nc")     # CSVs written from density_comparison.nc
OUT <- "sbc/malte_m4"; dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
TINY <- 1e-300

wald_ref <- function(t, v, s, b) {
  lp <- log(b) - log(s) - 0.5 * log(2 * pi) - 1.5 * log(t) - (b - v * t)^2 / (2 * s^2 * t)
  z1 <- (v * t - b) / (s * sqrt(t)); z2 <- -(v * t + b) / (s * sqrt(t))
  # survival in log form: S = Phi(-z1) - exp(2vb/s^2) Phi(z2)
  la <- pnorm(-z1, log.p = TRUE); lb <- 2 * v * b / s^2 + pnorm(z2, log.p = TRUE)
  lS <- la + log1p(-pmin(exp(lb - la), 1))
  list(lp = lp, lF = log(pmax(-expm1(lS), TINY)), lS = lS)
}
volt_ref <- function(t, th, dt, t_max) {
  r <- dcrdm_volterra(t, th[["v"]], th[["amp"]], th[["tau"]], th[["s"]], th[["b"]], dt, t_max)
  cdf <- pmin(pmax(r$cdf, 0), 1)
  list(lp = log(pmax(r$pdf, TINY)), lF = log(pmax(cdf, TINY)), lS = log(pmax(1 - cdf, TINY)))
}
flow_at <- function(fl, th, t) {
  ev <- flow_eval(fl, as.numeric(th[fl$context_names]), t); stopifnot(ev$in_box)
  list(lp = ev$log_pdf, lF = pnorm(ev$z, log.p = TRUE), lS = ev$log_sf)
}
trapz <- function(y, x) sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
# his divergences(), line for line
diverge <- function(ref, nn, t) {
  n <- length(t); p <- exp(ref$lp)
  body <- trapz(ifelse(p > 0, p * (ref$lp - nn$lp), 0), t)
  below <- exp(ref$lF[1]) * (ref$lF[1] - nn$lF[1])
  above <- exp(ref$lS[n]) * (ref$lS[n] - nn$lS[n])
  ad <- abs(exp(ref$lF) - exp(nn$lF))
  c(kl = below + body + above, w1 = trapz(ad, t) + 0.5 * t[1] * ad[1], ks = max(ad),
    mass = exp(ref$lF[1]) + trapz(p, t) + exp(ref$lS[n]))
}
qs <- function(x) sprintf("median %.2e  90%% %.2e  max %.2e", median(x), quantile(x, .9), max(x))
card <- function(n) flow_load(file.path(EXP, paste0("malte_", n, ".json")))
b_name <- function(fl) { n <- fl$context_names; n[n == "B"] <- "b"; n }

# ---------------------------------------------------------------- part 1
if (PART %in% c("1", "both")) {
  cat("=== Part 1: Malte's density_comparison.nc reproduced ===\n")
  tg <- exp(seq(log(1e-3), log(4), length.out = 2000))
  res1 <- list()
  # Wald, final flow
  m <- read.csv(file.path(NC, "malte_wald.csv")); fl <- card("rdm_aff_log_h2_k12_clip_100k")
  r <- t(sapply(seq_len(nrow(m)), function(i) {
    th <- c(v = m$v[i], s = m$s[i], b = m$b[i])
    diverge(wald_ref(tg, th[["v"]], th[["s"]], th[["b"]]), flow_at(fl, th, tg), tg) }))
  res1$wald <- cbind(m[, c("v", "s", "b", "kl", "w1", "ks")], ours = r)
  rep1 <- function(tag, m, r) {
    for (k in c("kl", "w1", "ks")) {
      d <- abs(r[, k] - m[[k]])
      cat(sprintf("  %-14s %s  his: %s\n  %-14s     ours: %s\n  %-14s     max |ours - his| %.2e, max relative %.2e, cor %.6f\n",
                  tag, k, qs(m[[k]]), "", qs(r[, k]), "", max(d), max(d / pmax(abs(m[[k]]), 1e-12)), cor(r[, k], m[[k]])))
    }
  }
  rep1("RDM final", m, r)
  ok_w <- max(abs(r[, "ks"] - m$ks)) < 1e-6 && max(abs(r[, "kl"] - m$kl)) < 1e-7
  cat(sprintf("%s  RDM final: his KL and KS reproduced set by set (KL < 1e-7, KS < 1e-6)\n", if (ok_w) "PASS" else "FAIL"))
  # the plain flow was not in his file's last run; same grid, for the record
  flp <- card("rdm_plain_h1_k12_noclip_500k")
  rp <- t(sapply(seq_len(nrow(m)), function(i) {
    th <- c(v = m$v[i], s = m$s[i], b = m$b[i])
    diverge(wald_ref(tg, th[["v"]], th[["s"]], th[["b"]]), flow_at(flp, th, tg), tg) }))
  cat(sprintf("  RDM plain, his grid (not in his file): KL %s\n%42sKS %s\n", qs(rp[, "kl"]), "", qs(rp[, "ks"])))
  res1$wald_plain <- cbind(m[, c("v", "s", "b")], rp)

  # CRDM: reference once at dt 5e-4 (his) and once at 1.25e-4
  mc <- read.csv(file.path(NC, "malte_crdm_dt0.0005.csv"))
  TH <- data.frame(v = mc$v_c, amp = mc$amp, tau = mc$tau, s = mc$s, b = mc$b)
  t0 <- proc.time()[3]
  refs <- mclapply(seq_len(nrow(TH)), function(i) {
    th <- unlist(TH[i, ]); list(a = volt_ref(tg, th, 5e-4, 4), f = volt_ref(tg, th, 1.25e-4, 4)) },
    mc.cores = CORES)
  cat(sprintf("  Volterra references, 540 sets at dt 5e-4 and 1.25e-4, t_max 4: %.0f s at %d cores\n", proc.time()[3] - t0, CORES))
  # the reference's own error, scored as if the coarse reference were a flow
  re <- t(sapply(refs, function(x) diverge(x$f, x$a, tg)))
  cat(sprintf("  reference dt 5e-4 against dt 1.25e-4: KL %s\n%48sKS %s\n", qs(re[, "kl"]), "", qs(re[, "ks"])))
  cat(sprintf("  mass of the dt 1.25e-4 reference on [0, 4] + censored tail: min %.6f max %.6f\n", min(re[, "mass"]), max(re[, "mass"])))
  ok_c <- TRUE
  for (dt in c("0.0005", "0.005", "0.05")) {
    m <- read.csv(file.path(NC, sprintf("malte_crdm_dt%s.csv", dt)))
    fl <- card(paste0("crdm_aff_log_h2_k12_clip_100k_dt", dt))
    ev <- lapply(seq_len(nrow(TH)), function(i) flow_at(fl, unlist(TH[i, ]), tg))
    ra <- t(sapply(seq_len(nrow(TH)), function(i) diverge(refs[[i]]$a, ev[[i]], tg)))
    rf <- t(sapply(seq_len(nrow(TH)), function(i) diverge(refs[[i]]$f, ev[[i]], tg)))
    rep1(paste("CRDM dt", dt), m, ra)
    cat(sprintf("  %-14s against the dt 1.25e-4 reference: KL %s\n%55sKS %s\n", paste("CRDM dt", dt), qs(rf[, "kl"]), "", qs(rf[, "ks"])))
    ok <- max(abs(ra[, "ks"] - m$ks)) < 1e-6 && max(abs(ra[, "kl"] - m$kl)) < 1e-6
    ok_c <- ok_c && ok
    cat(sprintf("%s  CRDM dt %s: his KL and KS reproduced set by set (< 1e-6)\n", if (ok) "PASS" else "FAIL", dt))
    res1[[paste0("crdm_dt", dt)]] <- cbind(TH, his = m[, c("kl", "w1", "ks")], ours = ra, fine = rf)
  }
  res1$ref_error <- cbind(TH, re)
  saveRDS(res1, file.path(OUT, "density_part1.rds"))
}

# ---------------------------------------------------------------- part 2
if (PART %in% c("2", "both")) {
  cat("=== Part 2: the first look on its own parameter draws, crude and proper ===\n")
  .libPaths(c(path.expand("~/R/nle-sbc-lib"), .libPaths())); suppressPackageStartupMessages(library(EMC2))
  stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(path.expand("~/R/nle-sbc-lib"))))
  TW <- CRDM(dt = 5e-4)                         # as first_look.R: default t_max 3, grid to 5 s
  set.seed(1)                                   # first_look.R's stream: one runif(d) per theta, cards in its order
  crude_grid <- function(crdm) exp(seq(log(1e-3), log(if (crdm) 5 else 20), length.out = if (crdm) 600 else 1500))
  crude <- function(ref, nn, t) {               # first_look.R's arithmetic
    rp <- exp(ref$lp); d <- exp(nn$lp); w <- c(diff(t), 0) * rp; ok <- rp > 0 & d > 0
    c(kl = sum((w * (log(rp) - log(d)))[ok]), ks = max(abs(exp(nn$lF) - exp(ref$lF))), mass = sum(w))
  }
  res2 <- list()
  look <- function(name, crdm, n_th, inner) {
    fl <- card(name); nm <- b_name(fl)
    lo <- setNames(fl$bounds_natural$lower, nm); hi <- setNames(fl$bounds_natural$upper, nm)
    f <- if (inner) .15 else .02
    TH <- t(sapply(seq_len(n_th), function(i) lo + (hi - lo) * runif(length(lo), f, 1 - f)))
    one <- function(i) {
      th <- TH[i, ]; tc <- crude_grid(crdm)
      if (crdm) {
        # as run: CRDM(dt = 5e-4) has t_max 3 and holds its end values beyond it
        P <- cbind(v = th[["v"]], B = th[["b"]], A = 0, t0 = 0, s = th[["s"]], amp = th[["amp"]], tau = th[["tau"]], b = th[["b"]])
        P <- P[rep(1, length(tc)), , drop = FALSE]
        rp <- TW$dfun(tc, P); rc <- TW$pfun(tc, P)
        ref_c <- list(lp = log(rp), lF = log(rc), lS = log1p(-rc))
        tp <- exp(seq(log(1e-3), log(6), length.out = 3000)); ref_p <- volt_ref(tp, th, 1.25e-4, 6)
      } else {
        ref_c <- wald_ref(tc, th[["v"]], th[["s"]], th[["b"]])
        # quantiles by root finding on log t (statmod::qinvgauss returns NA at extreme shape)
        wq <- function(lt, up) { w <- wald_ref(exp(lt), th[["v"]], th[["s"]], th[["b"]]); (if (up) w$lS else w$lF) - log(1e-10) }
        q <- exp(c(uniroot(wq, c(-40, 25), up = FALSE, tol = 1e-10)$root, uniroot(wq, c(-40, 25), up = TRUE, tol = 1e-10)$root))
        tp <- exp(seq(log(q[1]), log(q[2]), length.out = 4000)); ref_p <- wald_ref(tp, th[["v"]], th[["s"]], th[["b"]])
      }
      c(crude = crude(ref_c, flow_at(fl, th, tc), tc), proper = diverge(ref_p, flow_at(fl, th, tp), tp),
        in_crude_grid = if (crdm) exp(ref_p$lS[which.min(abs(tp - 5))]) else diff(exp(wald_ref(c(1e-3, 20), th[["v"]], th[["s"]], th[["b"]])$lF)))
    }
    r <- mclapply(seq_len(n_th), one, mc.cores = CORES); bad <- vapply(r, inherits, NA, "try-error"); if (any(bad)) stop(r[[which(bad)[1]]]); r <- do.call(rbind, r)
    tag <- sprintf("%s, inner %s", name, if (inner) "70%" else "96%")
    cat(sprintf("%s (%d theta)\n   crude : KL %s | KS %s\n   proper: KL %s | KS %s | W1 %s\n", tag, n_th,
                qs(r[, "crude.kl"]), qs(r[, "crude.ks"]), qs(r[, "proper.kl"]), qs(r[, "proper.ks"]), qs(r[, "proper.w1"])))
    cat(sprintf("   crude quadrature's reference mass: min %.4f median %.4f max %.4f; proper: min %.8f max %.8f\n",
                min(r[, "crude.mass"]), median(r[, "crude.mass"]), max(r[, "crude.mass"]), min(r[, "proper.mass"]), max(r[, "proper.mass"])))
    cat(sprintf("   KL crude / proper: median ratio %.2f, range %.2g to %.2g; rank correlation %.3f\n",
                median(r[, "crude.kl"] / r[, "proper.kl"]), min(r[, "crude.kl"] / r[, "proper.kl"]),
                max(r[, "crude.kl"] / r[, "proper.kl"]), cor(r[, "crude.kl"], r[, "proper.kl"], method = "spearman")))
    res2[[tag]] <<- cbind(TH, r)
  }
  for (n in c("rdm_aff_log_h2_k12_clip_100k", "rdm_plain_h1_k12_noclip_500k")) for (inn in c(FALSE, TRUE)) look(n, FALSE, 300, inn)
  for (dt in c("0.0005", "0.005", "0.05")) look(paste0("crdm_aff_log_h2_k12_clip_100k_dt", dt), TRUE, 60, TRUE)
  saveRDS(res2, file.path(OUT, "density_part2.rds"))
  # where the plain flow fails
  x <- as.data.frame(res2[["rdm_plain_h1_k12_noclip_500k, inner 96%"]])
  cat("Plain RDM flow, proper KS by region of its box [0,8] x [0,5] x [0,5]:\n")
  cell <- function(sel, lab) cat(sprintf("   %-34s n %3d  KS %s | KL %s\n", lab, sum(sel), qs(x$proper.ks[sel]), qs(x$proper.kl[sel])))
  cell(x$s >= .25 & x$b >= .25 & x$s <= 3.5 & x$b <= 3.5, "s, b in [0.25, 3.5] (final's box)")
  cell(x$s < .25 | x$b < .25, "s or b below 0.25")
  cell(x$s >= .5 & x$b >= .5, "s, b at least 0.5")
  cell(x$b < .5, "b below 0.5"); cell(x$s < .5 & x$b >= .5, "s below 0.5, b at least 0.5")
  cell(x$s > 3.5 | x$b > 3.5, "s or b above 3.5")
}

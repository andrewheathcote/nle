# Measure likelihood-SURFACE distortion directly, rather than at the truth.
#
# Phase 35: t0 is displaced even when the flow has no representation of it, so
# the fault arrives through the other parameters. Phase 33: score bias is a
# derivative AT the truth and says nothing about the surface a posterior
# explores. This measures the surface.
#
# For a FIXED simulated dataset, define
#
#     Delta(theta) = log L_flow(data | theta) - log L_analytic(data | theta)
#
# A perfect approximation makes Delta constant in theta. Any VARIATION in Delta
# tilts the likelihood surface and displaces the mode; the tilt matters, the
# level does not. So we report, per parameter, over a wide grid (+-3 prior sd,
# which is the region a posterior actually visits):
#
#   peak shift  where log L_flow peaks minus where log L_analytic peaks, in
#               prior-sd units -- the displacement this parameter would suffer
#               if the others were held fixed
#   tilt        Delta's total swing across the grid, in log units. This is the
#               quantity score bias cannot see: g is Delta's slope at ONE point.
#   curv        Delta's quadratic coefficient, which changes posterior WIDTH and
#               so inflates standardised bias even when the mode is right
#
#   Rscript sbc/test_surface_distortion.R <tag> [n_datasets]
# Env: SD_CORES (default 9), NTRIAL (default 400)
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(WienR); library(parallel)})
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")

a <- commandArgs(trailingOnly = TRUE)
TAG <- if (length(a)) a[1] else "cap256w_c4"
ND  <- if (length(a) > 1) as.integer(a[2]) else 12L
CORES <- as.integer(Sys.getenv("SD_CORES", "9"))
N <- as.integer(Sys.getenv("NTRIAL", "400"))
fl <- ddm_load(sprintf("port/export/ddm_%s_weights.json", TAG))
ENC <- fl$context_encoding
# Detect the layout from the CONTEXT, not from context_encoding: the shipped
# flow predates that field and reports NULL, which silently selected the
# 7-parameter layout against an 8-column flow and made every grid point invalid.
ST0EST <- "st0" %in% fl$context_names

# theta layout the port takes, and which entries are free
FULL <- if (ST0EST) {
  c("v","a","t0","s","Z","SZ","sv","st0")
} else {
  c("v","a","t0","s","Z","SZ","sv")
}
FREE <- setdiff(FULL, "s")
PSD  <- c(v = .15, a = .15, t0 = .1, Z = .05, SZ = .15, sv = .15, st0 = .15)

an_l <- function(th, rt, R) { w <- pnorm(th[match("Z", FULL)])
  WienR::dWDM(rt, response = ifelse(R == 1, "lower", "upper"),
    a = exp(th[match("a", FULL)]), v = th[1], w = w,
    t0 = exp(th[match("t0", FULL)]), sv = exp(th[match("sv", FULL)]),
    sw = 2 * pnorm(th[match("SZ", FULL)]) * min(w, 1 - w),
    st0 = if (ST0EST) exp(th[match("st0", FULL)]) else 0,
    precision = 1e-8)$logvalue }
fl_l <- function(th, rt, R) ddm_eval(fl, th, rt, R)$log_pdf

# No -1e3 floor: flooring invents a huge fake tilt wherever a grid point leaves
# the support (t0 raised above some rt), which swamps the real distortion. Return
# NA instead so that grid position is DROPPED from the comparison.
totll <- function(f, th, rt, R) {
  v <- sapply(1:2, function(r) { k <- R == r
    if (!any(k)) return(0)
    l <- f(th, rt[k], r)
    if (any(!is.finite(l))) NA_real_ else sum(l) })
  if (any(is.na(v))) NA_real_ else sum(v)
}

x <- readRDS("sbc/nci/results_ens5/DDMnn_N400_R500.rds")
truth <- as.matrix(x$SBC$med$alpha) - as.matrix(x$SBC$bias$alpha)
cat(sprintf("flow %s  (encoding %s)   %d datasets x %d trials   cores %d\n\n",
            TAG, ENC, ND, N, CORES))

one <- function(i) {
  tr <- truth[i, ]
  th0 <- setNames(numeric(length(FULL)), FULL)
  for (p in FULL) th0[p] <- if (p == "s") 0 else tr[[p]]
  t0n <- exp(th0[["t0"]])
  set.seed(9000 + i)
  grid <- t0n + exp(seq(log(1e-4), log(20), length.out = 3000))
  dens <- lapply(1:2, function(R) exp(an_l(th0, grid, R)))
  dr <- c(diff(grid), tail(diff(grid), 1))
  pR <- sapply(dens, function(d) sum(d * dr)); pR <- pR / sum(pR)
  R <- sample(1:2, N, TRUE, prob = pR)
  rt <- sapply(R, function(r) { cw <- cumsum(dens[[r]] * dr); cw <- cw / tail(cw, 1)
                                grid[which.max(cw >= runif(1))] })
  out <- matrix(NA_real_, length(FREE), 3,
                dimnames = list(FREE, c("peak_shift_sd", "tilt", "curv")))
  for (p in FREE) {
    sdp <- PSD[[p]]
    off <- seq(-3, 3, length.out = 41) * sdp
    la <- lf <- numeric(length(off))
    for (k in seq_along(off)) {
      th <- th0; th[p] <- th0[[p]] + off[k]
      la[k] <- totll(an_l, th, rt, R); lf[k] <- totll(fl_l, th, rt, R)
    }
    ok <- is.finite(la) & is.finite(lf)
    if (sum(ok) < 10) next
    D <- lf[ok] - la[ok]; o <- off[ok]
    out[p, "peak_shift_sd"] <- (o[which.max(lf[ok])] - o[which.max(la[ok])]) / sdp
    out[p, "tilt"] <- max(D) - min(D)
    out[p, "curv"] <- tryCatch(unname(coef(lm(D ~ poly(o, 2, raw = TRUE)))[3]) * sdp^2,
                               error = function(e) NA_real_)
  }
  out
}
res <- mclapply(seq_len(ND), one, mc.cores = CORES)
res <- Filter(function(r) is.matrix(r) && any(is.finite(r)), res)
cat(sprintf("usable datasets: %d / %d\n\n", length(res), ND))

cat("Delta = log L_flow - log L_analytic, over +-3 prior sd, others at the truth\n\n")
cat(sprintf("%-5s %15s %12s %12s\n", "par", "peak shift (sd)", "tilt (log)", "curv"))
for (p in FREE) {
  v <- sapply(res, function(r) r[p, "peak_shift_sd"])
  t <- sapply(res, function(r) r[p, "tilt"])
  c2 <- sapply(res, function(r) r[p, "curv"])
  cat(sprintf("%-5s %+15.3f %12.2f %12.3f\n", p,
              mean(v, na.rm = TRUE), mean(t, na.rm = TRUE), mean(c2, na.rm = TRUE)))
}
saveRDS(res, sprintf("sbc/surface_%s.rds", TAG))

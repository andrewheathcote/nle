# Stage M2 acceptance: port/src/crdm.cpp vs eamax (Malte Lueken) and vs the Wald limit.
#   Rscript /Users/andrew/Downloads/julian/nle/port/tests/validate_crdm.R
# Needs malte/crdm_ref/ (python malte/make_crdm_reference.py).
suppressMessages(library(Rcpp))
root <- "/Users/andrew/Downloads/julian/nle"
sourceCpp(file.path(root, "port/src/crdm.cpp"))
ref <- file.path(root, "malte/crdm_ref")
sets <- read.csv(file.path(ref, "sets.csv"))
line <- function(ok, name, msg) cat(sprintf("%s %s: %s\n", if (ok) "PASS" else "FAIL", name, msg))
all_ok <- TRUE
chk <- function(ok, name, msg) { line(ok, name, msg); all_ok <<- all_ok && ok }

## 0. mean / drift
{
  t <- seq(0.01, 2, length.out = 50); h <- 1e-6
  a <- crdm_mean(t, 1.3, 0.4, 0.2)
  fd <- (crdm_mean(t + h, 1.3, 0.4, 0.2)$mean - crdm_mean(t - h, 1.3, 0.4, 0.2)$mean) / (2 * h)
  pk <- max(crdm_mean(seq(0.001, 2, by = 0.001), 0, 0.4, 0.2)$mean)
  d <- max(abs(a$drift - fd)); 
  chk(d < 1e-6 && abs(pk - 0.4) < 1e-4, "crdm_mean", sprintf("drift vs finite diff %.1e, pulse peak %.6f (amp 0.4)", d, pk))
}

## 1. dcrdm_volterra vs his solver, same dt
for (dt in c(0.002, 0.0005)) {
  g <- read.csv(gzfile(file.path(ref, sprintf("grid_dt%s.csv.gz", c("0.002", "0.0005")[match(dt, c(0.002, 0.0005))]))))
  worst <- c(pdf = 0, cdf = 0); nn <- 0
  for (i in sets$set) {
    p <- sets[sets$set == i, ]; r <- g[g$set == i, ]
    o <- dcrdm_volterra(r$t, p$v, p$amp, p$tau, p$s, p$b, dt, 3)
    k <- r$pdf > 1e-12; nn <- nn + sum(k)
    worst["pdf"] <- max(worst["pdf"], max(abs(o$pdf[k] / r$pdf[k] - 1)))
    kc <- r$cdf > 1e-12
    worst["cdf"] <- max(worst["cdf"], max(abs(o$cdf[kc] / r$cdf[kc] - 1)))
  }
  chk(worst["pdf"] < 1e-8, sprintf("volterra_vs_eamax dt=%g", dt),
      sprintf("max rel diff pdf %.2e, cdf %.2e (%d density points > 1e-12, 20 sets)", worst["pdf"], worst["cdf"], nn))
}

## 2. amp = 0 vs closed-form Wald; convergence in dt
wald_pdf <- function(t, v, s, b) b / (s * sqrt(2 * pi * t^3)) * exp(-(b - v * t)^2 / (2 * s^2 * t))
wald_cdf <- function(t, v, s, b) pnorm((v * t - b) / (s * sqrt(t))) +
  exp(2 * v * b / s^2 + pnorm(-(v * t + b) / (s * sqrt(t)), log.p = TRUE))
wp <- data.frame(v = c(1, 2, 4, 3, 1.5, 6, 8, 2.5), s = c(1, 1, 1, .5, 2, 1, .25, 1.5), b = c(1, 1, 1.5, .5, 1, 2, .25, 2))
tab <- NULL
for (dt in c(0.002, 0.001, 0.0005)) {
  e <- numeric(nrow(wp)); mass <- numeric(nrow(wp))
  for (i in seq_len(nrow(wp))) {
    p <- wp[i, ]
    tq <- uniroot(function(t) wald_cdf(t, p$v, p$s, p$b) - 0.995, c(1e-3, 500))$root
    t_max <- ceiling(tq * 1.05 / 0.5) * 0.5
    gr <- crdm_volterra_grid(p$v, 0, 0.2, p$s, p$b, dt, t_max)
    F <- wald_cdf(gr$t, p$v, p$s, p$b)
    k <- F > 0.005 & F < 0.995
    e[i] <- max(abs(gr$pdf[k] / wald_pdf(gr$t[k], p$v, p$s, p$b) - 1))
    mass[i] <- gr$cdf[length(gr$cdf)] - F[length(F)]
  }
  tab <- rbind(tab, data.frame(dt = dt, max_rel = max(e), worst_set = which.max(e), max_cdf_abs_end = max(abs(mass))))
}
cat("Wald limit (amp = 0), central 99% of mass, 8 (v,s,b) sets:\n"); print(tab, row.names = FALSE)
chk(tab$max_rel[3] < 2e-3, "volterra_vs_wald dt=0.0005", sprintf("max rel diff %.2e (tolerance 2e-3)", tab$max_rel[3]))
cat("note: at amp = 0 the Fortet kernel factor (-v + m_diff/dt) is identically 0, so the recursion sum vanishes and the density is\n",
    "the analytic Wald term itself; the Wald check is exact by construction (also true of eamax) and says nothing about dt.\n")

## 2b. convergence in dt where the recursion matters (amp > 0): vs dt = 0.00025, at common grid points t = k * 0.002
act <- sets[sets$amp > 0, ]
gr_at <- function(p, dt) { g <- crdm_volterra_grid(p$v, p$amp, p$tau, p$s, p$b, dt, 3); ix <- round(seq(0.002, 3, by = 0.002) / dt); list(pdf = g$pdf[ix], cdf = g$cdf[ix]) }
fine <- lapply(seq_len(nrow(act)), function(i) gr_at(act[i, ], 0.00025))
conv <- NULL
for (dt in c(0.002, 0.001, 0.0005)) {
  ep <- ec <- 0
  for (i in seq_len(nrow(act))) {
    g <- gr_at(act[i, ], dt)
    ep <- max(ep, max(abs(g$pdf - fine[[i]]$pdf)) / max(fine[[i]]$pdf))
    ec <- max(ec, max(abs(g$cdf - fine[[i]]$cdf)))
  }
  conv <- rbind(conv, data.frame(dt = dt, max_pdf_err_over_peak = ep, max_cdf_abs_err = ec))
}
conv$order_pdf <- c(NA, log2(conv$max_pdf_err_over_peak[1:2] / conv$max_pdf_err_over_peak[2:3]))
cat("Convergence in dt, amp > 0 (17 sets), error vs dt = 0.00025:\n"); print(conv, row.names = FALSE, digits = 3)

## 3. simulator vs his draws and vs the Volterra CDF
d <- read.csv(gzfile(file.path(ref, "draws.csv.gz")))
set.seed(20260928)
cum <- function(i) { p <- sets[sets$set == i, ]; function(q) dcrdm_volterra(pmin(q, 3), p$v, p$amp, p$tau, p$s, p$b, 0.0005, 3)$cdf }
res <- NULL
for (i in unique(d$set)) {
  p <- sets[sets$set == i, ]
  x <- rcrdm_acc(1e5, p$v, p$amp, p$tau, p$s, p$b, 0.001, 3)
  y <- d$draw[d$set == i]
  cap <- function(z) ifelse(is.finite(z), z, 10)
  k2 <- suppressWarnings(ks.test(cap(x), cap(y)))
  Fv <- cum(i)
  k1 <- suppressWarnings(ks.test(cap(x), function(q) ifelse(q >= 3, 1, Fv(q))))
  res <- rbind(res, data.frame(set = i, p_vs_eamax = k2$p.value, D_vs_eamax = unname(k2$statistic),
                               p_vs_volterra = k1$p.value, D_vs_volterra = unname(k1$statistic),
                               cens_mine = mean(is.infinite(x)), cens_his = mean(is.infinite(y))))
}
print(res, row.names = FALSE, digits = 3)
chk(all(res$p_vs_eamax > 0.01), "simulator_vs_eamax_draws", sprintf("min KS p %.3f over 5 sets", min(res$p_vs_eamax)))
chk(all(res$p_vs_volterra > 0.01), "simulator_vs_volterra_cdf", sprintf("min KS p %.3f over 5 sets", min(res$p_vs_volterra)))

## 4. timing (t_max = 3, one call)
tm <- sapply(c(0.002, 0.001, 0.0005), function(dt)
  median(replicate(3, system.time(dcrdm_volterra(seq(0.1, 2, length.out = 200), 3, 0.5, 0.2, 1, 1, dt, 3))[["elapsed"]])))
cat(sprintf("timing dcrdm_volterra, t_max 3: dt=0.002 %.3fs, 0.001 %.3fs, 0.0005 %.3fs\n", tm[1], tm[2], tm[3]))
cat(if (all_ok) "ALL PASS\n" else "SOME FAIL\n")

# Validate the R reference AND the C++ (Rcpp) implementation of an exported
# race-model flow against the Python golden vectors, then benchmark.
#
# Run from the repo root:  Rscript port/tests/validate.R [model] [budget]
# e.g.                     Rscript port/tests/validate.R LNR small

args <- commandArgs(trailingOnly = TRUE)
model <- if (length(args) >= 1) args[1] else "LNR"
budget <- if (length(args) >= 2) args[2] else "small"
tag <- sprintf("%s_%s", tolower(model), budget)

source("port/R/flow_race.R")
cat("Compiling port/src/flow_race.cpp ...\n")
Rcpp::sourceCpp("port/src/flow_race.cpp")

exp_dir <- file.path("port", "export")
fl <- flow_load(file.path(exp_dir, sprintf("%s_weights.json", tag)))
golden <- read.csv(file.path(exp_dir, sprintf("%s_golden.csv", tag)))
inter <- jsonlite::fromJSON(file.path(exp_dir, sprintf("%s_intermediates.json", tag)))
ctx_names <- fl$context_names
ptr <- flow_build(fl)

fails <- 0L
check <- function(label, diff, tol) {
  ok <- is.finite(diff) && diff < tol
  cat(sprintf("%-46s max|diff| = %10.3e  tol = %g  %s\n",
              label, diff, tol, if (ok) "PASS" else "FAIL"))
  if (!ok) fails <<- fails + 1L
}

# --- 1. R reference: layer-by-layer intermediates (first 3 parameter sets) ---
kn <- flow_knots(fl, as.matrix(inter$theta))
check("R: ctx_std (scaler)",      max(abs(kn$ctx_std - as.matrix(inter$ctx_std))), 1e-12)
check("R: conditioner raw (MLP)", max(abs(kn$raw - as.matrix(inter$conditioner_raw))), 1e-9)
check("R: x_pos (knots)",         max(abs(kn$x_pos - as.matrix(inter$x_pos))), 1e-9)
check("R: y_pos (knots)",         max(abs(kn$y_pos - as.matrix(inter$y_pos))), 1e-9)
check("R: knot slopes",           max(abs(kn$slopes - as.matrix(inter$knot_slopes))), 1e-9)

# --- 2. R and C++ against the golden grid; C++ against R ---
d <- list(r_z = 0, r_lp = 0, r_cdf = 0, r_lsf = 0,
          c_lp = 0, c_cdf = 0, c_lsf = 0, rc = 0)
for (pid in unique(golden$param_id)) {
  g <- golden[golden$param_id == pid, ]
  theta <- as.numeric(g[1L, ctx_names])
  ev <- flow_eval(fl, theta, g$rt)
  cv <- flow_eval_cpp(ptr, theta, g$rt)
  stopifnot(ev$in_box, cv$in_box)
  d$r_z   <- max(d$r_z,   abs(ev$z - g$z))
  d$r_lp  <- max(d$r_lp,  abs(ev$log_pdf - g$log_pdf))
  d$r_cdf <- max(d$r_cdf, abs(ev$cdf - g$cdf))
  d$r_lsf <- max(d$r_lsf, abs(ev$log_sf - g$log_sf))
  d$c_lp  <- max(d$c_lp,  abs(cv$log_pdf - g$log_pdf))
  d$c_cdf <- max(d$c_cdf, abs(cv$cdf - g$cdf))
  d$c_lsf <- max(d$c_lsf, abs(cv$log_sf - g$log_sf))
  d$rc    <- max(d$rc,    abs(cv$log_pdf - ev$log_pdf),
                          abs(cv$cdf - ev$cdf), abs(cv$log_sf - ev$log_sf))
}
n_sets <- length(unique(golden$param_id))
cat(sprintf("\nGolden grid: %d parameter sets x %d rts\n",
            n_sets, nrow(golden) / n_sets))
check("R: latent z",    d$r_z,   1e-9)
check("R: log_pdf",     d$r_lp,  1e-9)
check("R: cdf",         d$r_cdf, 1e-9)
check("R: log_sf",      d$r_lsf, 1e-9)
check("C++: log_pdf",   d$c_lp,  1e-9)
check("C++: cdf",       d$c_cdf, 1e-9)
check("C++: log_sf",    d$c_lsf, 1e-9)
# tolerance allows BLAS-vs-loop summation-order noise in the MLP mat-vecs
check("C++ vs R (all outputs)", d$rc, 1e-10)

# --- 3. Bounding box: just outside each face -> pdf = 0, cdf = 0 ---
lo <- fl$bounds_sampled$lower; hi <- fl$bounds_sampled$upper
mid <- (lo + hi) / 2
rt_test <- c(0.3, 1, 3)
box_ok <- TRUE
for (j in seq_along(mid)) {
  for (side in c("lower", "upper")) {
    th <- mid
    th[j] <- if (side == "lower") lo[j] - 1e-9 else hi[j] + 1e-9
    for (evb in list(flow_eval(fl, th, rt_test), flow_eval_cpp(ptr, th, rt_test))) {
      if (isTRUE(evb$in_box) || any(evb$pdf != 0) || any(evb$cdf != 0) ||
          any(is.finite(evb$log_pdf))) box_ok <- FALSE
    }
  }
}
ev_mid <- flow_eval_cpp(ptr, mid, rt_test)
ev_lo <- flow_eval_cpp(ptr, lo, rt_test)
if (!ev_mid$in_box || !ev_lo$in_box || any(ev_mid$pdf <= 0)) box_ok <- FALSE
cat(sprintf("%-46s %s\n",
            sprintf("bounding box (R & C++, %d faces + boundary)", 2 * length(mid)),
            if (box_ok) "PASS" else "FAIL"))
if (!box_ok) fails <- fails + 1L

# --- 4. Trial-wise C++ path: duplicate-row caching + out-of-box rows ---
g1 <- golden[golden$param_id == 1L, ]
g2 <- golden[golden$param_id == 2L, ]
th_bad <- mid; th_bad[1] <- hi[1] + 1  # out of box
Theta <- rbind(
  matrix(as.numeric(g1[1L, ctx_names]), nrow(g1), length(mid), byrow = TRUE),
  matrix(th_bad, 3, length(mid), byrow = TRUE),
  matrix(as.numeric(g2[1L, ctx_names]), nrow(g2), length(mid), byrow = TRUE))
rts <- c(g1$rt, rt_test, g2$rt)
tv <- flow_eval_trials_cpp(ptr, Theta, rts)
expect_lp <- c(g1$log_pdf, rep(-Inf, 3), g2$log_pdf)
expect_cdf <- c(g1$cdf, rep(0, 3), g2$cdf)
tw_diff <- max(abs((tv$log_pdf - expect_lp)[is.finite(expect_lp)]),
               abs(tv$cdf - expect_cdf))
tw_ok <- tw_diff < 1e-9 && all(tv$log_pdf[nrow(g1) + 1:3] == -Inf)
cat(sprintf("%-46s %s\n", "C++ trial-wise (caching + out-of-box rows)",
            if (tw_ok) "PASS" else "FAIL"))
if (!tw_ok) fails <- fails + 1L

# --- 5. Sanity: pdf/cdf consistency and normalization ---
# pdf must equal d cdf / d rt (central finite differences), and the cdf must
# approach 0 and 1 at the extremes (proper normalized density — checked
# without integrate(), which is fragile for peaked densities).
d_fd <- 0; d_norm <- 0
for (pid in seq_len(min(5, n_sets))) {
  g <- golden[golden$param_id == pid, ]
  theta <- as.numeric(g[1L, ctx_names])
  rt0 <- exp(seq(log(0.05), log(15), length.out = 60))
  h <- rt0 * 1e-6
  fd <- (flow_eval_cpp(ptr, theta, rt0 + h)$cdf -
         flow_eval_cpp(ptr, theta, rt0 - h)$cdf) / (2 * h)
  pdf0 <- flow_eval_cpp(ptr, theta, rt0)$pdf
  d_fd <- max(d_fd, abs(fd - pdf0) / pmax(pdf0, 1))
  ends <- flow_eval_cpp(ptr, theta, c(1e-6, 1e6))$cdf
  d_norm <- max(d_norm, ends[1], 1 - ends[2])
}
check("C++: pdf vs d(cdf)/d(rt), rel.", d_fd, 1e-4)
check("C++: cdf endpoints (0 / 1)", d_norm, 1e-5)

# --- 6. Benchmark ---
cat("\n--- Benchmark (per call, evaluating pdf+cdf at n rts) ---\n")
theta_b <- as.numeric(golden[golden$param_id == 1L, ctx_names][1L, ])
set.seed(1)
for (n in c(100L, 1000L, 10000L)) {
  rt_b <- exp(rnorm(n, 0, 0.6))
  t_cpp <- system.time(for (i in 1:200) flow_eval_cpp(ptr, theta_b, rt_b))[3] / 200
  t_r <- system.time(for (i in 1:20) flow_eval(fl, theta_b, rt_b))[3] / 20
  line <- sprintf("n=%6d   C++ %9.1f us   R %9.1f us", n, t_cpp * 1e6, t_r * 1e6)
  if (model == "LNR") {
    m_ <- theta_b[1]; s_ <- exp(theta_b[2]); t0_ <- exp(theta_b[3])
    t_an <- system.time(for (i in 1:200) {
      dlnorm(rt_b - t0_, m_, s_)
      plnorm(rt_b - t0_, m_, s_)
    })[3] / 200
    line <- paste0(line, sprintf("   analytic dlnorm+plnorm %9.1f us", t_an * 1e6))
  }
  cat(line, "\n")
}

cat(sprintf("\n%s: %d check group(s) failed\n",
            if (fails == 0L) "ALL PASSED" else "FAILURE", fails))
if (fails > 0L) quit(status = 1L)

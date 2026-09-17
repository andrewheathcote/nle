# Validate the R reference AND the C++ implementation of the exported DDM
# flow + choice classifier against the Python golden vectors, then benchmark.
#
# Run from the repo root:  Rscript port/tests/validate_ddm.R [budget]

args <- commandArgs(trailingOnly = TRUE)
budget <- if (length(args) >= 1) args[1] else "large"
tag <- sprintf("ddm_%s", budget)

source("port/R/flow_race.R")   # shared numerical core
source("port/R/flow_ddm.R")
cat("Compiling port/src/flow_ddm.cpp ...\n")
Rcpp::sourceCpp("port/src/flow_ddm.cpp")

exp_dir <- file.path("port", "export")
fl <- ddm_load(file.path(exp_dir, sprintf("%s_weights.json", tag)))
golden <- read.csv(file.path(exp_dir, sprintf("%s_golden.csv", tag)))
inter <- jsonlite::fromJSON(file.path(exp_dir, sprintf("%s_intermediates.json", tag)))
# The golden CSV records theta on EMC2's scale (the port's public interface),
# which is what classifier_context_names holds; fl$context_names may be the
# flow's re-encoded names under --triple. Older exports lack the field.
ctx_names <- if (!is.null(fl$classifier_context_names))
  fl$classifier_context_names else fl$context_names
ptr <- ddm_build(fl)

fails <- 0L
check <- function(label, diff, tol) {
  ok <- is.finite(diff) && diff < tol
  cat(sprintf("%-50s max|diff| = %10.3e  tol = %g  %s\n",
              label, diff, tol, if (ok) "PASS" else "FAIL"))
  if (!ok) fails <<- fails + 1L
}

# --- 1. R reference: intermediates (first 3 parameter sets, R = 1) ---
d_ctx <- d_raw <- d_x <- d_y <- d_s <- d_lg <- 0
for (i in 1:3) {
  theta <- as.numeric(inter$theta[i, ])
  cond <- ddm_condition(fl, theta)
  raw <- .mlp_fwd(fl$flow_mlp, matrix(c(cond$ctx_std, 1), nrow = 1L))
  d_ctx <- max(d_ctx, abs(cond$ctx_std - as.numeric(inter$ctx_std[i, ])))
  d_raw <- max(d_raw, abs(as.numeric(raw) - as.numeric(inter$conditioner_raw_R1[i, ])))
  d_x <- max(d_x, abs(cond$knots[[1]]$x_pos - as.numeric(inter$x_pos_R1[i, ])))
  d_y <- max(d_y, abs(cond$knots[[1]]$y_pos - as.numeric(inter$y_pos_R1[i, ])))
  d_s <- max(d_s, abs(cond$knots[[1]]$slopes - as.numeric(inter$knot_slopes_R1[i, ])))
  d_lg <- max(d_lg, abs(cond$logit - inter$classifier_logit[i]))
}
check("R: ctx_std (scaler)", d_ctx, 1e-12)
check("R: flow conditioner raw (R=1)", d_raw, 1e-9)
check("R: x_pos / y_pos / slopes (R=1)", max(d_x, d_y, d_s), 1e-9)
check("R: classifier logit", d_lg, 1e-9)

# --- 2. R and C++ against the golden grid; C++ vs R ---
d <- list(r_lp = 0, r_cdf = 0, c_lp = 0, c_cdf = 0, c_pr = 0, rc = 0)
for (pid in unique(golden$param_id)) {
  g <- golden[golden$param_id == pid, ]
  theta <- as.numeric(g[1L, ctx_names])
  ev <- ddm_eval(fl, theta, g$rt, g$R)
  cv <- ddm_eval_cpp(ptr, theta, g$rt, as.integer(g$R))
  stopifnot(ev$in_box, cv$in_box)
  d$r_lp  <- max(d$r_lp,  abs(ev$log_pdf - g$log_pdf_joint))
  d$r_cdf <- max(d$r_cdf, abs(ev$cdf - g$cdf_defective))
  d$c_lp  <- max(d$c_lp,  abs(cv$log_pdf - g$log_pdf_joint))
  d$c_cdf <- max(d$c_cdf, abs(cv$cdf - g$cdf_defective))
  d$c_pr  <- max(d$c_pr,  abs(log(cv$p_R) - g$log_p_R))
  d$rc    <- max(d$rc, abs(cv$log_pdf - ev$log_pdf), abs(cv$cdf - ev$cdf))
}
n_sets <- length(unique(golden$param_id))
cat(sprintf("\nGolden grid: %d parameter sets x %d (rt, R) pairs\n",
            n_sets, nrow(golden) / n_sets))
check("R: joint log_pdf",     d$r_lp,  1e-9)
check("R: defective cdf",     d$r_cdf, 1e-9)
check("C++: joint log_pdf",   d$c_lp,  1e-9)
check("C++: defective cdf",   d$c_cdf, 1e-9)
check("C++: log P(R)",        d$c_pr,  1e-9)
# tolerance allows BLAS-vs-loop summation-order noise in the MLP mat-vecs
check("C++ vs R (all outputs)", d$rc, 1e-10)

# --- 3. Bounding box: just outside each face -> pdf = 0, cdf = 0 ---
lo <- fl$bounds_sampled$lower; hi <- fl$bounds_sampled$upper
mid <- (lo + hi) / 2
rt_test <- c(0.3, 1, 3); R_test <- c(1L, 2L, 1L)
box_ok <- TRUE
for (j in seq_along(mid)) {
  for (side in c("lower", "upper")) {
    th <- mid
    th[j] <- if (side == "lower") lo[j] - 1e-9 else hi[j] + 1e-9
    for (evb in list(ddm_eval(fl, th, rt_test, R_test),
                     ddm_eval_cpp(ptr, th, rt_test, R_test))) {
      if (isTRUE(evb$in_box) || any(evb$pdf != 0) || any(evb$cdf != 0) ||
          any(is.finite(evb$log_pdf))) box_ok <- FALSE
    }
  }
}
ev_mid <- ddm_eval_cpp(ptr, mid, rt_test, R_test)
if (!ev_mid$in_box || any(ev_mid$pdf <= 0)) box_ok <- FALSE
cat(sprintf("%-50s %s\n",
            sprintf("bounding box (R & C++, %d faces + interior)", 2 * length(mid)),
            if (box_ok) "PASS" else "FAIL"))
if (!box_ok) fails <- fails + 1L

# --- 4. Trial-wise C++ path: duplicate-row caching + out-of-box rows ---
g1 <- golden[golden$param_id == 1L, ]
g2 <- golden[golden$param_id == 2L, ]
th_bad <- mid; th_bad[1] <- hi[1] + 1
Theta <- rbind(
  matrix(as.numeric(g1[1L, ctx_names]), nrow(g1), length(mid), byrow = TRUE),
  matrix(th_bad, 3, length(mid), byrow = TRUE),
  matrix(as.numeric(g2[1L, ctx_names]), nrow(g2), length(mid), byrow = TRUE))
rts <- c(g1$rt, rt_test, g2$rt)
Rs <- as.integer(c(g1$R, R_test, g2$R))
tv <- ddm_eval_trials_cpp(ptr, Theta, rts, Rs)
expect_lp <- c(g1$log_pdf_joint, rep(-Inf, 3), g2$log_pdf_joint)
tw_diff <- max(abs((tv$log_pdf - expect_lp)[is.finite(expect_lp)]))
tw_ok <- tw_diff < 1e-9 && all(tv$log_pdf[nrow(g1) + 1:3] == -Inf) &&
  all(tv$cdf[nrow(g1) + 1:3] == 0)
cat(sprintf("%-50s %s\n", "C++ trial-wise (caching + out-of-box rows)",
            if (tw_ok) "PASS" else "FAIL"))
if (!tw_ok) fails <- fails + 1L

# --- 5. Sanity: defective-density structure ---
# P(R=1) + P(R=2) = 1; pdf = d cdf / d rt within each response; the
# defective cdf approaches P(R) as rt -> Inf and 0 as rt -> 0; total mass 1.
d_fd <- d_norm <- d_psum <- 0
for (pid in seq_len(min(5, n_sets))) {
  g <- golden[golden$param_id == pid, ]
  theta <- as.numeric(g[1L, ctx_names])
  rt0 <- exp(seq(log(0.05), log(15), length.out = 60))
  mass <- 0
  for (r in 1:2) {
    h <- rt0 * 1e-6
    Rv <- rep(as.integer(r), length(rt0))
    fd <- (ddm_eval_cpp(ptr, theta, rt0 + h, Rv)$cdf -
           ddm_eval_cpp(ptr, theta, rt0 - h, Rv)$cdf) / (2 * h)
    ev0 <- ddm_eval_cpp(ptr, theta, rt0, Rv)
    d_fd <- max(d_fd, abs(fd - ev0$pdf) / pmax(ev0$pdf, 1))
    ends <- ddm_eval_cpp(ptr, theta, c(1e-6, 1e6), c(r, r))
    d_norm <- max(d_norm, ends$cdf[1], abs(ends$cdf[2] - ends$p_R[2]))
    mass <- mass + ends$cdf[2]
  }
  d_psum <- max(d_psum, abs(mass - 1))
}
check("C++: pdf vs d(cdf)/d(rt), rel.", d_fd, 1e-4)
check("C++: cdf endpoints (0 / P(R))", d_norm, 1e-5)
check("C++: total mass over R (vs 1)", d_psum, 1e-5)

# --- 6. Benchmark: flow DDM vs WienR numerical DDM ---
cat("\n--- Benchmark (per call, joint pdf+cdf at n (rt, R) pairs) ---\n")
theta_b <- as.numeric(golden[golden$param_id == 1L, ctx_names][1L, ])
have_wienr <- requireNamespace("WienR", quietly = TRUE)
set.seed(1)
for (n in c(100L, 1000L, 10000L)) {
  rt_b <- exp(rnorm(n, -0.2, 0.5))
  R_b <- sample(1:2, n, replace = TRUE)
  t_cpp <- system.time(for (i in 1:200) ddm_eval_cpp(ptr, theta_b, rt_b, R_b))[3] / 200
  line <- sprintf("n=%6d   C++ flow %9.1f us", n, t_cpp * 1e6)
  if (have_wienr) {
    # analytic/numerical full DDM density (WienR dWDM), natural-scale params
    nat <- theta_b
    nat[c(2, 3, 4, 7, 8)] <- exp(nat[c(2, 3, 4, 7, 8)])   # a, t0, s, sv, st0
    nat[c(5, 6)] <- pnorm(nat[c(5, 6)])                    # Z, SZ
    t_an <- system.time(for (i in 1:20)
      WienR::dWDM(rt_b, response = as.character(R_b),
                  a = nat[2] / nat[4], v = nat[1] / nat[4], w = nat[5],
                  t0 = nat[3], sv = nat[7] / nat[4],
                  sw = 2 * nat[6] * min(nat[5], 1 - nat[5]),
                  st0 = nat[8]))[3] / 20
    line <- paste0(line, sprintf("   WienR dWDM %9.1f us   speedup x%.0f",
                                 t_an * 1e6, t_an / t_cpp))
  }
  cat(line, "\n")
}

cat(sprintf("\n%s: %d check group(s) failed\n",
            if (fails == 0L) "ALL PASSED" else "FAILURE", fails))
if (fails > 0L) quit(status = 1L)

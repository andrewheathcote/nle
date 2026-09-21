# Where does the flow's likelihood put the JOINT mode, against the analytic?
#
# Phase 32b: the st0=0 failure is a joint displacement of (t0, a, Z, SZ) that a
# single-parameter profile cannot see. This optimises over ALL free parameters
# at once, on matched simulated data, and reports the displacement of each.
#
# Both likelihoods are maximised from the SAME start (the truth) with the SAME
# optimiser and the SAME box (the flow's training bounds), so the comparison is
# of likelihood surfaces, not of optimisers. The analytic's own displacement is
# finite-sample MLE bias and is the baseline to subtract.
#
#   Rscript sbc/test_joint_mle.R <tag> [n_draws]
# Env: MLE_CORES (default 8)
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(WienR); library(parallel)})
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")

a <- commandArgs(trailingOnly = TRUE)
TAG <- if (length(a)) a[1] else "cap256w_c4"
ND  <- if (length(a) > 1) as.integer(a[2]) else 40L
CORES <- as.integer(Sys.getenv("MLE_CORES", "8"))
ST0Z <- grepl("^z0_", TAG)
fl <- ddm_load(sprintf("port/export/ddm_%s_weights.json", TAG))
N <- as.integer(Sys.getenv("NTRIAL", "400"))
FREE <- if (ST0Z) c(1,2,3,5,6,7) else c(1,2,3,5,6,7,8)   # index into theta; 4 = s, fixed
NM <- c("v","a","t0","s","Z","SZ","sv","st0")[FREE]
LO <- fl$bounds_sampled$lower; HI <- fl$bounds_sampled$upper
if (!ST0Z) { LO <- c(LO); HI <- c(HI) }

an_l <- function(th, rt, R) { w <- pnorm(th[5])
  WienR::dWDM(rt, response = ifelse(R == 1, "lower", "upper"), a = exp(th[2]),
    v = th[1], w = w, t0 = exp(th[3]), sv = exp(th[7]),
    sw = 2 * pnorm(th[6]) * min(w, 1 - w),
    st0 = if (ST0Z) 0 else exp(th[8]), precision = 1e-8)$logvalue }
fl_l <- function(th, rt, R) ddm_eval(fl, th, rt, R)$log_pdf

nll <- function(f) function(p, th0, rt, R) {
  th <- th0; th[FREE] <- p
  v <- sapply(1:2, function(r) { k <- R == r
    if (!any(k)) return(0)
    l <- f(th, rt[k], r); l[!is.finite(l)] <- -1e3; sum(l) })
  -sum(v)
}

x <- readRDS("sbc/nci/results_ens5/DDMnn_N400_R500.rds")
truth <- as.matrix(x$SBC$med$alpha) - as.matrix(x$SBC$bias$alpha)
cat(sprintf("flow %s   %d draws x %d trials   free: %s   cores %d\n\n",
            TAG, ND, N, paste(NM, collapse = ","), CORES))

one <- function(i) {
  tr <- truth[i, ]
  th0 <- c(tr[["v"]],tr[["a"]],tr[["t0"]],0,tr[["Z"]],tr[["SZ"]],tr[["sv"]],tr[["st0"]])
  if (ST0Z) th0 <- th0[1:7]
  t0n <- exp(tr[["t0"]])
  set.seed(5000 + i)
  grid <- t0n + exp(seq(log(1e-4), log(20), length.out = 3000))
  dens <- lapply(1:2, function(R) exp(an_l(th0, grid, R)))
  dr <- c(diff(grid), tail(diff(grid), 1))
  pR <- sapply(dens, function(d) sum(d * dr)); pR <- pR / sum(pR)
  R <- sample(1:2, N, TRUE, prob = pR)
  rt <- sapply(R, function(r) { cw <- cumsum(dens[[r]] * dr); cw <- cw / tail(cw, 1)
                                grid[which.max(cw >= runif(1))] })
  fit <- function(f) tryCatch({
    o <- optim(th0[FREE], nll(f), th0 = th0, rt = rt, R = R, method = "L-BFGS-B",
               lower = LO[FREE] + 1e-6, upper = HI[FREE] - 1e-6,
               control = list(maxit = 300, factr = 1e9))
    o$par - th0[FREE] }, error = function(e) rep(NA_real_, length(FREE)))
  rbind(analytic = fit(an_l), flow = fit(fl_l))
}
res <- mclapply(seq_len(ND), one, mc.cores = CORES)
ok <- !sapply(res, function(r) any(!is.finite(r)))
cat(sprintf("usable draws: %d / %d\n\n", sum(ok), ND))
A <- do.call(rbind, lapply(res[ok], function(r) r["analytic", ]))
F <- do.call(rbind, lapply(res[ok], function(r) r["flow", ]))
colnames(A) <- colnames(F) <- NM

cat("MLE displacement from the truth, sampled scale (mean over draws)\n")
cat(sprintf("%-6s %11s %11s %11s %9s\n", "par", "analytic", "flow", "flow-analytic", "t"))
for (j in seq_along(NM)) {
  d <- F[, j] - A[, j]
  cat(sprintf("%-6s %+11.5f %+11.5f %+13.5f %9.2f\n", NM[j],
              mean(A[, j]), mean(F[, j]), mean(d), mean(d)/(sd(d)/sqrt(length(d)))))
}
k <- which(NM == "t0")
cat(sprintf("\nt0 in ms at t0 = 0.3 s:  analytic %+.2f   flow %+.2f   excess %+.2f\n",
            mean(A[,k])*300, mean(F[,k])*300, (mean(F[,k])-mean(A[,k]))*300))
saveRDS(list(A = A, F = F, tag = TAG), sprintf("sbc/joint_mle_%s.rds", TAG))

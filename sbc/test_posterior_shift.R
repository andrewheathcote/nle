# Predict the SBC verdict from the flow's POSTERIOR displacement, measured
# directly on fixed data.
#
# Phases 33 and 36: score bias, joint MLE, multi-start and 1D profiles all fail
# to predict SBC, and every one conditions on the truth in the other parameters.
# SBC's standardised bias is a property of the whole posterior -- prior
# included, all parameters moving together. So measure exactly that.
#
# For a fixed dataset y the flow posterior is the analytic posterior tilted by
#
#     Delta(theta) = log L_flow(y | theta) - log L_analytic(y | theta)
#
# We draw from a defensive mixture of the two Laplace approximations (analytic
# MAP and flow MAP, covariances inflated 1.5^2) and importance-weight to BOTH
# posteriors. Per dataset this gives the flow and analytic posterior means and
# sds; over datasets drawn from the SBC prior,
#
#     std bias = mean(post mean - truth) / sd(post mean - truth)
#
# which is SBC's statistic (with the mean for the median). The ANALYTIC column
# is this estimator's control: it must return ~0 everywhere.
#
#   Rscript sbc/test_posterior_shift.R <tag> [n_datasets] [n_draws]
# Env: PS_CORES (default 9), NTRIAL (default 400)
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(WienR); library(parallel)})
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")

a <- commandArgs(trailingOnly = TRUE)
TAG <- if (length(a)) a[1] else "cap256w_c4"
ND  <- if (length(a) > 1) as.integer(a[2]) else 48L
M   <- if (length(a) > 2) as.integer(a[3]) else 2000L
CORES <- as.integer(Sys.getenv("PS_CORES", "9"))
N <- as.integer(Sys.getenv("NTRIAL", "400"))
fl <- ddm_load(sprintf("port/export/ddm_%s_weights.json", TAG))
ST0EST <- "st0" %in% fl$context_names     # context, not context_encoding (Phase 36)

FULL <- if (ST0EST) c("v","a","t0","s","Z","SZ","sv","st0") else
                    c("v","a","t0","s","Z","SZ","sv")
FREE <- setdiff(FULL, "s")
# SBC priors (sbc_specs.R / SBCddm7.R), by NAME
PM <- c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5), SZ = qnorm(.2),
        sv = log(.1), st0 = log(.05))[FREE]
PS <- c(v = .15, a = .15, t0 = .1, Z = .05, SZ = .15, sv = .15, st0 = .15)[FREE]

full_th <- function(x) { th <- setNames(numeric(length(FULL)), FULL); th[FREE] <- x; th }
an_l <- function(th, rt, R) { w <- pnorm(th[["Z"]])
  WienR::dWDM(rt, response = ifelse(R == 1, "lower", "upper"),
    a = exp(th[["a"]]), v = th[["v"]], w = w, t0 = exp(th[["t0"]]),
    sv = exp(th[["sv"]]), sw = 2 * pnorm(th[["SZ"]]) * min(w, 1 - w),
    st0 = if (ST0EST) exp(th[["st0"]]) else 0, precision = 1e-8)$logvalue }
fl_l <- function(th, rt, R) ddm_eval(fl, th, rt, R)$log_pdf
totll <- function(f, th, rt, R) {
  s <- 0
  for (r in 1:2) { k <- R == r; if (!any(k)) next
    l <- tryCatch(f(th, rt[k], r), error = function(e) -Inf)
    if (any(!is.finite(l))) return(-Inf); s <- s + sum(l) }
  s }
lprior <- function(x) sum(dnorm(x, PM, PS, log = TRUE))
lpost  <- function(f, x, rt, R) lprior(x) + totll(f, full_th(x), rt, R)

sim <- function(th0) {
  grid <- exp(th0[["t0"]]) + exp(seq(log(1e-4), log(20), length.out = 3000))
  dens <- lapply(1:2, function(R) exp(an_l(th0, grid, R)))
  dr <- c(diff(grid), tail(diff(grid), 1))
  pR <- sapply(dens, function(d) sum(d * dr)); pR <- pR / sum(pR)
  R <- sample(1:2, N, TRUE, prob = pR)
  rt <- sapply(R, function(r) { cw <- cumsum(dens[[r]] * dr); cw <- cw / tail(cw, 1)
                                grid[which.max(cw >= runif(1))] })
  list(rt = rt, R = R) }

laplace <- function(f, x0, y) {
  nlp <- function(x) { v <- -lpost(f, x, y$rt, y$R); if (is.finite(v)) v else 1e10 }
  o <- optim(x0, nlp, method = "BFGS", control = list(maxit = 500, reltol = 1e-10))
  H <- optimHess(o$par, nlp)
  e <- eigen((H + t(H)) / 2, symmetric = TRUE)
  # the prior alone bounds posterior precision below; floor there so a flat or
  # non-PD numerical Hessian cannot produce an absurd proposal
  ev <- pmax(e$values, min(1 / PS^2))
  list(m = o$par, S = e$vectors %*% diag(1 / ev) %*% t(e$vectors)) }

dmvn <- function(X, m, S) { L <- chol(S); z <- backsolve(L, t(X) - m, transpose = TRUE)
  -colSums(z^2) / 2 - sum(log(diag(L))) - ncol(X) / 2 * log(2 * pi) }
rmvn <- function(n, m, S) sweep(matrix(rnorm(n * length(m)), n) %*% chol(S), 2, m, "+")

wstats <- function(lw, X) {
  w <- exp(lw - max(lw)); w[!is.finite(w)] <- 0; w <- w / sum(w)
  mu <- colSums(w * X)
  list(mean = mu, sd = sqrt(colSums(w * sweep(X, 2, mu)^2)), ess = 1 / sum(w^2)) }

one <- function(i) {
  set.seed(7000 + i)
  truth <- rnorm(length(FREE), PM, PS); names(truth) <- FREE
  y <- sim(full_th(truth))
  if (min(y$rt) <= exp(truth[["t0"]])) return(NULL)
  la <- laplace(an_l, truth, y); lf <- laplace(fl_l, truth, y)
  Sa <- 1.5^2 * la$S; Sf <- 1.5^2 * lf$S
  X <- rbind(rmvn(M / 2, la$m, Sa), rmvn(M / 2, lf$m, Sf)); colnames(X) <- FREE
  q1 <- dmvn(X, la$m, Sa); q2 <- dmvn(X, lf$m, Sf); qm <- pmax(q1, q2)
  lq <- qm + log(0.5 * exp(q1 - qm) + 0.5 * exp(q2 - qm))
  pa <- apply(X, 1, function(x) lpost(an_l, x, y$rt, y$R))
  pf <- apply(X, 1, function(x) lpost(fl_l, x, y$rt, y$R))
  list(truth = truth, an = wstats(pa - lq, X), fl = wstats(pf - lq, X),
       map_a = la$m, map_f = lf$m)
}

cat(sprintf("flow %s   %d datasets x %d trials, %d draws, cores %d\n",
            TAG, ND, N, M, CORES))
t_start <- proc.time()[3]
res <- Filter(Negate(is.null), mclapply(seq_len(ND), one, mc.cores = CORES))
cat(sprintf("usable datasets %d / %d   %.1f min\n\n", length(res), ND,
            (proc.time()[3] - t_start) / 60))

get <- function(k, s) t(sapply(res, function(r) r[[k]][[s]]))
tr <- t(sapply(res, `[[`, "truth"))
ea <- get("an", "mean") - tr; ef <- get("fl", "mean") - tr
sb <- function(e) colMeans(e) / apply(e, 2, sd)
shift <- colMeans((get("fl", "mean") - get("an", "mean")) / get("an", "sd"))
wr <- colMeans(get("fl", "sd") / get("an", "sd"))
essa <- sapply(res, function(r) r$an$ess); essf <- sapply(res, function(r) r$fl$ess)
cat("predicted SBC std bias (mean(post mean - truth) / sd), and the shift of the\n")
cat("flow posterior from the analytic one on the SAME data, in analytic post sd\n\n")
cat(sprintf("%-5s %10s %10s %12s %10s\n", "par", "analytic", "flow", "shift (sd)", "sd ratio"))
for (p in FREE) cat(sprintf("%-5s %+10.3f %+10.3f %+12.3f %10.3f\n", p,
                            sb(ea)[p], sb(ef)[p], shift[p], wr[p]))
cat(sprintf("\nESS  analytic median %.0f (min %.0f)   flow median %.0f (min %.0f)\n",
            median(essa), min(essa), median(essf), min(essf)))
saveRDS(list(tag = TAG, res = res, sb_an = sb(ea), sb_fl = sb(ef), shift = shift,
             sd_ratio = wr), sprintf("sbc/postshift_%s.rds", TAG))

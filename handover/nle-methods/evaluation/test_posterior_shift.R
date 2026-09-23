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
#   Rscript evaluation/test_posterior_shift.R <tag> [n_datasets] [n_draws]
# Env: PS_CORES (default 9), NTRIAL (default 400)
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(WienR); library(parallel)})
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")

a <- commandArgs(trailingOnly = TRUE)
TAG <- if (length(a)) a[1] else "cap256w_c4"
ND  <- if (length(a) > 1) as.integer(a[2]) else 48L
M   <- if (length(a) > 2) as.integer(a[3]) else 2000L
CORES <- as.integer(Sys.getenv("PS_CORES", "9"))
# MULTISTART=k adds k optimiser starts dispersed at 2 prior sd for EACH
# likelihood, and puts the best-scoring mode found into the proposal. With k = 0
# both Laplace centres are found from the truth, which (Phase 33) never visits
# the flow's better-scoring optima away from it.
KMS <- as.integer(Sys.getenv("MULTISTART", "0"))
# SIM=emc2 simulates with EMC2's own sampler (WienR::rWDM, exact), as SBC does,
# instead of inverse-CDF on a 3000-point log grid.
SIMEMC <- Sys.getenv("SIM", "grid") == "emc2"
N <- as.integer(Sys.getenv("NTRIAL", "400"))
# TAG beginning "reg_" selects the direct-regression baseline (an MLP over
# (theta, log rt, R)) instead of a flow export. Same estimator either way.
REG <- grepl("^reg_", TAG)
if (REG) {
  suppressPackageStartupMessages(library(jsonlite)); source("evaluation/DDMreg.R")
  RNET <- regnet_get(sprintf("results/model_states/%s_budget/regress.json", TAG))
  ST0EST <- TRUE                          # the regression net is 8-parameter
} else {
  fl <- ddm_load(sprintf("port/export/ddm_%s_weights.json", TAG))
  ST0EST <- "st0" %in% fl$context_names   # context, not context_encoding (Phase 36)
}

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
fl_l <- if (REG) function(th, rt, R) {
  pars <- cbind(v = th[["v"]], a = exp(th[["a"]]), t0 = exp(th[["t0"]]), s = 1,
                Z = pnorm(th[["Z"]]), SZ = pnorm(th[["SZ"]]), sv = exp(th[["sv"]]),
                st0 = exp(th[["st0"]]))[rep(1, length(rt)), , drop = FALSE]
  reg_log_pdf(rt, R, pars, RNET)
} else function(th, rt, R) ddm_eval(fl, th, rt, R)$log_pdf
totll <- function(f, th, rt, R) {
  s <- 0
  for (r in 1:2) { k <- R == r; if (!any(k)) next
    l <- tryCatch(f(th, rt[k], r), error = function(e) -Inf)
    if (any(!is.finite(l))) return(-Inf); s <- s + sum(l) }
  s }
lprior <- function(x) sum(dnorm(x, PM, PS, log = TRUE))
lpost  <- function(f, x, rt, R) lprior(x) + totll(f, full_th(x), rt, R)

sim_emc <- function(th0) {
  p <- data.frame(v = th0[["v"]], a = exp(th0[["a"]]), t0 = exp(th0[["t0"]]),
                  z = pnorm(th0[["Z"]]), sv = exp(th0[["sv"]]),
                  sz = 2 * pnorm(th0[["SZ"]]) * min(pnorm(th0[["Z"]]), 1 - pnorm(th0[["Z"]])),
                  st0 = if (ST0EST) exp(th0[["st0"]]) else 0)
  r <- WienR::rWDM(N, a = p$a, v = p$v, w = p$z, t0 = p$t0, sv = p$sv, method = "p-ars",
                      sw = p$sz, st0 = p$st0)
  list(rt = r$q, R = ifelse(r$response == "lower", 1L, 2L)) }

sim <- function(th0) {
  if (SIMEMC) return(sim_emc(th0))
  grid <- exp(th0[["t0"]]) + exp(seq(log(1e-4), log(20), length.out = 3000))
  dens <- lapply(1:2, function(R) exp(an_l(th0, grid, R)))
  dr <- c(diff(grid), tail(diff(grid), 1))
  pR <- sapply(dens, function(d) sum(d * dr)); pR <- pR / sum(pR)
  R <- sample(1:2, N, TRUE, prob = pR)
  rt <- sapply(R, function(r) { cw <- cumsum(dens[[r]] * dr); cw <- cw / tail(cw, 1)
                                grid[which.max(cw >= runif(1))] })
  list(rt = rt, R = R) }

laplace <- function(f, x0, y, k = 0) {
  nlp <- function(x) { v <- -lpost(f, x, y$rt, y$R); if (is.finite(v)) v else 1e10 }
  fit <- function(s) optim(s, nlp, method = "BFGS", control = list(maxit = 500, reltol = 1e-10))
  o <- fit(x0)
  for (j in seq_len(k)) { oj <- fit(x0 + 2 * PS * rnorm(length(x0)))
                          if (oj$value < o$value) o <- oj }
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
  la <- laplace(an_l, truth, y, KMS); lf <- laplace(fl_l, truth, y, KMS)
  comps <- list(la, lf)
  if (KMS > 0) comps <- c(comps, list(laplace(fl_l, truth, y, 0)))   # keep the local flow mode too
  K <- length(comps)
  X <- do.call(rbind, lapply(comps, function(cc) rmvn(M %/% K, cc$m, 1.5^2 * cc$S)))
  colnames(X) <- FREE
  Q <- sapply(comps, function(cc) dmvn(X, cc$m, 1.5^2 * cc$S)); qm <- apply(Q, 1, max)
  lq <- qm + log(rowMeans(exp(Q - qm)))
  pa <- apply(X, 1, function(x) lpost(an_l, x, y$rt, y$R))
  pf <- apply(X, 1, function(x) lpost(fl_l, x, y$rt, y$R))
  list(truth = truth, an = wstats(pa - lq, X), fl = wstats(pf - lq, X),
       map_a = la$m, map_f = lf$m,
       d_map_t0 = unname(lf$m["t0"] - la$m["t0"]))
}

cat(sprintf("%s %s   %d datasets x %d trials, %d draws, cores %d\n",
            if (REG) "regression net" else "flow", TAG, ND, N, M, CORES))
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
             sd_ratio = wr), sprintf("results/score_bias/postshift_%s%s%s.rds", TAG, if (KMS > 0) sprintf("_ms%d", KMS) else "",
                         if (SIMEMC) "_emcsim" else ""))

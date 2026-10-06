# Stage M6 on Gadi (2026-10-06): M6a's posterior shift (P), which could not finish on the laptop. Same
# estimator as sbc/malte_m6.R mode P (registered in handover/MALTE_STAGES.md, "Stage M6a"; Laplace
# approximations of the Volterra-reference and of each card's posterior, defensive mixture with covariances
# x 1.5^2, M importance draws, dataset i simulated under set.seed(60000 + i)). What differs is only how it
# runs: one result file per dataset (P_ds/ds_<i>.rds, so a job resumes and datasets split across jobs),
# the Laplace fits of the models in parallel, the importance weights in parallel over draws, and a timing
# line per phase so the full run can be priced from one dataset.
#
#   Rscript sbc/malte_m6_P_gadi.R <first> <last> <cores> <tag1>=<name1> [...]   [M6_N=400 M6_M=4000 M6_OUT=...]
#   Rscript sbc/malte_m6_P_gadi.R join   0      1       <tag1>=<name1> [...]   (the summary of sbc/malte_m6.R's P mode)
# Run from a root holding port/R/flow_race.R, port/src/crdm.cpp, port/export/<cards>, sbc/malte_reference.R,
# sbc/malte_m6_lib.R. The EMC2 used (make_data, design) is asserted under M6_LIB (default ~/R/nle-lib).
a <- commandArgs(trailingOnly = TRUE)
MODE <- if (a[1] == "join") "join" else "run"
I1 <- if (MODE == "run") as.integer(a[1]) else NA; I2 <- if (MODE == "run") as.integer(a[2]) else NA
CORES <- as.integer(a[3])
kv <- strsplit(a[-(1:3)], "="); TAGS <- setNames(sapply(kv, `[`, 2), sapply(kv, `[`, 1))
LIB <- path.expand(Sys.getenv("M6_LIB", "~/R/nle-lib")); .libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(parallel); library(Rcpp)})
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
cat(sprintf("EMC2 %s at %s | built %s | Rcpp %s\n", packageVersion("EMC2"), find.package("EMC2"),
            packageDescription("EMC2")$Built, packageVersion("Rcpp")))
source("port/R/flow_race.R")
sourceCpp("port/src/crdm.cpp", cacheDir = Sys.getenv("M6_CPPCACHE", file.path(tempdir(), "sourceCpp")))
source("sbc/malte_reference.R"); source("sbc/malte_m6_lib.R")
OUT <- Sys.getenv("M6_OUT", "sbc/malte_m6"); DS <- file.path(OUT, "P_ds"); dir.create(DS, recursive = TRUE, showWarnings = FALSE)
N <- as.integer(Sys.getenv("M6_N", "400")); M <- as.integer(Sys.getenv("M6_M", "4000"))
FL <- lapply(TAGS, m6_card)
ACC <- c(list(analytic = function(p, rt) m6_volterra(p, rt)), lapply(FL, function(fl) function(p, rt) m6_flow(fl, p, rt)))
cat(sprintf("M6 P (Gadi) %s | cards %s | N %d | M %d | cores %d\n", MODE, paste(names(TAGS), TAGS, sep = " = ", collapse = ", "), N, M, CORES))

# ---- as sbc/malte_m6.R mode P -------------------------------------------------------------------------------
lprior <- function(x) sum(dnorm(x, PRIOR_M6[PARS6], PRIOR_S6[PARS6], log = TRUE))
lpost <- function(pf, x, y) lprior(x) + m6_loglik(pf, as.list(setNames(x, PARS6)), y)
laplace <- function(pf, x0, y) {
  n_ev <- 0L
  nlp <- function(x) { n_ev <<- n_ev + 1L; v <- -lpost(pf, x, y); if (is.finite(v)) v else 1e10 }
  o <- optim(x0, nlp, method = "BFGS", control = list(maxit = 200, reltol = 1e-8))
  n_opt <- n_ev
  H <- optimHess(o$par, nlp); e <- eigen((H + t(H)) / 2, symmetric = TRUE)
  ev <- pmax(e$values, min(1 / PRIOR_S6[PARS6]^2))
  list(m = o$par, S = e$vectors %*% diag(1 / ev) %*% t(e$vectors), convergence = o$convergence, n_opt = n_opt, n_ev = n_ev)
}
dmvn <- function(X, m, S) { L <- chol(S); z <- backsolve(L, t(X) - m, transpose = TRUE)
  -colSums(z^2) / 2 - sum(log(diag(L))) - ncol(X) / 2 * log(2 * pi) }
rmvn <- function(n, m, S) sweep(matrix(rnorm(n * length(m)), n) %*% chol(S), 2, m, "+")
wstats <- function(lw, X) { w <- exp(lw - max(lw)); w[!is.finite(w)] <- 0; w <- w / sum(w); mu <- colSums(w * X)
  list(mean = mu, sd = sqrt(colSums(w * sweep(X, 2, mu)^2)), ess = 1 / sum(w^2)) }

one <- function(i) {
  f <- file.path(DS, sprintf("ds_%03d.rds", i)); if (file.exists(f)) { cat(sprintf("ds %d: exists, skipped\n", i)); return(invisible()) }
  set.seed(60000 + i)
  truth <- setNames(rnorm(length(PARS6), PRIOR_M6[PARS6], PRIOR_S6[PARS6]), PARS6)
  y <- m6_sim(as.list(truth), N)
  tl <- proc.time()[3]
  comps <- mclapply(ACC, function(pf) { t1 <- proc.time()[3]; r <- tryCatch(laplace(pf, truth, y), error = function(e) NULL)
                                        if (!is.null(r)) r$sec <- proc.time()[3] - t1; r }, mc.cores = min(CORES, length(ACC)))
  tl <- proc.time()[3] - tl
  for (k in names(comps)) if (!is.null(comps[[k]])) cat(sprintf("ds %d laplace %-9s %6.0f s, %4d evals (%d optim), conv %d, %.3f s/eval\n", i, k,
      comps[[k]]$sec, comps[[k]]$n_ev, comps[[k]]$n_opt, comps[[k]]$convergence, comps[[k]]$sec / comps[[k]]$n_ev))
  if (any(sapply(comps, is.null))) { cat(sprintf("ds %d: a Laplace fit failed, dataset unusable\n", i)); saveRDS(NULL, f); return(invisible()) }
  K <- length(comps)
  X <- do.call(rbind, lapply(comps, function(cc) rmvn(M %/% K, cc$m, 1.5^2 * cc$S))); colnames(X) <- PARS6
  Q <- sapply(comps, function(cc) dmvn(X, cc$m, 1.5^2 * cc$S)); qm <- apply(Q, 1, max)
  lq <- qm + log(rowMeans(exp(Q - qm)))
  ti <- proc.time()[3]; sec_is <- c()
  out <- c(list(truth = truth), lapply(setNames(names(ACC), names(ACC)), function(k) {
    t1 <- proc.time()[3]
    blk <- split(seq_len(nrow(X)), cut(seq_len(nrow(X)), CORES, labels = FALSE))
    lp <- unlist(mclapply(blk, function(r) apply(X[r, , drop = FALSE], 1, function(x) lpost(ACC[[k]], x, y)), mc.cores = CORES), use.names = FALSE)
    sec_is[[k]] <<- proc.time()[3] - t1
    wstats(lp - lq, X)
  }))
  ti <- proc.time()[3] - ti
  for (k in names(ACC)) cat(sprintf("ds %d IS %-9s %6.0f s wall at %d cores for %d draws (%.3f core-s/eval), ESS %.0f\n", i, k, sec_is[[k]], CORES,
                                  nrow(X), sec_is[[k]] * CORES / nrow(X), out[[k]]$ess))
  out$laplace <- lapply(comps, function(cc) cc[c("m", "S", "convergence", "n_opt", "n_ev", "sec")])
  out$timing <- c(laplace_wall = tl, is_wall = ti, cores = CORES)
  saveRDS(out, f)
  cat(sprintf("ds %d: done, Laplace %.0f s wall, IS %.0f s wall, at %d cores\n", i, tl, ti, CORES))
}

if (MODE == "run") {
  for (i in I1:I2) one(i)
}

# ---- join: the printout of sbc/malte_m6.R mode P, then the files it saves for its screen mode ------------------
if (MODE == "join") {
  fs <- sort(list.files(DS, "^ds_[0-9]+\\.rds$", full.names = TRUE))
  res <- Filter(Negate(is.null), lapply(fs, readRDS))
  cat(sprintf("P: usable datasets %d of %d files, N = %d, %d draws\n", length(res), length(fs), N, M))
  saveRDS(list(tags = TAGS, N = N, res = res), file.path(OUT, sprintf("P_N%d.rds", N)))
  get <- function(k, s) t(sapply(res, function(r) r[[k]][[s]]))
  tr <- t(sapply(res, `[[`, "truth")); am <- get("analytic", "mean"); as <- get("analytic", "sd")
  sb <- function(e) colMeans(e) / apply(e, 2, sd)
  cat(sprintf("\nP, N = %d, %d datasets. Predicted standardised bias, shift from the analytic posterior in its sd (se), sd ratio\n", N, length(res)))
  cat(sprintf("%-9s %-11s %9s %16s %9s %9s\n", "flow", "par", "std bias", "shift (se)", "sd ratio", "med ESS"))
  ca <- sb(am - tr)
  for (p in PARS6) cat(sprintf("%-9s %-11s %+9.3f %16s %9s %9.0f\n", "analytic", p, ca[[p]], "", "", median(sapply(res, function(r) r$analytic$ess))))
  cat(sprintf("CONTROL P: max |analytic std bias| %.3f (must be below 0.15)\n", max(abs(ca))))
  SH <- list()
  for (tg in names(TAGS)) {
    sh <- (get(tg, "mean") - am) / as; SH[[tg]] <- sh; b <- sb(get(tg, "mean") - tr); w <- colMeans(get(tg, "sd") / as)
    for (p in PARS6) cat(sprintf("%-9s %-11s %+9.3f %+8.3f (%.3f) %9.3f %9.0f\n", tg, p, b[[p]], mean(sh[, p]), sd(sh[, p]) / sqrt(nrow(sh)), w[[p]],
                                 median(sapply(res, function(r) r[[tg]]$ess))))
  }
  saveRDS(SH, file.path(OUT, sprintf("P_shift_N%d.rds", N)))
}

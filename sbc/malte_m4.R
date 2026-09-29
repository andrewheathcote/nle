# Stage M4: density error (D), posterior shift (P) and the verdict, for the A = 0 RDM cell.
# The design, the scalars, the floors and the rule are registered in
# handover/MALTE_STAGES.md, "Stages M4 - M6, re-scoped 2026-09-29"; this script applies them.
# Score bias (G) is sbc/score_bias_race.R with SBR_CELL=A0 SBR_SKIP=200.
#
#   Rscript sbc/malte_m4.R density   <cores> HF=<tag> HP=<tag> O=<tag> OMED=<tag>
#   Rscript sbc/malte_m4.R postshift <cores> HF=... HP=... O=... OMED=...   [M4_ND=200 M4_N=400 M4_M=4000]
#   Rscript sbc/malte_m4.R verdict   1       HF=... HP=... O=... OMED=...   (needs the two above and sbc/malte_m4/G.rds)
a <- commandArgs(trailingOnly = TRUE)
MODE <- a[1]; CORES <- as.integer(a[2]); stopifnot(CORES <= 8L)
kv <- strsplit(a[-(1:2)], "="); TAGS <- setNames(sapply(kv, `[`, 2), sapply(kv, `[`, 1))
LIB <- path.expand("~/R/nle-sbc-lib"); .libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(parallel)})
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
source("port/R/flow_race.R"); source("sbc/malte_m4_lib.R")
OUT <- Sys.getenv("M4_OUT", "sbc/malte_m4"); dir.create(OUT, showWarnings = FALSE)
FL <- lapply(TAGS, m4_card)
REPORT <- 201:400                      # draws the selection never saw
trapz <- function(y, x) sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
cat(sprintf("M4 %s | flows %s | cores %d | EMC2 %s\n", MODE, paste(names(TAGS), TAGS, sep = " = ", collapse = ", "), CORES, find.package("EMC2")))

if (MODE == "density") {
  truth <- m4_truth()[REPORT, ]
  one <- function(i) {
    P <- m4_nat(as.list(truth[i, ]))
    do.call(rbind, lapply(1:2, function(k) {
      p <- P[[k]]
      wq <- function(lt, up) { w <- m4_wald(p, p[["t0"]] + exp(lt)); (if (up) w$lS else w$lF) - log(1e-10) }
      q <- c(uniroot(wq, c(-40, 25), up = FALSE, tol = 1e-10)$root, uniroot(wq, c(-40, 25), up = TRUE, tol = 1e-10)$root)
      t <- exp(seq(q[1], q[2], length.out = 4000)); rt <- p[["t0"]] + t; n <- length(t)
      ref <- m4_wald(p, rt); pr <- exp(ref$lp)
      # the analytic density the score-bias and posterior-shift controls use, against this file's Wald
      an <- EMC2:::dRDM(rt, matrix(p, n, 5, byrow = TRUE, dimnames = list(NULL, names(p))))
      r <- sapply(names(FL), function(f) {
        e <- m4_flow(FL[[f]], p, rt)
        c(kl = exp(ref$lF[1]) * (ref$lF[1] - e$lF[1]) + trapz(pr * (ref$lp - e$lp), t) + exp(ref$lS[n]) * (ref$lS[n] - e$lS[n]),
          ks = max(abs(exp(ref$lF) - exp(e$lF))), w1 = trapz(abs(exp(ref$lF) - exp(e$lF)), t),
          mad = trapz(pr * abs(ref$lp - e$lp), t))
      })
      data.frame(draw = REPORT[i], acc = k, v = p[["v"]], b = p[["B"]], flow = rep(colnames(r), each = nrow(r)),
                 stat = rep(rownames(r), ncol(r)), value = c(r), mass = trapz(pr, t) + exp(ref$lF[1]) + exp(ref$lS[n]),
                 emc2 = max(abs(an - pr) / max(pr)))
    }))
  }
  r <- do.call(rbind, mclapply(seq_len(nrow(truth)), one, mc.cores = CORES))
  cat(sprintf("CONTROL D: Wald mass on the grid in [%.8f, %.8f]; EMC2 dRDM(A = 0) against the closed form, max |diff| / peak %.1e\n",
              min(r$mass), max(r$mass), max(r$emc2)))
  for (f in names(FL)) for (s in c("kl", "ks", "mad", "w1")) { x <- r$value[r$flow == f & r$stat == s]
    cat(sprintf("D %-5s %-3s median %.3e  90%% %.3e  max %.3e  (n %d)\n", f, s, median(x), quantile(x, .9), max(x), length(x))) }
  saveRDS(r, file.path(OUT, "D.rds"))
}

if (MODE == "postshift") {
  ND <- as.integer(Sys.getenv("M4_ND", "200")); N <- as.integer(Sys.getenv("M4_N", "400")); M <- as.integer(Sys.getenv("M4_M", "4000"))
  ACC <- c(list(analytic = function(p, rt) m4_wald(p, rt)), lapply(FL, function(fl) function(p, rt) m4_flow(fl, p, rt)))
  lprior <- function(x) sum(dnorm(x, PRIOR_M, PRIOR_S, log = TRUE))
  lpost <- function(acc, x, y) lprior(x) + m4_loglik(acc, as.list(setNames(x, PARS4)), y)
  laplace <- function(acc, x0, y) {
    nlp <- function(x) { v <- -lpost(acc, x, y); if (is.finite(v)) v else 1e10 }
    o <- optim(x0, nlp, method = "BFGS", control = list(maxit = 500, reltol = 1e-10))
    H <- optimHess(o$par, nlp); e <- eigen((H + t(H)) / 2, symmetric = TRUE)
    ev <- pmax(e$values, min(1 / PRIOR_S^2))
    list(m = o$par, S = e$vectors %*% diag(1 / ev) %*% t(e$vectors))
  }
  dmvn <- function(X, m, S) { L <- chol(S); z <- backsolve(L, t(X) - m, transpose = TRUE)
    -colSums(z^2) / 2 - sum(log(diag(L))) - ncol(X) / 2 * log(2 * pi) }
  rmvn <- function(n, m, S) sweep(matrix(rnorm(n * length(m)), n) %*% chol(S), 2, m, "+")
  wstats <- function(lw, X) { w <- exp(lw - max(lw)); w[!is.finite(w)] <- 0; w <- w / sum(w); mu <- colSums(w * X)
    list(mean = mu, sd = sqrt(colSums(w * sweep(X, 2, mu)^2)), ess = 1 / sum(w^2)) }
  one <- function(i) {
    set.seed(7000 + i)
    truth <- setNames(rnorm(4, PRIOR_M, PRIOR_S), PARS4)
    y <- m4_sim(as.list(truth), N)
    comps <- lapply(ACC, function(acc) tryCatch(laplace(acc, truth, y), error = function(e) NULL))
    if (any(sapply(comps, is.null))) return(NULL)
    K <- length(comps)
    X <- do.call(rbind, lapply(comps, function(cc) rmvn(M %/% K, cc$m, 1.5^2 * cc$S))); colnames(X) <- PARS4
    Q <- sapply(comps, function(cc) dmvn(X, cc$m, 1.5^2 * cc$S)); qm <- apply(Q, 1, max)
    lq <- qm + log(rowMeans(exp(Q - qm)))
    c(list(truth = truth), lapply(ACC, function(acc) wstats(apply(X, 1, function(x) lpost(acc, x, y)) - lq, X)))
  }
  t0 <- proc.time()[3]
  res <- Filter(Negate(is.null), mclapply(seq_len(ND), one, mc.cores = CORES))
  cat(sprintf("usable datasets %d of %d, N = %d, %d draws, %.1f min at %d cores\n", length(res), ND, N, M, (proc.time()[3] - t0) / 60, CORES))
  saveRDS(list(tags = TAGS, N = N, res = res), file.path(OUT, sprintf("P_N%d.rds", N)))
}

if (MODE %in% c("postshift", "verdict")) {
  N <- as.integer(Sys.getenv("M4_N", "400"))
  res <- readRDS(file.path(OUT, sprintf("P_N%d.rds", N)))$res
  get <- function(k, s) t(sapply(res, function(r) r[[k]][[s]]))
  tr <- t(sapply(res, `[[`, "truth")); am <- get("analytic", "mean"); as <- get("analytic", "sd")
  sb <- function(e) colMeans(e) / apply(e, 2, sd)
  cat(sprintf("\nP, N = %d, %d datasets. Predicted standardised bias, shift from the analytic posterior in its sd (se), sd ratio\n", N, length(res)))
  cat(sprintf("%-9s %-9s %9s %16s %9s %9s\n", "flow", "par", "std bias", "shift (se)", "sd ratio", "med ESS"))
  ca <- sb(am - tr)
  for (p in PARS4) cat(sprintf("%-9s %-9s %+9.3f %16s %9s %9.0f\n", "analytic", p, ca[[p]], "", "", median(sapply(res, function(r) r$analytic$ess))))
  cat(sprintf("CONTROL P: max |analytic std bias| %.3f (must be below 0.15)\n", max(abs(ca))))
  SH <- list()
  for (f in names(TAGS)) {
    sh <- (get(f, "mean") - am) / as; SH[[f]] <- sh; b <- sb(get(f, "mean") - tr); w <- colMeans(get(f, "sd") / as)
    for (p in PARS4) cat(sprintf("%-9s %-9s %+9.3f %+8.3f (%.3f) %9.3f %9.0f\n", f, p, b[[p]], mean(sh[, p]), sd(sh[, p]) / sqrt(nrow(sh)), w[[p]],
                                 median(sapply(res, function(r) r[[f]]$ess))))
  }
  saveRDS(SH, file.path(OUT, sprintf("P_shift_N%d.rds", N)))
}

if (MODE == "verdict") {
  set.seed(20260929); B <- 2000
  D <- readRDS(file.path(OUT, "D.rds")); G <- readRDS(file.path(OUT, "G.rds")); SH <- readRDS(file.path(OUT, "P_shift_N400.rds"))
  ess <- sapply(names(TAGS), function(f) median(sapply(readRDS(file.path(OUT, "P_N400.rds"))$res, function(r) r[[f]]$ess)))
  # per-unit matrices: rows are the bootstrap units, the scalar is a function of the rows
  klm <- sapply(names(TAGS), function(f) { x <- D[D$flow == f & D$stat == "kl", ]; tapply(x$value, x$draw, c) }, simplify = FALSE)
  Dmat <- lapply(klm, function(l) do.call(rbind, l))                       # draws x 2 accumulators
  Gmat <- lapply(TAGS, function(tg) sapply(PARS4, function(p) if (p == "t0") G[, paste0("t0.", tg)] - G[, "t0.analytic"] else G[, paste0(p, ".", tg)]))
  scal <- list(D = function(m) median(m), G = function(m) max(abs(colMeans(m))), P = function(m) max(abs(colMeans(m))))
  mats <- list(D = Dmat, G = Gmat, P = SH); floor <- c(D = 1e-4, G = 0.005, P = 0.10)
  ctl_G <- max(abs(colMeans(G[, paste0(setdiff(PARS4, "t0"), ".analytic")])))
  tr <- t(sapply(readRDS(file.path(OUT, "P_N400.rds"))$res, `[[`, "truth")); am <- t(sapply(readRDS(file.path(OUT, "P_N400.rds"))$res, function(r) r$analytic$mean))
  ctl_P <- max(abs(colMeans(am - tr) / apply(am - tr, 2, sd)))
  cat(sprintf("\nCONTROLS  G analytic %.5f (< 0.001: %s)   P analytic %.3f (< 0.15: %s)   median ESS %s\n", ctl_G, ctl_G < 1e-3, ctl_P, ctl_P < .15,
              paste(names(ess), round(ess), collapse = ", ")))
  cat(sprintf("\n%-3s %-5s %12s %12s\n", "", "flow", "scalar", "boot se"))
  X <- list()
  for (k in names(mats)) for (f in names(TAGS)) { m <- mats[[k]][[f]]; n <- nrow(m)
    bs <- replicate(B, scal[[k]](m[sample.int(n, n, TRUE), , drop = FALSE]))
    X[[k]][[f]] <- scal[[k]](m); cat(sprintf("%-3s %-5s %12.3e %12.2e\n", k, f, X[[k]][[f]], sd(bs))) }
  read1 <- function(k, o, h) { mo <- mats[[k]][[o]]; mh <- mats[[k]][[h]]; n <- nrow(mo); stopifnot(nrow(mh) == n)
    d <- replicate(B, { i <- sample.int(n, n, TRUE); scal[[k]](mo[i, , drop = FALSE]) - scal[[k]](mh[i, , drop = FALSE]) })
    xo <- X[[k]][[o]]; xh <- X[[k]][[h]]; se <- sd(d)
    r <- if ((xo < floor[[k]] && xh < floor[[k]]) || abs(xo - xh) <= 2 * se) "tie" else if (xo < xh && xh >= floor[[k]]) "ours better" else if (xo > xh && xo >= floor[[k]]) "ours worse" else "tie"
    cat(sprintf("   %s: %s %.3e, %s %.3e, difference %+.3e (se %.2e), floor %g -> %s\n", k, o, xo, h, xh, xo - xh, se, floor[[k]], r)); r }
  for (o in intersect(c("O", "OMED"), names(TAGS))) for (h in intersect(c("HF", "HP"), names(TAGS))) {
    cat(sprintf("\n%s against %s\n", o, h)); r <- sapply(names(mats), read1, o = o, h = h)
    invalid <- !(ctl_G < 1e-3) || !(ctl_P < .15) || ess[[o]] < 200 || ess[[h]] < 200
    v <- if (invalid) "NOT SEPARATED (a screen is invalid): SBC cell needed" else if (!any(r == "ours worse")) "OURS AS GOOD OR BETTER: no SBC"
         else if (!any(r == "ours better") && r[["P"]] == "ours worse") "HIS BETTER: no SBC" else "NOT SEPARATED: SBC cell needed"
    cat(sprintf("VERDICT %s vs %s: %s%s\n", o, h, v, if (o == "OMED") "   (outside the registered verdict: the unselected checkpoint)" else ""))
  }
}

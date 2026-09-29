# Stage M6a: CRDM screens for the dt 0.0005 / 0.005 / 0.05 cards, against the Volterra reference
# (dt 1.25e-4, t_max 6). Design registered in handover/MALTE_STAGES.md, "Stage M6a" (the cell of
# sbc/gate_malte_dev_nle.R check 6). One pulsed accumulator (the card, or the Volterra reference)
# races a Wald (amp = 0 exactly, closed form, identical for card and reference).
#
#   Rscript sbc/malte_m6.R nest      <cores> CARD=<tag>                       # amp = 0 vs Wald
#   Rscript sbc/malte_m6.R G         <cores> <tag1>=<name1> [<tag2>=<name2> ...]   [M6_ND=200]
#   Rscript sbc/malte_m6.R P         <cores> <tag1>=<name1> [...]             [M6_ND=200 M6_N=400 M6_M=4000]
#   Rscript sbc/malte_m6.R screen    1       <tag1>=<name1> [...]             (needs G.rds, P_N400.rds)
# Smoke: set M6_SMOKE=1 to cut ND (and, for postshift, N) down for a masked dry run.
a <- commandArgs(trailingOnly = TRUE)
MODE <- a[1]; CORES <- as.integer(a[2]); stopifnot(CORES <= 8L)
kv <- strsplit(a[-(1:2)], "="); TAGS <- setNames(sapply(kv, `[`, 2), sapply(kv, `[`, 1))
LIB <- path.expand("~/R/nle-sbc-lib"); .libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(parallel); library(Rcpp)})
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
source("port/R/flow_race.R"); sourceCpp("port/src/crdm.cpp"); source("sbc/malte_reference.R"); source("sbc/malte_m6_lib.R")
OUT <- Sys.getenv("M6_OUT", "sbc/malte_m6"); dir.create(OUT, showWarnings = FALSE)
SMOKE <- nzchar(Sys.getenv("M6_SMOKE"))
FL <- lapply(TAGS, m6_card)
trapz <- function(y, x) sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
cat(sprintf("M6 %s%s | cards %s | cores %d | EMC2 %s\n", MODE, if (SMOKE) " [SMOKE: masked, code check only]" else "",
            paste(names(TAGS), TAGS, sep = " = ", collapse = ", "), CORES, find.package("EMC2")))

## ---- nesting check: the card at amp = 0 (network, not the hybrid switch) vs the Wald --------
if (MODE == "nest") {
  fl <- FL[[1]]; th <- as.list(PRIOR_M6)
  one <- function(cell) {
    nat <- m6_nat(th)[[cell]]
    p <- nat$pulsed; p[["amp"]] <- 0                       # force amp = 0, still routed to the NETWORK
    wq <- function(lt, up) { w <- m6_wald(p, p[["t0"]] + exp(lt)); (if (up) w$lS else w$lF) - log(1e-10) }
    q <- c(uniroot(wq, c(-40, 25), up = FALSE, tol = 1e-10)$root, uniroot(wq, c(-40, 25), up = TRUE, tol = 1e-10)$root)
    t <- exp(seq(q[1], q[2], length.out = 4000)); rt <- p[["t0"]] + t; n <- length(t)
    ref <- m6_wald(p, rt); pr <- exp(ref$lp); ev <- m6_flow(fl, p, rt)
    if (!ev$in_box) return(c(cell = cell, kl = NA, ks = NA, mad = NA, mass = NA))
    ev$lF <- log(pmax(-expm1(ev$lS), 1e-300))               # censored-ends KL, as sbc/malte_m4.R's density mode
    kl <- exp(ref$lF[1]) * (ref$lF[1] - ev$lF[1]) + trapz(pr * (ref$lp - ev$lp), t) + exp(ref$lS[n]) * (ref$lS[n] - ev$lS[n])
    ks <- max(abs(exp(ref$lF) - exp(ev$lF)))
    mad <- trapz(pr * abs(ref$lp - ev$lp), t)
    c(cell = cell, kl = kl, ks = ks, mad = mad, mass = trapz(pr, t) + exp(ref$lF[1]) + exp(ref$lS[n]))
  }
  r <- do.call(rbind, lapply(c("cong", "incong"), one))
  print(r)
  kl <- max(as.numeric(r[, "kl"])); ks <- max(as.numeric(r[, "ks"]))
  cat(sprintf("NESTING %s: max KL %.3e, max KS %.3e (tests the flow at amp = 0, not the solver)\n", TAGS[1], kl, ks))
}

## ---- G: score bias, central difference, H = 0.01, grid weighted by the Volterra reference ---
if (MODE == "G") {
  ND <- as.integer(Sys.getenv("M6_ND", if (SMOKE) "4" else "200"))
  H <- 0.01
  set.seed(60000)
  draws <- do.call(rbind, lapply(PARS6, function(p) rnorm(ND, PRIOR_M6[[p]], PRIOR_S6[[p]])))
  rownames(draws) <- PARS6; draws <- t(draws)
  models <- c(list(analytic = function(p, rt) m6_volterra(p, rt)),
              lapply(FL, function(fl) function(p, rt) m6_flow(fl, p, rt)))
  Lbar <- function(th, grid, f) {
    num <- 0; den <- 0
    for (cell in c("cong", "incong")) {
      w <- grid$w[[cell]]; ok <- w > 0; if (!any(ok)) next
      nat <- m6_nat(th)[[cell]]; e <- f(nat$pulsed, grid$rt[[cell]][ok])
      wok <- w[ok]; ok2 <- is.finite(e$lp)                  # e$lp and wok are both already length sum(ok)
      num <- num + sum(wok[ok2] * e$lp[ok2]); den <- den + sum(wok[ok2])
    }
    if (den <= 0) NA_real_ else num / den
  }
  PARS6o <- setdiff(PARS6, "t0")     # t0's hard support boundary needs the restricted-region grid below
  mkgrid <- function(nat, cell, lo) {
    t0n <- nat[[cell]]$pulsed[["t0"]]
    rt <- lo + exp(seq(log(1e-4), log(20), length.out = 300))
    dr <- c(diff(rt)[1], (head(diff(rt), -1) + tail(diff(rt), -1)) / 2, tail(diff(rt), 1))
    ref <- m6_volterra(nat[[cell]]$pulsed, rt)
    list(rt = rt, w = exp(ref$lp) * dr)
  }
  one <- function(i) {
    th0 <- as.list(draws[i, ])
    nat <- m6_nat(th0)
    grid <- list(rt = list(), w = list())
    for (cell in c("cong", "incong")) { g <- mkgrid(nat, cell, nat[[cell]]$pulsed[["t0"]]); grid$rt[[cell]] <- g$rt; grid$w[[cell]] <- g$w }
    out <- sapply(PARS6o, function(p) {
      tp <- th0; tm <- th0; tp[[p]] <- tp[[p]] + H; tm[[p]] <- tm[[p]] - H
      sapply(models, function(f) (Lbar(tp, grid, f) - Lbar(tm, grid, f)) / (2 * H))
    })
    # t0: as sbc/score_bias_race.R, restricted to rt >= cut (past the perturbed onset); target is
    # d/dt0 log P(rt >= cut | theta), not 0, so the flow is reported relative to the analytic value.
    grid_t0 <- list(rt = list(), w = list())
    for (cell in c("cong", "incong")) {
      cut <- nat[[cell]]$pulsed[["t0"]] * exp(H) * 1.05
      g <- mkgrid(nat, cell, cut); grid_t0$rt[[cell]] <- g$rt; grid_t0$w[[cell]] <- g$w
    }
    tp <- th0; tm <- th0; tp[["t0"]] <- tp[["t0"]] + H; tm[["t0"]] <- tm[["t0"]] - H
    t0v <- sapply(models, function(f) (Lbar(tp, grid_t0, f) - Lbar(tm, grid_t0, f)) / (2 * H))
    cbind(out, t0 = t0v)   # rows: models (analytic, tags); cols: PARS6
  }
  t0 <- proc.time()[3]
  res <- mclapply(seq_len(ND), one, mc.cores = CORES)
  cat(sprintf("G: %d draws, %.1f min at %d cores (%.1f s / draw)\n", ND, (proc.time()[3] - t0) / 60, CORES, (proc.time()[3] - t0) / ND))
  ok <- sapply(res, function(r) all(is.finite(r)))
  cat(sprintf("usable draws: %d / %d\n", sum(ok), ND)); res <- res[ok]
  arr <- array(unlist(res), dim = c(length(models), length(PARS6), length(res)), dimnames = list(names(models), PARS6, NULL))
  an_t0 <- arr["analytic", "t0", ]        # not 0 by design (hard t0 boundary); flows are reported relative to it
  for (tg in names(FL)) arr[tg, "t0", ] <- arr[tg, "t0", ] - an_t0
  G <- t(sapply(seq_along(res), function(i) c(arr[, , i])))
  colnames(G) <- as.vector(outer(PARS6, names(models), function(p, m) paste0(p, ".", m)))
  cat(sprintf("\nCONTROL G: max |g(analytic)| over %s %.5f (must be below 1e-3; t0 excluded, see above)\n",
              paste(PARS6o, collapse = ", "), max(abs(rowMeans(arr["analytic", PARS6o, ])))))
  cat(sprintf("%-12s %s\n", "flow", paste(sprintf("%10s", PARS6), collapse = " ")))
  for (m in names(models)) cat(sprintf("%-12s %s\n", m, paste(sprintf("%+10.5f", rowMeans(arr[m, , ])), collapse = " ")))
  saveRDS(list(tags = TAGS, draws = draws[ok, ], G = G, arr = arr), file.path(OUT, "G.rds"))
}

## ---- P: posterior shift, Laplace approximation, defensive mixture importance sampling -------
if (MODE == "P") {
  ND <- as.integer(Sys.getenv("M6_ND", if (SMOKE) "4" else "200"))
  N <- as.integer(Sys.getenv("M6_N", if (SMOKE) "50" else "400")); M <- as.integer(Sys.getenv("M6_M", "4000"))
  ACC <- c(list(analytic = function(p, rt) m6_volterra(p, rt)), lapply(FL, function(fl) function(p, rt) m6_flow(fl, p, rt)))
  lprior <- function(x) sum(dnorm(x, PRIOR_M6[PARS6], PRIOR_S6[PARS6], log = TRUE))
  lpost <- function(pf, x, y) lprior(x) + m6_loglik(pf, as.list(setNames(x, PARS6)), y)
  laplace <- function(pf, x0, y) {
    nlp <- function(x) { v <- -lpost(pf, x, y); if (is.finite(v)) v else 1e10 }
    o <- optim(x0, nlp, method = "BFGS", control = list(maxit = 200, reltol = 1e-8))
    H <- optimHess(o$par, nlp); e <- eigen((H + t(H)) / 2, symmetric = TRUE)
    ev <- pmax(e$values, min(1 / PRIOR_S6[PARS6]^2))
    list(m = o$par, S = e$vectors %*% diag(1 / ev) %*% t(e$vectors))
  }
  dmvn <- function(X, m, S) { L <- chol(S); z <- backsolve(L, t(X) - m, transpose = TRUE)
    -colSums(z^2) / 2 - sum(log(diag(L))) - ncol(X) / 2 * log(2 * pi) }
  rmvn <- function(n, m, S) sweep(matrix(rnorm(n * length(m)), n) %*% chol(S), 2, m, "+")
  wstats <- function(lw, X) { w <- exp(lw - max(lw)); w[!is.finite(w)] <- 0; w <- w / sum(w); mu <- colSums(w * X)
    list(mean = mu, sd = sqrt(colSums(w * sweep(X, 2, mu)^2)), ess = 1 / sum(w^2)) }
  one <- function(i) {
    set.seed(60000 + i)
    truth <- setNames(rnorm(length(PARS6), PRIOR_M6[PARS6], PRIOR_S6[PARS6]), PARS6)
    y <- m6_sim(as.list(truth), N)
    comps <- lapply(ACC, function(pf) tryCatch(laplace(pf, truth, y), error = function(e) NULL))
    if (any(sapply(comps, is.null))) return(NULL)
    K <- length(comps)
    X <- do.call(rbind, lapply(comps, function(cc) rmvn(M %/% K, cc$m, 1.5^2 * cc$S))); colnames(X) <- PARS6
    Q <- sapply(comps, function(cc) dmvn(X, cc$m, 1.5^2 * cc$S)); qm <- apply(Q, 1, max)
    lq <- qm + log(rowMeans(exp(Q - qm)))
    c(list(truth = truth), lapply(ACC, function(pf) wstats(apply(X, 1, function(x) lpost(pf, x, y)) - lq, X)))
  }
  t0 <- proc.time()[3]
  res <- Filter(Negate(is.null), mclapply(seq_len(ND), one, mc.cores = CORES))
  cat(sprintf("P: usable datasets %d of %d, N = %d, %d draws, %.1f min at %d cores (%.1f s / dataset)\n",
              length(res), ND, N, M, (proc.time()[3] - t0) / 60, CORES, (proc.time()[3] - t0) / max(1, ND)))
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

## ---- screen: PASS/FAIL per card, from the saved G and P results -----------------------------
if (MODE == "screen") {
  N <- as.integer(Sys.getenv("M6_N", "400"))
  Gr <- readRDS(file.path(OUT, "G.rds")); Pl <- readRDS(file.path(OUT, sprintf("P_N%d.rds", N)))
  SH <- readRDS(file.path(OUT, sprintf("P_shift_N%d.rds", N)))
  ess <- sapply(names(TAGS), function(tg) median(sapply(Pl$res, function(r) r[[tg]]$ess)))
  tr <- t(sapply(Pl$res, `[[`, "truth")); am <- t(sapply(Pl$res, function(r) r$analytic$mean))
  ctl_P <- max(abs(colMeans(am - tr) / apply(am - tr, 2, sd)))
  ctl_G <- max(abs(rowMeans(Gr$arr["analytic", setdiff(PARS6, "t0"), ])))   # t0 excluded: not 0 by design
  cat(sprintf("\nCONTROLS  G analytic %.5f (< 0.001: %s)   P analytic %.3f (< 0.15: %s)\n", ctl_G, ctl_G < 1e-3, ctl_P, ctl_P < .15))
  for (tg in names(TAGS)) {
    gmax <- max(abs(rowMeans(Gr$arr[tg, , ])))
    pmax_ <- max(abs(colMeans(SH[[tg]])))
    essOK <- ess[[tg]] > 200 & median(sapply(Pl$res, function(r) r$analytic$ess)) > 200
    which_fail <- c(if (!(ctl_G < 1e-3)) "G control", if (!(ctl_P < .15)) "P control", if (!essOK) "ESS",
                     if (gmax >= 0.005) "G", if (pmax_ >= 0.10) "P")
    pass <- length(which_fail) == 0
    cat(sprintf("\ncard %s: G max|g| %.5f (floor 0.005), P max|shift| %.3f posterior sd (floor 0.10), median ESS %.0f\n",
                TAGS[tg], gmax, pmax_, ess[[tg]]))
    cat(sprintf("SCREEN %s: %s\n", TAGS[tg], if (pass) "PASS" else sprintf("FAIL (%s)", paste(which_fail, collapse = ", "))))
  }
}

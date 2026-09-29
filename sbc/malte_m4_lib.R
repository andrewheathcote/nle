# Shared by the M4 scripts: one accumulator of the A = 0 RDM cell, under any card and under
# the closed-form Wald. Registered design: handover/MALTE_STAGES.md, "Stages M4 - M6, re-scoped".
# Needs port/R/flow_race.R sourced.
PARS4 <- c("v", "v_lMTRUE", "B", "t0")
PRIOR_M <- c(v = log(1.5), v_lMTRUE = .4, B = log(1), t0 = log(.2))
PRIOR_S <- c(v = .1, v_lMTRUE = .1, B = .1, t0 = .1)

# the 500 stored prior draws, A dropped
m4_truth <- function() {
  x <- readRDS("sbc/nci/results/RDMnn_N100_R500.rds")
  tr <- as.matrix(x$SBC$med$alpha) - as.matrix(x$SBC$bias$alpha)
  tr[, PARS4, drop = FALSE]
}
# sampled scale -> natural parameters of the two accumulators (1 = mismatch, 2 = match)
m4_nat <- function(th) {
  b <- exp(th[["B"]]); t0 <- exp(th[["t0"]])
  list(c(v = exp(th[["v"]]), B = b, A = 0, t0 = t0, s = 1),
       c(v = exp(th[["v"]] + th[["v_lMTRUE"]]), B = b, A = 0, t0 = t0, s = 1))
}
# the network's input for natural parameters p, in the card's order and coordinates
# (the rules of sbc/score_bias_race.R: zerobox inputs exact at 0, b = B + A, natural or log)
m4_context <- function(fl, p) {
  cn <- unlist(fl$context_names); natural <- identical(fl$context_scale, "natural")
  zb <- names(fl$zerobox_c)
  sapply(cn, function(nm) {
    if (nm %in% zb) return(asinh(p[[nm]] / fl$zerobox_c[[nm]]))
    x <- if (nm == "b") p[["B"]] + p[["A"]] else p[[nm]]
    if (nm == "A") stop("card conditions on log A: not an A = 0 card")
    if (natural) x else log(x)
  })
}
# log pdf, log survival and cdf of one accumulator at response times rt
m4_flow <- function(fl, p, rt) {
  t_arg <- if (identical(fl$pre, "t0")) rt - p[["t0"]] else rt
  ok <- t_arg > 0
  lp <- rep(-Inf, length(rt)); lS <- rep(0, length(rt)); z <- rep(-Inf, length(rt))
  if (any(ok)) {
    ev <- flow_eval(fl, m4_context(fl, p), t_arg[ok])
    if (!ev$in_box) return(list(lp = rep(-Inf, length(rt)), lS = rep(-Inf, length(rt)), lF = rep(-Inf, length(rt)), in_box = FALSE))
    lp[ok] <- ev$log_pdf; lS[ok] <- ev$log_sf; z[ok] <- ev$z
  }
  list(lp = lp, lS = lS, lF = pnorm(z, log.p = TRUE), in_box = TRUE)
}
m4_wald <- function(p, rt) {
  t <- rt - p[["t0"]]; v <- p[["v"]]; s <- p[["s"]]; b <- p[["B"]] + p[["A"]]
  ok <- t > 0; lp <- rep(-Inf, length(rt)); lS <- rep(0, length(rt)); t <- t[ok]
  lp[ok] <- log(b) - log(s) - 0.5 * log(2 * pi) - 1.5 * log(t) - (b - v * t)^2 / (2 * s^2 * t)
  z1 <- (v * t - b) / (s * sqrt(t)); z2 <- -(v * t + b) / (s * sqrt(t))
  la <- pnorm(-z1, log.p = TRUE); lb <- 2 * v * b / s^2 + pnorm(z2, log.p = TRUE)
  lS[ok] <- la + log1p(-pmin(exp(lb - la), 1))
  list(lp = lp, lS = lS, lF = log(pmax(-expm1(lS), 1e-300)), in_box = TRUE)
}
# total log likelihood of a race data set; y$rt, y$win (1 = mismatching accumulator won, 2 = matching)
m4_loglik <- function(acc, th, y) {
  P <- m4_nat(th)
  if (min(y$rt) <= P[[1]][["t0"]]) return(-Inf)
  e <- lapply(P, function(p) acc(p, y$rt))
  if (!e[[1]]$in_box || !e[[2]]$in_box) return(-Inf)
  w2 <- y$win == 2
  ll <- sum(e[[2]]$lp[w2] + e[[1]]$lS[w2]) + sum(e[[1]]$lp[!w2] + e[[2]]$lS[!w2])
  if (is.finite(ll)) ll else -Inf
}
# exact inverse-Gaussian draws (Michael, Schucany and Haas, 1976)
rig <- function(n, mu, lam) {
  y <- rnorm(n)^2
  x <- mu + mu^2 * y / (2 * lam) - mu / (2 * lam) * sqrt(4 * mu * lam * y + mu^2 * y^2)
  ifelse(runif(n) <= mu / (mu + x), x, mu^2 / x)
}
m4_sim <- function(th, n) {
  P <- m4_nat(th)
  T1 <- rig(n, P[[1]][["B"]] / P[[1]][["v"]], P[[1]][["B"]]^2)
  T2 <- rig(n, P[[2]][["B"]] / P[[2]][["v"]], P[[2]][["B"]]^2)
  list(rt = P[[1]][["t0"]] + pmin(T1, T2), win = ifelse(T2 < T1, 2L, 1L))
}
m4_card <- function(tag) {
  f <- file.path("port/export", c(paste0(tag, "_weights.json"), paste0(tag, ".json")))
  flow_load(f[file.exists(f)][1])
}

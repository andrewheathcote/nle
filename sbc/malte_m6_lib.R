# Shared by the M6 scripts: the CRDM screening cell (a pulsed accumulator racing a Wald),
# under any card and under the Volterra reference. Registered design: handover/MALTE_STAGES.md,
# "Stage M6a" (the cell of sbc/gate_malte_dev_nle.R check 6).
# Needs port/R/flow_race.R sourced, Rcpp::sourceCpp("port/src/crdm.cpp") run, and
# sbc/malte_reference.R sourced (wald_pdf/cdf, row_groups, CRDM_ref, malte_ref_model).
PARS6 <- c("v", "v_lMTRUE", "B", "t0", "tau", "amp_lDTRUE")
PRIOR_M6 <- c(v = log(2), v_lMTRUE = .5, B = log(1), t0 = log(.3), tau = log(.1), amp_lDTRUE = log(.4))
PRIOR_S6 <- c(v = .15, v_lMTRUE = .1, B = .15, t0 = .1, tau = .2, amp_lDTRUE = .15)

# the design of check 6: two accumulators (left, right); lM = response matches stimulus S;
# lD = response matches distractor D. v depends on lM (mismatch / match), amp on lD (0 / card).
DES6 <- list(factors = list(subjects = 1, S = c("left", "right"), D = c("left", "right")), Rlevels = c("left", "right"),
             matchfun = function(d) d$S == d$lR,
             functions = list(lD = function(d) factor(d$D == d$lR, levels = c(FALSE, TRUE))))
F6 <- list(v ~ lM, B ~ 1, t0 ~ 1, tau ~ 1, amp ~ 0 + lD)
CONST6 <- c(amp_lDFALSE = -Inf)

# theta (sampled scale) -> the two trial types' (congruent: S == D, incongruent: S != D)
# pulsed and Wald accumulators, natural scale. The pulsed accumulator is always the one
# whose lR == D; it is the matching accumulator when congruent, the mismatching one when not.
m6_nat <- function(th) {
  v_b <- exp(th[["v"]]); v_m <- exp(th[["v"]] + th[["v_lMTRUE"]])
  B <- exp(th[["B"]]); t0 <- exp(th[["t0"]]); amp <- exp(th[["amp_lDTRUE"]]); tau <- exp(th[["tau"]])
  wald <- function(v) c(v = v, B = B, A = 0, t0 = t0, s = 1)
  pulsed <- function(v) c(v = v, B = B, A = 0, t0 = t0, s = 1, amp = amp, tau = tau)
  list(cong = list(pulsed = pulsed(v_m), wald = wald(v_b)),
       incong = list(pulsed = pulsed(v_b), wald = wald(v_m)))
}
# the network's input for natural parameters p (v, amp, tau, s, b), in the card's order/coordinates
m6_context <- function(fl, p) {
  cn <- unlist(fl$context_names); natural <- identical(fl$context_scale, "natural")
  zb <- names(fl$zerobox_c)
  sapply(cn, function(nm) {
    if (nm %in% zb) return(asinh(p[[nm]] / fl$zerobox_c[[nm]]))
    x <- if (nm == "b") p[["B"]] + p[["A"]] else p[[nm]]
    if (natural) x else log(x)
  })
}
# log pdf, log survival of the FLOW at the pulsed accumulator (card takes decision time)
m6_flow <- function(fl, p, rt) {
  t_arg <- if (identical(fl$pre, "t0")) rt - p[["t0"]] else rt
  ok <- t_arg > 0
  lp <- rep(-Inf, length(rt)); lS <- rep(0, length(rt))
  if (any(ok)) {
    ev <- flow_eval(fl, m6_context(fl, p), t_arg[ok])
    if (!ev$in_box) return(list(lp = rep(-Inf, length(rt)), lS = rep(-Inf, length(rt)), in_box = FALSE))
    lp[ok] <- ev$log_pdf; lS[ok] <- ev$log_sf
  }
  list(lp = lp, lS = lS, in_box = TRUE)
}
# closed-form Wald, as sbc/malte_m4_lib.R::m4_wald
m6_wald <- function(p, rt) {
  t <- rt - p[["t0"]]; v <- p[["v"]]; s <- p[["s"]]; b <- p[["B"]] + p[["A"]]
  ok <- t > 0; lp <- rep(-Inf, length(rt)); lS <- rep(0, length(rt)); t <- t[ok]
  lp[ok] <- log(b) - log(s) - 0.5 * log(2 * pi) - 1.5 * log(t) - (b - v * t)^2 / (2 * s^2 * t)
  z1 <- (v * t - b) / (s * sqrt(t)); z2 <- -(v * t + b) / (s * sqrt(t))
  la <- pnorm(-z1, log.p = TRUE); lb <- 2 * v * b / s^2 + pnorm(z2, log.p = TRUE)
  lS[ok] <- la + log1p(-pmin(exp(lb - la), 1))
  list(lp = lp, lS = lS, lF = log(pmax(-expm1(lS), 1e-300)), in_box = TRUE)
}
# Volterra reference at the pulsed accumulator: dt/t_max as in the registration (1.25e-4, 6)
m6_volterra <- function(p, rt, dt = 1.25e-4, t_max = 6) {
  t <- rt - p[["t0"]]; ok <- t > 0
  lp <- rep(-Inf, length(rt)); lS <- rep(0, length(rt))
  if (any(ok)) {
    o <- dcrdm_volterra(t[ok], p[["v"]], p[["amp"]], p[["tau"]], p[["s"]], p[["B"]] + p[["A"]], dt, t_max)
    lp[ok] <- log(pmax(o$pdf, 0)); lS[ok] <- log(pmax(1 - o$cdf, 1e-300))
  }
  list(lp = lp, lS = lS, in_box = TRUE)
}
# total log likelihood of one CRDM race dataset (y$rt, y$cong, y$win_pulsed), under a given
# pulsed-accumulator density function pf(p, rt) -> list(lp, lS). The Wald accumulator is the
# same closed form for every model (amp == 0 exactly, network or Volterra agree with it there).
MIN_LL6 <- log(1e-10)
m6_loglik <- function(pf, th, y) {
  nat <- m6_nat(th)
  ll <- numeric(length(y$rt))
  for (cell in c("cong", "incong")) {
    idx <- y$cong == (cell == "cong")
    if (!any(idx)) next
    pp <- nat[[cell]]$pulsed; wp <- nat[[cell]]$wald
    ep <- pf(pp, y$rt[idx]); if (!isTRUE(ep$in_box)) return(-Inf)
    ew <- m6_wald(wp, y$rt[idx])
    wpu <- y$win_pulsed[idx]
    ll[idx] <- ifelse(wpu, ep$lp + ew$lS, ew$lp + ep$lS)
  }
  ll <- pmax(ll, MIN_LL6)
  if (all(is.finite(ll))) sum(ll) else -Inf
}
# simulate a dataset of the M6a cell via EMC2's make_data on the CRDM design, ground truth the
# Volterra reference at the registered resolution (dt 1.25e-4, t_max 6, simulator step 5e-4).
m6_des <- suppressMessages(do.call(design, c(DES6, list(formula = F6, constants = CONST6,
  model = function() CRDM_ref(dt = 1.25e-4, t_max = 6, sim_dt = 5e-4), report_p_vector = FALSE))))
m6_sim <- function(th, n) {
  # make_data()'s n_trials is PER design cell (S x D, 4 cells here: gate check 5b's own
  # n_trials = 500 -> nrow(dat) == 2000), so n total trials needs n_trials = n / 4.
  dat <- make_data(unlist(th)[PARS6], m6_des, n_trials = ceiling(n / 4))
  list(rt = dat$rt, cong = as.character(dat$S) == as.character(dat$D), win_pulsed = as.character(dat$R) == as.character(dat$D))
}
m6_card <- function(tag) {
  f <- file.path("port/export", c(paste0(tag, "_weights.json"), paste0(tag, ".json")))
  flow_load(f[file.exists(f)][1])
}

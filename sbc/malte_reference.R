# Reference implementations, in plain R, of what dev-nle must provide for
# Malte Lueken's flows (stage M3). Two uses:
#   1. the specification handed to the EMC2 session (MALTE_FLOWS_PROMPT.md):
#      the package's compiled versions must agree with these;
#   2. the references sbc/gate_malte_dev_nle.R compares dev-nle against, and,
#      with GATE_MALTE_MOCK=1, stand-ins for dev-nle so the gate's own
#      arithmetic can be tested before dev-nle has the feature.
#
# Needs, already loaded by the caller:
#   source("port/R/flow_race.R")             flow_load(), flow_eval()   (stage M1)
#   Rcpp::sourceCpp("port/src/crdm.cpp")     rcrdm_acc(), dcrdm_volterra()  (stage M2)
#
# Parameterisation everywhere: EMC2's RDM parameters v, B, A, t0, s (natural
# scale), with A = 0 exactly (his flows and the Volterra solver have no
# start-point variability), so the network's threshold input b = B + A = B;
# CRDM adds amp (pulse height, exactly 0 = no pulse) and tau (pulse time
# constant). The flows model DECISION time, rt - t0 (card field pre = "t0").

## ---- closed-form Wald, start point 0, threshold b, drift v, noise s --------
wald_pdf <- function(t, v, s, b) b / (s * sqrt(2 * pi * t^3)) * exp(-(b - v * t)^2 / (2 * s^2 * t))
wald_cdf <- function(t, v, s, b) pnorm((v * t - b) / (s * sqrt(t))) +
  exp(2 * v * b / s^2 + pnorm(-(v * t + b) / (s * sqrt(t)), log.p = TRUE))
# exact draws (Michael, Schucany & Haas 1976): mean b / v, shape (b / s)^2
wald_r <- function(n, v, s, b) {
  mu <- b / v; lam <- (b / s)^2
  y <- rnorm(n)^2
  x <- mu + mu^2 * y / (2 * lam) - mu / (2 * lam) * sqrt(4 * mu * lam * y + mu^2 * y^2)
  ifelse(runif(n) <= mu / (mu + x), x, mu^2 / x)
}

## ---- grouping: one evaluation per distinct parameter row --------------------
# Exact (hexadecimal) keys, so rows equal to the last bit share a solve and
# rows that differ in the last bit do not.
row_groups <- function(X) {
  key <- do.call(paste, c(lapply(seq_len(ncol(X)), function(j) sprintf("%a", X[, j])), sep = "|"))
  split(seq_len(nrow(X)), factor(key, levels = unique(key)))
}

## ---- one accumulator: the network, or the Wald where amp is exactly 0 -------
# fl: a card loaded by flow_load(); nat: natural-scale parameter rows (columns
# v, B, A, t0, s and, for CRDM, amp, tau), one per rt. hybrid = TRUE: rows with
# amp exactly 0 are the closed-form Wald and never reach the network.
# Decision time <= 0: density 0, CDF 0. Rows outside the card's box (closed,
# natural scale): density 0, CDF 0 (flow_eval's sentinel).
malte_ref_acc <- function(fl, nat, rt, hybrid = FALSE) {
  stopifnot(nrow(nat) == length(rt), all(nat[, "A"] == 0))
  X <- cbind(nat, b = nat[, "B"] + nat[, "A"])
  ctx <- unlist(fl$context_names)
  t <- rt - X[, "t0"]
  pdf <- cdf <- numeric(length(rt)); net <- logical(length(rt))
  for (g in row_groups(X[, ctx, drop = FALSE])) {
    i <- g[t[g] > 0]
    if (!length(i)) next
    th <- X[g[1], ctx]
    if (hybrid && th[["amp"]] == 0) {
      pdf[i] <- wald_pdf(t[i], th[["v"]], th[["s"]], th[["b"]])
      cdf[i] <- wald_cdf(t[i], th[["v"]], th[["s"]], th[["b"]])
    } else {
      ev <- flow_eval(fl, unname(th), t[i])
      pdf[i] <- ev$pdf; cdf[i] <- ev$cdf; net[i] <- TRUE
    }
  }
  list(pdf = pdf, cdf = cdf, network = net)
}

## ---- race log-likelihood per trial, as log_likelihood_race assembles it -----
# dadm: EMC2 data-augmented design (one row per accumulator, accumulators of a
# trial contiguous); pars: natural-scale rows for it; acc(rt, pars) returns
# list(pdf, cdf). Winner: log pdf; losers: log(1 - cdf); floored at min_ll.
race_ref_ll <- function(dadm, pars, acc, min_ll = log(1e-10)) {
  w <- dadm$winner
  ev <- acc(dadm$rt, pars)
  lds <- ifelse(w, log(ev$pdf), log(1 - ev$cdf))
  lds[is.na(lds)] <- min_ll
  n_acc <- length(levels(dadm$R))
  ll <- lds[w] + colSums(matrix(lds[!w], nrow = n_acc - 1))
  pmax(min_ll, ll)[attr(dadm, "expand")]
}

## ---- the hand-written CRDM twin ---------------------------------------------
# An ordinary EMC2 model function: parameterisation, simulator and the Volterra
# reference density. dt: step of the Volterra grid (5e-4: error ~6e-4 of peak
# density, 86 ms per solve at t_max 3; 2e-3: ~6e-3, 5 ms). t_max: the grid
# reaches at least this far, and further when a group's largest decision time
# needs it (the recursion is causal, so a longer grid leaves earlier values
# unchanged). sim_dt: step of the simulator's grid.
#
# dfun/pfun: rows with amp exactly 0 are the closed-form Wald; every other
# distinct parameter row is ONE Volterra solve, interpolated at that row's
# decision times.
# rfun: unpulsed accumulators are exact Wald draws; a pulsed accumulator is
# simulated only as far as the fastest unpulsed accumulator of its trial (it
# matters only if it finishes first), so no horizon is assumed. A trial whose
# accumulators are all pulsed is simulated to t_max and must produce a finisher.
CRDM_ref <- function(dt = 5e-4, t_max = 3, sim_dt = 5e-4) {
  pnames <- c("v", "amp", "tau", "s", "b")
  dens <- function(rt, pars, what) {
    out <- numeric(length(rt))
    ok <- stats::complete.cases(pars[, c(pnames, "t0"), drop = FALSE])
    t <- rt - pars[, "t0"]
    ok <- ok & !is.na(t) & t > 0
    if (!any(ok)) return(out)
    ix <- which(ok)
    for (g in row_groups(pars[ix, pnames, drop = FALSE])) {
      i <- ix[g]; th <- pars[i[1], pnames]
      if (th[["amp"]] == 0) {
        out[i] <- if (what == "pdf") wald_pdf(t[i], th[["v"]], th[["s"]], th[["b"]]) else
          wald_cdf(t[i], th[["v"]], th[["s"]], th[["b"]])
      } else {
        tm <- max(t_max, ceiling(max(t[i]) / dt) * dt)
        out[i] <- dcrdm_volterra(t[i], th[["v"]], th[["amp"]], th[["tau"]], th[["s"]], th[["b"]], dt, tm)[[what]]
      }
    }
    out
  }
  list(
    type = "RACE",
    p_types = c(v = log(1), B = log(1), A = log(0), t0 = log(0), s = log(1), amp = log(0), tau = log(0.1)),
    transform = list(func = c(v = "exp", B = "exp", A = "exp", t0 = "exp", s = "exp", amp = "exp", tau = "exp")),
    # RDM()'s bounds; A admits only 0; amp = 0 (no pulse) admitted
    bound = list(minmax = cbind(v = c(1e-3, Inf), B = c(0, Inf), A = c(0, 0), t0 = c(0.05, Inf),
                                s = c(0, Inf), amp = c(0, Inf), tau = c(0, Inf)),
                 exception = c(A = 0, v = 0, amp = 0)),
    Ttransform = function(pars, dadm) cbind(pars, b = pars[, "B"] + pars[, "A"]),
    rfun = function(data = NULL, pars) {
      lR <- data$lR; nr <- length(levels(lR)); n <- nrow(pars)
      ok <- attr(pars, "ok"); if (is.null(ok)) ok <- rep(TRUE, n)
      if (any(pars[ok, "A"] != 0)) stop("CRDM: A must be exactly 0")
      trial <- rep(seq_len(n / nr), each = nr)
      ft <- rep(Inf, n)
      pulsed <- ok & pars[, "amp"] > 0
      wd <- which(ok & !pulsed)
      if (length(wd)) ft[wd] <- wald_r(length(wd), pars[wd, "v"], pars[wd, "s"], pars[wd, "b"])
      if (any(pulsed)) {
        best <- tapply(ft, trial, min)[trial]               # fastest unpulsed accumulator of the trial
        ix <- which(pulsed)
        for (g in row_groups(pars[ix, pnames, drop = FALSE])) {
          i <- ix[g]; th <- pars[i[1], pnames]
          hz <- max(ifelse(is.finite(best[i]), best[i], t_max))
          hz <- max(sim_dt, ceiling(hz / sim_dt) * sim_dt)
          ft[i] <- rcrdm_acc(length(i), th[["v"]], th[["amp"]], th[["tau"]], th[["s"]], th[["b"]], sim_dt, hz)
        }
      }
      dtm <- matrix(ft, nrow = nr)
      if (any(!is.finite(apply(dtm, 2, min))))
        stop("CRDM: a trial without an unpulsed accumulator had no finisher by t_max; raise t_max")
      R <- apply(dtm, 2, which.min)
      pick <- cbind(R, seq_len(ncol(dtm)))
      data.frame(R = factor(levels(lR)[R], levels = levels(lR)),
                 rt = matrix(pars[, "t0"], nrow = nr)[pick] + dtm[pick])
    },
    dfun = function(rt, pars) dens(rt, pars, "pdf"),
    pfun = function(rt, pars) dens(rt, pars, "cdf"),
    log_likelihood = function(pars, dadm, model, min_ll = log(1e-10))
      EMC2:::log_likelihood_race(pars = pars, dadm = dadm, model = model, min_ll = min_ll))
}

## ---- stand-in for register_nn_model() on a Malte card (mock mode only) ------
# The twin's parameterisation and simulator with the R port as likelihood: an
# ordinary R-path EMC2 model, no `nn` registration, no compiled likelihood.
malte_ref_model <- function(card, twin, hybrid = NULL) {
  fl <- flow_load(card); tl <- twin(); hy <- !is.null(hybrid)
  ml <- tl[c("type", "p_types", "transform", "bound", "rfun")]
  ml$Ttransform <- function(pars, dadm) cbind(pars, b = pars[, "B"] + pars[, "A"])
  ml$dfun <- function(rt, pars) malte_ref_acc(fl, pars, rt, hy)$pdf
  ml$pfun <- function(rt, pars) malte_ref_acc(fl, pars, rt, hy)$cdf
  ml$log_likelihood <- function(pars, dadm, model, min_ll = log(1e-10))
    EMC2:::log_likelihood_race(pars = pars, dadm = dadm, model = model, min_ll = min_ll)
  function() ml
}

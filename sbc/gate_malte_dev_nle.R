# Acceptance gate for Malte Lueken's flows on dev-nle, from the NLE side
# (stage M3; the prompt it accepts is SBC-staging/2026-09-24_dev-nle_phase6/
# MALTE_FLOWS_PROMPT.md). Runs on the three random-weight fixtures of stage M1,
# so it needs no trained card. Nine verdict lines, a few minutes at 8 cores.
#
#   Rscript sbc/gate_malte_dev_nle.R [library] [fixtures dir]     # from the repo root
#   GATE_MALTE_MOCK=1 Rscript sbc/gate_malte_dev_nle.R            # self-test of the gate
#
# References are this project's validated code: the R port (port/R/flow_race.R,
# 1e-11 against his evaluate_pdf_sf), the closed-form Wald, and the Volterra
# solver and simulator of port/src/crdm.cpp (1e-13 against his solver). They
# are assembled in sbc/malte_reference.R.
#
# Check 0 needs nothing new and must pass on any dev-nle: it proves the gate's
# own race assembly against the shipped rdm_small. Checks 1-6 fail until
# dev-nle has the feature.
#
# Mock mode replaces register_nn_model() and CRDM by the R references, which
# tests the gate's arithmetic and design; it says nothing about dev-nle, and
# the checks that need a registration (parts of 1, the compiled likelihood in
# 4, all of 6) are skipped.
a <- commandArgs(trailingOnly = TRUE)
LIB <- path.expand(if (length(a)) a[1] else "~/R/nle-sbc-lib")
FIX <- path.expand(if (length(a) > 1) a[2] else "malte/fixtures")
MOCK <- nzchar(Sys.getenv("GATE_MALTE_MOCK"))
.libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(jsonlite); library(Rcpp)})
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
source("port/R/flow_race.R")
sourceCpp("port/src/crdm.cpp")
source("sbc/malte_reference.R")
card <- function(name) file.path(FIX, name, "card.json")
stopifnot(file.exists(card("rdm_final")), file.exists(card("rdm_plain")), file.exists(card("crdm_final")))

npass <- 0L; nfail <- 0L; nskip <- 0L
verdict <- function(ok, lab) { ok <- isTRUE(as.logical(ok)); cat(sprintf("%s  %s\n", if (ok) "PASS" else "FAIL", lab)); if (ok) npass <<- npass + 1L else nfail <<- nfail + 1L; invisible(ok) }
skip <- function(lab) { cat(sprintf("SKIP  %s (mock)\n", lab)); nskip <<- nskip + 1L }
try_check <- function(expr) tryCatch(expr, error = function(e) { cat("   error:", conditionMessage(e), "\n"); FALSE })
cat(sprintf("=== Malte-flow acceptance%s  EMC2 %s at %s  %s ===\n", if (MOCK) "  [MOCK: tests the gate, not dev-nle]" else "",
            as.character(packageVersion("EMC2")), find.package("EMC2"), format(Sys.time(), "%Y-%m-%d %H:%M")))
t_all <- proc.time()[3]

# max |log a - log b| where the reference density is representable, and whether
# zeros agree. Random-weight flows give densities down to e-600; those count.
cmp_log <- function(new, ref) {
  fin <- ref > 1e-290
  c(diff = if (any(fin)) max(abs(log(new[fin]) - log(ref[fin]))) else 0,
    zeros_ok = all(new[ref == 0] == 0) && all(is.finite(new)), n = sum(fin), n0 = sum(ref == 0))
}
MIN_LL <- log(1e-10)
trialwise <- function(P, dadm, model) EMC2:::nn_trialwise(P, dadm, model)
nat_pars <- function(p, dadm, model) EMC2:::get_pars_matrix_oo(p, dadm, model)
# dev-nle's trial log-likelihoods against the reference race, for the rows of P
ll_vs_ref <- function(P, dat, des, acc) {
  model <- des$model                       # the design's model, as the kit and the sampler use it
  dadm <- EMC2:::nn_dadm(dat, des)
  new <- trialwise(P, dadm, model)
  worst <- 0; free <- 0L
  for (i in seq_len(nrow(P))) {
    ref <- race_ref_ll(dadm, nat_pars(P[i, ], dadm, model), acc, MIN_LL)
    worst <- max(worst, abs(new[, i] - ref)); free <- free + sum(ref > MIN_LL)
  }
  c(diff = worst, unfloored = free, n = length(new))
}
DES <- list(factors = list(subjects = 1, S = c("left", "right"), D = c("left", "right")), Rlevels = c("left", "right"),
            matchfun = function(d) d$S == d$lR,
            functions = list(lD = function(d) factor(d$D == d$lR, levels = c(FALSE, TRUE))))

## ---- 0. the gate's own race assembly, on what dev-nle already ships ---------
ok0 <- try_check({
  # the artefact dev-nle ships (checkpoint 195; port/export's e207 is a different checkpoint), through the R port
  fl <- readRDS(system.file("extdata", "flownn", "rdm_small.rds", package = "EMC2"))
  fl$mlp$layers <- lapply(fl$mlp$layers, function(l) list(W = as.matrix(l$W), b = as.numeric(l$b)))
  fl$affine <- FALSE; ctx <- unlist(fl$context_names)
  stopifnot(!isTRUE(fl$mlp$use_norm), all(unlist(fl$context_transforms) == "log"))
  des <- suppressMessages(design(factors = DES$factors[1:2], Rlevels = DES$Rlevels, matchfun = DES$matchfun,
                                 formula = list(v ~ lM, B ~ 1, A ~ 1, t0 ~ 1), model = RDMnn, report_p_vector = FALSE))
  p <- c(v = log(1.5), v_lMTRUE = .6, B = log(1), A = log(.3), t0 = log(.3))
  set.seed(1); dat <- make_data(p, des, n_trials = 100)
  P <- rbind(p, p + .1, p - .1)
  acc <- function(rt, pars) {
    pdf <- cdf <- numeric(length(rt))
    for (g in row_groups(pars[, ctx, drop = FALSE])) {
      ev <- flow_eval(fl, log(unname(pars[g[1], ctx])), rt[g]); pdf[g] <- ev$pdf; cdf[g] <- ev$cdf }
    list(pdf = pdf, cdf = cdf)
  }
  r <- ll_vs_ref(P, dat, des, acc)
  cat(sprintf("   rdm_small: max |ll(dev-nle compiled) - ll(gate reference)| %.1e over %d trial x particle, %d unfloored\n",
              r[["diff"]], r[["n"]], r[["unfloored"]]))
  r[["diff"]] < 1e-7 && r[["unfloored"]] > r[["n"]] / 2
})
verdict(ok0, "0. gate self-test: reference race likelihood == dev-nle on shipped rdm_small; crdm.cpp compiles")

## ---- 1. registration --------------------------------------------------------
reg <- if (MOCK) function(path, pars, twin, exceptions, hybrid = NULL) malte_ref_model(path, twin, hybrid) else register_nn_model
if (MOCK) CRDM <- CRDM_ref
mk <- list()
ok1 <- try_check({
  mk$rdm_final <- reg(card("rdm_final"), pars = c("v", "s", "B"), twin = RDM, exceptions = c(A = 0))
  mk$rdm_plain <- reg(card("rdm_plain"), pars = c("v", "s", "B"), twin = RDM, exceptions = c(A = 0))
  if (!exists("CRDM", mode = "function")) stop("dev-nle exports no CRDM() model function")
  mk$hybrid <- reg(card("crdm_final"), pars = c("v", "amp", "tau", "s", "B"), twin = CRDM, exceptions = c(A = 0), hybrid = "amp")
  mk$allflow <- reg(card("crdm_final"), pars = c("v", "amp", "tau", "s", "B"), twin = CRDM, exceptions = c(A = 0))
  TRUE
})
verdict(ok1, "1a. rdm_final, rdm_plain, crdm_final (hybrid and all-flow) register")
if (MOCK) skip("1b. registration contents and refusals") else verdict(ok1 && try_check({
  refused <- function(expr) inherits(tryCatch(expr, error = function(e) e), "error")
  for (nm in names(mk)) {
    m <- mk[[nm]]()
    cat(sprintf("   %-9s pre %s | transforms %s | c_name %s | hybrid %s | A bound [%g, %g]\n", nm, format(m$nn$pre),
                paste(unique(m$nn$transforms), collapse = ","), format(m$c_name), format(m$nn$hybrid),
                m$bound$minmax[1, "A"], m$bound$minmax[2, "A"]))
    stopifnot(identical(m$nn$pre, "t0"), all(m$nn$transforms == "identity"), identical(m$c_name, "NN"),
              all(m$bound$minmax[, "A"] == 0), m$bound$exception[["A"]] == 0, m$p_types[["A"]] == -Inf)
  }
  mh <- mk$hybrid()
  cat(sprintf("   hybrid: p_types amp %g, tau %g | exception amp %g | amp bound [%g, %g]\n", mh$p_types[["amp"]],
              mh$p_types[["tau"]], mh$bound$exception[["amp"]], mh$bound$minmax[1, "amp"], mh$bound$minmax[2, "amp"]))
  stopifnot(identical(mh$nn$hybrid, "amp"), is.null(mk$allflow()$nn$hybrid), mh$p_types[["amp"]] == -Inf,
            mh$bound$exception[["amp"]] == 0, mh$bound$minmax[2, "amp"] == 1)
  r <- c(A_not_fixed = refused(register_nn_model(card("rdm_final"), pars = c("v", "s", "B"), twin = RDM)),
         hybrid_without_amp = refused(register_nn_model(card("rdm_final"), pars = c("v", "s", "B"), twin = RDM,
                                                        exceptions = c(A = 0), hybrid = "amp")),
         pre_contradicts_card = refused(register_nn_model(card("rdm_final"), pars = c("v", "s", "B"), twin = RDM,
                                                          exceptions = c(A = 0), pre = "s")))
  cat("   refused:", paste(names(r), r, collapse = "  "), "\n")
  all(r)
}), "1b. pre = t0 from the card; identity inputs; compiled; A fixed at 0; amp default -Inf with exception 0; three refusals")

## ---- 2. exact: RDM-final and RDM-plain dfun / pfun == R port -----------------
draw_rt <- function(n, t0) c(runif(n / 3, t0 - .05, t0 + .05), runif(n - n / 3, t0 + .05, t0 + 2.5))
ok2 <- ok1 && try_check({
  set.seed(5); n <- 60; ok <- TRUE
  for (nm in c("rdm_final", "rdm_plain")) {
    fl <- flow_load(card(nm)); m <- mk[[nm]](); w <- c(pdf = 0, cdf = 0); np <- n0 <- 0; z <- TRUE
    for (k in 1:25) {
      t0 <- runif(1, .15, .4)
      nat <- cbind(v = runif(1, .05, 7.9), B = runif(1, .3, 3.4), A = 0, t0 = t0, s = runif(1, .3, 3.4))[rep(1, n), ]
      if (k == 25) nat[, "v"] <- 8.5                                   # outside the box: density and CDF 0
      rt <- draw_rt(n, t0)
      ref <- malte_ref_acc(fl, nat, rt); d <- m$dfun(rt, nat); p <- m$pfun(rt, nat)
      cl <- cmp_log(d, ref$pdf)
      w["pdf"] <- max(w["pdf"], cl[["diff"]]); w["cdf"] <- max(w["cdf"], abs(p - ref$cdf))
      np <- np + cl[["n"]]; n0 <- n0 + cl[["n0"]]; z <- z && cl[["zeros_ok"]] == 1
    }
    cat(sprintf("   %-9s max |log pdf diff| %.1e (%d points), max |cdf diff| %.1e; zeros agree at %d points (rt <= t0, out of box): %s\n",
                nm, w[["pdf"]], np, w[["cdf"]], n0, z))
    ok <- ok && w[["pdf"]] < 1e-7 && w[["cdf"]] < 1e-8 && z && n0 > 0
  }
  ok
})
verdict(ok2, "2. rdm_final and rdm_plain: dfun, pfun == R port (< 1e-7 log pdf, < 1e-8 cdf); 0 for rt <= t0 and out of box")

## ---- 3. exact: the hybrid CRDM race, and the all-flow route ------------------
ok3 <- ok1 && try_check({
  set.seed(6); n <- 60; fl <- flow_load(card("crdm_final")); ok <- TRUE
  for (nm in c("hybrid", "allflow")) {
    m <- mk[[nm]](); hy <- nm == "hybrid"; w <- c(pdf = 0, cdf = 0, wald = 0); nnet <- nwald <- 0; z <- TRUE
    for (k in 1:25) {
      t0 <- runif(1, .15, .4)
      one <- c(B = runif(1, .3, 2.9), A = 0, t0 = t0, s = runif(1, .3, 2.9), tau = runif(1, .02, .45))
      # accumulators interleaved as in a race: pulsed (amp > 0) and unpulsed (amp exactly 0), different rates
      nat <- cbind(v = rep(runif(2, .05, 7.9), n / 2), matrix(one, n, 5, byrow = TRUE, dimnames = list(NULL, names(one))),
                   amp = rep(c(runif(1, .02, .98), 0), n / 2))
      rt <- draw_rt(n, t0)
      ref <- malte_ref_acc(fl, nat, rt, hybrid = hy); d <- m$dfun(rt, nat); p <- m$pfun(rt, nat)
      cl <- cmp_log(d, ref$pdf)
      w["pdf"] <- max(w["pdf"], cl[["diff"]]); w["cdf"] <- max(w["cdf"], abs(p - ref$cdf)); z <- z && cl[["zeros_ok"]] == 1
      un <- nat[, "amp"] == 0 & rt > t0
      nnet <- nnet + sum(ref$network); nwald <- nwald + sum(un & !ref$network)
      if (hy) w["wald"] <- max(w["wald"], cmp_log(d[un], wald_pdf(rt[un] - t0, nat[un, "v"], nat[un, "s"], nat[un, "B"]))[["diff"]])
    }
    cat(sprintf("   %-8s max |log pdf diff| %.1e, max |cdf diff| %.1e; %d network rows, %d Wald rows%s; zeros agree: %s\n",
                nm, w[["pdf"]], w[["cdf"]], nnet, nwald,
                if (hy) sprintf(" (unpulsed vs closed-form Wald %.1e)", w[["wald"]]) else " (amp = 0 goes to the network)", z))
    ok <- ok && w[["pdf"]] < 1e-7 && w[["cdf"]] < 1e-8 && z && (if (hy) nwald > 0 && w[["wald"]] < 1e-7 else nwald == 0)
  }
  ok
})
verdict(ok3, "3. crdm_final: hybrid (network where amp > 0, Wald where amp == 0) and all-flow dfun, pfun == references")

## ---- 4. the design: lD, exact zeros, and the likelihood dev-nle samples with --
p_c <- c(v = log(2), v_lMTRUE = .5, B = log(1), t0 = log(.3), tau = log(.1), amp_lDTRUE = log(.4))
f_c <- list(v ~ lM, B ~ 1, t0 ~ 1, tau ~ 1, amp ~ 0 + lD)
ok4 <- ok1 && try_check({
  des <- suppressMessages(do.call(design, c(DES, list(formula = f_c, constants = c(amp_lDFALSE = -Inf),
                                                      model = mk$hybrid, report_p_vector = FALSE))))
  mp <- mapped_pars(des, p_c)
  print(mp[, intersect(c("S", "D", "lR", "lM", "lD", "v", "B", "A", "t0", "s", "amp", "tau"), colnames(mp))], digits = 4)
  lD <- as.logical(as.character(mp$lD))
  zeros <- all(mp$amp[!lD] == 0) && all(abs(mp$amp[lD] - .4) < 1e-12) && all(mp$A == 0) && all(abs(mp$tau - .1) < 1e-12)
  fl <- flow_load(card("crdm_final"))
  set.seed(2); dat <- make_data(p_c, des, n_trials = 50)
  P <- rbind(p_c, p_c + .05, p_c - .05)
  r <- ll_vs_ref(P, dat, des, function(rt, pars) malte_ref_acc(fl, pars, rt, hybrid = TRUE))
  cat(sprintf("   hybrid CRDM: max |ll(dev-nle%s) - ll(reference race)| %.1e over %d trial x particle, %d unfloored\n",
              if (MOCK) " R path" else " compiled", r[["diff"]], r[["n"]], r[["unfloored"]]))
  # and the RDM card through the same route
  flr <- flow_load(card("rdm_final"))
  desr <- suppressMessages(design(factors = DES$factors[1:2], Rlevels = DES$Rlevels, matchfun = DES$matchfun,
                                  formula = list(v ~ lM, B ~ 1, t0 ~ 1), model = mk$rdm_final, report_p_vector = FALSE))
  pr <- p_c[c("v", "v_lMTRUE", "B", "t0")]
  set.seed(3); datr <- make_data(pr, desr, n_trials = 100)
  rr <- ll_vs_ref(rbind(pr, pr + .05, pr - .05), datr, desr, function(rt, pars) malte_ref_acc(flr, pars, rt))
  cat(sprintf("   rdm_final:   max |ll - ll(reference race)| %.1e over %d, %d unfloored; mapped A %s\n",
              rr[["diff"]], rr[["n"]], rr[["unfloored"]], paste(unique(mapped_pars(desr, pr)$A), collapse = ",")))
  zeros && r[["diff"]] < 1e-7 && rr[["diff"]] < 1e-7 && r[["unfloored"]] > r[["n"]] / 2 && rr[["unfloored"]] > rr[["n"]] / 2 &&
    all(mapped_pars(desr, pr)$A == 0)
})
verdict(ok4, "4. amp ~ 0 + lD with amp_lDFALSE = -Inf: mapped_pars exact zeros, A = 0; sampled likelihood == reference race (< 1e-7)")

## ---- 5. the twin: Volterra dfun hoisted per parameter row; simulator ----------
ok5a <- try_check({
  if (!exists("CRDM", mode = "function")) stop("dev-nle exports no CRDM() model function")
  tw <- CRDM(); set.seed(7); U <- 6; n <- 40
  th <- cbind(v = runif(U, .5, 6), B = runif(U, .4, 2.5), A = 0, t0 = runif(U, .15, .4), s = runif(U, .4, 2.5),
              amp = c(runif(U - 2, .05, .95), 0, 0), tau = runif(U, .02, .45))
  row <- sample(rep(seq_len(U), n))                                      # rows of a parameter vector are not adjacent
  pars <- cbind(th[row, ], b = th[row, "B"]); rt <- pars[, "t0"] + runif(U * n, -.03, 2.4)
  el <- system.time({ d <- tw$dfun(rt, pars); p <- tw$pfun(rt, pars) })[["elapsed"]]
  rd <- rp <- numeric(length(rt))
  for (u in seq_len(U)) {
    i <- which(row == u & rt > th[u, "t0"]); t <- rt[i] - th[u, "t0"]
    if (th[u, "amp"] == 0) { rd[i] <- wald_pdf(t, th[u, "v"], th[u, "s"], th[u, "B"]); rp[i] <- wald_cdf(t, th[u, "v"], th[u, "s"], th[u, "B"]) }
    else { o <- dcrdm_volterra(t, th[u, "v"], th[u, "amp"], th[u, "tau"], th[u, "s"], th[u, "B"], 5e-4, 3); rd[i] <- o$pdf; rp[i] <- o$cdf }
  }
  e <- c(max(abs(d - rd)) / max(rd), max(abs(p - rp)))
  cat(sprintf("   CRDM() dfun/pfun vs dcrdm_volterra (dt 5e-4) and Wald at amp = 0: max |pdf diff| / peak %.1e, max |cdf diff| %.1e\n", e[1], e[2]))
  cat(sprintf("   %d rows, %d distinct parameter rows (%d pulsed), dfun + pfun in %.2f s (one solve per row and call: ~%.1f s; per trial: ~%.0f s)\n",
              U * n, U, U - 2, el, 2 * (U - 2) * .086, 2 * (U - 2) * n * .086))
  all(e < 1e-10) && all(d[rt <= pars[, "t0"]] == 0) && el < 5
})
verdict(ok5a, "5a. CRDM() twin: dfun, pfun == Volterra reference (dt 5e-4), Wald at amp = 0, one solve per distinct row")
ok5b <- ok1 && try_check({
  des <- suppressMessages(do.call(design, c(DES, list(formula = f_c, constants = c(amp_lDFALSE = -Inf),
                                                      model = mk$hybrid, report_p_vector = FALSE))))
  set.seed(4); dat <- make_data(p_c, des, n_trials = 500)
  stopifnot(nrow(dat) == 2000, all(is.finite(dat$rt)), min(dat$rt) > exp(p_c[["t0"]]))
  # reference race by quadrature on the Volterra grid: P(pulsed accumulator wins) and its rt distribution
  vm <- exp(p_c[["v"]] + p_c[["v_lMTRUE"]]); vx <- exp(p_c[["v"]]); ok <- TRUE
  for (cong in c(TRUE, FALSE)) {
    vp <- if (cong) vm else vx; vw <- if (cong) vx else vm             # pulsed accumulator matches iff congruent
    g <- crdm_volterra_grid(vp, .4, .1, 1, 1, 5e-4, 8)
    G <- cumsum(g$pdf * (1 - wald_cdf(g$t, vw, 1, 1))) * 5e-4
    sel <- (dat$S == dat$D) == cong
    won <- as.character(dat$R[sel]) == as.character(dat$D[sel])        # the pulsed accumulator is the one D points to
    ph <- mean(won); pr <- G[length(G)]; se <- sqrt(pr * (1 - pr) / sum(sel))
    ks <- suppressWarnings(stats::ks.test(dat$rt[sel][won] - exp(p_c[["t0"]]), function(q) approx(g$t, G / pr, q, rule = 2)$y))
    cat(sprintf("   %-11s P(pulsed wins) simulated %.3f, reference %.3f (se %.3f); rt of those %d trials vs reference: KS p %.3f\n",
                if (cong) "congruent" else "incongruent", ph, pr, se, sum(won), ks$p.value))
    ok <- ok && abs(ph - pr) < 4 * se && ks$p.value > 1e-3
  }
  ok
})
verdict(ok5b, "5b. make_data() on the hybrid model: simulator races the pulse against a Wald (choice within 4 se, KS p > 0.001)")

## ---- 6. SBC plumbing ----------------------------------------------------------
if (MOCK) skip("6. nn_cell() and 2-replicate nn_sbc_cell()") else verdict(ok1 && try_check({
  cell <- do.call(nn_cell, c(list(mk$hybrid, mean = p_c, sd = c(v = .15, v_lMTRUE = .1, B = .15, t0 = .1, tau = .2, amp_lDTRUE = .15),
                                  formula = f_c, constants = c(amp_lDFALSE = -Inf)), DES[c("Rlevels", "matchfun", "functions")],
                             list(factors = DES$factors[-1])))
  out <- nn_sbc_cell(cell, trials = 50, replicates = 2, run_control = FALSE, cores_per_chain = 4, cores_for_chains = 2)
  print(out$summary)
  cellr <- nn_cell(mk$rdm_final, mean = p_c[1:4], sd = c(v = .15, v_lMTRUE = .1, B = .15, t0 = .1),
                   formula = list(v ~ lM, B ~ 1, t0 ~ 1), factors = DES$factors[2], Rlevels = DES$Rlevels, matchfun = DES$matchfun)
  outr <- nn_sbc_cell(cellr, trials = 100, replicates = 2, run_control = FALSE, cores_per_chain = 4, cores_for_chains = 2)
  print(outr$summary)
  cat(sprintf("   controls: %s (CRDM cell), %s (RDM cell)\n", cell$control_label, cellr$control_label))
  !is.null(out$nn) && !is.null(outr$nn) && !is.null(cell$control_design) && !is.null(cellr$control_design)
}), "6. nn_cell(functions = ) builds the cell with its control; 2-replicate nn_sbc_cell() completes (hybrid CRDM, rdm_final)")

cat(sprintf("\n=== %d passed, %d failed, %d skipped in %.1f min ===\n%s\n", npass, nfail, nskip, (proc.time()[3] - t_all) / 60,
            if (MOCK) "MOCK RUN: the gate's references agree with themselves; this is not a result about dev-nle"
            else if (nfail == 0) sprintf("GATE PASSED: dev-nle at %s takes Malte's flows and the hybrid CRDM race", LIB) else "GATE FAILED"))
quit(status = nfail)

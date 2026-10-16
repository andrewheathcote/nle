# Eight-case DDM review: likelihood speed of each case's flow against the
# analytic DDM (EMC2 DDM, the registered twin) on the same data.
#
# Cases (listed parameters free, the rest pinned at exactly 0):
#   1 none  2 st0+SZ  3 st0+sv  4 sv+SZ  5 st0  6 sv  7 SZ  8 all
# Flow per case: st0 pinned -> v1 zb_s903_e271; st0 free -> v2 zb2_s901_e274;
# case 8 -> cap256w_c4 (t0 correction is post hoc, so it does not affect speed).
# The cell is built as in zerobox_s4_cell.R (same prior, design, pins).
# Timing: calc_ll_manager over K prior draws at N trials, single thread, the
# call a sampler makes; reported as ms per likelihood evaluation.
#
#   Rscript sbc/ddm_8case_bench.R [trials] [K] [reps]
# Env: LIB (dev-nle library)
a <- commandArgs(trailingOnly = TRUE)
TRIALS <- if (length(a) > 0) as.integer(a[1]) else 400L
K <- if (length(a) > 1) as.integer(a[2]) else 100L
REPS <- if (length(a) > 2) as.integer(a[3]) else 5L
LIB <- path.expand(Sys.getenv("LIB", "~/R/nle-lib"))
.libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages(library(EMC2))
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))

V1 <- "port/export/ddm_zb_s903_e271_weights.json"
V2 <- "handover/nle-methods/port/export/ddm_zb2_s901_e274_weights.json"
C4 <- "port/export/ddm_cap256w_c4_weights.json"
CASES <- list(
  `1 none`   = list(pin = c(sv = -Inf, SZ = -Inf, st0 = -Inf), card = V1),
  `2 st0+SZ` = list(pin = c(sv = -Inf),                        card = V2),
  `3 st0+sv` = list(pin = c(SZ = -Inf),                        card = V2),
  `4 sv+SZ`  = list(pin = c(st0 = -Inf),                       card = V1),
  `5 st0`    = list(pin = c(sv = -Inf, SZ = -Inf),             card = V2),
  `6 sv`     = list(pin = c(SZ = -Inf, st0 = -Inf),            card = V1),
  `7 SZ`     = list(pin = c(sv = -Inf, st0 = -Inf),            card = V1),
  `8 all`    = list(pin = c(),                                 card = C4))

PM <- c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5), sv = log(.1), SZ = qnorm(.2), st0 = log(.05))
PS <- c(v = .15, a = .15, t0 = .1, Z = .05, sv = .15, SZ = .15, st0 = .15)
REG <- list()

tm <- function(X, dadm, model) {
  invisible(EMC2:::calc_ll_manager(X, dadm, model))  # warm-up
  s <- vapply(seq_len(REPS), function(r)
    system.time(EMC2:::calc_ll_manager(X, dadm, model))[["elapsed"]], numeric(1))
  1000 * stats::median(s) / nrow(X)
}

cat(sprintf("=== 8-case likelihood bench  N=%d  K=%d draws  %d reps  EMC2 %s  %s ===\n", TRIALS, K, REPS,
            as.character(packageVersion("EMC2")), format(Sys.time(), "%Y-%m-%d %H:%M")))
res <- do.call(rbind, lapply(names(CASES), function(nm) {
  cs <- CASES[[nm]]; PIN <- cs$pin
  FREE <- setdiff(names(PM), names(PIN))
  if (is.null(REG[[cs$card]])) REG[[cs$card]] <<- register_nn_model(cs$card, twin = DDM)
  cell <- nn_cell(REG[[cs$card]], mean = PM[FREE], sd = PS[FREE],
                  formula = lapply(FREE, function(p) as.formula(paste(p, "~ 1"))),
                  constants = if (length(PIN)) PIN else NULL,
                  factors = list(S = c("left", "right")), Rlevels = c("left", "right"))
  for (d in list(cell$design, cell$control_design)) {
    mp <- mapped_pars(d, cell$mean)
    for (p in names(PIN)) stopifnot(all(mp[, p] == 0))
    for (p in setdiff(c("sv", "SZ", "st0"), names(PIN))) stopifnot(all(mp[, p] > 0))
  }
  set.seed(11)
  dat <- suppressMessages(make_data(cell$mean, cell$control_design, n_trials = TRIALS / 2))
  X <- EMC2:::nn_cell_draws(cell, K)
  dn <- EMC2:::nn_dadm(dat, cell$design); dc <- EMC2:::nn_dadm(dat, cell$control_design)
  ms_n <- tm(X, dn, cell$design$model); ms_c <- tm(X, dc, cell$control_design$model)
  r <- data.frame(case = nm, flow = sub("_weights.json", "", sub("ddm_", "", basename(cs$card))),
                  trials = nrow(dat), ms_flow = ms_n, ms_analytic = ms_c, speedup = ms_c / ms_n)
  print(r, digits = 3, row.names = FALSE); flush.console()
  r
}))
cat("\n"); print(res, digits = 3, row.names = FALSE)
write.csv(res, Sys.getenv("BENCH_OUT", "sbc/ddm_8case_bench.csv"), row.names = FALSE)

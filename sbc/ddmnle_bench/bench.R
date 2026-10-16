# DDMnle (compiled, EMC2 branch dev-nle-scaled) against WienR, per variability case, N = 100 and 400.
# WienR at EMC2's default precision 5e-3 (EMC2's compiled DDM) and at 1e-8 (the project's reference
# precision; EMC2's compiled DDM fixes 5e-3, so 1e-8 runs through EMC2's R path, dDDM(precision = 1e-8)).
# The R-path cost at 5e-3 is also timed, to show what the R path itself adds.
# Single thread, ms per likelihood evaluation, median of REPS timings over K prior draws.
#   LIB=~/R/nle-scaled-lib Rscript sbc/ddmnle_bench/bench.R [K] [REPS]
a <- commandArgs(trailingOnly = TRUE)
K <- if (length(a) > 0) as.integer(a[1]) else 400L; REPS <- if (length(a) > 1) as.integer(a[2]) else 7L
CARD <- "handover/nle-methods/port/export/ddm_sc_s1106_e245_weights.json"
CASES <- c(`2 st0+SZ` = "sv0", `3 st0+sv` = "SZ0", `4 sv+SZ` = "st00", `5 st0` = "st0only", `7 SZ` = "SZonly", `8 all` = "full")
src <- readLines("sbc/scaled_cell.R"); stop_at <- grep("^# ---- checks", src)[1] - 1
Sys.setenv(NATIVE = "1", BENCH = "0")
res <- list()
for (cs in names(CASES)) {
  ARGS <- c(CASES[[cs]], CARD)
  suppressMessages(capture.output(eval(parse(text = src[1:stop_at]))))
  ctl <- cell$control_design$model
  r_prec <- function(prec) function() { m <- ctl(); m$c_name <- NULL; tt <- m$Ttransform   # DDM()'s Ttransform drops the "ok" attribute the R path needs
    m$Ttransform <- function(pars, dadm) { out <- tt(pars, dadm); attr(out, "ok") <- attr(pars, "ok"); out }
    m$dfun <- eval(bquote(function(rt, R, pars) dDDM(rt, R, pars, precision = .(prec)))); environment(m$dfun) <- asNamespace("EMC2"); m }
  X <- EMC2:::nn_cell_draws(cell, K)
  for (N in c(100, 400)) {
    set.seed(7); dat <- suppressMessages(make_data(cell$mean, cell$control_design, n_trials = N / 2))
    dc <- EMC2:::nn_dadm(dat, cell$control_design); dn <- EMC2:::nn_dadm(dat, cell$design)
    tm <- function(d, model, X, reps = REPS) { EMC2:::calc_ll_manager(X[1:3, , drop = FALSE], d, model)
      1000 * median(replicate(reps, system.time(EMC2:::calc_ll_manager(X, d, model))[[3]])) / nrow(X) }
    stopifnot(all(abs(EMC2:::calc_ll_manager(X[1:3, , drop = FALSE], dc, r_prec(1e-8)) - EMC2:::calc_ll_manager(X[1:3, , drop = FALSE], dc, ctl)) < 3))   # R path really evaluates, and agrees with the compiled control
    Xs <- X[1:min(nrow(X), 20), , drop = FALSE]           # the slow R-path timings use fewer draws and repeats
    row <- data.frame(case = cs, N = N, wienr_5e3 = tm(dc, ctl, X), wienr_5e3_Rpath = tm(dc, r_prec(5e-3), Xs, 3),
                      wienr_1e8 = tm(dc, r_prec(1e-8), Xs, 3), ddmnle = tm(dn, sc_native, X))
    print(row, digits = 3, row.names = FALSE); flush.console()
    res[[length(res) + 1]] <- row
  }
}
R <- do.call(rbind, res)
R$speedup_5e3 <- R$wienr_5e3 / R$ddmnle; R$speedup_1e8 <- R$wienr_1e8 / R$ddmnle
cat("\n"); print(R, digits = 3, row.names = FALSE)
write.csv(R, "sbc/ddmnle_bench/bench.csv", row.names = FALSE)

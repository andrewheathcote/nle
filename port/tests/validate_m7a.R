# Stage M7a acceptance for cards exported by flow/scripts/export_flow.py with
# the options `affine` and `pre = "t0"` (decision time), and for plain cards.
# Zero-box cards (context_transforms "zerobox", handover/RDM_A0_BOX_PLAN.md) are
# checked at A = 0 exactly and above it; the dev-nle line then reports both.
# Two lines per tag:
#   golden   R port (port/R/flow_race.R) vs the exporter's float64 golden
#            vectors, max |diff| over z, log_pdf, cdf, log_sf       (< 1e-10)
#   dev-nle  the card registers, and the registered model's dfun / pfun equal
#            the R port across theta, log density and CDF            (< 1e-7)
# The reference is always THIS repository's port/R/flow_race.R: EMC2 carries
# its own copy of the port, so dev-nle is never checked against itself.
#
#   Rscript port/tests/validate_m7a.R [--lib=~/R/nle-sbc-lib] [--port=port] tag1 [tag2 ...]
# Tags are export basenames: rdm_m7a_smoke_aff -> <port>/export/rdm_m7a_smoke_aff_weights.json
# Run from the repo root. Exit status 1 if any line fails.
a <- commandArgs(trailingOnly = TRUE)
opt <- function(key, default) {
  hit <- grep(paste0("^--", key, "="), a, value = TRUE)
  if (length(hit)) sub(paste0("^--", key, "="), "", hit[1]) else default
}
LIB <- path.expand(opt("lib", "~/R/nle-sbc-lib"))
PORT <- opt("port", "port")
tags <- a[!grepl("^--", a)]
stopifnot(length(tags) > 0, dir.exists(LIB))
.libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(jsonlite)})
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
source(file.path(PORT, "R", "flow_race.R"))
cat(sprintf("=== M7a card check  EMC2 %s at %s  port %s ===\n",
            as.character(packageVersion("EMC2")), find.package("EMC2"), normalizePath(PORT)))

nfail <- 0L
line <- function(ok, tag, what, val, tol, note = "") {
  ok <- isTRUE(ok)
  cat(sprintf("%s %-8s %-24s max|diff| %.1e (< %g)%s\n", if (ok) "PASS" else "FAIL",
              what, tag, val, tol, note))
  if (!ok) nfail <<- nfail + 1L
}

for (tag in tags) {
  path <- file.path(PORT, "export", sprintf("%s_weights.json", tag))
  fl <- flow_load(path)
  ctx <- unlist(fl$context_names)
  dectime <- identical(fl$pre, "t0")
  tfs <- unlist(fl$context_transforms)[ctx]
  stopifnot(is.null(fl$pre) || dectime, all(tfs %in% c("log", "zerobox")))
  zb <- ctx[tfs == "zerobox"]
  stopifnot(all(zb %in% names(fl$zerobox_c)), all(zb == "A"))
  # natural-scale parameters -> the network's input coordinate
  to_input <- function(nat) sapply(ctx, function(nm)
    if (nm %in% zb) asinh(nat[[nm]] / fl$zerobox_c[[nm]]) else log(nat[[nm]]))

  ## ---- golden ---------------------------------------------------------------
  g <- read.csv(file.path(PORT, "export", sprintf("%s_golden.csv", tag)))
  stopifnot(all(ctx %in% names(g)), !dectime || "t0" %in% names(g))
  worst <- 0
  for (pid in unique(g$param_id)) {
    gi <- g[g$param_id == pid, ]
    # the caller's arithmetic: rt - t0, t0 from its sampled (log) value
    t_arg <- if (dectime) gi$rt - exp(gi$t0[1]) else gi$rt
    ev <- flow_eval(fl, as.numeric(gi[1L, ctx]), t_arg)
    stopifnot(ev$in_box)
    worst <- max(worst, abs(ev$z - gi$z), abs(ev$log_pdf - gi$log_pdf),
                 abs(ev$cdf - gi$cdf), abs(ev$log_sf - gi$log_sf))
  }
  line(worst < 1e-10, tag, "golden", worst, 1e-10,
       sprintf("  [%d sets x %d times; affine %s, pre %s%s]", length(unique(g$param_id)),
               nrow(g) / length(unique(g$param_id)), isTRUE(fl$affine), format(fl$pre),
               if (length(zb)) sprintf("; %d sets at A = 0", sum(tapply(g$A, g$param_id, function(x) x[1] == 0))) else ""))

  ## ---- dev-nle --------------------------------------------------------------
  res <- tryCatch({
    m <- register_nn_model(path, twin = RDM)()
    stopifnot(identical(m$nn$pre, if (dectime) "t0" else NULL) || (!dectime && is.null(m$nn$pre)))
    set.seed(11); n <- 60; w <- 0; w0 <- 0
    lo <- setNames(fl$bounds_sampled$lower, ctx); hi <- setNames(fl$bounds_sampled$upper, ctx)
    for (k in 1:25) {
      # natural-scale accumulator parameters inside the box, away from its faces
      p <- c(v = runif(1, .5, 4), B = runif(1, .3, 2.5), A = runif(1, .01, 1.5),
             s = runif(1, .3, 1.8), t0 = runif(1, .1, .6))
      zero <- length(zb) && k %% 2 == 0           # a zero-box card: every second theta at A = 0
      if (zero) p[["A"]] <- 0
      th <- to_input(p)
      stopifnot(all(th >= lo & th <= hi), !zero || th[["A"]] == 0)
      rt <- p[["t0"]] + exp(runif(n, log(.02), log(8)))
      nat <- matrix(p, n, length(p), byrow = TRUE, dimnames = list(NULL, names(p)))
      ref <- flow_eval(fl, th, if (dectime) rt - p[["t0"]] else rt)
      stopifnot(ref$in_box)
      d_new <- m$dfun(rt, nat); p_new <- m$pfun(rt, nat)
      dd <- max(abs(log(d_new) - ref$log_pdf), abs(p_new - ref$cdf))
      w <- max(w, dd); if (zero) w0 <- max(w0, dd)
    }
    c(w, w0) }, error = function(e) { cat("   error:", conditionMessage(e), "\n"); c(NA_real_, NA_real_) })
  line(is.finite(res[1]) && res[1] < 1e-7, tag, "dev-nle", res[1], 1e-7,
       sprintf("  [registers; dfun and pfun vs this repo's R port, 25 theta x 60 rt%s]",
               if (length(zb)) sprintf("; 12 theta at A = 0 exactly, max|diff| %.1e", res[2]) else ""))
}
cat(sprintf("=== %d tag(s), %d failing line(s) ===\n", length(tags), nfail))
quit(status = if (nfail) 1L else 0L)

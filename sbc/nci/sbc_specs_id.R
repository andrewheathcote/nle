# Cell specs for the Phase 4 identification study: the standard DDM cell with
# the sv and SZ generating values (and hence prior means) overridden.
#
# Rationale. At the Phase 1-3 generating point sv=0.1 the unit information is
# 3e-6, so N would have to reach ~14 MILLION trials for the likelihood to match
# the prior: sv's posterior is the prior, and its SBC "pass" was therefore
# almost content-free. SZ=0.2 has I=0.0134, reachable at N~3300. Raising sv to
# 1.5 and SZ to 0.4 makes both likelihood-dominated at N=3200 while keeping the
# prior (mean +- 5 psd) inside the flow's training box; sv=2.0 does NOT fit
# (log 1.44 > box 1.386).
#
# Sourced by SBCid.R. Requires sbc_specs.R alongside it.

source("sbc_specs.R")

# likelihood-info / prior-info at the four Phase 4 cells (measured, analytic
# DDM; prior precision 1/0.15^2 = 44.4). >1 means the likelihood dominates.
#   sv=0.1 SZ=0.2  N=400 : sv 0.00  SZ 0.12     <- neither identified
#   sv=0.1 SZ=0.2  N=3200: sv 0.00  SZ 0.97     <- N identifies SZ only
#   sv=1.5 SZ=0.4  N=400 : sv 0.33  SZ 0.67     <- magnitude alone is not enough
#   sv=1.5 SZ=0.4  N=3200: sv 2.61  SZ 5.38     <- both identified
make_sbc_cell_id <- function(model_name, sv = 0.1, SZ = 0.2) {
  cell <- make_sbc_cell(model_name)
  pn <- names(sampled_pars(cell$design))
  stopifnot(all(c("sv", "SZ") %in% pn))
  pmean <- cell$pmean
  names(pmean) <- pn
  pmean[["sv"]] <- log(sv)
  pmean[["SZ"]] <- qnorm(SZ)
  # box guard: every prior face must sit inside the flow's training box
  box_lo <- c(sv = -4.605, SZ = -1.645); box_hi <- c(sv = 1.386, SZ = 2.326)
  for (p in c("sv", "SZ")) {
    lo <- pmean[[p]] - 5 * cell$psd[which(pn == p)]
    hi <- pmean[[p]] + 5 * cell$psd[which(pn == p)]
    if (lo < box_lo[[p]] || hi > box_hi[[p]])
      stop(sprintf("prior for %s spans [%.3f, %.3f], outside the training box [%.3f, %.3f]",
                   p, lo, hi, box_lo[[p]], box_hi[[p]]))
  }
  prior_in <- prior(cell$design, type = "single", pmean = pmean, psd = cell$psd)
  list(design = cell$design, prior = prior_in, pmean = pmean, psd = cell$psd,
       sv = sv, SZ = SZ)
}

run_sbc_cell_id <- function(model_name, trials, replicates, cores, out_dir,
                            sv = 0.1, SZ = 0.2) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  cell <- make_sbc_cell_id(model_name, sv = sv, SZ = SZ)
  tag <- sprintf("%s_sv%s_SZ%s_N%d_R%d", model_name,
                 sub("\\.", "", format(sv)), sub("\\.", "", format(SZ)),
                 trials, replicates)
  cat(sprintf("\n===== SBC cell %s (%d cores) started %s =====\n",
              tag, cores, format(Sys.time())))
  t0 <- proc.time()[3]
  SBC <- run_sbc(cell$design, cell$prior, replicates = replicates,
                 trials = trials, cores_per_chain = cores, verbose = TRUE,
                 fileName = file.path(out_dir, paste0(tag, "_tmp.RData")))
  mins <- (proc.time()[3] - t0) / 60
  saveRDS(list(SBC = SBC, model = model_name, trials = trials,
               replicates = replicates, pmean = cell$pmean, psd = cell$psd,
               sv = sv, SZ = SZ, mins = mins),
          file.path(out_dir, paste0(tag, ".rds")))
  cat(sprintf("===== %s done in %.1f min =====\n", tag, mins))
  cov <- apply(SBC$coverage$alpha, 2, mean, na.rm = TRUE)
  bias <- apply(SBC$bias$alpha, 2, mean, na.rm = TRUE)
  cat("parameter  coverage  mean-bias\n")
  for (i in seq_along(cov))
    cat(sprintf("%-9s  %.3f     %+.4f\n", names(cov)[i], cov[i], bias[i]))
  invisible(TRUE)
}

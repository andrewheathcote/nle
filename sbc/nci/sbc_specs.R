# Shared SBC cell specifications: designs and priors for the six models
# (analytic LNR/RDM/DDM and their nn twins), single participant, following
# the EMC2 SBC vignette. Sourced by the NCI R scripts (colocated) and by
# sbc/run_sbc_nn.R (laptop).
#
# Priors sit well inside the nn training boxes (+-4 sd) — required for the
# nn models (out-of-box draws have floored likelihoods, corrupting ranks)
# and shared with the analytic twins for comparability. Deviation from the
# vignette: DDM SZ prior mean qnorm(.2), not qnorm(.05) (box edge).

make_sbc_cell <- function(model_name) {
  matchfun <- function(d) d$S == d$lR
  spec <- switch(model_name,
    LNR = , LNRnn = list(
      model = get(model_name),
      formula = list(m ~ lM, s ~ 1, t0 ~ 1),
      constants = NULL, use_matchfun = TRUE,
      pmean = c(0.7, -0.5, log(.4), log(.2)),
      psd = c(.15, .15, .1, .1)),
    RDM = , RDMnn = list(
      model = get(model_name),
      formula = list(v ~ lM, B ~ 1, A ~ 1, t0 ~ 1, s ~ 1),
      constants = c(s = log(1)), use_matchfun = TRUE,
      pmean = c(log(1.5), .4, log(1), log(.3), log(.2)),
      psd = c(.1, .1, .1, .1, .1)),
    DDM = , DDMnn = list(
      model = get(model_name),
      formula = list(v ~ 1, a ~ 1, t0 ~ 1, s ~ 1, Z ~ 1, sv ~ 1,
                     SZ ~ 1, st0 ~ 1),
      constants = c(s = log(1)), use_matchfun = FALSE,
      pmean = c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5),
                sv = log(.1), SZ = qnorm(.2), st0 = log(.05)),
      psd = c(.15, .15, .1, .05, .15, .15, .15)),
    stop("Unknown model: ", model_name)
  )
  des_args <- list(factors = list(subjects = 1, S = c("left", "right")),
                   Rlevels = c("left", "right"),
                   formula = spec$formula, model = spec$model, verbose = FALSE)
  if (spec$use_matchfun) des_args$matchfun <- matchfun
  if (!is.null(spec$constants)) des_args$constants <- spec$constants
  design_in <- do.call(design, des_args)
  stopifnot(length(names(sampled_pars(design_in))) == length(spec$pmean))
  prior_in <- prior(design_in, type = "single",
                    pmean = spec$pmean, psd = spec$psd)
  list(design = design_in, prior = prior_in,
       pmean = spec$pmean, psd = spec$psd)
}

# Run one SBC cell and save results + a console summary
run_sbc_cell <- function(model_name, trials, replicates, cores, out_dir) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  cell <- make_sbc_cell(model_name)
  tag <- sprintf("%s_N%d_R%d", model_name, trials, replicates)
  cat(sprintf("\n===== SBC cell %s (%d cores) started %s =====\n",
              tag, cores, format(Sys.time())))
  t0 <- proc.time()[3]
  SBC <- run_sbc(cell$design, cell$prior, replicates = replicates,
                 trials = trials, cores_per_chain = cores, verbose = TRUE,
                 fileName = file.path(out_dir, paste0(tag, "_tmp.RData")))
  mins <- (proc.time()[3] - t0) / 60
  saveRDS(list(SBC = SBC, model = model_name, trials = trials,
               replicates = replicates, pmean = cell$pmean, psd = cell$psd,
               mins = mins),
          file.path(out_dir, paste0(tag, ".rds")))
  cat(sprintf("===== %s done in %.1f min =====\n", tag, mins))
  cov <- apply(SBC$coverage$alpha, 2, mean, na.rm = TRUE)
  bias <- apply(SBC$bias$alpha, 2, mean, na.rm = TRUE)
  prec <- apply(SBC$med$alpha, 2, sd, na.rm = TRUE)
  cat("parameter  coverage  mean-bias  sd(post med)\n")
  for (i in seq_along(cov))
    cat(sprintf("%-9s  %.3f     %+.4f    %.4f\n",
                names(cov)[i], cov[i], bias[i], prec[i]))
  invisible(TRUE)
}

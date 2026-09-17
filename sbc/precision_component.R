# Reusable precision-study component for nn-likelihood models in EMC2.
#
# Measures single-participant posterior precision (sd and 95% CI width per
# parameter) as a function of trial count, for a set of model specs fitted
# to the SAME simulated datasets (paired comparison). Fit settings mirror
# EMC2's SBC_single (type = "single", compress = FALSE,
# rt_resolution = 1e-12, stop_criteria min_es = 100 / max_gd = 1.1), so
# precision estimates transfer to the SBC operating point.
#
# Designed for reuse: a "spec" is list(label = "DDMnn", model = DDMnn).
# Later studies (e.g., flows trained at larger budgets) add specs without
# touching this file — e.g., a DDMnn variant whose weights .rds differs.
#
# Cells are (spec x n_trials x replicate). Data for a given
# (n_trials, replicate) are generated ONCE (from gen_spec, default the
# first spec) and shared across specs. Each worker appends a line to
# log_file so long runs show progress.

# prefer the dedicated nn-EMC2 library (nle/.Rlibs) so installs of other
# EMC2 branches into the default library cannot affect these studies
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(EMC2))

# specs may carry their own $n_grid (subset of the study grid) so cheap and
# expensive likelihoods can be measured over different ranges, and their own
# $make_emc_args (e.g., list(compress = TRUE, rt_resolution = 1/60)) to vary
# data-processing conditions; give such specs distinct labels, since the
# label keys the per-cell cache
make_precision_cells <- function(specs, n_grid, n_reps) {
  cells <- do.call(rbind, lapply(seq_along(specs), function(i) {
    g <- if (is.null(specs[[i]]$n_grid)) n_grid else specs[[i]]$n_grid
    expand.grid(spec = i, n_trials = g, rep = seq_len(n_reps),
                KEEP.OUT.ATTRS = FALSE)
  }))
  cells$label <- vapply(cells$spec, function(i) specs[[i]]$label, "")
  cells
}

run_precision_study <- function(specs, design_maker, prior_maker, p_true,
                                n_grid, n_reps, seed0 = 4000,
                                cores = 1L, log_file = NULL,
                                fit_args = list(), cache_dir = NULL) {
  designs <- lapply(specs, function(sp) design_maker(sp$model))
  priors <- lapply(designs, prior_maker)
  cells <- make_precision_cells(specs, n_grid, n_reps)
  if (!is.null(cache_dir)) dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

  # pre-generate shared datasets: one per (n_trials, rep), from spec 1
  datasets <- list()
  for (n in n_grid) for (r in seq_len(n_reps)) {
    set.seed(seed0 + 1000 * r + n)
    datasets[[paste(n, r)]] <- make_data(p_true, designs[[1]], n_trials = n)
  }

  one_cell <- function(k) {
    row <- cells[k, ]
    sp <- specs[[row$spec]]
    # per-cell disk cache: completed fits survive crashes/interrupts, and
    # reruns (or later studies adding specs) only pay for missing cells
    cache_f <- if (!is.null(cache_dir))
      file.path(cache_dir, sprintf("%s_N%d_rep%d.rds",
                                   row$label, row$n_trials, row$rep)) else NULL
    if (!is.null(cache_f) && file.exists(cache_f)) {
      if (!is.null(log_file))
        cat(sprintf("[%s] cell %d/%d  %s N=%d rep=%d  cached\n",
                    format(Sys.time(), "%H:%M:%S"), k, nrow(cells),
                    row$label, row$n_trials, row$rep),
            file = log_file, append = TRUE)
      return(readRDS(cache_f))
    }
    t0 <- proc.time()[3]
    res <- tryCatch({
      dat <- datasets[[paste(row$n_trials, row$rep)]]
      me_args <- list(data = dat, design = designs[[row$spec]],
                      prior_list = priors[[row$spec]], type = "single",
                      compress = FALSE, rt_resolution = 1e-12)
      if (!is.null(fit_args$make_emc))
        me_args <- utils::modifyList(me_args, fit_args$make_emc)
      if (!is.null(sp$make_emc_args))
        me_args <- utils::modifyList(me_args, sp$make_emc_args)
      emc <- do.call(make_emc, me_args)
      emc <- do.call(fit, c(list(
        emc = emc, verbose = FALSE,
        stop_criteria = list(min_es = 100, max_gd = 1.1, selection = "alpha"),
        cores_per_chain = 1, cores_for_chains = 1), fit_args$fit))
      draws <- get_pars(emc, selection = "alpha", return_mcmc = FALSE,
                        merge_chains = TRUE, flatten = TRUE)[, 1, ]
      ci <- credint(emc)[[1]]
      data.frame(label = row$label, n_trials = row$n_trials, rep = row$rep,
                 par = rownames(ci),
                 post_sd = apply(draws, 1, sd),
                 ci_width = ci[, 3] - ci[, 1],
                 post_med = ci[, 2],
                 bias = ci[, 2] - p_true[rownames(ci)],
                 ess = as.numeric(ess_summary(emc, stat = NULL)[[1]]),
                 secs = proc.time()[3] - t0, failed = FALSE,
                 row.names = NULL)
    }, error = function(e) {
      data.frame(label = row$label, n_trials = row$n_trials, rep = row$rep,
                 par = NA, post_sd = NA, ci_width = NA, post_med = NA,
                 bias = NA, ess = NA, secs = proc.time()[3] - t0,
                 failed = TRUE, row.names = NULL)
    })
    if (!is.null(cache_f) && !res$failed[1]) saveRDS(res, cache_f)
    if (!is.null(log_file))
      cat(sprintf("[%s] cell %d/%d  %s N=%d rep=%d  %.0fs%s\n",
                  format(Sys.time(), "%H:%M:%S"), k, nrow(cells), row$label,
                  row$n_trials, row$rep, res$secs[1],
                  if (res$failed[1]) "  FAILED" else ""),
          file = log_file, append = TRUE)
    rm(list = setdiff(ls(), "res")); gc(verbose = FALSE)
    res
  }

  out <- parallel::mclapply(seq_len(nrow(cells)), one_cell,
                            mc.cores = cores, mc.preschedule = FALSE)
  do.call(rbind, out)
}

# Effective-information summary. Model: posterior precision = prior
# precision + N * unit information i. The reference unit information is
# pooled over the reference spec's (possibly shorter) trial grid; kappa for
# the target at each N is i_target(N) / i_ref. kappa < 1 means the target
# likelihood extracts less information per trial; matching the reference
# precision at N_base needs ~ N_base / kappa target trials.
effective_info <- function(res, ref, target, prior_sd) {
  agg <- stats::aggregate(post_sd ~ label + n_trials + par, data = res,
                          FUN = function(x) sqrt(mean(x^2)))
  unit_info <- function(sd, n, p) (1 / sd^2 - 1 / prior_sd[p]^2) / n
  out <- data.frame()
  for (p in unique(agg$par)) {
    r <- agg[agg$label == ref & agg$par == p, ]
    i_ref <- mean(unit_info(r$post_sd, r$n_trials, p))
    tg <- agg[agg$label == target & agg$par == p, ]
    for (j in seq_len(nrow(tg))) {
      i_t <- unit_info(tg$post_sd[j], tg$n_trials[j], p)
      out <- rbind(out, data.frame(par = p, n_trials = tg$n_trials[j],
                                   sd_target = tg$post_sd[j],
                                   i_ref = i_ref, i_target = i_t,
                                   kappa = i_t / i_ref))
    }
  }
  out
}

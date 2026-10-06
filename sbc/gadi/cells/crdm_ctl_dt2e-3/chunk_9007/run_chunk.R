# Auto-written by build_crdm_cell.R: runs (or resumes) this control chunk.
LIB <- path.expand("~/R/nle-lib"); .libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages(library(EMC2))
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
cat("EMC2", as.character(packageVersion("EMC2")), "at", find.package("EMC2"), "| built", packageDescription("EMC2")$Built, "\n")
sp <- readRDS("chunk_spec.rds")
# CRDM()'s Volterra cdf exceeds 1 by up to ~1e-5 in the far right tail for ~6% of parameter rows within 2 prior sd
# (2026-10-06), so log(1 - cdf) in log_likelihood_race warns 'NaNs produced' (the NaN is already floored to min_ll
# there) and run_sbc() discards any replicate whose fit warns. Muffle that one warning; every value is unchanged.
ml <- sp$design$model(); ll0 <- ml$log_likelihood
ml$log_likelihood <- function(pars, dadm, model, min_ll = log(1e-10)) withCallingHandlers(ll0(pars, dadm, model, min_ll),
  warning = function(w) if (identical(conditionMessage(w), "NaNs produced")) invokeRestart("muffleWarning"))
sp$design$model <- function() ml
set.seed(sp$seed)
out <- run_sbc(sp$design, sp$prior, replicates = sp$replicates, trials = sp$trials, n_subjects = 1, fileName = "sbc_chunk.RData", cores_per_chain = 3, cores_for_chains = 1)
saveRDS(out, "chunk.rds")

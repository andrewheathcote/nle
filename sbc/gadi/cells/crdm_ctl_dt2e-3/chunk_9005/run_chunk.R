# Auto-written by build_crdm_cell.R: runs (or resumes) this control chunk.
LIB <- path.expand("/Users/andrew/R/nle-sbc-lib"); .libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages(library(EMC2))
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
sp <- readRDS("chunk_spec.rds")
set.seed(sp$seed)
out <- run_sbc(sp$design, sp$prior, replicates = sp$replicates, trials = sp$trials, n_subjects = 1, fileName = 'sbc/gadi/cells/crdm_ctl_dt2e-3/chunk_9005/chunk.rds', cores_per_chain = 3, cores_for_chains = 1)
saveRDS(out, 'sbc/gadi/cells/crdm_ctl_dt2e-3/chunk_9005/chunk.rds')

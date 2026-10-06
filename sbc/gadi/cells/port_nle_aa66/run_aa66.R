# Port-NLE (2026-10-06): rerun of an M5/M4 RDM SBC cell on EMC2 dev-nle 142bd497
# (= dev-sampler aa66e11d tree; sampler H5-H8, split-Rhat stop rule).
# Wrapper, not run_cell.R (nn_sbc_cell rewrites that): pre-warms the evaluator cache from
# the bundled card, because cell.rds holds the laptop's absolute card path
# (memory bundle-absolute-card-path-trap). From the bundle directory:
#   Rscript run_aa66.R <trials> <replicates> <run_control TRUE/FALSE> <cores_per_chain>
a <- commandArgs(TRUE)
LIB <- path.expand("~/R/nle-lib"); .libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages(library(EMC2))
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
cat("EMC2", as.character(packageVersion("EMC2")), "at", find.package("EMC2"), "| built", packageDescription("EMC2")$Built, "\n")
cat("BLAS:", sessionInfo()$BLAS, "\n")
cell <- readRDS("cell.rds")
reg <- cell$reg
cand <- file.path("port", "export", basename(reg$path))
if (!file.exists(cand)) { cand <- list.files(file.path("port", "export"), "\\.json$", full.names = TRUE); stopifnot(length(cand) == 1) }
reg$path <- normalizePath(cand)
invisible(EMC2:::nn_ptr(reg))
cat("evaluator pre-warmed from", reg$path, "\n")
EMC2:::nle_artefact_info(cell$model)
res <- nn_sbc_cell(cell, trials = as.integer(a[1]), replicates = as.integer(a[2]), run_control = as.logical(a[3]),
                   archive_dir = "..", cores_per_chain = as.integer(a[4]), cores_for_chains = 1L)
print(res$summary)

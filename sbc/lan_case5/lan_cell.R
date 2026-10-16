# SBC of HSSM's ddm_uniform_st LAN (case 5: st0 the only variability) in ITS training box, with the
# analytic DDM in the same parameterisation, design and prior as control. AH 2026-10-16.
#   Rscript sbc/lan_case5/lan_cell.R <nn|control|summary> [replicates] [trials]
# Env: LIB (default ~/R/nle-lib), CPC, CFC, OUT (default sbc/lan_case5/results)
a <- commandArgs(trailingOnly = TRUE)
WHAT <- a[1]; REPS <- if (length(a) > 1) as.integer(a[2]) else 500L; TRIALS <- if (length(a) > 2) as.integer(a[3]) else 400L
LIB <- path.expand(Sys.getenv("LIB", "~/R/nle-lib")); .libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages(library(EMC2))
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
CPC <- as.integer(Sys.getenv("CPC", "1")); CFC <- as.integer(Sys.getenv("CFC", "3"))
OUT <- Sys.getenv("OUT", "sbc/lan_case5/results")
source("sbc/lan_case5/lan_model.R")
cell <- nn_cell(lan_model(), mean = PM, sd = PS, control = DDMlan, factors = list(S = c("left", "right")),
                formula = lapply(names(PM), function(p) as.formula(paste(p, "~ 1"))), Rlevels = c("left", "right"))
cat(sprintf("=== LAN case 5 %s  N=%d reps=%d  cores %dx%d  %s ===\n", WHAT, TRIALS, REPS, CPC, CFC, format(Sys.time())))
print(cell)
if (WHAT == "control") {
  dir.create(file.path(OUT, "results"), recursive = TRUE, showWarnings = FALSE)
  EMC2:::nn_sbc_run(cell$control_design, cell$control_prior, REPS, TRIALS, file.path(OUT, "results", "sbc_control.RData"),
                    list(cores_per_chain = CPC, cores_for_chains = CFC))
} else {
  out <- nn_sbc_cell(cell, trials = TRIALS, replicates = REPS, run_control = WHAT == "summary", archive_dir = OUT,
                     cores_per_chain = CPC, cores_for_chains = CFC,
                     info = list(case = 5, network = "HSSM ddm_uniform_st (LAN)", box = "LAN training box"))
  print(out$summary)
}
cat("done", format(Sys.time()), "\n")

# Build a self-contained CRDM SBC-cell bundle for Gadi (stage M6a, part B): either the
# network cell (a card, hybrid race) or a chunk of a Volterra-only control (dev-nle's exported
# CRDM(dt, t_max), no card). Writes cell.rds/run_cell.R/session.txt (network) or a run_sbc()
# script (control chunk); nothing is fitted here. The design is the cell of
# sbc/gate_malte_dev_nle.R check 6 (handover/MALTE_STAGES.md, "Stage M6a").
#
#   network chunk:  Rscript build_crdm_cell.R net   <cardtag> <out_dir> <trials> <replicates> [LIB]
#   control chunk:  Rscript build_crdm_cell.R ctl <dt> <chunk_out_dir> <trials> <reps_per_chunk> <seed> [LIB]
#
# Traps paid for: ~/R/nle-lib prepended and asserted; CRDM()'s default t_max is 3 (control
# chunks pass t_max explicitly); s = 1 (constant, as in every RDM/CRDM control cell here).
a <- commandArgs(trailingOnly = TRUE)
KIND <- a[1]
LIB <- path.expand(if (KIND == "net" && length(a) > 5) a[6] else if (KIND == "ctl" && length(a) > 6) a[7] else "~/R/nle-lib")
.libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages(library(EMC2))
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
if (!exists("CRDM", mode = "function")) stop("dev-nle exports no CRDM() model function")

DES6 <- list(factors = list(subjects = 1, S = c("left", "right"), D = c("left", "right")), Rlevels = c("left", "right"),
             matchfun = function(d) d$S == d$lR,
             functions = list(lD = function(d) factor(d$D == d$lR, levels = c(FALSE, TRUE))))
F6 <- list(v ~ lM, B ~ 1, t0 ~ 1, tau ~ 1, amp ~ 0 + lD)
CONST6 <- c(amp_lDFALSE = -Inf)
PM6 <- c(v = log(2), v_lMTRUE = .5, B = log(1), t0 = log(.3), tau = log(.1), amp_lDTRUE = log(.4))
PS6 <- c(v = .15, v_lMTRUE = .1, B = .15, t0 = .1, tau = .2, amp_lDTRUE = .15)

if (KIND == "net") {
  TAG <- a[2]; OUT <- a[3]; TRIALS <- as.integer(a[4]); REPS <- as.integer(a[5])
  card_local <- file.path("port/export", c(paste0(TAG, "_weights.json"), paste0(TAG, ".json")))
  card_local <- card_local[file.exists(card_local)][1]; stopifnot(!is.na(card_local))
  sha <- system(sprintf("shasum -a 256 %s", shQuote(card_local)), intern = TRUE); cat(sprintf("SHA256 %s\n", sha))
  mk <- register_nn_model(card_local, pars = c("v", "amp", "tau", "s", "B"), twin = CRDM, exceptions = c(A = 0), hybrid = "amp")
  cell <- do.call(nn_cell, c(list(mk, mean = PM6, sd = PS6, formula = F6, constants = CONST6),
                             DES6[c("Rlevels", "matchfun", "functions")], list(factors = DES6$factors[-1])))
  print(cell)
  mp <- mapped_pars(cell$design, cell$mean)
  stopifnot(all(mp$A == 0), all(mp$amp[!as.logical(as.character(mp$lD))] == 0))
  cat("mapped_pars asserted: A == 0, amp == 0 on the unpulsed (lD FALSE) rows\n")
  dat <- make_data(cell$mean, cell$design, n_trials = 50); stopifnot(nrow(dat) == 200, all(is.finite(dat$rt)))  # 50 trials x 4 S/D cells
  cat("simulator OK (50-trial smoke draw)\n")
  CPC <- as.integer(Sys.getenv("CPC", "3")); CFC <- as.integer(Sys.getenv("CFC", "1"))
  out <- nn_sbc_cell(cell, trials = TRIALS, replicates = REPS, run_control = FALSE, archive_dir = OUT, run = FALSE,
                     cores_per_chain = CPC, cores_for_chains = CFC,
                     info = list(branch = "dev-sbc-calibration", where = "Gadi", topic = sprintf("M6a crdm_net %s", TAG)))
  # run_cell.R runs from bundle/, so the card must resolve at the same relative path from there.
  dir.create(file.path(OUT, "bundle", "port", "export"), recursive = TRUE, showWarnings = FALSE)
  stopifnot(file.copy(card_local, file.path(OUT, "bundle", card_local), overwrite = TRUE))
  cat(sprintf("bundle written to %s (run = FALSE): %s; card copied to bundle/%s\n", OUT, paste(list.files(OUT), collapse = ", "), card_local))

} else if (KIND == "ctl") {
  DT <- as.numeric(a[2]); OUT <- a[3]; TRIALS <- as.integer(a[4]); REPS <- as.integer(a[5]); SEED <- as.integer(a[6])
  des <- suppressMessages(do.call(design, c(DES6, list(formula = F6, constants = CONST6,
    model = function() CRDM(dt = DT, t_max = 6), report_p_vector = FALSE))))
  mp <- mapped_pars(des, PM6); stopifnot(all(mp$A == 0), all(mp$amp[!as.logical(as.character(mp$lD))] == 0))
  cat(sprintf("mapped_pars asserted (dt %g). ", DT))
  pr <- suppressMessages(prior(des, type = "single", pmean = PM6, psd = PS6))
  dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
  fn <- file.path(OUT, "chunk.rds")
  saveRDS(list(design = des, prior = pr, dt = DT, trials = TRIALS, replicates = REPS, seed = SEED), file.path(OUT, "chunk_spec.rds"))
  writeLines(c(
    "# Auto-written by build_crdm_cell.R: runs (or resumes) this control chunk.",
    sprintf("LIB <- path.expand(\"%s\"); .libPaths(c(LIB, .libPaths()))", LIB),
    "suppressPackageStartupMessages(library(EMC2))",
    "stopifnot(startsWith(normalizePath(find.package(\"EMC2\")), normalizePath(LIB)))",
    "sp <- readRDS(\"chunk_spec.rds\")",
    "set.seed(sp$seed)",
    sprintf("out <- run_sbc(sp$design, sp$prior, replicates = sp$replicates, trials = sp$trials, n_subjects = 1, fileName = %s, cores_per_chain = 3, cores_for_chains = 1)", shQuote(fn)),
    sprintf("saveRDS(out, %s)", shQuote(fn))
  ), file.path(OUT, "run_chunk.R"))
  cat(sprintf("chunk bundle written to %s: %s\n", OUT, paste(list.files(OUT), collapse = ", ")))
}

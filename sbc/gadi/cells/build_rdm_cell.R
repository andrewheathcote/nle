# Build a self-contained RDM SBC-cell bundle for Gadi (stage M6a, part B). Writes cell.rds,
# run_cell.R, session.txt to <out> via nn_sbc_cell(..., run = FALSE); nothing is fitted here.
# The card is copied alongside (port/export is not tracked by git).
#
#   Rscript build_rdm_cell.R <cell:a0|std> <cardtag> <out_dir> <trials> <replicates> <run_control:0|1> [LIB]
# Env: CPC (cores_per_chain, default 3 for a laptop smoke), CFC (cores_for_chains, default 1)
#
# a0:  A pinned at exactly 0 (constants A = -Inf, sampled scale); v ~ lM, B ~ 1, t0 ~ 1, s = 1.
#      Card without A support (Malte's): pars = c(v, s, B), exceptions = c(A = 0) instead of a
#      design constant (its card takes no A input at all).
# std: the registered standard cell of sbc/nci/sbc_specs.R (A sampled, A ~ 1).
# Traps paid for: ~/R/nle-lib prepended and asserted; s = 1 in every RDM-control cell
# (pWald overflows for v b / s^2 > 354).
a <- commandArgs(trailingOnly = TRUE)
CELL <- a[1]; TAG <- a[2]; OUT <- a[3]; TRIALS <- as.integer(a[4]); REPS <- as.integer(a[5])
RUN_CONTROL <- as.logical(as.integer(a[6])); LIB <- path.expand(if (length(a) > 6) a[7] else "~/R/nle-lib")
stopifnot(CELL %in% c("a0", "std"))
.libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages(library(EMC2))
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))

card_local <- file.path("port/export", c(paste0(TAG, "_weights.json"), paste0(TAG, ".json")))
card_local <- card_local[file.exists(card_local)][1]
stopifnot(!is.na(card_local))
sha <- system(sprintf("shasum -a 256 %s", shQuote(card_local)), intern = TRUE)
cat(sprintf("SHA256 %s\n", sha))

# does this card take A as its own input? (our cards: zerobox on A; Malte's: no A, threshold b = B)
has_A <- {
  j <- jsonlite::fromJSON(card_local, simplifyVector = TRUE)
  "A" %in% unlist(j$context_names)
}
matchfun <- function(d) d$S == d$lR
DESF <- list(factors = list(subjects = 1, S = c("left", "right")), Rlevels = c("left", "right"), matchfun = matchfun)

if (CELL == "a0") {
  PM <- c(v = log(1.5), v_lMTRUE = .4, B = log(1), t0 = log(.2))
  PS <- c(v = .1, v_lMTRUE = .1, B = .1, t0 = .1)
  formula <- list(v ~ lM, B ~ 1, t0 ~ 1)
  if (has_A) {
    mk <- register_nn_model(card_local, twin = RDM)               # card takes A itself, pinned by a design constant
    constants <- c(s = log(1), A = -Inf)
  } else {
    mk <- register_nn_model(card_local, pars = c("v", "s", "B"), twin = RDM, exceptions = c(A = 0))
    constants <- c(s = log(1))
  }
} else {
  stopifnot(has_A)                                                  # std cell needs A sampled: only O's card supports it
  PM <- c(v = log(1.5), v_lMTRUE = .4, B = log(1), A = log(.3), t0 = log(.2))
  PS <- c(v = .1, v_lMTRUE = .1, B = .1, A = .1, t0 = .1)
  formula <- list(v ~ lM, B ~ 1, A ~ 1, t0 ~ 1)
  mk <- register_nn_model(card_local, twin = RDM)
  constants <- c(s = log(1))
}
cell <- do.call(nn_cell, c(list(mk, mean = PM, sd = PS, formula = formula, constants = constants), DESF))
print(cell)
for (d in list(cell$design, cell$control_design)) {
  mp <- mapped_pars(d, cell$mean)
  if (CELL == "a0") stopifnot(all(mp[, "A"] == 0))
  stopifnot(all(mp[, "s"] == 1))
}
cat("mapped_pars asserted: A pinned as expected (a0) or free (std), s == 1\n")
dat <- make_data(cell$mean, cell$design, n_trials = 50)
stopifnot(nrow(dat) == 100, all(is.finite(dat$rt)))
cat("simulator OK (50-trial smoke draw)\n")

CPC <- as.integer(Sys.getenv("CPC", "3")); CFC <- as.integer(Sys.getenv("CFC", "1"))
out <- nn_sbc_cell(cell, trials = TRIALS, replicates = REPS, run_control = RUN_CONTROL, archive_dir = OUT,
                   run = FALSE, cores_per_chain = CPC, cores_for_chains = CFC,
                   info = list(branch = "dev-sbc-calibration", where = "Gadi", topic = sprintf("M6a %s %s", CELL, TAG)))
# run_cell.R runs from bundle/ (its own comment: "from this bundle directory"), so the card
# must resolve at the SAME relative path from there, not from the archive root.
dir.create(file.path(OUT, "bundle", "port", "export"), recursive = TRUE, showWarnings = FALSE)
stopifnot(file.copy(card_local, file.path(OUT, "bundle", card_local), overwrite = TRUE))
cat(sprintf("bundle written to %s (run = FALSE): %s; card copied to bundle/%s\n", OUT, paste(list.files(OUT), collapse = ", "), card_local))

# Writes a zero-box MODEL CARD FIXTURE for dev-nle's register_nn_model() work,
# usable before any trained zero-box flow exists: the shipped cap256w_c4 export
# relabelled with context_encoding = "zerobox", the plan's constants and
# network-scale bounds (lower = 0 for sv, SZ, st0), exactly as
# port/tests/validate_zerobox_plumbing.R builds its tempfile. The numbers it
# produces are not a model of anything; the point is that dev-nle's dfun must
# equal the NLE R port on it, including at exact zeros.
#   Rscript handover/zerobox/make_zerobox_fixture.R [out]   # default ~/R/nle-cards/ddm_zerobox_fixture.json
suppressPackageStartupMessages(library(jsonlite))
a <- commandArgs(trailingOnly = TRUE)
out <- path.expand(if (length(a)) a[1] else "~/R/nle-cards/ddm_zerobox_fixture.json")
fl <- fromJSON("handover/nle-methods/port/export/ddm_cap256w_c4_weights.json", simplifyDataFrame = FALSE)
ZC <- list(sv = 0.01, SZ = 0.05, st0 = 0.002)
fl$context_encoding <- "zerobox"; fl$zerobox_c <- ZC
fl$context_transforms[names(ZC)] <- "zerobox"          # the new transform name, per input
cn <- fl$context_names
lo <- fl$bounds_sampled$lower; hi <- fl$bounds_sampled$upper
for (p in names(ZC)) { j <- match(p, cn); lo[j] <- 0
  hi[j] <- asinh(c(sv = 4, SZ = 0.99, st0 = 0.49)[[p]] / ZC[[p]]) }
fl$bounds_sampled <- list(lower = lo, upper = hi)
bn <- fl$bounds_natural                                # order = names(context_transforms)
for (p in names(ZC)) bn$lower[match(p, names(fl$context_transforms))] <- 0
fl$bounds_natural <- bn
fl$tag <- "ddm_zerobox_fixture"; fl$fixture_note <- "cap256w_c4 weights relabelled as zerobox; plumbing fixture only"
dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
write_json(fl, out, auto_unbox = TRUE, digits = NA, null = "null")
cat("wrote", out, "\n")

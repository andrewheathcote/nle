# Build the EMC2-side bundle for the st0 = 0 (7-parameter) DDM flow.
#
#   Rscript sbc/make_bundle_st0zero.R <out.rds> <tag1> [tag2 ...]
#
# Mirrors sbc/make_ens5_rds.R but infers the theta width from context_names
# instead of hardcoding 8 columns, and asserts the context is the 7-parameter
# one -- a silently permuted or wrong-width context is the failure this project
# has already paid for twice.
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")
suppressPackageStartupMessages(library(EMC2))

args <- commandArgs(trailingOnly = TRUE)
out <- args[1]; tags <- args[-1]
stopifnot(length(tags) >= 1)
EXPECT <- c("v", "a", "s", "Z", "SZ", "sv")

members <- lapply(tags, function(m)
  ddm_load(sprintf("port/export/ddm_%s_weights.json", m)))
for (i in seq_along(members)) {
  stopifnot(identical(members[[i]]$context_names, EXPECT))
  stopifnot(identical(members[[i]]$context_encoding, "dectime"))
  stopifnot(isTRUE(all.equal(members[[i]]$scaler$mean,  members[[1]]$scaler$mean)),
            isTRUE(all.equal(members[[i]]$scaler$scale, members[[1]]$scaler$scale)),
            identical(members[[i]]$spline$num_bins, members[[1]]$spline$num_bins))
}
bundle <- list(model = "DDM", type = "ensemble", member_tags = tags,
               architecture = members[[1]]$budget, members = members)
saveRDS(bundle, out)

# round-trip: the saved bundle must evaluate identically to the loaded exports
ptr_a <- EMC2:::ddm_build_ensemble(readRDS(out)$members)
ptr_b <- EMC2:::ddm_build_ensemble(members)
# The COMPILED port takes the FLOW context (6 columns) and is evaluated at the
# decision time d = rt - t0. It does NOT implement the shift -- that is the model
# wrapper's job (DDMnn6.R) -- so the round-trip feeds it what it gets at inference.
t0n <- 0.3
th <- c(1, log(0.8), 0, qnorm(0.5), qnorm(0.2), log(0.1))            # 6 columns
rt <- c(0.35, 0.5, 0.8) - t0n; Th <- matrix(rep(th, each = 3), nrow = 3)
a <- EMC2:::ddm_ens_eval_trials_cpp(ptr_a, Th, rt, rep(1L, 3))$log_pdf
b <- EMC2:::ddm_ens_eval_trials_cpp(ptr_b, Th, rt, rep(1L, 3))$log_pdf
stopifnot(all(is.finite(a)), max(abs(a - b)) < 1e-12)
cat(sprintf("wrote %s\n  %d member(s): %s\n  context: %s\n  round-trip max|diff| %.2e\n",
            out, length(tags), paste(tags, collapse = ", "),
            paste(EXPECT, collapse = ","), max(abs(a - b))))

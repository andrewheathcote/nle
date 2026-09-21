# Build the EMC2-side ensemble bundle that DDMnn evaluates.
#
# DDMnn -> dDDMnn -> ddmnn_get("ddm_ens5") -> readRDS of
#   <EMC2>/extdata/flownn/ddm_ens5.rds
# so the ONLY way to change which flows DDMnn uses is to replace that file.
# The name is hardcoded, hence the backup below rather than a new name.
#
#   Rscript sbc/make_ens5_rds.R <out.rds> <tag1> <tag2> ...
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")
suppressPackageStartupMessages(library(EMC2))

args <- commandArgs(trailingOnly = TRUE)
out  <- args[1]; tags <- args[-1]
stopifnot(length(tags) >= 1)
members <- lapply(tags, function(m)
  ddm_load(sprintf("port/export/ddm_%s_weights.json", m)))

# every member must share the scaler and spline geometry, or the mixture is
# evaluating its components on different input scales
sc1 <- members[[1]]$scaler
for (i in seq_along(members)) {
  stopifnot(isTRUE(all.equal(members[[i]]$scaler$mean,  sc1$mean)),
            isTRUE(all.equal(members[[i]]$scaler$scale, sc1$scale)),
            identical(members[[i]]$spline$num_bins, members[[1]]$spline$num_bins))
}
bundle <- list(model = "DDM", type = "ensemble", member_tags = tags,
               architecture = members[[1]]$budget, members = members)
saveRDS(bundle, out)

# round-trip: the bundle must build and evaluate identically to the tags
ptr_a <- EMC2:::ddm_build_ensemble(readRDS(out)$members)
ptr_b <- EMC2:::ddm_build_ensemble(members)
th <- c(1, log(0.8), log(0.3), 0, qnorm(0.5), qnorm(0.2), log(0.1), log(0.05))
rt <- c(0.35, 0.5, 0.8); Th <- matrix(rep(th, each = 3), nrow = 3)
a <- EMC2:::ddm_ens_eval_trials_cpp(ptr_a, Th, rt, rep(1L, 3))$log_pdf
b <- EMC2:::ddm_ens_eval_trials_cpp(ptr_b, Th, rt, rep(1L, 3))$log_pdf
stopifnot(max(abs(a - b)) < 1e-12)
cat(sprintf("wrote %s\n  %d members: %s\n  round-trip max|diff| %.2e\n",
            out, length(tags), paste(tags, collapse = ", "), max(abs(a - b))))

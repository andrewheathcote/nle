# Splice a trained classifier into existing flow exports.
#
# The flow and the classifier are modular — plain-MLE flows never see the
# classifier during training — so a better classifier can be dropped into any
# export without retraining the flow. Used to give the batch-1024 runs the
# corrected-target (c4) classifier; they were trained with the copied
# large_budget one, so their measured score bias still carries its error.
#
#   Rscript sbc/splice_classifier.R <donor_tag> <suffix> <tag1> <tag2> ...
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(jsonlite))
a <- commandArgs(trailingOnly = TRUE)
donor <- a[1]; suffix <- a[2]; tags <- a[-(1:2)]
p <- function(t) sprintf("port/export/ddm_%s_weights.json", t)
D <- fromJSON(p(donor), simplifyVector = TRUE)
cat(sprintf("donor %s: classifier step %s\n", donor, as.character(D$checkpoint_step_classifier)))
for (t in tags) {
  w <- fromJSON(p(t), simplifyVector = TRUE)
  dm <- max(abs(w$scaler$mean - D$scaler$mean))
  if (dm > 1e-3) { cat(sprintf("  SKIP %s: scaler differs by %.2e\n", t, dm)); next }
  w$classifier_mlp <- D$classifier_mlp
  w$checkpoint_step_classifier <- D$checkpoint_step_classifier
  w$budget <- paste0(t, suffix)
  write(toJSON(w, auto_unbox = TRUE, digits = 17), p(paste0(t, suffix)))
  cat(sprintf("  %s%s (scaler diff %.1e)\n", t, suffix, dm))
}

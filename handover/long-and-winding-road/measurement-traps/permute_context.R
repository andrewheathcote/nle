# Re-express a race flow in a different context ORDER without retraining.
#
# EMC2's dRDMnn hardcodes theta = log(cbind(v, B, t0, s, A)) and ignores the
# export's context_names, so a flow trained on a different column order cannot
# be dropped into EMC2 — even though the network itself is perfectly valid.
#
# The fix is exact, not an approximation. The network sees
# standardise(theta)[perm] %*% W. Permuting the scaler entries AND the rows of
# the first layer's weight matrix by the same permutation yields a network that
# takes theta in the NEW order and computes an identical function.
#
#   Rscript sbc/permute_context.R <in_tag> <out_tag> <target order, comma sep>
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(jsonlite))
a <- commandArgs(trailingOnly = TRUE)
inp <- a[1]; out <- a[2]; target <- strsplit(a[3], ",")[[1]]
p <- function(t) sprintf("port/export/%s_weights.json", t)
w <- fromJSON(p(inp), simplifyVector = TRUE)
old <- w$context_names
stopifnot(setequal(old, target), length(old) == length(target))
perm <- match(target, old)          # position in OLD order of each NEW column
cat(sprintf("old: %s\nnew: %s\nperm: %s\n", paste(old, collapse=","),
            paste(target, collapse=","), paste(perm, collapse=",")))
w$scaler$mean  <- w$scaler$mean[perm]
w$scaler$scale <- w$scaler$scale[perm]
w$mlp$layers$W[[1]] <- w$mlp$layers$W[[1]][perm, , drop = FALSE]
if (!is.null(w$bounds_sampled)) {
  w$bounds_sampled$lower <- w$bounds_sampled$lower[perm]
  w$bounds_sampled$upper <- w$bounds_sampled$upper[perm]
}
if (!is.null(w$bounds_natural)) {
  w$bounds_natural$lower <- w$bounds_natural$lower[perm]
  w$bounds_natural$upper <- w$bounds_natural$upper[perm]
}
if (!is.null(w$context_transforms)) w$context_transforms <- w$context_transforms[perm]
w$context_names <- target
w$budget <- out
write(toJSON(w, auto_unbox = TRUE, digits = 17), p(out))
cat(sprintf("wrote %s\n", p(out)))

# verify: identical density at a random theta expressed in each order
source("port/R/flow_race.R")
f1 <- flow_load(p(inp)); f2 <- flow_load(p(out))
set.seed(1)
th_old <- setNames(c(log(1.5), log(1.4), log(0.3), log(1), log(1)), old)
th_new <- th_old[target]
rt <- c(0.4, 0.7, 1.3, 2.5)
d1 <- flow_eval(f1, as.numeric(th_old), rt)$log_pdf
d2 <- flow_eval(f2, as.numeric(th_new), rt)$log_pdf
cat(sprintf("verify: max |log_pdf difference| = %.3e  %s\n", max(abs(d1 - d2)),
            if (max(abs(d1 - d2)) < 1e-10) "EXACT" else "MISMATCH"))

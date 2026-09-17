# Stochastic Weight Averaging across checkpoints of ONE training run.
#
# Phase 14: the optimiser wanders on a plateau that is flat in validation loss
# (spread 0.004 over epochs 161-237) but not in score (g_a swings 0.11). That is
# precisely the situation weight averaging is for — it lands at a central point
# rather than an arbitrary one.
#
# This is NOT the density-mixture ensembling that failed in Phase 8. A mixture
# averages DENSITIES, and its score is provably the average of member scores
# (measured: -0.0398 against a member mean of -0.0393), so it cannot beat its
# members. Averaging WEIGHTS produces a single network at a different point in
# parameter space; its score is not the average of the members' scores, and it
# costs the same to evaluate as one flow.
#
# Averaging weights is only meaningful within one basin, which consecutive
# checkpoints of a single run are. It would be meaningless across seeds.
#
#   Rscript sbc/make_swa.R <out_tag> <tag1> <tag2> ...
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
out_tag <- args[1]; tags <- args[-1]
stopifnot(length(tags) >= 2)
path <- function(t) sprintf("port/export/ddm_%s_weights.json", t)
J <- lapply(tags, function(t) fromJSON(path(t), simplifyVector = TRUE))

# everything except the flow weights must be identical, or averaging is invalid
ref <- J[[1]]
for (k in seq_along(J)) {
  stopifnot(identical(J[[k]]$context_names, ref$context_names),
            isTRUE(all.equal(J[[k]]$scaler$mean,  ref$scaler$mean)),
            isTRUE(all.equal(J[[k]]$scaler$scale, ref$scaler$scale)),
            identical(J[[k]]$spline$num_bins, ref$spline$num_bins),
            isTRUE(all.equal(J[[k]]$classifier_mlp, ref$classifier_mlp)))
}
cat(sprintf("averaging %d checkpoints; scaler, spline and classifier verified identical\n",
            length(J)))

nl <- length(ref$flow_mlp$layers$W)
out <- ref
for (i in seq_len(nl)) {
  Ws <- lapply(J, function(j) j$flow_mlp$layers$W[[i]])
  bs <- lapply(J, function(j) j$flow_mlp$layers$b[[i]])
  out$flow_mlp$layers$W[[i]] <- Reduce(`+`, Ws) / length(Ws)
  out$flow_mlp$layers$b[[i]] <- Reduce(`+`, bs) / length(bs)
  cat(sprintf("  layer %d: %s  mean|W| before %.4f  after %.4f\n", i,
              paste(dim(Ws[[1]]), collapse="x"),
              mean(sapply(Ws, function(w) mean(abs(w)))),
              mean(abs(out$flow_mlp$layers$W[[i]]))))
}
out$budget <- out_tag
out$checkpoint_step_flow <- paste0("swa-", length(J))
write(toJSON(out, auto_unbox = TRUE, digits = 17), path(out_tag))
cat(sprintf("wrote %s\n", path(out_tag)))

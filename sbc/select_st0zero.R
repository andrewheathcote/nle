# Select the st0 = 0 checkpoint to send to SBC, on TOTAL |g| across all seven
# parameters. With --pick, print only the winning tag.
#
#   Rscript sbc/select_st0zero.R <g_rds> <t0_rds> [--pick]
#
# Total |g| is the criterion used for the RDM selection. Score bias ranks
# checkpoints reliably WITHIN one architecture (cor -0.989 with SBC rank), which
# is what this is; it does NOT rank across architectures (Phase 25), so this
# must not be used to compare the st0=0 build against the 8-parameter one --
# that comparison is SBC's job.
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
a <- commandArgs(trailingOnly = TRUE)
pick <- "--pick" %in% a; a <- setdiff(a, "--pick")
G <- readRDS(a[1])$G                       # draws x pars x tags
t0 <- readRDS(a[2])
tags <- intersect(dimnames(G)[[3]], setdiff(colnames(t0), c("analytic", "closed")))
stopifnot(length(tags) > 0)
pars <- dimnames(G)[[2]]
gm  <- sapply(tags, function(z) sapply(pars, function(p) mean(G[, p, z])))
gt0 <- sapply(tags, function(z) mean(t0[, z] - t0[, "analytic"]))
tot <- colSums(abs(gm)) + abs(gt0)
best <- tags[which.min(tot)]
if (pick) { cat(best, "\n"); quit(save = "no") }

seed_of <- function(z) sub("^(z0_s[0-9]+)_e[0-9]+$", "\\1", z)
cat(sprintf("%d checkpoints across %d seeds\n\n", length(tags),
            length(unique(seed_of(tags)))))
cat("Per-seed means (the run-level picture):\n")
cat(sprintf("  %-10s %s %10s %10s\n", "seed",
            paste(sprintf("%9s", pars), collapse = ""), "t0", "total|g|"))
for (s in sort(unique(seed_of(tags)))) {
  k <- seed_of(tags) == s
  cat(sprintf("  %-10s %s %+10.4f %10.4f\n", s,
      paste(sprintf("%+9.4f", rowMeans(gm[, k, drop = FALSE])), collapse = ""),
      mean(gt0[k]), mean(tot[k])))
}
o <- order(tot)
cat("\nBest 8 checkpoints by total |g|:\n")
cat(sprintf("  %-18s %s %10s %10s\n", "tag",
            paste(sprintf("%9s", pars), collapse = ""), "t0", "total|g|"))
for (i in head(o, 8))
  cat(sprintf("  %-18s %s %+10.4f %10.4f%s\n", tags[i],
      paste(sprintf("%+9.4f", gm[, i]), collapse = ""), gt0[i], tot[i],
      if (tags[i] == best) "  <- SELECTED" else ""))

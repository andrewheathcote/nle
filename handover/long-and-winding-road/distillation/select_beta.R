# Aggregate a distillation arm's checkpoints and report mean g_a / g_t0.
# With --pick, print ONLY the tag prefix of the arm whose mean |g_t0| is
# smallest, for the pipeline to consume.
#
#   Rscript sbc/select_beta.R <g_rds> <t0_rds> [--pick]
#
# g_t0 is the selection target because t0 is the parameter that fails SBC and
# the one distillation was shown to move (Phase 24). g_a is reported as the
# price, not optimised -- no beta zeroes both.
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
a <- commandArgs(trailingOnly = TRUE)
pick <- "--pick" %in% a; a <- setdiff(a, "--pick")
G  <- readRDS(a[1])$G                       # draws x pars x tags
t0 <- readRDS(a[2])
arm_of <- function(tg) sub("_e[0-9]+(_c4)?$", "", tg)

tags <- intersect(dimnames(G)[[3]], setdiff(colnames(t0), c("analytic", "closed")))
ga  <- apply(G[, "a", tags, drop = FALSE], 3, mean)
gt0 <- sapply(tags, function(z) mean(t0[, z] - t0[, "analytic"]))
arm <- arm_of(tags)

ma <- tapply(ga, arm, mean); mt <- tapply(gt0, arm, mean)
sa <- tapply(ga, arm, sd);   st <- tapply(gt0, arm, sd)
best <- names(mt)[which.min(abs(mt))]

if (pick) { cat(best, "\n"); quit(save = "no") }

cat(sprintf("%-16s %10s %10s %10s %10s %6s\n",
            "arm", "g_a", "(ckpt sd)", "g_t0", "(ckpt sd)", "n"))
for (k in names(mt))
  cat(sprintf("%-16s %+10.4f %10.4f %+10.4f %10.4f %6d%s\n", k, ma[k], sa[k],
              mt[k], st[k], sum(arm == k), if (k == best) "  <- min |g_t0|" else ""))
cat("\nbest single checkpoints by |g_t0|:\n")
o <- order(abs(gt0))
for (i in head(o, 5))
  cat(sprintf("  %-26s g_t0 %+.4f   g_a %+.4f\n", tags[i], gt0[i], ga[i]))

# Rank every measured flow by the SBC bias its score bias PREDICTS.
#
# |g_a| alone is the wrong criterion: the posterior displacement is
# delta = (N*I + Sigma0^-1)^-1 (N*g), and V is far from diagonal (a couples to
# Z, t0 to SZ), so a model can have small g_a and still be displaced in a by
# the rest of the g vector. This ranks on max |standardised predicted bias|
# over the parameters, at the N the SBC gate uses.
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
N <- as.numeric(Sys.getenv("SB_N", "100"))
F <- readRDS("sbc/fisher_ddm.rds"); P <- names(F$pmean)
# F$predict was saved as a closure over globalenv(), which readRDS does NOT
# restore -- its `I` resolves to base::I. Rebuild it from the saved matrix.
predict_bias <- function(g, N) {
  V <- solve(N * F$I + diag(1 / F$psd^2))
  as.numeric(V %*% (N * g[P])) / sqrt(diag(V))
}

six <- do.call(cbind, lapply(Sys.glob("sbc/rep_sb/*chunk*.rds"), function(f) {
  G <- readRDS(f)$G; apply(G, c(2, 3), mean, na.rm = TRUE) }))
t0 <- unlist(lapply(c("sbc/rep_sb/t0_both_arms.rds", "sbc/rep_sb/t0_ref.rds"), function(f) {
  r <- readRDS(f); tg <- setdiff(colnames(r), c("analytic", "closed"))
  sapply(tg, function(t) mean(r[, t] - r[, "analytic"])) }))

tags <- intersect(colnames(six), names(t0))
cat(sprintf("models with BOTH measurements: %d  (N = %g)\n\n", length(tags), N))
G <- rbind(six[, tags, drop = FALSE], t0 = t0[tags])[P, , drop = FALSE]

pred <- apply(G, 2, function(g) predict_bias(g, N))
rownames(pred) <- P
score <- apply(abs(pred), 2, max)
o <- order(score)

cat("Predicted standardised SBC bias (gate |.| <= 0.5). Ranked by worst parameter.\n\n")
cat(sprintf("%-24s %7s %s\n", "model", "worst", paste(sprintf("%7s", P), collapse = "")))
show <- function(idx) for (i in idx)
  cat(sprintf("%-24s %7.3f %s%s\n", tags[i], score[i],
              paste(sprintf("%+7.3f", pred[, i]), collapse = ""),
              if (score[i] > 0.5) "  <-- FAILS" else ""))
cat("--- best 15 ---\n");  show(head(o, 15))
cat("--- current production reference ---\n")
show(which(tags %in% c("cap256w_c4", "b1k_amp0_e216_c4")))
cat("--- worst 5 ---\n"); show(tail(o, 5))
cat(sprintf("\nof %d models, %d are inside the |0.5| gate on every parameter\n",
            length(tags), sum(score <= 0.5)))
saveRDS(list(G = G, pred = pred, score = score, N = N), "sbc/rep_sb/ranking.rds")

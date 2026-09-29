# Selection of our flow for M4, by the rule registered in handover/MALTE_STAGES.md:
# W = max over both cells (std, A0) and all parameters (t0 included) of |g_p|, draws 1-200;
# O = the checkpoint of smallest W; OMED = the checkpoint of median W (rank 60 of 120).
# A checkpoint whose export failed has W = Inf.
t <- read.csv("sbc/seed_study/ss/table_base.csv"); t <- t[t$arm == "base" & is.finite(t$g), ]; t$tag <- sprintf("rdm_ss_base_s%d_e%d", t$seed, t$checkpoint)
W <- aggregate(abs(g) ~ tag, t, max); names(W)[2] <- "W"
for (cl in c("std", "A0")) { x <- aggregate(abs(g) ~ tag, t[t$cell == cl, ], max); W[[cl]] <- x[match(W$tag, x$tag), 2] }
wp <- aggregate(abs(g) ~ tag, t, function(x) 0); wp <- t[order(t$tag, -abs(t$g)), ]; wp <- wp[!duplicated(wp$tag), ]
W$worst <- paste(wp$cell, wp$par)[match(W$tag, wp$tag)]
failed <- readLines("sbc/seed_study/ss/failed_exports.txt"); failed <- failed[nzchar(failed)]
if (length(failed)) W <- rbind(W, data.frame(tag = failed, W = Inf, std = Inf, A0 = Inf, worst = "export failed"))
W <- W[order(W$W), ]; W$rank <- seq_len(nrow(W)); stopifnot(nrow(W) == 120)
write.csv(W, "sbc/malte_m4/selection.csv", row.names = FALSE)
cat(sprintf("checkpoints %d (export failed: %d). W: min %.4f  lower quartile %.4f  median %.4f  upper quartile %.4f  max finite %.4f\n",
            nrow(W), length(failed), W$W[1], W$W[30], W$W[60], W$W[90], max(W$W[is.finite(W$W)])))
print(head(W, 5), row.names = FALSE, digits = 3)
cat(sprintf("SELECTED O = %s (W %.4f: std %.4f, A0 %.4f, worst %s)\nMEDIAN OMED = %s (W %.4f, worst %s)\n", W$tag[1], W$W[1], W$std[1], W$A0[1], W$worst[1], W$tag[60], W$W[60], W$worst[60]))
for (s in 801:804) { k <- grepl(paste0("_s", s, "_"), W$tag); cat(sprintf("  seed %d: best rank %d, W min %.4f median %.4f\n", s, min(W$rank[k]), min(W$W[k]), median(W$W[k]))) }

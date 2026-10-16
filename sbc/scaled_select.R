# Selection for the scaled DDM (handover/SCALED_DDM_PLAN.md section 5, declared before training):
# rank every checkpoint on sum over regimes of (|g_v| + |g_a| + |g_Z|), top 3 to SBC.
# Also reports st0's |g| (the plan's prediction: clean everywhere) and the best per seed.
#   Rscript sbc/scaled_select.R [dir (default sbc/scaled_screen/stage2)] [top (3)]
a <- commandArgs(trailingOnly = TRUE)
DIR <- if (length(a) > 0) a[1] else "sbc/scaled_screen/stage2"
TOP <- if (length(a) > 1) as.integer(a[2]) else 3L
REG <- c("full", "sv0", "SZ0", "st00", "st0only", "SZonly", "all0")
tabs <- lapply(REG, function(r) { t <- readRDS(file.path(DIR, paste0(r, ".rds")))$table
  t <- t[grepl("^sc:sc_s11", t$tag), ]
  data.frame(tag = sub("^sc:", "", t$tag), regime = r,
             vaZ = abs(t$v) + abs(t$a) + abs(t$Z), a = t$a,
             st0 = if ("st0" %in% names(t)) abs(t$st0) else NA_real_) })
L <- do.call(rbind, tabs)
W <- reshape(L[, c("tag", "regime", "vaZ")], idvar = "tag", timevar = "regime", direction = "wide")
names(W) <- sub("^vaZ\\.", "", names(W))
W$score <- rowSums(W[, REG])
W$worst_regime <- apply(W[, REG], 1, max)
W$max_st0 <- tapply(L$st0, L$tag, max, na.rm = TRUE)[W$tag]
W$seed <- sub("_e[0-9]+$", "", W$tag)
W <- W[order(W$score), ]
fmt <- function(d) { d[] <- lapply(d, function(x) if (is.numeric(x)) signif(x, 3) else x); d }
cat(sprintf("%d checkpoints, %d regimes, from %s\n\nTop 12 by sum over regimes of |g_v|+|g_a|+|g_Z|:\n",
            nrow(W), length(REG), DIR))
print(fmt(W[1:min(12, nrow(W)), c("tag", REG, "score", "worst_regime", "max_st0")]), row.names = FALSE)
cat("\nBest checkpoint per seed:\n")
B <- W[!duplicated(W$seed), ]
print(fmt(B[, c("seed", "tag", "score", "worst_regime", "max_st0")]), row.names = FALSE)
cat(sprintf("\nmax |g_st0| over all checkpoints and st0-free regimes: %.4f\n", max(L$st0, na.rm = TRUE)))
cat(sprintf("SELECTED (top %d): %s\n", TOP, paste(W$tag[1:TOP], collapse = " ")))
saveRDS(list(table = W, long = L, top = W$tag[1:TOP]), file.path(DIR, "ranking.rds"))

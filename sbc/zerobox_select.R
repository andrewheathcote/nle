# Zero-box S3: apply the declared selection rule to the per-regime screens.
#
# Rule (sbc/zerobox_screen.R header, written before any number was read):
# rank checkpoints on mean |g_a| across the five zero regimes (full, sv0, SZ0,
# st00, all0); ties on the sum of |g| over the sampled parameters. The top three
# go to the posterior-shift screen per regime. Q2 reference: cap256w_c4's g in
# the full regime on the same draws. Q3: the gap regimes for the top three.
#
#   Rscript sbc/zerobox_select.R [top]      # default 3
a <- commandArgs(trailingOnly = TRUE); TOP <- if (length(a)) as.integer(a[1]) else 3L
REG <- c("full", "sv0", "SZ0", "st00", "all0")
tabs <- lapply(REG, function(r) { f <- sprintf("sbc/zerobox_screen/%s.rds", r)
  if (!file.exists(f)) stop("missing ", f); readRDS(f)$table })
names(tabs) <- REG
ctl <- sapply(tabs, function(t) t$ctl_max[1])
cat("analytic control, max |g| per regime:", paste(sprintf("%s %.1e", REG, ctl), collapse = "  "), "\n")
stopifnot(all(ctl < 5e-4))
tags <- Reduce(intersect, lapply(tabs, `[[`, "tag"))
zb <- grep("^zb_", tags, value = TRUE)
ga <- matrix(sapply(REG, function(r) { t <- tabs[[r]]; t$a[match(zb, t$tag)] }), length(zb), length(REG), dimnames = list(zb, REG))
sg <- matrix(sapply(REG, function(r) { t <- tabs[[r]]; t$sum_abs_g[match(zb, t$tag)] }), length(zb), length(REG), dimnames = list(zb, REG))
score <- rowMeans(abs(ga)); tie <- rowSums(sg)
o <- order(score, tie)
rk <- data.frame(tag = zb[o], mean_abs_g_a = score[o], sum_abs_g = tie[o],
                 round(ga[o, , drop = FALSE], 5), row.names = NULL)
names(rk)[4:8] <- paste0("g_a_", REG)
cat(sprintf("\n%d checkpoints ranked; top %d:\n", nrow(rk), TOP)); print(head(rk, TOP), digits = 4)
cat("\nworst 3:\n"); print(tail(rk, 3), digits = 4)
if ("cap256w_c4" %in% tabs$full$tag) {
  ref <- tabs$full[tabs$full$tag == "cap256w_c4", ]
  cat(sprintf("\nQ2 reference cap256w_c4, full regime, same draws: g_a %+.5f (se %.4f); sum|g| %.5f\n",
              ref$a, ref$se_a, ref$sum_abs_g))
  best <- tabs$full[tabs$full$tag == rk$tag[1], ]
  cat(sprintf("selected %s, full regime:                        g_a %+.5f (se %.4f); sum|g| %.5f  -> %s\n",
              rk$tag[1], best$a, best$se_a, best$sum_abs_g,
              if (abs(best$a) <= abs(ref$a) + 2 * sqrt(best$se_a^2 + ref$se_a^2)) "not worse than the shipped flow (Q2 pass)"
              else "WORSE than the shipped flow on a: ship both (Q2)"))
}
# per-seed spread: is selection buying anything over the seed's own median?
seed <- sub("_e[0-9]+$", "", zb)
cat("\nper seed: n, median and min of mean|g_a|\n")
print(do.call(rbind, lapply(split(score, seed), function(s) c(n = length(s), median = median(s), min = min(s)))), digits = 3)
for (r in c("gap1", "gap3")) { f <- sprintf("sbc/zerobox_screen/%s.rds", r)
  if (file.exists(f)) { t <- readRDS(f)$table; t <- t[t$tag %in% head(rk$tag, TOP), ]
    cat(sprintf("\nQ3 %s (st0 = %s ms), control max |g| %.1e:\n", r, sub("gap", "", r), t$ctl_max[1]))
    print(t[, c("tag", "v", "a", "Z", "sv", "SZ", "sum_abs_g")], digits = 4, row.names = FALSE) } }
dir.create("sbc/zerobox_screen", showWarnings = FALSE)
saveRDS(list(ranking = rk, top = head(rk$tag, TOP), rule = "mean |g_a| over full,sv0,SZ0,st00,all0; ties on sum|g|"),
        "sbc/zerobox_screen/ranking.rds")
cat(sprintf("\nselected for the posterior-shift screen: %s\nsaved sbc/zerobox_screen/ranking.rds\n", paste(head(rk$tag, TOP), collapse = ", ")))

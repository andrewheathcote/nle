# Eight-case table for the scaled DDM SBC (AWS run1, 2026-10-15) against the N = 400 reference cells.
#   Rscript sbc/scaled_8case_table.R
CASES <- c(`2` = "sv0", `3` = "SZ0", `4` = "st00", `5` = "st0only", `7` = "SZonly", `8` = "full")
REF <- c(sv0 = "sbc/nci/results_8case/sv0", SZ0 = "sbc/nci/results_8case/SZ0",
         st00 = "sbc/nci/results_zerobox_v1_aa66/st00", st0only = "sbc/nci/results_8case/st0only",
         SZonly = "sbc/nci/results_8case/SZonly", full = "sbc/nci/results_8case/full")
CARDS <- c("sc_s1106_e245", "sc_s1101_e276", "sc_s1103_e272")
rd <- function(f) if (file.exists(f)) read.csv(f) else NULL
out <- function(s) if (is.null(s)) "?" else { o <- s$parameter[!s$in_envelope]; if (length(o)) paste(o, collapse = ", ") else "none" }
worst <- function(s) if (is.null(s)) NA else { i <- which.max(abs(s$std_bias)); sprintf("%+.2f %s", s$std_bias[i], s$parameter[i]) }
rows <- list(); long <- list()
for (k in names(CASES)) { r <- CASES[[k]]
  ref <- rd(file.path(REF[[r]], "results", "summary.csv"))
  rn <- ref[ref$model == "network", ]; rc <- ref[ref$model == "control", ]
  row <- data.frame(case = k, regime = r, old_net = out(rn), old_worst = worst(rn),
                    control = if (nrow(rc)) out(rc) else "-")
  for (c in CARDS) { s <- rd(sprintf("sbc/aws/scaled_sbc/run1/results/ddm_%s/%s/results/summary.csv", c, r))
    row[[c]] <- out(s); row[[paste0(c, "_worst")]] <- worst(s)
    s$card <- c; s$case <- k; long[[length(long) + 1]] <- s }
  rows[[k]] <- row }
T <- do.call(rbind, rows); L <- do.call(rbind, long)
print(T[, c("case", "regime", "old_net", "control", CARDS)], row.names = FALSE)
cat("\nworst |std bias|:\n"); print(T[, c("case", "old_worst", paste0(CARDS, "_worst"))], row.names = FALSE)
cat("\nstd bias by parameter (rows case x card):\n")
W <- reshape(L[, c("case", "card", "parameter", "std_bias")], idvar = c("case", "card"), timevar = "parameter", direction = "wide")
names(W) <- sub("std_bias\\.", "", names(W)); W[, -(1:2)] <- round(W[, -(1:2)], 2)
print(W[order(W$case, W$card), ], row.names = FALSE)
cat("\ncoverage range:", range(L$coverage), "  min KS p:", signif(min(L$ks_p), 2), "\n")
saveRDS(list(table = T, long = L), "sbc/aws/scaled_sbc/run1/table_8case.rds")

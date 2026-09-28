# Seed study: assemble the score-bias table and apply the registered criterion.
# The criterion is in the header of sbc/seed_study_launch_tux.sh and is not
# restated or adjusted here; this script computes what it says.
#
#   Rscript sbc/seed_study_table.R <dir> <prefix>       # dir = sbc/seed_study/<prefix>
# Reads <dir>/g_<run>_<cell>.csv (sbc/score_bias_race.R, SBR_CSV) and
# <dir>/failed_exports.txt. Writes <dir>/table.csv (arm, seed, checkpoint, cell,
# par, g, abs_g, se, n, control) and <dir>/summary.csv; prints the summary.
a <- commandArgs(trailingOnly = TRUE)
DIR <- a[1]; PREFIX <- a[2]
ARMS <- c("base", "aff", "dt", "both", "malte")
PARS <- list(std = c("v", "v_lMTRUE", "B", "A", "t0"), A0 = c("v", "v_lMTRUE", "B", "t0"))
CELL_OF <- c(aff = "std", dt = "std", both = "std", malte = "A0")     # the cell each arm is judged in

rx <- sprintf("^rdm_%s_(%s)_s([0-9]+)_e([0-9]+)$", PREFIX, paste(ARMS, collapse = "|"))
split_tag <- function(tag) {
  stopifnot(all(grepl(rx, tag)))
  data.frame(arm = sub(rx, "\\1", tag), seed = as.integer(sub(rx, "\\2", tag)),
             checkpoint = as.integer(sub(rx, "\\3", tag)))
}
files <- list.files(DIR, pattern = "^g_.*\\.csv$", full.names = TRUE)
stopifnot(length(files) > 0)
g <- do.call(rbind, lapply(files, read.csv, stringsAsFactors = FALSE))
tab <- cbind(split_tag(g$tag), cell = g$cell, par = g$par, g = g$g, abs_g = abs(g$g),
             se = g$se, n = g$n, control = g$control)
# a failed export counts with |g| = Inf, in every cell its arm is screened in
ff <- file.path(DIR, "failed_exports.txt")
failed <- if (file.exists(ff)) readLines(ff) else character(0)
failed <- failed[nzchar(failed)]
if (length(failed)) {
  f <- split_tag(failed)
  add <- do.call(rbind, lapply(seq_len(nrow(f)), function(i) do.call(rbind, lapply(names(PARS), function(cl) {
    if (f$arm[i] == "malte" && cl == "std") return(NULL)
    cbind(f[rep(i, length(PARS[[cl]])), ], cell = cl, par = PARS[[cl]], g = NA_real_, abs_g = Inf,
          se = NA_real_, n = 0L, control = NA_real_)
  }))))
  tab <- rbind(tab, add)
}
tab$arm <- factor(tab$arm, levels = ARMS)
tab <- tab[order(tab$cell, tab$arm, tab$seed, tab$checkpoint, match(tab$par, PARS$std)), ]
rownames(tab) <- NULL
write.csv(tab, file.path(DIR, "table.csv"), row.names = FALSE)

cat(sprintf("=== seed study %s: %d rows -> %s ===\n", PREFIX, nrow(tab), file.path(DIR, "table.csv")))
cnt <- unique(tab[, c("arm", "seed", "checkpoint")])
cat("\nRuns and checkpoints per arm (failed exports included, as |g| = Inf):\n")
for (am in ARMS) if (any(cnt$arm == am))
  cat(sprintf("  %-6s %d seeds, %d checkpoints, %d failed exports\n", am, length(unique(cnt$seed[cnt$arm == am])),
              sum(cnt$arm == am), if (length(failed)) sum(split_tag(failed)$arm == am) else 0L))
ctl <- tapply(tab$control, tab$cell, max, na.rm = TRUE)
cat(sprintf("\nAnalytic control, max over screens: %s   (registered ceiling 1e-3)\n",
            paste(sprintf("%s %.5f", names(ctl), ctl), collapse = ", ")))

q <- function(x, p) as.numeric(quantile(x, p, names = FALSE, type = 7))
rows <- list()
for (cl in names(PARS)) {
  tc <- tab[tab$cell == cl, ]
  if (!nrow(tc) || !any(tc$arm == "base")) next
  key <- paste(tc$arm, tc$seed, tc$checkpoint)
  m_p <- sapply(PARS[[cl]], function(p) median(tc$abs_g[tc$arm == "base" & tc$par == p]))
  G <- tapply(tc$abs_g / m_p[tc$par], key, mean)                 # the primary statistic, per checkpoint
  Garm <- tc$arm[match(names(G), key)]
  for (stat in c("G", PARS[[cl]])) {
    val <- function(am) if (stat == "G") G[Garm == am] else tc$abs_g[tc$arm == am & tc$par == stat]
    base_q25 <- q(val("base"), .25)
    for (am in ARMS) {
      x <- val(am); if (!length(x)) next
      judged <- am != "base" && identical(unname(CELL_OF[am]), cl)
      rows[[length(rows) + 1]] <- data.frame(
        cell = cl, stat = stat, arm = am, n = length(x), min = min(x), q25 = q(x, .25), median = median(x),
        q75 = q(x, .75), max = max(x), base_q25 = base_q25,
        verdict = if (!judged) "" else if (median(x) < base_q25) "HELPS" else "does not help",
        rule = if (!judged) "" else if (stat == "G") "primary" else "secondary")
    }
  }
}
S <- do.call(rbind, rows)
write.csv(S, file.path(DIR, "summary.csv"), row.names = FALSE)
for (cl in unique(S$cell)) {
  cat(sprintf("\nCell %s: distribution of |g| over checkpoints (G = mean over parameters of |g| / baseline median)\n", cl))
  cat(sprintf("%-9s %-6s %4s %9s %9s %9s %9s %9s %9s  %s\n", "stat", "arm", "n", "min", "q25", "median", "q75",
              "max", "base q25", "registered rule: median < base q25"))
  for (i in which(S$cell == cl))
    cat(sprintf("%-9s %-6s %4d %9.5f %9.5f %9.5f %9.5f %9.5f %9.5f  %s\n", S$stat[i], S$arm[i], S$n[i], S$min[i],
                S$q25[i], S$median[i], S$q75[i], S$max[i], S$base_q25[i],
                if (nzchar(S$verdict[i])) sprintf("%s (%s)", S$verdict[i], S$rule[i]) else ""))
}
cat("\nPRIMARY VERDICTS\n")
P <- S[S$rule == "primary", ]
if (nrow(P)) for (i in seq_len(nrow(P)))
  cat(sprintf("  %-6s vs base, cell %-3s: median G %.4f against base lower quartile %.4f -> %s\n", P$arm[i], P$cell[i],
              P$median[i], P$base_q25[i], P$verdict[i])) else cat("  none (no baseline, or no judged arm)\n")

# Batch-size replication: two arms, same machine, same seeds, same classifier.
#
# Reads the chunked score_bias_draws.R outputs and reports, per parameter:
#   - per-checkpoint g (10 plateau checkpoints per seed)
#   - the SEED-level statistic, mean of a run's checkpoints
#   - arm mean, between-seed sd, Welch t on the arm difference, and the
#     variance ratio against F(4,4)
# The seed is the unit of replication: checkpoints within a run are not
# independent, so a t-test over 50 checkpoints would be pseudo-replication.
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
arms <- commandArgs(trailingOnly = TRUE)
if (!length(arms)) arms <- c("rep_b128", "rep_b1024")

load_arm <- function(arm) {
  fs <- Sys.glob(sprintf("sbc/rep_sb/%s_chunk*.rds", arm))
  if (!length(fs)) return(NULL)
  Gs <- lapply(fs, function(f) readRDS(f)$G)
  pars <- dimnames(Gs[[1]])[[2]]
  G <- do.call(abind_tags, list(Gs))            # draws x pars x tags
  list(G = G, pars = pars)
}
# small dependency-free bind along the tag axis
abind_tags <- function(Gs) {
  nd <- dim(Gs[[1]])[1]; np <- dim(Gs[[1]])[2]
  tg <- unlist(lapply(Gs, function(g) dimnames(g)[[3]]))
  out <- array(NA_real_, c(nd, np, length(tg)),
               dimnames = list(NULL, dimnames(Gs[[1]])[[2]], tg))
  k <- 0
  for (g in Gs) { n <- dim(g)[3]; out[, , (k+1):(k+n)] <- g; k <- k + n }
  out
}

A <- lapply(arms, load_arm); names(A) <- arms
ok <- !vapply(A, is.null, TRUE)
if (!all(ok)) cat("missing arms:", paste(arms[!ok], collapse = ", "), "\n")
A <- A[ok]; arms <- arms[ok]
pars <- A[[1]]$pars

seed_of <- function(tg) sub("^.*_(s[0-9]+)_e[0-9]+_c4$", "\\1", tg)

cat(sprintf("draws per checkpoint: %d\n", dim(A[[1]]$G)[1]))
for (a in arms)
  cat(sprintf("%-10s %d checkpoints, seeds %s\n", a, dim(A[[a]]$G)[3],
              paste(sort(unique(seed_of(dimnames(A[[a]]$G)[[3]]))), collapse = ",")))
cat("\n")

# checkpoint-level g = mean over draws
ckpt <- lapply(A, function(x) apply(x$G, c(2, 3), mean, na.rm = TRUE))  # pars x tags

for (p in pars) {
  cat(sprintf("=== g_%s ===\n", p))
  sm <- list()
  for (a in arms) {
    v <- ckpt[[a]][p, ]
    s <- seed_of(names(v))
    m <- tapply(v, s, mean)
    sm[[a]] <- m
    cat(sprintf("  %-10s seed means: %s\n", a,
                paste(sprintf("%s=%+.4f", names(m), m), collapse = "  ")))
    cat(sprintf("  %-10s arm mean %+.4f   between-seed sd %.4f   ckpt sd %.4f   n- %d/%d\n",
                "", mean(m), sd(m), sd(v), sum(v < 0), length(v)))
  }
  if (length(arms) == 2) {
    x <- sm[[arms[1]]]; y <- sm[[arms[2]]]
    tt <- t.test(y, x)
    F <- var(x) / var(y)
    cat(sprintf("  DIFF (%s - %s) %+.4f   t = %.2f (df %.1f, p = %.3f)\n",
                arms[2], arms[1], mean(y) - mean(x), tt$statistic,
                tt$parameter, tt$p.value))
    cat(sprintf("  VAR RATIO %s/%s = %.2fx   F(4,4) crit 6.39   p = %.3f\n",
                arms[1], arms[2], F,
                2 * min(pf(F, 4, 4), 1 - pf(F, 4, 4))))
  }
  cat("\n")
}

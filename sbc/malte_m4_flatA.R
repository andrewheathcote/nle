# RDM_A0_BOX_PLAN.md section 4, test 2, for the selected flow: E_true |log f_flow - log f|
# at A = 0, 0.001, 0.01, 0.05, 0.1, 0.3, 1 (target: flat in A, no worse than 0.014).
# Draws 201-300 of the stored prior draws, both accumulators. 4 cores.
a <- commandArgs(trailingOnly = TRUE)
LIB <- path.expand("~/R/nle-sbc-lib"); .libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(parallel)})
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
source("port/R/flow_race.R"); source("sbc/malte_m4_lib.R")
trapz <- function(y, x) sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
tr <- m4_truth()[201:300, ]; AS <- c(0, .001, .01, .05, .1, .3, 1)
for (tag in a) { fl <- m4_card(tag)
  r <- do.call(rbind, mclapply(seq_len(nrow(tr)), function(i) { P <- m4_nat(as.list(tr[i, ]))
    sapply(AS, function(A) mean(sapply(P, function(p) { p[["A"]] <- A
      t <- exp(seq(log(1e-4), log(100), length.out = 3000)); rt <- p[["t0"]] + t
      f <- EMC2:::dRDM(rt, matrix(p, length(rt), 5, byrow = TRUE, dimnames = list(NULL, names(p)))); ok <- f > 1e-300
      e <- flow_eval(fl, m4_context(fl, p), rt); stopifnot(e$in_box)
      trapz((f * abs(log(f) - e$log_pdf))[ok], t[ok]) / trapz(f[ok], t[ok]) }))) }, mc.cores = 4))
  cat(sprintf("%-24s %s\n", "A", paste(sprintf("%7g", AS), collapse = " ")))
  cat(sprintf("%-24s %s   max %.4f (target 0.014: %s)\n", tag, paste(sprintf("%7.4f", colMeans(r)), collapse = " "), max(colMeans(r)), if (max(colMeans(r)) <= .014) "PASS" else "FAIL"))
}

# The simulator against its twin: pLBA(rt | theta) of every simulated row is
# uniform when the data follow EMC2's LBA. Prints the KS p-value and deciles
# overall, at A = 0 exactly and at v < 0. Usage (from this directory):
#   R_LIBS=<library with EMC2> Rscript check_simulator.R [data_smoke/train]
suppressMessages(library(EMC2))
dir <- commandArgs(TRUE)[1]; if (is.na(dir)) dir <- "data_smoke/train"
files <- list.files(dir, "\\.(parquet|csv)$", full.names = TRUE)
read1 <- function(f) if (grepl("parquet$", f)) as.data.frame(arrow::read_parquet(f)) else read.csv(f)
d <- do.call(rbind, lapply(files, read1))
pars <- cbind(v = d$v, sv = exp(d$sv), b = exp(d$B) + exp(d$A), A = exp(d$A), t0 = exp(d$t0))
u <- EMC2:::pLBA(d$rt, pars)
for (g in list(all = rep(TRUE, nrow(d)), `A = 0` = d$A == -Inf, `v < 0` = d$v < 0)) {
  x <- u[g]
  cat(sprintf("%7d rows: KS p = %.3f; deciles %s\n", length(x),
              suppressWarnings(ks.test(x, "punif"))$p.value,
              paste(sprintf("%.3f", table(cut(x, seq(0, 1, .1))) / length(x)), collapse = " ")))
}

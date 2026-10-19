# The simulator against WienR (at s = 1, SZ = 0, the spec's fixed values): given the response, the PIT
# pWDM(rt, R | theta) / P(R | theta) of every simulated row is uniform when the
# data follow the DDM. Also the rows' ll against dWDM recomputed at a finer
# precision. Prints the KS p-value and deciles overall, at sv = 0, at st0 = 0, at
# both 0, at |v| > 3 and for each response. Usage (from this directory):
#   R_LIBS=<library with EMC2> Rscript check_simulator.R [data_smoke/val] [rows = 24000]
suppressMessages({library(EMC2); library(WienR)})
a <- commandArgs(TRUE)
dir <- if (length(a) > 0) a[1] else "data_smoke/val"
n_max <- if (length(a) > 1) as.integer(a[2]) else 24000L
files <- list.files(dir, "\\.(parquet|csv)$", full.names = TRUE)
read1 <- function(f) if (grepl("parquet$", f)) as.data.frame(arrow::read_parquet(f)) else read.csv(f)
d <- do.call(rbind, lapply(files, read1))
if (nrow(d) > n_max) { set.seed(1); d <- d[sort(sample.int(nrow(d), n_max)), ] }
resp <- ifelse(d$R == 2, "upper", "lower")
nat <- list(v = d$v, a = exp(d$a), t0 = exp(d$t0), w = pnorm(d$Z), sv = exp(d$sv), st0 = exp(d$st0))
cdf <- function(t) pWDM(t, response = resp, a = nat$a, v = nat$v, t0 = nat$t0, w = nat$w, sv = nat$sv,
                        sw = 0, st0 = nat$st0, precision = 1e-8)$value
u <- cdf(d$rt) / cdf(rep(1e6, nrow(d)))
groups <- list(all = rep(TRUE, nrow(d)), `sv = 0` = d$sv == -Inf, `st0 = 0` = d$st0 == -Inf,
               `sv = st0 = 0` = d$sv == -Inf & d$st0 == -Inf, `|v| > 3` = abs(d$v) > 3,
               `R = 1` = d$R == 1, `R = 2` = d$R == 2)
for (g in names(groups)) {
  x <- u[groups[[g]]]
  cat(sprintf("%-13s %6d rows: KS p = %.3f; deciles %s\n", g, length(x),
              suppressWarnings(ks.test(x, "punif"))$p.value,
              paste(sprintf("%.3f", table(cut(x, seq(0, 1, .1))) / length(x)), collapse = " ")))
}
if ("ll" %in% names(d) && all(is.finite(d$ll))) {
  ll2 <- pmax(log(dWDM(d$rt, response = resp, a = nat$a, v = nat$v, t0 = nat$t0, w = nat$w, sv = nat$sv,
                       sw = 0, st0 = nat$st0, precision = 1e-10)$value), log(1e-10))
  cat(sprintf("ll column vs dWDM at precision 1e-10: max |diff| %.2e, mean -ll %.4f (the analytic floor of the loss)\n",
              max(abs(ll2 - d$ll)), -mean(d$ll)))
}
cat(sprintf("log rt: min %.2f, 0.1%% %.2f, 99.9%% %.2f, max %.2f\n", min(log(d$rt)),
            quantile(log(d$rt), .001), quantile(log(d$rt), .999), max(log(d$rt))))

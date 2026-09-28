# Sampler test for the A = 0 RDM data (handover/RDM_A0_BOX_PLAN.md, section 2).
# One line per assertion, exit status 1 if any fails.
#
#   Rscript simulation/test_rdm_a0.R [dir]        # default data/train/rdm24m_a0
# Reads up to 20 files of <dir>/RDM/trials. Env A0_LIB as in sim_rdm_a0.R.
LIB <- Sys.getenv("A0_LIB", "")
if (nzchar(LIB)) {
  .libPaths(c(normalizePath(path.expand(LIB)), .libPaths()))
} else if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(arrow)})
a <- commandArgs(trailingOnly = TRUE)
DIR <- if (length(a)) a[1] else "data/train/rdm24m_a0"
f <- list.files(file.path(DIR, "RDM", "trials"), pattern = "parquet$", full.names = TRUE)
stopifnot(length(f) > 0)
f <- f[unique(round(seq(1, length(f), length.out = min(20, length(f)))))]
d <- do.call(rbind, lapply(f, function(x) as.data.frame(read_parquet(x))))
A <- exp(d$A); n <- nrow(d)

nfail <- 0L
check <- function(ok, lab) { cat(sprintf("%s  %s\n", if (isTRUE(ok)) "PASS" else "FAIL", lab)); if (!isTRUE(ok)) nfail <<- nfail + 1L }
share <- function(x) mean(x)
se <- function(p) 4 * sqrt(p * (1 - p) / n)               # 4 sd of a binomial share

cat(sprintf("=== A0 sampler test: %s rows from %d files of %s ===\n", format(n, big.mark = ","), length(f), DIR))
check(identical(names(d), c("param_id", "S", "R", "rt", "C", "ll", "v", "B", "t0", "A", "s")),
      "columns are those of the 16m set, in its order")
p0 <- share(A == 0)
check(abs(p0 - .15) < se(.15) && all(d$A[A == 0] == -Inf), sprintf("A = 0 exactly in %.4f of rows (15%%), stored as -Inf", p0))
check(all(A >= 0 & A <= 2), sprintf("A in [0, 2]: range [%g, %.4f]", min(A), max(A)))
# expected shares of the mixture, from its definition
c_A <- 0.05; as_max <- asinh(2 / c_A)
ex <- function(lo, hi) .70 * (hi - lo) / 2 + .15 * (asinh(hi / c_A) - asinh(lo / c_A)) / as_max
p <- share(A >= .2 & A <= .45); e <- ex(.2, .45)
check(abs(p - e) < se(e) && p > .11, sprintf("A in [0.2, 0.45], the standard cell's region: %.4f (expected %.4f, floor 0.11)", p, e))
p <- share(A > 0 & A < .01); e <- ex(0, .01)
check(abs(p - e) < se(e) && p > .008, sprintf("0 < A < 0.01, the approach to zero: %.4f (expected %.4f, floor 0.008)", p, e))
for (q in c("v", "B", "t0", "s")) {
  lo <- c(v = 1e-3, B = .1, t0 = .05, s = .1)[[q]]; hi <- c(v = 5, B = 3, t0 = 1, s = 2)[[q]]
  x <- exp(d[[q]])
  check(all(x >= lo & x <= hi) && abs(mean(x) - (lo + hi) / 2) < 4 * (hi - lo) / sqrt(12 * n),
        sprintf("%s uniform on [%g, %g]: range [%.4f, %.4f], mean %.4f", q, lo, hi, min(x), max(x), mean(x)))
}
check(all(is.finite(d$rt)) && all(d$rt > exp(d$t0)), "rt finite and above t0 in every row")
check(all(is.finite(d$ll)) && mean(d$ll <= log(1e-10) + 1e-9) < 1e-3,
      sprintf("ll finite; at the floor in %.5f of rows", mean(d$ll <= log(1e-10) + 1e-9)))
# the stored ll is the density of the stored parameters, zeros included
i <- c(which(A == 0)[1:500], which(A > 0)[1:500])
pars <- cbind(v = exp(d$v[i]), B = exp(d$B[i]), A = A[i], t0 = exp(d$t0[i]), s = exp(d$s[i]))
dl <- max(abs(pmax(log(EMC2:::dRDM(d$rt[i], pars)), log(1e-10)) - d$ll[i]))
check(dl < 1e-9, sprintf("stored ll == dRDM at the stored parameters, 500 rows at A = 0 and 500 above: max|diff| %.1e", dl))
# A = 0 rows are Wald draws: probability integral transform against the closed form
j <- which(A == 0)[1:min(20000, sum(A == 0))]
k <- exp(d$B[j]) / exp(d$s[j]); l <- exp(d$v[j]) / exp(d$s[j]); x <- d$rt[j] - exp(d$t0[j])
u <- pnorm((l * x - k) / sqrt(x)) + exp(2 * k * l + pnorm(-(l * x + k) / sqrt(x), log.p = TRUE))
pv <- suppressWarnings(ks.test(u, "punif")$p.value)
check(pv > .001, sprintf("A = 0 rows against the closed-form Wald CDF (PIT, %d rows): KS p = %.3f", length(j), pv))
cat(sprintf("=== %d failing line(s) ===\n", nfail))
quit(status = if (nfail) 1L else 0L)

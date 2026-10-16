# Oracle test of port/R/ddm_scaled.R (handover/SCALED_DDM_PLAN.md section 3, test 1): with the
# network replaced by the analytic unit-scale DDM, DDMsc's likelihood must equal EMC2's dDDM,
# st0 included, for random parameters with a and s varied. This tests the rescaling, the t0 shift
# and the CDF-difference convolution, independently of any network.
#   Rscript port/tests/test_ddm_scaled.R        (from handover/nle-methods)
.libPaths(c(path.expand(Sys.getenv("LIB", "~/R/nle-lib")), .libPaths()))
suppressPackageStartupMessages(library(EMC2))
source("port/R/ddm_scaled.R")
set.seed(20261013)
m_or <- DDMsc(evaluator = ddm_sc_oracle)()
m_an <- DDM()
res <- NULL
for (k in 1:400) {
  Z <- runif(1, .2, .8); SZp <- if (k %% 4 == 0) 0 else runif(1, .05, .9)
  st0 <- c(0, 1e-9, runif(1, .001, .02), runif(1, .02, .3))[k %% 4 + 1]
  p <- c(v = runif(1, -3, 3), a = runif(1, .4, 2.5), sv = if (k %% 3 == 0) 0 else runif(1, .05, 2),
         t0 = runif(1, .1, .5), st0 = st0, s = runif(1, .3, 2), Z = Z, SZ = 2 * SZp * min(Z, 1 - Z))
  if (abs(p[["v"]] * p[["a"]] / p[["s"]]^2) > 10 || p[["sv"]] * p[["a"]] / p[["s"]]^2 > 8) next
  sc <- (p[["a"]] / p[["s"]])^2
  rt <- p[["t0"]] + c(-0.01, 0, st0 / 3, st0, sc * rexp(6, 1 / .3) + st0)
  R <- factor(sample(1:2, length(rt), TRUE), levels = 1:2)
  P <- matrix(p, length(rt), length(p), byrow = TRUE, dimnames = list(NULL, names(p)))
  a <- EMC2:::dDDM(rt, R, P, precision = 1e-8); a[rt <= p[["t0"]]] <- 0
  o <- m_or$dfun(rt, R, P)
  res <- rbind(res, data.frame(k, st0, an = a, or = o))
}
res$err <- abs(log(res$or) - log(res$an))
z <- res$an == 0
cat(sprintf("%d parameter sets, %d densities\n", length(unique(res$k)), nrow(res)))
cat(sprintf("analytic zero (rt <= t0): oracle max %.2e\n", max(res$or[z])))
for (b in c(1e-4, 1e-2, 1)) {
  s <- !z & res$an > b
  cat(sprintf("density > %g: max |log ratio| %.2e (n %d)\n", b, max(res$err[s]), sum(s)))
}
by_st0 <- tapply(res$err[!z & res$an > 1e-2], cut(res$st0[!z & res$an > 1e-2], c(-1, 0, 1e-6, .02, 1)), max)
print(signif(by_st0, 3))
ok <- max(res$or[z]) == 0 && max(res$err[!z & res$an > 1e-2]) < 1e-3
cat("ORACLE TEST:", if (ok) "PASS" else "FAIL", "\n")

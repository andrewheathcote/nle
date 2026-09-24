# Out-of-sample test of the Phase 27 t0 correction: delta is estimated from the
# four ORIGINAL cells only, and applied to the new N = 250 cell, which played no
# part in calibrating it.
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
old <- c(Sys.glob("sbc/nci/results_c4single/*_R500.rds"),
         "sbc/nci/results_c4single_N800/DDMnn_N800_R300.rds")
get <- function(f) { x <- readRDS(f)
  list(N = x$trials, bias = x$SBC$bias$alpha$t0, rank = x$SBC$rank$alpha[, "t0"]) }
D <- lapply(old, get); names(D) <- sapply(D, `[[`, "N")
new <- get("sbc/nci/results_c4single_N250/DDMnn_N250_R300.rds")

deltas <- sapply(D, function(d) mean(d$bias))
delta <- mean(deltas)
cat(sprintf("delta from the calibrating cells only (N %s): %.5f  (x%.5f, %.2f ms at t0=0.3)\n",
            paste(names(deltas), collapse=","), delta, exp(-delta), delta*300))
cat(sprintf("new cell N=250, mean log-scale bias: %.5f   (predicted %.5f)\n\n",
            mean(new$bias), delta))

stats <- function(b, r) { r <- r[is.finite(r)]
  c(std = mean(b)/sd(b), mr = mean(r),
    ks = suppressWarnings(ks.test(r, "punif")$p.value)) }
shift <- function(d, del) { s <- -d$bias / qnorm(pmin(pmax(d$rank, 1e-6), 1-1e-6))
  s[!is.finite(s) | s <= 0] <- NA
  list(b = d$bias - del, r = pnorm(-(d$bias - del)/s)) }

cat(sprintf("%-28s %10s %10s %10s\n", "N=250 cell", "std bias", "mean rank", "KS p"))
u <- stats(new$bias, new$rank)
cat(sprintf("%-28s %+10.3f %10.3f %10.2e\n", "uncorrected", u["std"], u["mr"], u["ks"]))
z <- shift(new, delta); c1 <- stats(z$b, z$r)
cat(sprintf("%-28s %+10.3f %10.3f %10.2e\n", "corrected (out of sample)", c1["std"], c1["mr"], c1["ks"]))

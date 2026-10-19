# A card inside EMC2's race likelihood: compiled vs R path, and the network vs
# its analytic twin (LBA), over 50 prior draws of a two-accumulator design with
# 100 trials per stimulus. Usage: R_LIBS=<library with EMC2> Rscript check_in_emc2.R <card>
suppressMessages(library(EMC2)); nle <- function(f) getFromNamespace(f, "EMC2")
card <- commandArgs(TRUE)[1]
m <- suppressMessages(register_nn_model(card, twin = LBA))
PM <- c(v = 1.5, v_lMTRUE = 1, sv = log(1), B = log(1), A = log(.5), t0 = log(.25))
PS <- c(v = .3, v_lMTRUE = .3, sv = .1, B = .2, A = .2, t0 = .1)
FO <- list(v ~ lM, sv ~ 1, B ~ 1, A ~ 1, t0 ~ 1)
cell <- suppressWarnings(nn_cell(m, mean = PM, sd = PS, formula = FO, factors = list(S = c("left", "right")),
                                 Rlevels = c("left", "right"), matchfun = function(d) d$S == d$lR))
set.seed(7)
dat <- suppressMessages(make_data(cell$mean, cell$control_design, n_trials = 100))
X <- nle("nn_cell_draws")(cell, 50)
dadm <- nle("nn_dadm")(dat, cell$design)
prepared <- cell$design$model()
r_path <- function() { ml <- prepared; ml$c_name <- NULL; ml }
native <- as.numeric(nle("calc_ll_manager")(X, dadm, cell$design$model))
rp <- as.numeric(nle("calc_ll_manager")(X, dadm, r_path))
cat(sprintf("compiled vs R path over %d draws (N = %d): max |diff| %.2e; all finite %s\n",
            nrow(X), nrow(dat), max(abs(native - rp)), all(is.finite(native))))
dadm_c <- nle("nn_dadm")(dat, cell$control_design)
an <- as.numeric(nle("calc_ll_manager")(X, dadm_c, cell$control_design$model))
cat(sprintf("network vs analytic LBA (summed log lik over %d trials): mean |diff| %.3f, max %.3f, per trial %.4f; mean ll %.1f vs %.1f\n",
            nrow(dat), mean(abs(native - an)), max(abs(native - an)), mean(abs(native - an)) / nrow(dat), mean(native), mean(an)))

# Per accumulator: the card's density (the registered model's dfun) against the
# analytic log density `ll` of simulated rows (default 1e5 rows of data/val).
dir <- commandArgs(TRUE)[2]; if (is.na(dir)) dir <- "data/val"
files <- list.files(dir, "\\.(parquet|csv)$", full.names = TRUE)
if (length(files)) {
  read1 <- function(f) if (grepl("parquet$", f)) as.data.frame(arrow::read_parquet(f)) else read.csv(f)
  d <- do.call(rbind, lapply(files, read1)); d <- d[seq_len(min(nrow(d), 1e5)), ]
  P <- cbind(v = d$v, sv = exp(d$sv), B = exp(d$B), A = exp(d$A), t0 = exp(d$t0))
  P <- cbind(P, b = P[, "B"] + P[, "A"])
  lq <- log(m()$dfun(d$rt, P)); keep <- d$ll > log(1e-10)
  e <- (lq - d$ll)[keep]
  cat(sprintf("per accumulator, %d rows of %s: mean log p - log q %.4f (the KL estimate), |diff| median %.4f, 90%% %.4f, 99%% %.4f\n",
              sum(keep), dir, -mean(e), median(abs(e)), quantile(abs(e), .9), quantile(abs(e), .99)))
  for (g in list(`A = 0` = d$A[keep] == -Inf, `v < 0` = d$v[keep] < 0, `rt > 10 s` = d$rt[keep] > 10))
    cat(sprintf("   %6d rows: |diff| median %.4f, 90%% %.4f\n", sum(g), median(abs(e[g])), quantile(abs(e[g]), .9)))
}

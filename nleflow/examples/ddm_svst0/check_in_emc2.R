# A card inside EMC2's DDM likelihood, registered with twin = DDM and the card's
# `fixed` (s = 1, SZ = 0) as exceptions. (1) Compiled vs R path, and the network vs
# its analytic twin (DDM), over 50 prior draws of a two-stimulus design with 100
# trials per stimulus. (2) The card against WienR per trial: the registered
# model's dfun against the analytic log density `ll` of held-out rows (default
# 1e5 rows of data/val). (3) With a third argument, EMC2's evaluation over prior
# draws in the four regimes (sv and st0 free or exactly 0) written as reference
# cases for `nleflow onnx-check --cases` (columns: the context on the stored
# scale, rt, choice -1/+1, logp).
#   R_LIBS=<library with EMC2> Rscript check_in_emc2.R <card> [data/val] [cases.csv]
suppressMessages(library(EMC2)); nle <- function(f) getFromNamespace(f, "EMC2")
a <- commandArgs(TRUE)
card <- a[1]
FIXED <- unlist(jsonlite::fromJSON(card)$fixed)          # s = 1, SZ = 0: held in training
m <- suppressMessages(register_nn_model(card, twin = DDM, exceptions = FIXED))
reg <- m()$nn
cat(sprintf("card %s: kind %s, inputs %s\n", basename(card), reg$kind, paste(reg$pars, collapse = ", ")))

# --- (1) through a design ------------------------------------------------------------
PM <- c(v = 1, a = log(1.2), t0 = log(.3), Z = qnorm(.5), sv = log(.5), st0 = log(.1))
PS <- c(v = .3, a = .15, t0 = .1, Z = .1, sv = .2, st0 = .2)
FO <- lapply(names(PM), function(p) stats::as.formula(paste(p, "~ 1")))
cell <- suppressWarnings(nn_cell(m, mean = PM, sd = PS, formula = FO, factors = list(S = c("left", "right")),
                                 Rlevels = c("left", "right")))
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
cat(sprintf("network vs analytic DDM (summed log lik over %d trials): mean |diff| %.3f, max %.3f, per trial %.4f; mean ll %.1f vs %.1f\n",
            nrow(dat), mean(abs(native - an)), max(abs(native - an)), mean(abs(native - an)) / nrow(dat), mean(native), mean(an)))

# --- (2) per trial against WienR on held-out rows ---------------------------------------
natural <- function(d) cbind(v = d$v, a = exp(d$a), t0 = exp(d$t0), Z = pnorm(d$Z), SZ = FIXED[["SZ"]],
                             sv = exp(d$sv), st0 = exp(d$st0), s = FIXED[["s"]])
dir <- if (length(a) > 1) a[2] else "data/val"
files <- list.files(dir, "\\.(parquet|csv)$", full.names = TRUE)
if (length(files)) {
  read1 <- function(f) if (grepl("parquet$", f)) as.data.frame(arrow::read_parquet(f)) else read.csv(f)
  d <- do.call(rbind, lapply(files, read1)); d <- d[seq_len(min(nrow(d), 1e5)), ]
  lq <- log(m()$dfun(d$rt, factor(d$R, levels = 1:2), natural(d))); keep <- d$ll > log(1e-10)
  e <- (lq - d$ll)[keep]
  cat(sprintf("per trial, %d rows of %s: mean log p - log q %.4f (the KL estimate), |diff| median %.4f, 90%% %.4f, 99%% %.4f\n",
              sum(keep), dir, -mean(e), median(abs(e)), quantile(abs(e), .9), quantile(abs(e), .99)))
  dk <- d[keep, ]
  groups <- list(`sv = 0` = dk$sv == -Inf, `st0 = 0` = dk$st0 == -Inf, `sv = st0 = 0` = dk$sv == -Inf & dk$st0 == -Inf,
                 `|v| > 3` = abs(dk$v) > 3, `rt - t0 < 0.05 s` = dk$rt - exp(dk$t0) < .05, `rt > 3 s` = dk$rt > 3)
  for (g in names(groups)) { i <- groups[[g]]
    cat(sprintf("   %-17s %6d rows: |diff| median %.4f, 90%% %.4f, 99%% %.4f\n", g, sum(i),
                median(abs(e[i])), quantile(abs(e[i]), .9), quantile(abs(e[i]), .99))) }
}

# --- (3) reference cases for the ONNX graph ------------------------------------------------
if (length(a) > 2) {
  OUT <- a[3]; DRAWS <- 200L; TRIALS <- 100L
  PIN <- list(full = c(), sv0 = c(sv = -Inf), st00 = c(st0 = -Inf), both0 = c(sv = -Inf, st0 = -Inf))
  set.seed(20261019); rows <- list()
  for (regime in names(PIN)) for (k in seq_len(DRAWS)) {
    th <- PM + PS * rnorm(length(PM)); th[names(PIN[[regime]])] <- -Inf
    nat <- c(v = th[["v"]], a = exp(th[["a"]]), t0 = exp(th[["t0"]]), Z = pnorm(th[["Z"]]), SZ = FIXED[["SZ"]],
             sv = exp(th[["sv"]]), st0 = exp(th[["st0"]]), s = FIXED[["s"]])
    # times: before and just after the onset, inside the st0 window, and spread over the
    # decision time. Not at rt = t0 exactly: the graph takes log t0 and onnxruntime's
    # batched Exp may differ from R's exp by an ulp, which decides a tie at the
    # discontinuous onset of an st0 = 0 density either way.
    rt <- nat[["t0"]] + c(-0.01, -1e-6, nat[["st0"]] / 3 + 1e-6, nat[["st0"]] + 1e-6,
                          exp(runif(TRIALS - 4, log(1e-3), log(4))))
    R <- sample(1:2, length(rt), replace = TRUE)
    P <- matrix(nat, length(rt), length(nat), byrow = TRUE, dimnames = list(NULL, names(nat)))
    pdf <- m()$dfun(rt, factor(R, levels = 1:2), P)
    rows[[length(rows) + 1]] <- data.frame(regime = regime, draw = k, t(th), rt = rt, choice = ifelse(R == 2, 1, -1),
                                           logp = log(pdf), check.names = FALSE)
  }
  cs <- do.call(rbind, rows)
  write.csv(format(cs, digits = 17), OUT, row.names = FALSE, quote = FALSE)
  cat(sprintf("wrote %s: %d rows, %d regimes x %d draws x %d trials; finite logp %d, -Inf %d\n", OUT, nrow(cs),
              length(PIN), DRAWS, TRIALS, sum(is.finite(cs$logp)), sum(is.infinite(cs$logp))))
}

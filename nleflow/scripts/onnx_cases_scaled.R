# Reference cases for the ONNX graph of a scaled DDM card: EMC2's own evaluation
# (nn_eval_scaled, the R path of the flow_scaled kind, which equals the compiled
# path to ~1e-13) over prior draws in the six regimes of sbc/scaled_cell.R.
#
#   Rscript onnx_cases_scaled.R <card_weights.json> <out.csv> [draws per regime = 200] [trials = 100]
#
# Env: NLEFLOW_RLIB (library holding EMC2 dev-flow). Columns: regime, draw, the
# natural DDM parameters v, a, t0, s, Z, SZ (proportion), sv, st0, rt, choice
# (-1 lower, +1 upper), logp (log density; -Inf for rt <= t0 or pdf = 0).
LIB <- Sys.getenv("NLEFLOW_RLIB", "")
if (nzchar(LIB)) .libPaths(c(normalizePath(path.expand(LIB)), .libPaths()))
suppressPackageStartupMessages(library(EMC2))
a <- commandArgs(trailingOnly = TRUE)
CARD <- normalizePath(a[1], mustWork = TRUE); OUT <- a[2]
DRAWS <- if (length(a) > 2) as.integer(a[3]) else 200L
TRIALS <- if (length(a) > 3) as.integer(a[4]) else 100L
m <- register_nn_model(CARD, twin = DDM); reg <- m()$nn
stopifnot(reg$kind == "flow_scaled")
nn_eval_scaled <- getFromNamespace("nn_eval_scaled", "EMC2")

# the SBC cells' prior (sbc/scaled_cell.R), sampled scale; s = 1 is the cell convention,
# so half the draws also vary s (log-normal, sd .15) and a is drawn with it
PM <- c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5), sv = log(.1), SZ = qnorm(.2), st0 = log(.05))
PS <- c(v = .15, a = .15, t0 = .1, Z = .05, sv = .15, SZ = .15, st0 = .15)
PIN <- list(full = c(), sv0 = c(sv = -Inf), SZ0 = c(SZ = -Inf), st00 = c(st0 = -Inf),
            st0only = c(sv = -Inf, SZ = -Inf), SZonly = c(sv = -Inf, st0 = -Inf))
set.seed(20261017)
rows <- list()
for (regime in names(PIN)) {
  for (k in seq_len(DRAWS)) {
    th <- PM + PS * rnorm(length(PM)); th[names(PIN[[regime]])] <- -Inf
    s <- if (k %% 2 == 0) exp(rnorm(1, 0, .15)) else 1
    nat <- c(v = th[["v"]], a = exp(th[["a"]]), sv = exp(th[["sv"]]), t0 = exp(th[["t0"]]), st0 = exp(th[["st0"]]),
             s = s, Z = pnorm(th[["Z"]]), SZ = pnorm(th[["SZ"]]))
    c2 <- (nat[["s"]] / nat[["a"]])^2
    # times: a few before and at the onset, inside the st0 window, and spread over the decision time
    u <- c(-0.01, 0, nat[["st0"]] / 3, nat[["st0"]], exp(runif(TRIALS - 4, log(1e-3), log(4))) / c2 + nat[["st0"]])
    rt <- nat[["t0"]] + u
    R <- sample(1:2, length(rt), replace = TRUE)
    P <- matrix(nat, length(rt), length(nat), byrow = TRUE, dimnames = list(NULL, names(nat)))
    pdf <- nn_eval_scaled(reg, rt, R, P, "pdf")
    rows[[length(rows) + 1]] <- data.frame(regime = regime, draw = k, v = nat[["v"]], a = nat[["a"]], t0 = nat[["t0"]],
                                           s = nat[["s"]], Z = nat[["Z"]], SZ = nat[["SZ"]], sv = nat[["sv"]],
                                           st0 = nat[["st0"]], rt = rt, choice = ifelse(R == 2, 1, -1),
                                           logp = log(pdf))
  }
}
d <- do.call(rbind, rows)
write.csv(format(d, digits = 17), OUT, row.names = FALSE, quote = FALSE)
cat(sprintf("wrote %s: %d rows, %d regimes x %d draws x %d trials; finite logp %d, -Inf %d\n", OUT, nrow(d),
            length(PIN), DRAWS, TRIALS, sum(is.finite(d$logp)), sum(is.infinite(d$logp))))

# Score bias averaged over the SBC PRIOR DRAWS, for several exported flows.
#
# sbc/score_bias_flow.R measures g at ONE generating point (the precision
# study's). But SBC draws truths from the prior, and the score penalty is
# trained over the whole training box — so a single point can miss an effect
# that is real over the region. This evaluates g per draw and averages.
#
# Estimator is the validated grid one from sbc/st0_slope.R:
#   g = d/dtheta E_true[log f_flow] at the truth, central difference, with
#   E_true taken over a log-spaced rt grid weighted by the true defective
#   density TIMES the trapezoid interval widths (omitting the widths is not an
#   expectation and biases g badly — see st0_slope.R).
# Validation: the same estimator on the ANALYTIC likelihood must give g ~ 0;
# computed on the first --n-check draws and reported per parameter.
# t0 is excluded: it moves the support boundary and does not validate.
#
# Usage:
#   Rscript sbc/score_bias_draws.R <n_draws> <tag1> <tag2> ...

if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")
suppressPackageStartupMessages({library(EMC2); library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
n_draws <- as.integer(args[1]); tags <- args[-1]
# Optional: SV=<value> re-centres the sv prior for the draws, to test the
# score bias where sv is IDENTIFIED. At the SBC generating point sv=0.1 the
# unit information is 3e-6 (N* = 14M trials); at sv=1.0 it is 0.0143
# (N* = 3106), so calibration there is a real test rather than prior-driven.
sv_new <- NA_real_
hit <- grep("^SV=", tags)
if (length(hit)) { sv_new <- as.numeric(sub("^SV=", "", tags[hit])); tags <- tags[-hit] }
N_CHECK <- 40L
# Central differences carry O(H^2) truncation error. At H = 0.05 the analytic
# control for `a` is +0.00187 — 14% of cap256w_c4's signal; at H = 0.01 it is
# under 1e-4. The offset is common to every model measured the same way, so
# RANKINGS are unaffected, but absolute values should be quoted at 0.01.
H <- as.numeric(Sys.getenv("SB_H", "0.01"))
PARS <- c("v","a","Z","sv","SZ","st0")            # t0 excluded (moving support)
FI <- c(v=1L,a=2L,t0=3L,s=4L,Z=5L,SZ=6L,sv=7L,st0=8L)
# The st0 = 0 build has a 7-column context and no st0 at all. Detect it from the
# first export rather than from a flag, so the estimator cannot be pointed at a
# 7-column model while still differencing an 8th parameter that does not exist.
.ctx_of <- function(tg) {
  f <- sprintf("port/export/ddm_%s_weights.json", sub("^ENS:.*?,?", "", tg))
  if (!file.exists(f)) return(NULL)
  jsonlite::fromJSON(f, simplifyVector = TRUE)$context_names
}
.ctx <- .ctx_of(tags[1])
if (!is.null(.ctx) && !("st0" %in% .ctx)) {
  PARS <- c("v","a","Z","sv","SZ")
  FI   <- c(v=1L,a=2L,t0=3L,s=4L,Z=5L,SZ=6L,sv=7L)
  ST0_ZERO <- TRUE
  cat("context has no st0: 7-parameter (st0 = 0) mode; st0 dropped from PARS\n")
} else ST0_ZERO <- FALSE

# A tag of the form ENS:tagA,tagB,... builds an equal-weight density mixture
# of those members (EMC2's shipped ddm_build_ensemble), so an ensemble can be
# measured with exactly the same estimator as a single flow.
build_one <- function(tg) {
  if (grepl("^ENS:", tg)) {
    mem <- strsplit(sub("^ENS:", "", tg), ",")[[1]]
    EMC2:::ddm_build_ensemble(lapply(mem, function(m)
      ddm_load(sprintf("port/export/ddm_%s_weights.json", m))))
  } else {
    fl <- ddm_load(sprintf("port/export/ddm_%s_weights.json", tg))
    # EMC2 ships its own COMPILED copy of the port, and that copy predates the
    # (SZ0, SZ, SZ1) reparameterisation: it applies one scaler to one context
    # for both heads, so it silently feeds a triple-encoded flow raw EMC2
    # parameters (measured: 6.2 log units of error, which shows up as a huge
    # spurious score bias). EMC2 is installed binary-only here so it cannot be
    # rebuilt; route those models through the R port instead, which is
    # validated against JAX (15/15 golden checks). Everything else keeps the
    # fast C++ path.
    # "dectime" flows need the t0 shift applied before evaluation, which the
    # COMPILED port does not implement -- route them through the R port, as
    # "triple" already is. Sending them to ddm_build would silently evaluate
    # at rt instead of d.
    if (fl$context_encoding %in% c("triple", "dectime")) structure(list(fl=fl), class="ddm_flow_rport")
    else EMC2:::ddm_build(fl)
  }
}
ptrs <- lapply(tags, build_one)
names(ptrs) <- tags

an_lpdf <- function(th, rt, R) { w <- pnorm(th[FI[["Z"]]])
  WienR::dWDM(rt, response = ifelse(R==1,"lower","upper"), a = exp(th[FI[["a"]]]),
    v = th[FI[["v"]]], w = w, t0 = exp(th[FI[["t0"]]]), sv = exp(th[FI[["sv"]]]),
    sw = 2*pnorm(th[FI[["SZ"]]])*min(w,1-w),
    st0 = if (ST0_ZERO) 0 else exp(th[FI[["st0"]]]),
    precision = 1e-8)$logvalue }
fl_lpdf <- function(ptr, th, rt, R) {
  Th <- matrix(th, nrow=length(rt), ncol=length(th), byrow=TRUE)
  Ri <- rep(as.integer(R), length(rt))
  if (inherits(ptr, "ddm_flow_rport")) return(ddm_eval(ptr$fl, th, rt, R)$log_pdf)
  ev <- if (inherits(ptr, "ddm_flow_ensemble"))
    EMC2:::ddm_ens_eval_trials_cpp(ptr, Th, rt, Ri)
  else EMC2:::ddm_eval_trials_cpp(ptr, Th, rt, Ri)
  ev$log_pdf
}

Lbar <- function(th, grid, f) { num<-0; den<-0
  for (R in 1:2) { lp<-f(th,grid$rt,R); w<-grid$w[[R]]
    ok <- is.finite(lp) & is.finite(w) & w>0
    if (any(ok)) { num<-num+sum(w[ok]*lp[ok]); den<-den+sum(w[ok]) } }
  if (den<=0) NA_real_ else num/den }

gslope <- function(th0, par, grid, f) { j<-FI[[par]]
  tp<-th0; tp[j]<-tp[j]+H; tm<-th0; tm[j]<-tm[j]-H
  (Lbar(tp,grid,f) - Lbar(tm,grid,f))/(2*H) }

x <- readRDS("sbc/nci/results_ens5/DDMnn_N400_R500.rds")
truth <- as.matrix(x$SBC$med$alpha) - as.matrix(x$SBC$bias$alpha)
if (is.finite(sv_new)) {
  truth[, "sv"] <- truth[, "sv"] - mean(truth[, "sv"]) + log(sv_new)
  cat(sprintf("sv prior re-centred at sv = %.2f (log %.3f)\n", sv_new, log(sv_new)))
}
n_draws <- min(n_draws, nrow(truth))
cat(sprintf("draws: %d   tags: %s\n\n", n_draws, paste(tags, collapse=", ")))

G <- array(NA_real_, c(n_draws, length(PARS), length(tags)),
           dimnames=list(NULL, PARS, tags))
A <- array(NA_real_, c(N_CHECK, length(PARS)), dimnames=list(NULL, PARS))
t0s <- proc.time()[3]
for (i in seq_len(n_draws)) {
  tr <- truth[i,]
  th0 <- if (ST0_ZERO)
    c(tr[["v"]],tr[["a"]],tr[["t0"]],0,tr[["Z"]],tr[["SZ"]],tr[["sv"]])
  else
    c(tr[["v"]],tr[["a"]],tr[["t0"]],0,tr[["Z"]],tr[["SZ"]],tr[["sv"]],tr[["st0"]])
  t0n <- exp(tr[["t0"]])
  rt <- t0n + exp(seq(log(1e-3), log(8), length.out=120))
  dr <- c(diff(rt)[1], (head(diff(rt),-1)+tail(diff(rt),-1))/2, tail(diff(rt),1))
  grid <- list(rt=rt, w=lapply(1:2, function(R) exp(an_lpdf(th0,rt,R))*dr))
  for (p in PARS) {
    for (tg in tags) G[i,p,tg] <- gslope(th0,p,grid,function(th,rt,R) fl_lpdf(ptrs[[tg]],th,rt,R))
    if (i <= N_CHECK) A[i,p] <- gslope(th0,p,grid,an_lpdf)
  }
  if (i %% 50 == 0) cat(sprintf("  %d/%d (%.0f s)\n", i, n_draws, proc.time()[3]-t0s))
}
cat(sprintf("done in %.1f min\n\n", (proc.time()[3]-t0s)/60))

cat("VALIDATION  mean g on the ANALYTIC likelihood (must be ~0):\n ")
for (p in PARS) cat(sprintf(" %s=%+.5f", p, mean(A[,p], na.rm=TRUE))); cat("\n\n")
cat(sprintf("Mean g over %d prior draws (se in brackets):\n", n_draws))
cat(sprintf("%-14s %s\n", "tag", paste(sprintf("%14s", PARS), collapse="")))
for (tg in tags) {
  cat(sprintf("%-14s", tg))
  for (p in PARS) {
    v <- G[,p,tg]; cat(sprintf(" %+8.5f(%.4f)", mean(v,na.rm=TRUE),
                               sd(v,na.rm=TRUE)/sqrt(sum(is.finite(v)))))
  }
  cat("\n")
}
OUT <- Sys.getenv("SB_OUT", "sbc/score_bias_draws_results.rds")
saveRDS(list(G=G, A=A, tags=tags, pars=PARS), OUT)
cat(sprintf("\nSaved %s\n", OUT))

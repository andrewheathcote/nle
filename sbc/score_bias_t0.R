# Score bias for t0 — the parameter the standard estimator cannot measure.
#
# WHY THE USUAL ESTIMATOR FAILS. It computes g = d/dtheta E_true[log f_hat] with
# the expectation over the full support. Move t0 up by h and the ANALYTIC
# likelihood assigns zero density to [t0, t0+h), a region where the truth has
# mass, so E_true[log f] = -Inf exactly and the derivative does not exist. This
# is a property of the analytic reference, not of the flow: the flow splines
# log(rt) over the whole real line and has positive density everywhere, so its
# g_t0 is perfectly well defined. What was broken was the CONTROL, and because
# the control could not be validated, t0 was dropped from every diagnostic — and
# that is exactly where the residual in the SBC ended up (Phase 7).
#
# THE FIX. Evaluate the expectation over a fixed region A = [c, Inf) chosen so
# both perturbed densities are supported. On a restricted region the score
# identity no longer gives zero; it gives a known non-zero target
#
#     d/dtheta  E[log f | rt in A]  ->  analytic value  d/dtheta log P(A | theta)
#
# for the true model (A fixed, not depending on theta). So:
#
#     score bias = g_flow(A) - g_analytic(A)
#
# and g_analytic(A) is independently checkable against the closed form computed
# from the defective CDF. That check is the validation this estimator has and
# the old one lacked. Note the same construction reduces to the standard
# estimator when A is the full support, where d/dtheta log P(A) = 0.
#
# Usage:  Rscript sbc/score_bias_t0.R <n_draws> <tag1> [tag2 ...]
# Env:    SBT_CORES (default 4)

if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")
suppressPackageStartupMessages({library(EMC2); library(parallel); library(WienR); library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
n_draws <- as.integer(args[1]); tags <- args[-1]
CORES <- as.integer(Sys.getenv("SBT_CORES", "4"))
FI <- c(v=1L,a=2L,t0=3L,s=4L,Z=5L,SZ=6L,sv=7L,st0=8L)
H  <- 0.05            # step on the sampled (log) scale for t0
PARS <- c("t0")
# st0 = 0 build: 7-column context, detected from the first export rather than
# from a flag. With st0 = 0 the support boundary is a HARD jump -- there is no
# smear to clear -- so the restricted region can start just above t0, as in the
# RDM (sbc/score_bias_race.R), instead of past the st0 tail.
.ctx1 <- local({
  t1 <- sub("^REG:", "", tags[1])
  f <- sprintf("port/export/ddm_%s_weights.json", t1)
  if (file.exists(f)) jsonlite::fromJSON(f, simplifyVector = TRUE)$context_names
  else NULL
})
ST0_ZERO <- !is.null(.ctx1) && !("st0" %in% .ctx1)
if (ST0_ZERO) cat("context has no st0: 7-parameter mode, hard support edge\n")

# A tag of the form REG:<tag> is the DIRECT-REGRESSION baseline
# (results/model_states/<tag>_budget/regress.json), evaluated with the SAME
# restricted-region estimator so flow and regression are compared as models
# rather than as estimators. The net predicts the JOINT log-density, which is
# what the port's log_pdf and WienR's defective density also are.
.gelu <- function(x) 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
reg_logL <- function(net, th, rt, R) {
  n <- length(rt)
  raw <- cbind(matrix(th[1:8], nrow = n, ncol = 8, byrow = TRUE), log(rt))
  z <- sweep(sweep(raw, 2, net$scaler_mean, "-"), 2, net$scaler_scale, "/")
  x <- cbind(z, rep(as.numeric(R), n))
  nl <- length(net$layers$W)
  for (i in seq_len(nl)) {
    x <- sweep(x %*% net$layers$W[[i]], 2, net$layers$b[[i]], "+")
    if (i < nl) x <- .gelu(x)
  }
  as.numeric(x)
}
build_one <- function(tg) {
  if (grepl("^REG:", tg)) {
    nm <- sub("^REG:", "", tg)
    net <- jsonlite::fromJSON(sprintf("results/model_states/%s_budget/regress.json", nm),
                              simplifyVector = TRUE)
    return(structure(list(net = net), class = "ddm_regress"))
  }
  fl <- ddm_load(sprintf("port/export/ddm_%s_weights.json", tg))
  # "dectime" flows need the t0 shift applied before evaluation, which the
    # COMPILED port does not implement -- route them through the R port, as
    # "triple" already is. Sending them to ddm_build would silently evaluate
    # at rt instead of d.
    if (fl$context_encoding %in% c("triple", "dectime"))
    structure(list(fl=fl), class="ddm_flow_rport") else EMC2:::ddm_build(fl)
}
ptrs <- lapply(tags, build_one); names(ptrs) <- tags

an_lpdf <- function(th, rt, R) { w <- pnorm(th[FI[["Z"]]])
  WienR::dWDM(rt, response=ifelse(R==1,"lower","upper"), a=exp(th[FI[["a"]]]),
    v=th[FI[["v"]]], w=w, t0=exp(th[FI[["t0"]]]), sv=exp(th[FI[["sv"]]]),
    sw=2*pnorm(th[FI[["SZ"]]])*min(w,1-w),
    st0=if (ST0_ZERO) 0 else exp(th[FI[["st0"]]]),
    precision=1e-8)$logvalue }
an_cdf <- function(th, rt, R) { w <- pnorm(th[FI[["Z"]]])
  WienR::pWDM(rt, response=ifelse(R==1,"lower","upper"), a=exp(th[FI[["a"]]]),
    v=th[FI[["v"]]], w=w, t0=exp(th[FI[["t0"]]]), sv=exp(th[FI[["sv"]]]),
    sw=2*pnorm(th[FI[["SZ"]]])*min(w,1-w),
    st0=if (ST0_ZERO) 0 else exp(th[FI[["st0"]]]),
    precision=1e-8)$value }
fl_lpdf <- function(ptr, th, rt, R) {
  if (inherits(ptr, "ddm_regress")) return(reg_logL(ptr$net, th, rt, R))
  if (inherits(ptr, "ddm_flow_rport")) return(ddm_eval(ptr$fl, th, rt, R)$log_pdf)
  Th <- matrix(th, nrow=length(rt), ncol=length(th), byrow=TRUE)
  Ri <- rep(as.integer(R), length(rt))
  EMC2:::ddm_eval_trials_cpp(ptr, Th, rt, Ri)$log_pdf
}

# E[log f | rt in A] under the TRUE density at th0, restricted to the grid
Lbar <- function(th, grid, f) { num<-0; den<-0
  for (R in 1:2) { lp<-f(th,grid$rt,R); w<-grid$w[[R]]
    ok <- is.finite(lp) & is.finite(w) & w>0
    if (any(ok)) { num<-num+sum(w[ok]*lp[ok]); den<-den+sum(w[ok]) } }
  if (den<=0) NA_real_ else num/den }
gslope <- function(th0, grid, f) { j <- FI[["t0"]]
  tp<-th0; tp[j]<-tp[j]+H; tm<-th0; tm[j]<-tm[j]-H
  (Lbar(tp,grid,f) - Lbar(tm,grid,f))/(2*H) }

# closed form: d/dt0 log P(rt in A | theta), A = [c, Inf)
dlogPA <- function(th0, cut) { j <- FI[["t0"]]
  pA <- function(th) 1 - sum(sapply(1:2, function(R) an_cdf(th, cut, R)))
  tp<-th0; tp[j]<-tp[j]+H; tm<-th0; tm[j]<-tm[j]-H
  (log(pA(tp)) - log(pA(tm)))/(2*H) }

x <- readRDS("sbc/nci/results_ens5/DDMnn_N400_R500.rds")
truth <- as.matrix(x$SBC$med$alpha) - as.matrix(x$SBC$bias$alpha)
n_draws <- min(n_draws, nrow(truth))
cat(sprintf("draws: %d   tags: %s   cores: %d\n\n", n_draws, paste(tags, collapse=", "), CORES))

one <- function(i) {
  tr <- truth[i,]
  th0 <- if (ST0_ZERO)
    c(tr[["v"]],tr[["a"]],tr[["t0"]],0,tr[["Z"]],tr[["SZ"]],tr[["sv"]])
  else
    c(tr[["v"]],tr[["a"]],tr[["t0"]],0,tr[["Z"]],tr[["SZ"]],tr[["sv"]],tr[["st0"]])
  t0n <- exp(tr[["t0"]])
  # A must start above the END of the st0 smear, not merely above the onset.
  # Inside the smear the density varies on the scale of st0, so a step of H in
  # log t0 is far too coarse for a central difference there and the estimator
  # is badly biased (measured: -0.202 against a target of +0.138). Past the
  # smear the density is smooth in t0 and the estimator agrees with the closed
  # form to under 1%. The +H perturbation pushes the smear to t0*exp(H) + st0.
  # With st0 > 0 the cut must clear the END of the uniform smear, not just the
  # onset: inside the smear the density varies on the scale of st0 and a step of
  # H in log t0 is far too coarse. With st0 = 0 there is no smear, so the RDM's
  # simpler cut applies and keeps more of the informative region.
  cut <- if (ST0_ZERO) t0n * exp(H) * 1.05
         else (t0n * exp(H) + exp(tr[["st0"]])) * 1.05
  rt  <- cut + exp(seq(log(1e-3), log(8), length.out = 160))
  dr  <- c(diff(rt)[1], (head(diff(rt),-1)+tail(diff(rt),-1))/2, tail(diff(rt),1))
  grid <- list(rt=rt, w=lapply(1:2, function(R) exp(an_lpdf(th0,rt,R))*dr))
  c(analytic = gslope(th0, grid, an_lpdf),
    closed   = dlogPA(th0, cut),
    sapply(tags, function(tg) gslope(th0, grid, function(th,rt,R) fl_lpdf(ptrs[[tg]],th,rt,R))))
}
res <- do.call(rbind, mclapply(seq_len(n_draws), one, mc.cores = CORES))
ok <- apply(is.finite(res), 1, all)
cat(sprintf("usable draws: %d / %d\n\n", sum(ok), n_draws))
res <- res[ok, , drop = FALSE]

cat("VALIDATION — the estimator on the ANALYTIC likelihood must equal the closed form\n")
cat(sprintf("  estimator  %+.5f (se %.5f)\n", mean(res[,"analytic"]), sd(res[,"analytic"])/sqrt(nrow(res))))
cat(sprintf("  closed form%+.5f (se %.5f)\n", mean(res[,"closed"]),  sd(res[,"closed"])/sqrt(nrow(res))))
d <- res[,"analytic"] - res[,"closed"]
cat(sprintf("  difference %+.5f (se %.5f)   correlation %.4f\n\n",
            mean(d), sd(d)/sqrt(length(d)), cor(res[,"analytic"], res[,"closed"])))

cat("SCORE BIAS on t0  =  g_flow(A) - g_analytic(A)\n")
for (tg in tags) {
  b <- res[,tg] - res[,"analytic"]
  cat(sprintf("  %-16s %+.5f (se %.5f)\n", tg, mean(b), sd(b)/sqrt(length(b))))
}
saveRDS(res, Sys.getenv("SBT_OUT", "sbc/score_bias_t0_results.rds"))

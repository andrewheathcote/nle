# Does the flow's likelihood have a spurious optimum AWAY from the truth?
#
# The joint MLE started at the truth and landed near it (+0.37 ms excess on t0),
# yet SBC gives +12.6 ms with coverage 0.000. MCMC explores globally; an
# optimiser started at the truth does not. If the flow has a better-scoring mode
# elsewhere, that reconciles the two.
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(WienR); library(parallel)})
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")
TAG <- "z0_s503_e282"; fl <- ddm_load(sprintf("port/export/ddm_%s_weights.json", TAG))
N <- 400; ND <- 12; NS <- 12      # draws, random starts per draw
FREE <- c(1,2,3,5,6,7); NM <- c("v","a","t0","Z","SZ","sv")
LO <- fl$bounds_sampled$lower; HI <- fl$bounds_sampled$upper
PSD <- c(v=.15, a=.15, t0=.1, Z=.05, SZ=.15, sv=.15)[NM]

an_l <- function(th, rt, R) { w <- pnorm(th[5])
  WienR::dWDM(rt, response=ifelse(R==1,"lower","upper"), a=exp(th[2]), v=th[1], w=w,
    t0=exp(th[3]), sv=exp(th[7]), sw=2*pnorm(th[6])*min(w,1-w), st0=0,
    precision=1e-8)$logvalue }
fl_l <- function(th, rt, R) ddm_eval(fl, th, rt, R)$log_pdf
nll <- function(f) function(p, th0, rt, R) { th <- th0; th[FREE] <- p
  -sum(sapply(1:2, function(r) { k <- R==r; if (!any(k)) return(0)
    l <- f(th, rt[k], r); l[!is.finite(l)] <- -1e3; sum(l) })) }

x <- readRDS("sbc/nci/results_ens5/DDMnn_N400_R500.rds")
truth <- as.matrix(x$SBC$med$alpha) - as.matrix(x$SBC$bias$alpha)
cat(sprintf("%s: %d draws x %d random starts (prior-sd dispersed)\n\n", TAG, ND, NS))

one <- function(i) {
  tr <- truth[i,]
  th0 <- c(tr[["v"]],tr[["a"]],tr[["t0"]],0,tr[["Z"]],tr[["SZ"]],tr[["sv"]])
  t0n <- exp(tr[["t0"]]); set.seed(7000+i)
  grid <- t0n + exp(seq(log(1e-4), log(20), length.out=3000))
  dens <- lapply(1:2, function(R) exp(an_l(th0, grid, R)))
  dr <- c(diff(grid), tail(diff(grid),1))
  pR <- sapply(dens, function(d) sum(d*dr)); pR <- pR/sum(pR)
  R <- sample(1:2, N, TRUE, prob=pR)
  rt <- sapply(R, function(r){cw<-cumsum(dens[[r]]*dr); cw<-cw/tail(cw,1); grid[which.max(cw>=runif(1))]})
  go <- function(f, start) tryCatch({
    o <- optim(start, nll(f), th0=th0, rt=rt, R=R, method="L-BFGS-B",
               lower=LO[FREE]+1e-6, upper=HI[FREE]-1e-6,
               control=list(maxit=400, factr=1e9))
    list(par=o$par, val=o$value)}, error=function(e) NULL)
  starts <- c(list(th0[FREE]), lapply(1:NS, function(k)
    pmin(pmax(th0[FREE] + rnorm(length(FREE), 0, 2*PSD), LO[FREE]+1e-5), HI[FREE]-1e-5)))
  runs <- Filter(Negate(is.null), lapply(starts, function(s) go(fl_l, s)))
  vals <- sapply(runs, `[[`, "val")
  best <- runs[[which.min(vals)]]
  at_truth <- runs[[1]]
  c(t0_truthstart = (at_truth$par[3] - th0[3]) * 300,
    t0_best       = (best$par[3]      - th0[3]) * 300,
    nll_gain      = at_truth$val - best$val)
}
res <- do.call(rbind, mclapply(seq_len(ND), one, mc.cores = 9))
cat(sprintf("%-5s %16s %16s %14s\n","draw","t0 ms (truth-start)","t0 ms (best of 13)","nll improved by"))
for (i in seq_len(nrow(res)))
  cat(sprintf("%-5d %16.2f %16.2f %14.3f\n", i, res[i,1], res[i,2], res[i,3]))
cat(sprintf("\n%-5s %16.2f %16.2f %14.3f\n","MEAN", mean(res[,1]), mean(res[,2]), mean(res[,3])))

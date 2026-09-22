# MCMC posterior vs importance-sampled posterior, ONE dataset, SAME likelihood.
#
# Phase 37: importance reweighting through the R port says the st0 = 0 flow
# displaces t0 by ~0.2 posterior sd; SBC through EMC2's sampler says +3.1 sd
# (12.6 ms). Bundle, priors, likelihood (agree to 1e-12 laptop vs tux26) and
# data-generating model are all the same. So the difference is in how the
# posterior is computed. This decides which computation is wrong.
#
# The discriminator is the log posterior itself. If the MCMC's own draws sit
# where lpost is HIGHER than anything the importance sampler covered, the
# importance sampler missed mass and SBC is right. If the MCMC sits where lpost
# is LOWER, the sampler is not sampling this posterior.
#
#   Rscript sbc/test_is_vs_mcmc.R [seed]
# Env: SAMP_CORES (default 9), NTRIAL (default 400), M (draws, default 4000)
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(EMC2); library(WienR)})
source("handover/st0_zero/DDMnn7.R")
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")
a <- commandArgs(trailingOnly = TRUE)
SEED <- if (length(a)) as.integer(a[1]) else 11L
CORES <- as.integer(Sys.getenv("SAMP_CORES", "9"))
N <- as.integer(Sys.getenv("NTRIAL", "400"))
M <- as.integer(Sys.getenv("M", "4000"))

fl <- ddm_load("port/export/ddm_z0_s503_e282_weights.json")
b <- readRDS(system.file("extdata", "flownn", "ddm_st0zero.rds", package = "EMC2"))
m <- if (is.null(b$members)) b else b$members[[1]]
cat(sprintf("installed %s step %s   port %s step %s\n", m$budget,
            as.character(m$checkpoint_step_flow), fl$budget,
            as.character(fl$checkpoint_step_flow)))

mk <- function(model) design(factors = list(subjects = 1, S = c("left", "right")),
  Rlevels = c("left", "right"),
  formula = list(v ~ 1, a ~ 1, t0 ~ 1, s ~ 1, Z ~ 1, SZ ~ 1, sv ~ 1),
  model = model, use_matchfun = FALSE, constants = c(s = log(1)), verbose = FALSE)
des_fl <- mk(DDMnn7)
sp <- names(sampled_pars(des_fl))
PM <- c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5), SZ = qnorm(.2), sv = log(.1))[sp]
PS <- c(v = .15, a = .15, t0 = .1, Z = .05, SZ = .15, sv = .15)[sp]

set.seed(SEED)
truth <- rnorm(length(sp), PM, PS); names(truth) <- sp
dat <- make_data(truth, design = des_fl, n_trials = N / 2)   # DDMnn7 rfun = analytic rDDM
rt <- dat$rt; R <- as.integer(dat$R)
cat(sprintf("truth t0 = %.4f s   min rt = %.4f s   %d trials\n",
            exp(truth[["t0"]]), min(rt), length(rt)))

full <- function(x) { th <- setNames(numeric(7), c("v","a","t0","s","Z","SZ","sv"))
                      th[sp] <- x[sp]; th }
fl_ll <- function(x) { th <- full(x)
  s <- 0; for (r in 1:2) { k <- R == r; if (!any(k)) next
    l <- ddm_eval(fl, th, rt[k], r)$log_pdf
    if (any(!is.finite(l))) return(-Inf); s <- s + sum(l) }; s }
lprior <- function(x) sum(dnorm(x[sp], PM, PS, log = TRUE))
lpost <- function(x) lprior(x) + fl_ll(x)

## --- 1. importance-sampled posterior, as in test_posterior_shift.R -----------
nlp <- function(x) { v <- -lpost(x); if (is.finite(v)) v else 1e10 }
o <- optim(truth, nlp, method = "BFGS", control = list(maxit = 500, reltol = 1e-10))
for (j in 1:12) { oj <- optim(truth + 2 * PS * rnorm(length(sp)), nlp, method = "BFGS",
                              control = list(maxit = 500, reltol = 1e-10))
                  if (oj$value < o$value) o <- oj }
H <- optimHess(o$par, nlp); e <- eigen((H + t(H)) / 2, symmetric = TRUE)
S <- e$vectors %*% diag(1 / pmax(e$values, min(1 / PS^2))) %*% t(e$vectors)
X <- sweep(matrix(rnorm(M * length(sp)), M) %*% chol(2^2 * S), 2, o$par, "+")
colnames(X) <- sp
lp <- apply(X, 1, lpost)
lq <- -0.5 * rowSums((sweep(X, 2, o$par) %*% solve(chol(2^2 * S)))^2)
w <- exp(lp - lq - max(lp - lq, na.rm = TRUE)); w[!is.finite(w)] <- 0; w <- w / sum(w)
is_mean <- colSums(w * X); is_sd <- sqrt(colSums(w * sweep(X, 2, is_mean)^2))
cat(sprintf("IS: ESS %.0f of %d,  lpost at its mode %.3f\n", 1 / sum(w^2), M, -o$value))

## --- 2. MCMC posterior, the object SBC measures ------------------------------
pr <- prior(des_fl, type = "single", pmean = PM, psd = PS)
emc <- make_emc(dat, des_fl, type = "single", prior_list = pr, n_chains = 3)
t_s <- proc.time()[3]
emc <- fit(emc, cores_per_chain = max(1, CORES %/% 3), verbose = FALSE)
cat(sprintf("MCMC fitted in %.1f min\n", (proc.time()[3] - t_s) / 60))
g <- get_pars(emc, selection = "alpha", merge_chains = TRUE)
S_mc <- sapply(g[sp], function(p) as.numeric(if (is.list(p)) p[[1]] else p))
colnames(S_mc) <- sp
mc_med <- apply(S_mc, 2, median); mc_sd <- apply(S_mc, 2, sd)

## --- 3. the comparison -------------------------------------------------------
cat(sprintf("\n%-4s %9s %9s %9s %9s %9s %9s\n", "par", "truth", "IS mean", "IS sd",
            "MCMC med", "MCMC sd", "(MC-IS)/IS sd"))
for (p in sp) cat(sprintf("%-4s %+9.4f %+9.4f %9.4f %+9.4f %9.4f %+9.2f\n", p,
  truth[[p]], is_mean[[p]], is_sd[[p]], mc_med[[p]], mc_sd[[p]],
  (mc_med[[p]] - is_mean[[p]]) / is_sd[[p]]))

lp_mc <- apply(S_mc, 1, lpost)
cat(sprintf("\nlog posterior (port, float64):\n  truth          %.3f\n  IS mode        %.3f\n",
            lpost(truth), -o$value))
cat(sprintf("  MCMC median    %.3f\n  MCMC draws     max %.3f  median %.3f\n",
            lpost(mc_med), max(lp_mc), median(lp_mc)))
cat(sprintf("  IS draws       max %.3f  (weighted mean %.3f)\n", max(lp, na.rm = TRUE),
            sum(w * lp)))
cat(sprintf("\nt0: truth %.2f ms, IS %+.2f ms, MCMC %+.2f ms from truth\n",
            1000 * exp(truth[["t0"]]), 1000 * (exp(is_mean[["t0"]]) - exp(truth[["t0"]])),
            1000 * (exp(mc_med[["t0"]]) - exp(truth[["t0"]]))))
saveRDS(list(truth = truth, is_mean = is_mean, is_sd = is_sd, mc = S_mc,
             lp_mc = lp_mc, lp_is = lp, w = w, X = X, mode = o$par,
             lp_truth = lpost(truth), lp_mode = -o$value),
        sprintf("sbc/is_vs_mcmc_seed%d.rds", SEED))

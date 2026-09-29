# M6 cost: how many likelihood evaluations one SBC replicate of the CRDM cell takes, and
# what one replicate of the Volterra control costs, measured (3 cores: one per chain).
#   Rscript sbc/malte_m6/cost_per_replicate.R [replicates] [control dt]
a <- commandArgs(trailingOnly = TRUE)
R <- if (length(a)) as.integer(a[1]) else 2L
DT <- if (length(a) > 1) as.numeric(a[2]) else 2e-3
LIB <- path.expand("~/R/nle-sbc-lib"); .libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages(library(EMC2))
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))
twin <- function() CRDM(dt = DT)
hyb <- register_nn_model("port/export/malte_crdm_aff_log_h2_k12_clip_100k_dt0.0005.json", pars = c("v", "amp", "tau", "s", "B"),
                         twin = twin, exceptions = c(A = 0), hybrid = "amp")
DES <- list(factors = list(S = c("left", "right"), D = c("left", "right")), Rlevels = c("left", "right"),
            matchfun = function(d) d$S == d$lR,
            functions = list(lD = function(d) factor(d$D == d$lR, levels = c(FALSE, TRUE))))
p_c <- c(v = log(2), v_lMTRUE = .5, B = log(1), t0 = log(.3), tau = log(.1), amp_lDTRUE = log(.4))
cell <- do.call(nn_cell, c(list(hyb, mean = p_c, sd = c(v = .15, v_lMTRUE = .1, B = .15, t0 = .1, tau = .2, amp_lDTRUE = .15),
                                formula = list(v ~ lM, B ~ 1, t0 ~ 1, tau ~ 1, amp ~ 0 + lD), constants = c(amp_lDFALSE = -Inf)), DES))
cat("control:", cell$control_label, "\n")
set.seed(11)
t1 <- system.time(nn <- run_sbc(cell$design, cell$prior, replicates = R, trials = 100, cores_per_chain = 1, cores_for_chains = 3, verbose = FALSE))[["elapsed"]]
set.seed(11)
t2 <- system.time(ct <- run_sbc(cell$control_design, cell$control_prior, replicates = R, trials = 100, cores_per_chain = 1, cores_for_chains = 3, verbose = FALSE))[["elapsed"]]
cat(sprintf("COST network: %.1f s per replicate (wall, 3 chains on 3 cores) = %.1f core-s | control dt %g: %.1f s per replicate = %.1f core-s | ratio %.1f\n",
            t1 / R, 3 * t1 / R, DT, t2 / R, 3 * t2 / R, t2 / t1))
saveRDS(list(R = R, dt = DT, t_nn = t1, t_ctl = t2), sprintf("sbc/malte_m6/cost_dt%g.rds", DT))

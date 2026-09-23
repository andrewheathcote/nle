# Our DDM flow against the likelihood the FULL DDM actually uses.
#
# HSSM has no LAN for the full seven-parameter DDM: `full_ddm` (v, a, z, t, sz,
# sv, st) ships a BLACKBOX likelihood only -- hddm_wfpt's numerical integration
# at err = 1e-8 -- so that, not a network, is the thing our flow replaces at
# this parameterisation. Conventions verified against WienR: max |diff| 4.3e-07
# over 1000 trials (hddm's t is the midpoint and st the full width; its a is the
# full separation, which is why HSSM passes a*2).
#
#   Rscript sbc/bench_fullddm.R
# Env: PYBIN (a python with hddm_wfpt installed), BF_N (default 1000)
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages({library(WienR); library(EMC2); library(jsonlite)})
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")
PY <- Sys.getenv("PYBIN", "")
N <- as.integer(Sys.getenv("BF_N", "1000"))
fl <- ddm_load("port/export/ddm_cap256w_c4_weights.json")

# three points: our training-box floors, a moderate setting, and one with the
# variability parameters large -- where the numerical integration has to work
settings <- list(
  `floors  (sv .01, sz .05, st0 .10)` = c(v=1, a=1.6, w=.5, t0=.3, st0=.10, sv=.01, sz=.05),
  `moderate(sv .50, sz .20, st0 .20)` = c(v=1, a=1.6, w=.5, t0=.3, st0=.20, sv=.50, sz=.20),
  `large   (sv 1.5, sz .40, st0 .30)` = c(v=1, a=1.6, w=.5, t0=.3, st0=.30, sv=1.5, sz=.40))

ms <- function(f, reps = 20) { t <- system.time(for (k in seq_len(reps)) f())[3] / reps
                               unname(t * 1000 * 1000 / N) }
cat(sprintf("ms per 1000 trials, one core, n = %d per call\n\n", N))
cat(sprintf("%-34s %10s %10s %10s %10s\n", "setting", "HSSM", "WienR", "EMC2 dflt", "our flow"))
res <- NULL
for (nm in names(settings)) {
  p <- settings[[nm]]
  set.seed(7); rt <- runif(N, p[["t0"]] + .05, p[["t0"]] + 1.2); R <- sample(1:2, N, TRUE)
  wr <- function(prec) ms(function() WienR::dWDM(rt, response = ifelse(R == 1, "lower", "upper"),
    a = p[["a"]], v = p[["v"]], w = p[["w"]], t0 = p[["t0"]], sv = p[["sv"]],
    sw = p[["sz"]], st0 = p[["st0"]], precision = prec)$logvalue)
  SZ <- p[["sz"]] / (2 * min(p[["w"]], 1 - p[["w"]]))      # EMC2's SZ proportion
  pars <- cbind(v=p[["v"]], a=p[["a"]], t0=p[["t0"]], s=1, Z=p[["w"]], SZ=SZ,
                sv=p[["sv"]], st0=p[["st0"]])[rep(1, N), ]
  flow <- ms(function() EMC2:::dDDMnn(rt, as.integer(R), pars))
  py <- if (nzchar(PY)) fromJSON(system2(PY, c("/private/tmp/claude-501/-Users-andrew-Downloads-julian-nle/af944744-342a-49be-b95b-71e1a6028a54/scratchpad/bench_fullddm.py",
        p[["v"]], p[["a"]], p[["w"]], p[["t0"]], p[["st0"]], p[["sv"]], p[["sz"]], N, 20),
        stdout = TRUE))$ms_per_1000 else NA_real_
  cat(sprintf("%-34s %10.2f %10.2f %10.2f %10.2f\n", nm, py, wr(1e-8), wr(5e-3), flow))
  res <- rbind(res, data.frame(setting = nm, hssm_blackbox = py, wienr_1e8 = wr(1e-8),
                               emc2_default = wr(5e-3), flow_compiled = flow))
}
saveRDS(res, "sbc/bench_fullddm_results.rds")
cat("\nHSSM full_ddm = hddm_wfpt numerical integration at err=1e-8, the only\n")
cat("likelihood HSSM offers for this parameterisation (no LAN exists for it).\n")

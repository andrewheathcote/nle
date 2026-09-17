#!/bin/bash
# Analytic DDM control cells on tux26 (72 cores).
#
# WHY: the flow's t0 bias grows with N (+0.118 / +0.249 / +0.319 / +0.371 at
# N = 100/200/400/800) and we only have an analytic control at N=100, where it
# sits at -0.015. Without the analytic trend we cannot say the drift is the
# flow's rather than the pipeline's. The N=100 cell also cross-validates tux26
# against the existing Gadi result (DDM_N100_R500 in results/).
#
# Cost: the analytic likelihood is per-trial, so it scales with N far more
# steeply than the flow (which is dominated by fixed per-call overhead). Measured
# locally: ~80 core-minutes per replicate at N=400. 500 replicates is therefore
# ~40,000 core-minutes, i.e. ~8 h on 72 cores — feasible here, where it was 41 h
# on 4 local cores.
#
# Cores are split by estimated cost so both cells land together.
#
#   bash launch_analytic_tux.sh           # N=100 and N=400, 500 reps each
#   SBC_REPS=250 bash launch_analytic_tux.sh
set -e
export EMC2NN_LIB=${EMC2NN_LIB:-$HOME/emc2nn/Rlib-devnn}
[ -d "$EMC2NN_LIB" ] || { echo "ABORT: no library at $EMC2NN_LIB"; exit 1; }
REPS=${SBC_REPS:-500}
TOTAL=${SBC_TOTAL_CORES:-72}
# cost weights from measurement: N=100 ~ 6.6 core-min/rep, N=400 ~ 80
C100=$(( TOTAL * 7 / 100 )); [ "$C100" -lt 2 ] && C100=2
C400=$(( TOTAL - C100 ))

if pgrep -f "SBCanalytic\.R" >/dev/null; then
  echo "REFUSING: an analytic run is already active"; pgrep -af "SBCanalytic\.R"; exit 1
fi
Rscript -e '.libPaths(c(Sys.getenv("EMC2NN_LIB"), .libPaths())); library(EMC2)
  cat("EMC2:", find.package("EMC2"), as.character(packageVersion("EMC2")), "\n")
  stopifnot(exists("DDM", envir = asNamespace("EMC2")))'

echo "analytic DDM, $REPS reps: N=100 on $C100 cores, N=400 on $C400 cores"
for spec in "100 $C100" "400 $C400"; do
  set -- $spec; N=$1; C=$2
  LOG="SBCanalytic_N${N}.log"
  nohup env EMC2NN_LIB="$EMC2NN_LIB" SBC_CORES="$C" \
    Rscript SBCanalytic.R "results_analytic_N${N}" "$N" "$REPS" > "$LOG" 2>&1 &
  echo "  N=$N on $C cores -> results_analytic_N${N} (pid $!, log $LOG)"
done
echo
echo "progress:  grep -c 'Processing data set' SBCanalytic_N*.log"
echo "expect ~8 h; the N=100 cell should reproduce results/DDM_N100_R500.rds"

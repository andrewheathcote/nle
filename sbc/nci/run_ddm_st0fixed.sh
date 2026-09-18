#!/bin/bash
# Cell A then Cell B, sequentially: the DDM flow with st0 held constant.
#
#   bash sbc/nci/run_ddm_st0fixed.sh [cores_per_chain] [N] [reps]
#   default: 8 per chain (= 24 cores), N=400, 500 replicates
#
# N=400 because that is where the estimated-st0 cell shows the effect most
# clearly (t0 std bias +0.319, rank KS p 1.8e-07) with a matched 500-replicate
# comparison already in hand.
#
# EMC2 runs 3 chains in parallel, so TOTAL cores = 3 x cores_per_chain.
# Timing calibrated on the real estimated-st0 runs, which were FLAT in N
# (175/166/152 min at N=100/200/400, 48 cores) -- SBC cost here is dominated by
# per-replicate MCMC overhead, not by N. So expect ~2.5 h per cell at 48 cores,
# ~5 h at 24, and roughly double that again at 12.
#
# NOT set -e: cell B must still run, and the bundle must still be restored,
# if cell A fails.
set -u
cd "$(dirname "$0")/../.."
CORES=${1:-8}; N=${2:-400}; REPS=${3:-500}
OUT=sbc/nci/results_ddm_st0fixed
echo "cores_per_chain=$CORES (x3 chains = $((CORES*3)))  N=$N  reps=$REPS"

DST=$(Rscript -e 'if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()));
  cat(system.file("extdata","flownn","ddm_ens5.rds", package="EMC2", mustWork=TRUE))' 2>/dev/null)
[ -n "$DST" ] || { echo "ABORT: cannot locate EMC2 extdata"; exit 1; }
[ -f sbc/ddm_ens5_ORIGINAL.rds ] || cp "$DST" sbc/ddm_ens5_ORIGINAL.rds

# cap256w_c4 is the model under test and is already what is installed; verify
# rather than assume, because a stale bundle silently measures the wrong flow.
SRC=$HOME/emc2nn/nci/ddm_ens5_single_c4.rds
if [ -f "$SRC" ]; then cp "$SRC" "$DST"; echo "installed ddm_ens5_single_c4 -> $DST"
else echo "WARNING: $SRC absent; using whatever is installed"; fi

for V in 0.05 0.01; do
  echo ""
  echo "############ st0 fixed at $V ############"
  SBC_CORES=$CORES Rscript sbc/nci/SBCddm_st0fixed.R "$OUT" "$N" "$REPS" "$V"
  echo "############ st0=$V exit $? ############"
done

cp sbc/ddm_ens5_ORIGINAL.rds "$DST"
echo ""
echo "restored original DDM bundle -> results in $OUT"

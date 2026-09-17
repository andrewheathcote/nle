#!/bin/bash
# RDM SBC on tux26. Swaps the refined [64,32] flow in, runs the cell, restores.
#
# EMC2 runs 3 chains in parallel, so TOTAL cores = 3 x cores_per_chain. On the
# laptop cores_per_chain=8 meant 24 cores on a 12-core machine and saturated it;
# here 16 means 48, which is what tux26 has spare.
#
#   bash sbc/nci/run_rdm_sbc_tux.sh            # 500 reps, 48 cores
#   SBC_CORES=8 SBC_REPS=250 bash ...          # 24 cores, half the reps
set -uo pipefail
cd "$(dirname "$0")/../.."
CORES=${SBC_CORES:-16}          # x3 chains = 48
REPS=${SBC_REPS:-500}
echo "cores_per_chain=$CORES (x3 chains = $((CORES*3)) cores)  replicates=$REPS"

DST=$(Rscript -e 'if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()));
  cat(system.file("extdata","flownn","rdm_large.rds", package="EMC2", mustWork=TRUE))')
[ -n "$DST" ] || { echo "ABORT: cannot locate EMC2 extdata — is .Rlibs linked?"; exit 1; }
[ -f sbc/rdm_large_ORIGINAL.rds ] || cp "$DST" sbc/rdm_large_ORIGINAL.rds
cp sbc/rdm_small_best.rds "$DST"
echo "installed rdm_small_best -> $DST"

SBC_CORES=$CORES Rscript sbc/nci/SBCrdm.R sbc/nci/results_rdm_small 100 "$REPS"
rc=$?
# restore whatever the outcome: a left-behind bundle makes a later run silently
# measure the wrong model
cp sbc/rdm_large_ORIGINAL.rds "$DST"
echo "restored original bundle (exit $rc)"

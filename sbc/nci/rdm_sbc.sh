#!/bin/bash
# Generic RDM SBC runner: any bundle, any N, any replicate count.
#
#   bash sbc/nci/rdm_sbc.sh <bundle> [N] [reps] [cores_per_chain]
#
#   bash sbc/nci/rdm_sbc.sh rdm_small_best 100 500 16   # the reference cell
#   bash sbc/nci/rdm_sbc.sh rdm_small_best 400 500 16   # 2nd N, to check a t0 correction
#   bash sbc/nci/rdm_sbc.sh rdm_b1k_best   100 500 16   # the [256,128] fallback
#
# EMC2 runs 3 chains in parallel, so TOTAL cores = 3 x cores_per_chain.
# Output goes to sbc/nci/results_<bundle>_N<N>/ and the ORIGINAL rdm_large.rds
# is always restored, whatever the outcome -- a left-behind bundle makes a later
# run silently measure the wrong model.
set -uo pipefail
cd "$(dirname "$0")/../.."
BUNDLE=${1:?bundle name, e.g. rdm_small_best (no .rds)}
N=${2:-100}; REPS=${3:-500}; CORES=${4:-16}
SRC="sbc/${BUNDLE}.rds"
[ -f "$SRC" ] || { echo "ABORT: $SRC not found"; exit 1; }
OUT="sbc/nci/results_${BUNDLE}_N${N}"

echo "bundle=$BUNDLE  N=$N  reps=$REPS  cores_per_chain=$CORES (x3 = $((CORES*3)))"
DST=$(Rscript -e 'if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()));
  cat(system.file("extdata","flownn","rdm_large.rds", package="EMC2", mustWork=TRUE))' 2>/dev/null)
[ -n "$DST" ] || { echo "ABORT: cannot locate EMC2 extdata"; exit 1; }
[ -f sbc/rdm_large_ORIGINAL.rds ] || cp "$DST" sbc/rdm_large_ORIGINAL.rds
cp "$SRC" "$DST"; echo "installed $BUNDLE -> $DST"

SBC_CORES=$CORES Rscript sbc/nci/SBCrdm.R "$OUT" "$N" "$REPS"
rc=$?
cp sbc/rdm_large_ORIGINAL.rds "$DST"
echo "restored original bundle (exit $rc) -> results in $OUT"

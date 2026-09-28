#!/bin/bash
# Zero-box S4 on tux26: four cells (full, st0 = 0, all-zero, sv = 0), each the
# network plus its matched analytic control, N = 400, 500 replicates. Two cells
# at a time at 3 chains x 12 cores = 36 cores each (72 in total, AH's ceiling).
# Run from ~/emc2nn/nle:  setsid nohup bash sbc/zerobox_s4_launch.sh > sbc/zerobox_s4.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
export LIB=~/emc2nn/nle/.Rlibs-nle CPC=${CPC:-12} CFC=${CFC:-3} OUT=${OUT:-sbc/nci/results_zerobox}   # OUT=..._v2 for the retrain, so v1 is kept
REPS=${REPS:-500}; CARD=${CARD:-handover/nle-methods/port/export/ddm_zb_s903_e271_weights.json}
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
run() { Rscript sbc/zerobox_s4_cell.R $1 $CARD $REPS \
          > $OUT/cell_$1.log 2>&1 && say "  cell $1 done" || say "  cell $1 FAILED"; }
mkdir -p $OUT
say "=== S4: full + st00 ==="; run full & run st00 & wait
say "=== S4: all0 + sv0 ==="; run all0 & run sv0 & wait
say "=== S4 cells done ==="
for r in full st00 all0 sv0; do echo "--- $r"; sed -n '/=== cell .* done/,$p' $OUT/cell_$r.log; done
echo "zerobox s4 done $(date)" > $OUT/done.marker

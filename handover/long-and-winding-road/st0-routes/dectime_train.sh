#!/bin/bash
# Train the decision-time (rt - t0) DDM flow: 3 seeds, 3 cores each.
#
# Context is 6 parameters -- t0 is ABSENT. Phase 34 exonerated the analytic, so
# the st0=0 failure is the flow's; Phase 33 located it in how the flow's error
# varies ACROSS parameter space, distorting the likelihood surface away from the
# truth. Dropping t0 from the conditioning set means the network cannot distort
# the surface in the t0 direction at all.
#
# --val-data is pinned: the default val set is the 8-parameter one and carries
# no `d` column, and checkpoints are retained by best val loss.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd); PY="$ROOT/flow/.venv/bin/python"
LOG="$ROOT/sbc/dectime.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

train_one() {
  local tag=$1
  local seed=$2
  local ck="$ROOT/results/model_states/${tag}_budget/DDM_flow"
  if [ -d "$ck" ] && [ "$(ls "$ck" | grep -c '^[0-9]')" -ge 30 ]; then
    say "  $tag already trained; skipping"; return 0
  fi
  OMP_NUM_THREADS=3 "$PY" flow/scripts/train_ddm.py \
    --tag "$tag" --decision-time \
    --train-data data/train/dectime400k/DDM/trials \
    --val-data data/val/DDM_dectime/trials \
    --batch 1024 --epochs 300 --num-bins 64 --widths 256,128 \
    --keep-checkpoints 30 --seed "$seed" --classifier-from c4clf \
    > "$ROOT/sbc/train_${tag}.log" 2>&1
}
say "=== decision-time training, 3 seeds (3 cores each) ==="
train_one dt_s701 701 & a=$!
train_one dt_s702 702 & b=$!
train_one dt_s703 703 & c=$!
wait $a $b $c
say "=== training done ==="
say "NEXT (manual): export needs CONTEXT_DECTIME + the d-transform in the port,"
say "  and the golden vectors must be EXTENDED, not just re-run -- the port now"
say "  computes d = rt - t0 before evaluating, which the old goldens do not cover."

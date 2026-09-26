#!/bin/bash
# Zero-box DDM flow: three seeds, concurrently, on tux26.
#
# Runs from the PACKAGE root (~/emc2nn/nle/handover/nle-methods on tux26) with
# the repo's venv, and PYTHONPATH pointing at the package's own flow/src so the
# patched loader (apply_context_encoding) is the one imported -- the venv's
# editable install points at the unpatched repo tree.
#
# Waits for the training data to finish arriving (240 chunks) before starting,
# so it can be launched while the rsync is still running.
#
#   nohup bash flow/scripts/train_zerobox_tux.sh > sbc_zerobox_train.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
PY=~/emc2nn/nle/flow/.venv/bin/python
export PYTHONPATH="$ROOT/flow/src"
LOG="$ROOT/zerobox_train.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

CLF=${CLF:-c4clf}
TRAIN="data/train/zerobox600k/DDM/trials"
VAL="data/val/zerobox/DDM/trials"
say "waiting for $TRAIN to reach 240 chunks"
until [ "$(ls "$ROOT/$TRAIN" 2>/dev/null | grep -c parquet)" -ge 240 ]; do sleep 60; done
sleep 120   # let the last chunk finish writing
say "data complete: $(ls "$ROOT/$TRAIN" | grep -c parquet) chunks, $(du -sh "$ROOT/$TRAIN" | cut -f1)"
[ "$(ls "$ROOT/$VAL" 2>/dev/null | grep -c parquet)" -ge 10 ] || { say "ABORT: val set missing"; exit 1; }
[ -d "$ROOT/results/model_states/${CLF}_budget/DDM_classifier" ] || { say "ABORT: placeholder classifier ${CLF}_budget missing"; exit 1; }

# --classifier-from copies an EXISTING classifier directory into the run as a
# placeholder (train_ddm.py cannot train one itself; plain MLE training never
# evaluates it). It is replaced before export by one regressed on the exact
# P(R) targets (retrain_classifier.py --zerobox), as fix_z0_classifier.sh did
# for the st0 = 0 build. CLF names a budget dir linked into results/model_states.
train_one() {
  local tag=$1 seed=$2
  local ck="$ROOT/results/model_states/${tag}_budget/DDM_flow"
  if [ -d "$ck" ] && [ "$(ls "$ck" | grep -c '^[0-9]')" -ge 30 ]; then
    say "  $tag already trained; skipping"; return 0
  fi
  OMP_NUM_THREADS=4 "$PY" flow/scripts/train_ddm.py \
    --tag "$tag" --zerobox --classifier-from "$CLF" \
    --train-data "$TRAIN" --val-data "$VAL" \
    --batch 1024 --epochs 300 --num-bins 64 --widths 256,128 \
    --keep-checkpoints 30 --seed "$seed" \
    > "$ROOT/train_${tag}.log" 2>&1
  say "  $tag finished (exit $?)"
}
say "=== zero-box training, 3 seeds x 4 threads ==="
train_one zb_s901 901 & a=$!
train_one zb_s902 902 & b=$!
train_one zb_s903 903 & c=$!
wait $a $b $c
say "=== training done ==="
for t in zb_s901 zb_s902 zb_s903; do
  say "  $t: $(ls "$ROOT/results/model_states/${t}_budget/DDM_flow" 2>/dev/null | grep -c '^[0-9]') checkpoints; best val: $(grep -oE 'val[^0-9-]*-?[0-9.]+' "$ROOT/train_${t}.log" | tail -1)"
done
echo "zerobox training done $(date)" >> "$ROOT/zerobox_done.marker"

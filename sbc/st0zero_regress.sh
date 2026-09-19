#!/bin/bash
# st0 = 0 direct-regression arm: 3 seeds, SEQUENTIALLY, on the 3 cores the
# flow training leaves free. Phase 31 showed regression has no detectable
# systematic residual where the flow has one, and the st0 = 0 artefact
# compresses an analytic that exists -- exactly regression's regime.
#
# Sequential, not concurrent: the 3 flow runs already hold 9 of 12 cores and
# they are the critical path to the SBC handoff. One regression run at a time
# fits in the remainder without slowing them.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd); PY="$ROOT/flow/.venv/bin/python"
LOG="$ROOT/sbc/st0zero_regress.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

for s in 601 602 603; do
  tag="z0reg_s${s}"
  if [ -f "$ROOT/results/model_states/${tag}_budget/regress.json" ]; then
    say "$tag already done; skipping"; continue
  fi
  say "training $tag"
  OMP_NUM_THREADS=3 "$PY" flow/scripts/train_regress.py \
    --tag "$tag" --seed "$s" --st0-zero --target ll \
    --train-data data/train/st0zero400k/DDM/trials \
    --val-data data/val/DDM_st0zero/trials \
    --widths 256,256,256 --epochs 60 --batch 1024 --peak-lr 1e-3 \
    > "$ROOT/sbc/train_${tag}.log" 2>&1
  say "  $tag rc=$?"
done
say "ALL REGRESSION SEEDS DONE"

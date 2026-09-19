#!/bin/bash
# Seed-replicate the direct-regression baseline.
#
# Phase 30's rule: an intervention is not believed until it is compared against
# the measured between-seed spread at fixed settings. reg_b1k is ONE seed, and
# it is currently the only route beating plain MLE on the centre of g, so it is
# exactly the claim most in need of replication.
#
# Settings identical to reg_b1k (16M rows, [256,256,256], 60 epochs, batch
# 1024, lr 1e-3) with only the seed changed.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd); PY="$ROOT/flow/.venv/bin/python"
LOG="$ROOT/sbc/regress_seeds.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

train_one() {   # tag seed
  local tag=$1
  local seed=$2
  local out="$ROOT/results/model_states/${tag}_budget/regress.json"
  if [ -f "$out" ]; then say "  $tag already done; skipping"; return 0; fi
  OMP_NUM_THREADS=3 "$PY" flow/scripts/train_regress.py \
    --tag "$tag" --seed "$seed" \
    --train-data data/train/tpt40p400k/DDM/trials \
    --widths 256,256,256 --epochs 60 --batch 1024 --peak-lr 1e-3 \
    > "$ROOT/sbc/train_${tag}.log" 2>&1
}

say "=== training 3 regression seeds (3 cores each) ==="
train_one reg_s401 401 & a=$!
train_one reg_s402 402 & b=$!
train_one reg_s403 403 & c=$!
wait $a $b $c
say "=== training done; measuring ==="

TAGS="REG:reg_b1k REG:reg_s401 REG:reg_s402 REG:reg_s403"
SBG_CORES=9 Rscript sbc/score_bias_regress.R 200 reg_b1k reg_s401 reg_s402 reg_s403 \
  >> "$LOG" 2>&1
SBT_CORES=9 SBT_OUT="$ROOT/sbc/rep_sb/t0_regress_seeds.rds" \
  Rscript sbc/score_bias_t0.R 200 $TAGS >> "$LOG" 2>&1
say "=== ALL DONE ==="

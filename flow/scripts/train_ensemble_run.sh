#!/bin/bash
# Seed-ensemble chain: train arch64w seeds 102-104, export + validate each,
# then the ensemble gate study (5 members: seeds 100-104).
set -e
cd "$(dirname "$0")/.."    # flow/
for seed in 102 103 104; do
  tag="arch64w_s${seed}"
  echo "=== [$(date +%H:%M:%S)] training $tag ==="
  uv run python scripts/train_ddm.py --tag "$tag" --num-bins 64 --widths 64,32 \
      --epochs 300 --seed "$seed" > "../sbc/train_${tag}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] exporting $tag ==="
  uv run python scripts/export_flow_ddm.py --budget "$tag" >> "../sbc/train_${tag}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] validating $tag ==="
  (cd .. && Rscript port/tests/validate_ddm.R "$tag") >> "../sbc/train_${tag}.log" 2>&1 \
      && echo "    $tag golden validation PASSED" \
      || { echo "    $tag golden validation FAILED"; exit 1; }
done
echo "=== [$(date +%H:%M:%S)] ensemble gate study ==="
cd ..
Rscript sbc/run_ensemble_study_ddm.R > sbc/ensemble_study_stdout.log 2>&1
echo "=== [$(date +%H:%M:%S)] ALL DONE — gate table in sbc/ensemble_study_stdout.log ==="

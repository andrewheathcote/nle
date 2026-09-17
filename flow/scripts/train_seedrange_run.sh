#!/bin/bash
# Seed replicate (arch64w_s101) + narrowed spline range (arch64wR3), each
# trained -> exported -> golden-validated, then the joint gate study.
set -e
cd "$(dirname "$0")/.."    # flow/
for spec in "arch64w_s101 --seed 101" "arch64wR3 --range-min -3 --range-max 3"; do
  set -- $spec
  tag=$1; shift
  echo "=== [$(date +%H:%M:%S)] training $tag ($*) ==="
  uv run python scripts/train_ddm.py --tag "$tag" --num-bins 64 --widths 64,32 \
      --epochs 300 "$@" > "../sbc/train_${tag}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] exporting $tag ==="
  uv run python scripts/export_flow_ddm.py --budget "$tag" >> "../sbc/train_${tag}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] validating $tag ==="
  (cd .. && Rscript port/tests/validate_ddm.R "$tag") >> "../sbc/train_${tag}.log" 2>&1 \
      && echo "    $tag golden validation PASSED" \
      || { echo "    $tag golden validation FAILED"; exit 1; }
done
echo "=== [$(date +%H:%M:%S)] gate study ==="
cd ..
Rscript sbc/run_seedrange_study_ddm.R > sbc/seedrange_study_stdout.log 2>&1
echo "=== [$(date +%H:%M:%S)] ALL DONE — gate table in sbc/seedrange_study_stdout.log ==="

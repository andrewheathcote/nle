#!/bin/bash
# Architecture-arm training: variants targeting the DDM flow's leading-edge
# overshoot (see sbc/SBC_PLAN.md and sbc/appendix_precision_study.md).
# Trains each variant on the EXISTING large-budget (1M) data, then exports
# and golden-validates it, so the morning starts gate-ready.
#
# Launch detached from the nle repo root:
#   nohup bash flow/scripts/train_variants_tonight.sh > sbc/variants_tonight.log 2>&1 &
set -e
cd "$(dirname "$0")/.."    # flow/

for spec in "arch64 64 32,16" "arch64w 64 64,32"; do
  set -- $spec
  tag=$1; bins=$2; widths=$3
  echo "=== [$(date +%H:%M:%S)] training $tag (bins=$bins widths=$widths) ==="
  uv run python scripts/train_ddm.py --tag "$tag" --num-bins "$bins" \
      --widths "$widths" --epochs 300 > "../sbc/train_${tag}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] exporting $tag ==="
  uv run python scripts/export_flow_ddm.py --budget "$tag" \
      >> "../sbc/train_${tag}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] validating $tag ==="
  (cd .. && Rscript port/tests/validate_ddm.R "$tag") \
      >> "../sbc/train_${tag}.log" 2>&1 \
      && echo "    $tag golden validation PASSED" \
      || echo "    $tag golden validation FAILED (see sbc/train_${tag}.log)"
done
echo "=== [$(date +%H:%M:%S)] ALL VARIANTS DONE ==="

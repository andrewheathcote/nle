#!/bin/bash
# Censored-training run: arch64wC = arch64w architecture + censored MLE
# (upper censor 10 s), then export -> golden validation -> gate study.
# Launch detached from the nle repo root:
#   nohup bash flow/scripts/train_censor_run.sh > sbc/censor_run.log 2>&1 &
set -e
cd "$(dirname "$0")/.."    # flow/
echo "=== [$(date +%H:%M:%S)] training arch64wC (bins=64 widths=64,32 censor=10) ==="
uv run python scripts/train_ddm.py --tag arch64wC --num-bins 64 --widths 64,32 \
    --censor 10 --epochs 300 > ../sbc/train_arch64wC.log 2>&1
echo "=== [$(date +%H:%M:%S)] exporting arch64wC ==="
uv run python scripts/export_flow_ddm.py --budget arch64wC >> ../sbc/train_arch64wC.log 2>&1
echo "=== [$(date +%H:%M:%S)] validating arch64wC ==="
(cd .. && Rscript port/tests/validate_ddm.R arch64wC) >> ../sbc/train_arch64wC.log 2>&1 \
    && echo "    arch64wC golden validation PASSED" \
    || { echo "    arch64wC golden validation FAILED"; exit 1; }
echo "=== [$(date +%H:%M:%S)] gate study ==="
cd ..
Rscript sbc/run_censor_study_ddm.R > sbc/censor_study_stdout.log 2>&1
echo "=== [$(date +%H:%M:%S)] ALL DONE — gate table in sbc/censor_study_stdout.log ==="

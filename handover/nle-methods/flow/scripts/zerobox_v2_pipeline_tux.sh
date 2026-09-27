#!/bin/bash
# Zero-box VERSION 2 retrain, end to end on tux26 (plan section 11):
#   1. simulate the v2 training set (600k draws, ZB_VERSION=2) and its
#      validation set, concurrently;
#   2. train three seeds (zb2_s901..903) on it, THREADS each;
#   3. retrain the classifier with the new scaler and export every checkpoint.
# The v1 data, runs and exports are left untouched (different names throughout).
# Screening and the SBC cells follow from the laptop / sbc/ scripts.
#
# Run from the package root on tux26:
#   setsid nohup bash flow/scripts/zerobox_v2_pipeline_tux.sh > zb2_pipeline.out 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
export ZB_VERSION=2 EMC2NN_LIB=$HOME/emc2nn/Rlib-devnn   # the resolved dir: .Rlibs is a symlink and the sim scripts compare normalised paths
export PREFIX=zb2 SUFFIX=_v2 THREADS=${THREADS:-12} CLF=${CLF:-large}
SIMC=${SIMC:-48}

say "=== 1. simulate v2 training and validation sets (SIM_CORES $SIMC + 12) ==="
if [ "$(ls data/train/zerobox600k_v2/DDM/trials 2>/dev/null | grep -c parquet)" -ge 240 ]; then
  say "  training set already present"
else
  SIM_CORES=$SIMC Rscript simulation/sim_ddm_zerobox.R 600000 > sim_zb2_train.log 2>&1 & a=$!
  SIM_CORES=12 Rscript simulation/sim_val_zerobox.R > sim_zb2_val.log 2>&1 & b=$!
  wait $a || { say "ABORT: training simulation failed (sim_zb2_train.log)"; exit 1; }
  wait $b || { say "ABORT: validation simulation failed (sim_zb2_val.log)"; exit 1; }
fi
[ -f sim_zb2_train.log ] && say "  $(grep -E '=== done|ll at floor' sim_zb2_train.log | tr '\n' ' ')"
[ -f sim_zb2_val.log ] && say "  $(grep -E '=== val done' sim_zb2_val.log)"
[ "$(ls data/train/zerobox600k_v2/DDM/trials | grep -c parquet)" -ge 240 ] || { say "ABORT: fewer than 240 chunks"; exit 1; }
[ "$(ls data/val/zerobox_v2/DDM/trials 2>/dev/null | grep -c parquet)" -ge 10 ] || { say "ABORT: v2 validation set missing"; exit 1; }
# The sampler test reads a chunk with arrow; where arrow is absent (tux26 when
# the data was simulated elsewhere) the test is run where the data was made.
if R_LIBS=$EMC2NN_LIB Rscript -e 'quit(status = !requireNamespace("arrow", quietly = TRUE))' 2>/dev/null; then
  ZB_VERSION=2 R_LIBS=$EMC2NN_LIB Rscript simulation/test_zerobox.R > test_zb2.log 2>&1
  say "  sampler test: $(grep -c '^PASS' test_zb2.log) PASS, $(grep -c '^FAIL' test_zb2.log) FAIL"
  [ "$(grep -c '^FAIL' test_zb2.log)" -eq 0 ] || { say "ABORT: sampler test failed (test_zb2.log)"; exit 1; }
else
  say "  sampler test skipped here (no arrow); run where the data was simulated"
fi

say "=== 2. train (3 seeds x $THREADS threads) ==="
bash flow/scripts/train_zerobox_tux.sh > zb2_launcher.out 2>&1
[ -f zb2_done.marker ] || { say "ABORT: training did not write its marker"; exit 1; }
say "  $(tail -3 zb2_train.log | tr '\n' ' ')"

say "=== 3. classifier + export ==="
bash flow/scripts/zerobox_export_tux.sh > zb2_export.out 2>&1
[ -f zb2_export.marker ] || { say "ABORT: export did not write its marker"; exit 1; }
say "  $(grep -cE 'EXPORT FAILED' zb2_export.log) export failures; $(ls port/export | grep -c '^ddm_zb2_s90[0-9]_e[0-9]*_weights.json$') cards"
say "=== v2 pipeline done ==="
echo "zb2 pipeline done $(date)" > zb2_pipeline.marker

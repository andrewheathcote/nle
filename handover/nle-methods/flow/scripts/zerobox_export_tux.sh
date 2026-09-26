#!/bin/bash
# Zero-box S3, part 1: the classifier, then every retained checkpoint exported.
#
# Run from the package root on tux26 after train_zerobox_tux.sh has written
# zerobox_done.marker:
#   nohup bash flow/scripts/zerobox_export_tux.sh > zerobox_export.out 2>&1 &
#
# Sequence (the smoke run exercised every step end to end on 2026-09-26):
#   1. retrain_classifier.py --zerobox: regress P(R = upper) on the exact
#      targets in data/choice_prob/*_zerobox.parquet, encoded as the flow is,
#      standardised with ONE seed's flow scaler (all three seeds share the
#      training data, so their scalers agree to float precision; the export
#      asserts it).
#   2. json_to_classifier_ckpt.py: the JSON -> an orbax checkpoint dir.
#   3. Install that checkpoint into each seed's run, with the seed's own
#      train_loader.pkl beside it, replacing the placeholder copied at launch.
#   4. export_flow_ddm.py --zerobox for every retained step of every seed; the
#      port gate is retried once at the relaxed tolerance, as before, and the
#      distribution it prints is what to read, not the max.
set -u
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
PY=~/emc2nn/nle/flow/.venv/bin/python
export PYTHONPATH="$ROOT/flow/src"
LOG="$ROOT/zerobox_export.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
SEEDS="zb_s901 zb_s902 zb_s903"
REF=zb_s901                                     # whose scaler the classifier uses

[ -f "$ROOT/zerobox_done.marker" ] || { say "ABORT: training marker absent"; exit 1; }
for t in $SEEDS; do
  n=$(ls "$ROOT/results/model_states/${t}_budget/DDM_flow" 2>/dev/null | grep -c '^[0-9]')
  [ "$n" -ge 30 ] || { say "ABORT: $t has $n checkpoints"; exit 1; }
done

say "=== 1. classifier (config big) on the zero-box targets, scaler from $REF ==="
if [ ! -f "$ROOT/port/export/clf_zb_big_weights.json" ]; then
  OMP_NUM_THREADS=8 "$PY" flow/scripts/retrain_classifier.py --zerobox --config big \
    --scaler-from "$REF" --data-suffix _zerobox \
    --out-json port/export/clf_zb_big_weights.json > "$ROOT/clf_zb_big.log" 2>&1 \
    || { say "ABORT: classifier training failed"; exit 1; }
fi
say "  $(grep -E '^FINAL' "$ROOT/clf_zb_big.log")"

say "=== 2-3. checkpoint + install into each seed ==="
"$PY" flow/scripts/json_to_classifier_ckpt.py --from-export clf_zb_big --tag clfzb \
  --scaler-from "$REF" >> "$LOG" 2>&1 || { say "ABORT: checkpoint conversion failed"; exit 1; }
for t in $SEEDS; do
  d="$ROOT/results/model_states/${t}_budget"
  rm -rf "$d/DDM_classifier"
  cp -R "$ROOT/results/model_states/clfzb_budget/DDM_classifier" "$d/DDM_classifier"
  cp "$d/train_loader.pkl" "$d/DDM_classifier/train_loader.pkl"
  say "  $t: zero-box classifier installed"
done

say "=== 4. export every retained checkpoint (--zerobox) ==="
for t in $SEEDS; do
  d="$ROOT/results/model_states/${t}_budget/DDM_flow"
  for e in $(ls "$d" | grep '^[0-9]' | sort -n); do
    [ -f "$ROOT/port/export/ddm_${t}_e${e}_weights.json" ] && continue
    "$PY" flow/scripts/export_flow_ddm.py --budget "$t" --zerobox --step "$e" \
      --tag-suffix "_e${e}" >> "$ROOT/export_${t}.log" 2>&1 \
     || "$PY" flow/scripts/export_flow_ddm.py --budget "$t" --zerobox --step "$e" \
          --tag-suffix "_e${e}" --port-tol 2.5e-2 >> "$ROOT/export_${t}.log" 2>&1 \
     || say "  $t e$e EXPORT FAILED"
  done
  say "  $t: $(ls "$ROOT/port/export/" | grep -cE "^ddm_${t}_e[0-9]+_weights.json$") exported"
done
say "=== export done; screening is part 2 (regime-specific, see the plan) ==="
echo "zerobox export done $(date)" >> "$ROOT/zerobox_export.marker"

#!/bin/bash
# Scaled DDM network on tux26 (handover/SCALED_DDM_PLAN.md). One stage per call; AH approves each
# launch on tux26 (pilot, then full training) -- nothing here chains into the next approval.
#
#   STAGE=sim    simulate training (600k draws), validation (25k) and choice targets
#   STAGE=pilot  one seed, PILOT_EPOCHS (30), then classifier + export of its last checkpoint
#   STAGE=train  SEEDS (6) concurrently, THREADS (4) each, 300 epochs, 30 checkpoints kept
#   STAGE=export classifier on the exact P(R) targets, installed in every seed; every checkpoint exported
#
# Run from the package root (~/emc2nn/nle/handover/nle-methods):
#   STAGE=sim setsid nohup bash flow/scripts/scaled_pipeline_tux.sh > sc_sim.out 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
PY=${PY:-$HOME/emc2nn/nle/flow/.venv/bin/python}
export PYTHONPATH="$ROOT/flow/src"
# The simulator must run on the dev-nn EMC2 the zero-box data used: on the final-sampler library
# simulate_trials() floors every stored ll (found 2026-10-13; the script asserts < 5% at the floor).
export EMC2NN_LIB=${EMC2NN_LIB:-$HOME/emc2nn/Rlib-devnn}
# progressr, future, future.apply, parallelly: absent from that library on tux26, installed separately (2026-10-13)
export R_LIBS=${R_LIBS:-$HOME/emc2nn/Rlib-simdeps}
STAGE=${STAGE:?set STAGE=sim|pilot|train|export}
SIMC=${SIMC:-24}; THREADS=${THREADS:-4}
SEEDS=${SEEDS:-"sc_s1101 sc_s1102 sc_s1103 sc_s1104 sc_s1105 sc_s1106"}
PILOT=${PILOT:-sc_pilot}; PILOT_EPOCHS=${PILOT_EPOCHS:-30}
CLF_PLACEHOLDER=${CLF_PLACEHOLDER:-large}   # never evaluated in training; replaced before export
TRAIN=data/train/scaled600k/DDM/trials; VAL=data/val/scaled/DDM/trials
LOG="$ROOT/sc_pipeline.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $STAGE: $*" | tee -a "$LOG"; }
nchunks() { ls "$ROOT/$1" 2>/dev/null | grep -c parquet; }

train_one() {   # tag seed epochs keep
  OMP_NUM_THREADS=$THREADS "$PY" flow/scripts/train_ddm.py --tag "$1" --scaled \
    --classifier-from "$CLF_PLACEHOLDER" --train-data "$TRAIN" --val-data "$VAL" \
    --batch 1024 --epochs "$3" --num-bins 64 --widths 256,128 --keep-checkpoints "$4" \
    --seed "$2" --range-min -12 --range-max 2 > "$ROOT/train_$1.log" 2>&1
  say "  $1 finished (exit $?)"
}
classifier() {  # reference tag whose scaler the classifier takes
  local ref=$1
  if [ ! -f "$ROOT/port/export/clf_sc_big_weights.json" ]; then
    OMP_NUM_THREADS=8 "$PY" flow/scripts/retrain_classifier.py --scaled --config big --scaler-from "$ref" \
      --data-suffix _scaled --out-json port/export/clf_sc_big_weights.json > "$ROOT/clf_sc_big.log" 2>&1 \
      || { say "ABORT: classifier training failed"; exit 1; }
  fi
  say "  classifier: $(grep -E '^FINAL' "$ROOT/clf_sc_big.log")"
  "$PY" flow/scripts/json_to_classifier_ckpt.py --from-export clf_sc_big --tag clfsc --scaler-from "$ref" \
    >> "$LOG" 2>&1 || { say "ABORT: checkpoint conversion failed"; exit 1; }
}
install_clf() {
  local d="$ROOT/results/model_states/$1_budget"
  rm -rf "$d/DDM_classifier"; cp -R "$ROOT/results/model_states/clfsc_budget/DDM_classifier" "$d/DDM_classifier"
  cp "$d/train_loader.pkl" "$d/DDM_classifier/train_loader.pkl"
}
export_one() {  # tag step
  [ -f "$ROOT/port/export/ddm_$1_e$2_weights.json" ] && return 0
  "$PY" flow/scripts/export_flow_ddm.py --budget "$1" --scaled --step "$2" --tag-suffix "_e$2" >> "$ROOT/export_$1.log" 2>&1 \
   || "$PY" flow/scripts/export_flow_ddm.py --budget "$1" --scaled --step "$2" --tag-suffix "_e$2" --port-tol 2.5e-2 \
        >> "$ROOT/export_$1.log" 2>&1 || say "  $1 e$2 EXPORT FAILED"
}

case $STAGE in
sim)
  say "simulate (SIM_CORES $SIMC train, 8 val, then choice targets)"
  if [ "$(nchunks $TRAIN)" -ge 240 ]; then say "  training set present"; else
    SIM_CORES=$SIMC Rscript simulation/sim_ddm_scaled.R train 600000 > sim_sc_train.log 2>&1 & a=$!
    SIM_CORES=8 Rscript simulation/sim_ddm_scaled.R val 25000 > sim_sc_val.log 2>&1 & b=$!
    wait $a || { say "ABORT: training simulation failed"; exit 1; }
    wait $b || { say "ABORT: validation simulation failed"; exit 1; }
  fi
  SIM_CORES=$SIMC Rscript simulation/sim_ddm_scaled.R choice 200000 20000 > sim_sc_choice.log 2>&1 \
    || { say "ABORT: choice targets failed"; exit 1; }
  say "  $(grep '=== done' sim_sc_train.log) | val $(grep '=== done' sim_sc_val.log)"
  [ "$(nchunks $TRAIN)" -ge 240 ] && [ "$(nchunks $VAL)" -ge 10 ] || { say "ABORT: chunks missing"; exit 1; }
  echo "sim done $(date)" > sc_sim.marker ;;
pilot)
  [ -f sc_sim.marker ] || { say "ABORT: run STAGE=sim first"; exit 1; }
  say "pilot $PILOT: $PILOT_EPOCHS epochs, $THREADS threads"
  train_one "$PILOT" 1100 "$PILOT_EPOCHS" 3
  classifier "$PILOT"; install_clf "$PILOT"
  for e in $(ls "$ROOT/results/model_states/${PILOT}_budget/DDM_flow" | grep '^[0-9]' | sort -n); do export_one "$PILOT" "$e"; done
  say "  exported: $(ls port/export | grep -E "^ddm_${PILOT}_e[0-9]+_weights.json$" | tr '\n' ' ')"
  echo "pilot done $(date)" > sc_pilot.marker ;;
train)
  [ -f sc_sim.marker ] || { say "ABORT: run STAGE=sim first"; exit 1; }
  say "train: $SEEDS x $THREADS threads"
  pids=""
  for t in $SEEDS; do train_one "$t" "${t##*_s}" 300 30 & pids="$pids $!"; done
  wait $pids
  for t in $SEEDS; do say "  $t: $(ls "$ROOT/results/model_states/${t}_budget/DDM_flow" 2>/dev/null | grep -c '^[0-9]') checkpoints"; done
  echo "train done $(date)" > sc_train.marker ;;
export)
  [ -f sc_train.marker ] || { say "ABORT: training marker absent"; exit 1; }
  set -- $SEEDS; classifier "$1"
  for t in $SEEDS; do install_clf "$t"; done
  for t in $SEEDS; do
    for e in $(ls "$ROOT/results/model_states/${t}_budget/DDM_flow" | grep '^[0-9]' | sort -n); do export_one "$t" "$e"; done &
  done; wait
  say "  $(ls port/export | grep -cE '^ddm_sc_s11[0-9]{2}_e[0-9]+_weights.json$') cards; $(grep -c 'EXPORT FAILED' "$LOG") failures"
  echo "export done $(date)" > sc_export.marker ;;
*) say "unknown STAGE"; exit 1 ;;
esac

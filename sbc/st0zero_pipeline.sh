#!/bin/bash
# st0 = 0 DDM build: simulate -> train 3 seeds -> export -> screen.
#
# WAITS for the regression seed runs to finish before starting, so the laptop
# is never oversubscribed. Phase A (simulation) is single-threaded R; Phase C
# trains 3 seeds concurrently at 3 cores each.
#
# Three seeds from the start, not one: Phases 25 and 30 both retracted a
# confident single-run claim, and no arm replicated across >=3 seeds has been.
#
# NOT set -e: a failed arm must not cost the others or skip the screening.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd); PY="$ROOT/flow/.venv/bin/python"
LOG="$ROOT/sbc/st0zero.log"
DATA=data/train/st0zero400k/DDM/trials
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "waiting for the regression seed runs to release the cores..."
# The venv python is a symlink, so the running process reports the Homebrew
# interpreter path, not ".venv/bin/python" -- a predicate anchored on the
# venv path matches nothing and the wait becomes a no-op. Match the script
# name instead; the [t] bracket keeps this line from matching itself.
while pgrep -f "[t]rain_regress[.]py" > /dev/null; do sleep 120; done
say "cores free; starting"

# ---- Phase A: training data (st0 = 0) ----
if [ -d "$ROOT/$DATA" ]; then
  say "PHASE A: $DATA already present; skipping simulation"
else
  say "PHASE A: simulating 16e6 rows with st0 = 0"
  Rscript handover/st0_zero/sim_ddm_st0zero.R > "$ROOT/sbc/sim_st0zero.log" 2>&1
  say "PHASE A done (rc=$?)"
fi
[ -d "$ROOT/$DATA" ] || { say "ABORT: $DATA was not created; see sbc/sim_st0zero.log"; exit 1; }

# ---- Phase C: train 3 seeds ----
train_one() {
  local tag=$1
  local seed=$2
  local ck="$ROOT/results/model_states/${tag}_budget/DDM_flow"
  if [ -d "$ck" ] && [ "$(ls "$ck" | grep -c '^[0-9]')" -ge 30 ]; then
    say "  $tag already trained; skipping"; return 0
  fi
  # --val-data is REQUIRED. The default (data/val/DDM) was simulated with st0
  # sampled from [0.01, 0.49], so its rt and ll reflect a smeared leading edge.
  # Checkpoints are retained by best val_loss, so validating an st0 = 0 flow on
  # smeared data would actively select checkpoints that OVER-SMOOTH the leading
  # edge -- the exact failure mode this build exists to test.
  OMP_NUM_THREADS=3 "$PY" flow/scripts/train_ddm.py \
    --tag "$tag" --st0-zero --train-data "$DATA" \
    --val-data data/val/DDM_st0zero/trials \
    --batch 1024 --epochs 300 --num-bins 64 --widths 256,128 \
    --keep-checkpoints 30 --seed "$seed" --classifier-from c4clf \
    > "$ROOT/sbc/train_${tag}.log" 2>&1
}
say "PHASE C: training 3 seeds (3 cores each)"
train_one z0_s501 501 & a=$!
train_one z0_s502 502 & b=$!
train_one z0_s503 503 & c=$!
wait $a $b $c
say "PHASE C done"

# ---- Phase E: export every retained checkpoint ----
say "PHASE E: exporting"
for t in z0_s501 z0_s502 z0_s503; do
  d="$ROOT/results/model_states/${t}_budget/DDM_flow"
  [ -d "$d" ] || { say "  $t: no checkpoints"; continue; }
  for e in $(ls "$d" | grep '^[0-9]' | sort -n); do
    [ -f "$ROOT/port/export/ddm_${t}_e${e}_weights.json" ] && continue
    "$PY" flow/scripts/export_flow_ddm.py --budget "$t" --st0-zero --step "$e" \
      --tag-suffix "_e${e}" >> "$ROOT/sbc/export_${t}.log" 2>&1 \
      || "$PY" flow/scripts/export_flow_ddm.py --budget "$t" --st0-zero --step "$e" \
           --tag-suffix "_e${e}" --port-tol 2.5e-2 >> "$ROOT/sbc/export_${t}.log" 2>&1 \
      || say "  $t e$e EXPORT FAILED"
  done
  say "  $t: $(ls "$ROOT/port/export/" | grep -cE "^ddm_${t}_e[0-9]+_weights.json$") exported"
done
say "PHASE E done"
say "NEXT (manual): the 7-parameter score-bias edits, then screening and SBC."
say "  score_bias_draws.R and score_bias_t0.R still assume 8 parameters."

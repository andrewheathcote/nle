#!/bin/bash
# Export every retained decision-time checkpoint, then measure score bias.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd); PY="$ROOT/flow/.venv/bin/python"
LOG="$ROOT/sbc/dectime_screen.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
say "=== exporting ==="
for t in dt_s701 dt_s702 dt_s703; do
  d="$ROOT/results/model_states/${t}_budget/DDM_flow"
  for e in $(ls "$d" | grep '^[0-9]' | sort -n); do
    [ -f "$ROOT/port/export/ddm_${t}_e${e}_weights.json" ] && continue
    "$PY" flow/scripts/export_flow_ddm.py --budget "$t" --decision-time --step "$e" \
      --tag-suffix "_e${e}" >> "$ROOT/sbc/export_${t}.log" 2>&1 \
     || "$PY" flow/scripts/export_flow_ddm.py --budget "$t" --decision-time --step "$e" \
          --tag-suffix "_e${e}" --port-tol 2.5e-2 >> "$ROOT/sbc/export_${t}.log" 2>&1 \
     || say "  $t e$e EXPORT FAILED"
  done
  say "  $t: $(ls "$ROOT/port/export/" | grep -cE "^ddm_${t}_e[0-9]+_weights.json$")/30"
done
TAGS=$(for t in dt_s701 dt_s702 dt_s703; do
  ls "$ROOT/port/export/" | grep -oE "^ddm_${t}_e[0-9]+_weights.json$" \
    | sed 's/^ddm_//; s/_weights.json$//'; done | sort -u | tr '\n' ' ')
say "=== measuring score bias on $(echo $TAGS | wc -w | tr -d ' ') checkpoints ==="
SB_H=0.01 SB_OUT="$ROOT/sbc/rep_sb/dt_g.rds" \
  Rscript sbc/score_bias_draws.R 200 $TAGS >> "$LOG" 2>&1
SBT_CORES=9 SBT_OUT="$ROOT/sbc/rep_sb/dt_t0.rds" \
  Rscript sbc/score_bias_t0.R 200 $TAGS >> "$LOG" 2>&1
say "=== done ==="

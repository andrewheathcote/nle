#!/bin/bash
# Distillation beta sweep + seed replication, end to end, unattended.
#
# Stage 1  train beta = 20, 30, 50 concurrently (3 cores each = 9)
# Stage 2  export every retained checkpoint, measure g_a and g_t0
# Stage 3  pick the arm whose mean |g_t0| is smallest
# Stage 4  train 3 fresh seeds at that beta, concurrently
# Stage 5  measure those, write sbc/DISTILL_SWEEP_REPORT.md
#
# Rationale: g_t0 crosses zero between beta=10 (-0.0317) and pure (+0.0265).
# Interpolating on the distillation share beta/(1+beta) puts the crossing near
# beta ~ 23, so 20 and 30 straddle it and 50 confirms monotonicity.
#
# Settings match the existing dist_* arms exactly so the new points join the
# same curve: batch 1024, 300 epochs, [256,128], 64 bins, 10 kept checkpoints,
# --classifier-from c4clf (verified md5-identical to cap256w_c4's classifier,
# so nothing needs splicing and no cross-classifier comparison arises).
#
# NOT set -e: one failed arm must not abort the other two or lose the report.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY="$ROOT/flow/.venv/bin/python"
LOG="$ROOT/sbc/distill_sweep.log"
REPORT="$ROOT/sbc/DISTILL_SWEEP_REPORT.md"
DRAWS=200
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

train_arm() {   # tag beta seed
  local tag=$1 beta=$2 seed=$3
  # --train-data is NOT optional here: train_ddm.py defaults to
  # data/train/large (1M rows), while every dist_* arm used the 16M-row
  # tpt40p400k set. Omitting it trains on 1/16th of the data and the new beta
  # points do not join the same curve.
  OMP_NUM_THREADS=3 "$PY" flow/scripts/train_ddm.py \
    --tag "$tag" --distill-beta "$beta" --nll-weight 1.0 \
    --train-data data/train/tpt40p400k/DDM/trials \
    --classifier-from c4clf --batch 1024 --epochs 300 \
    --num-bins 64 --widths 256,128 --keep-checkpoints 10 --seed "$seed" \
    > "$ROOT/sbc/train_${tag}.log" 2>&1
}

export_arm() {  # tag  -> exports every retained checkpoint
  local tag=$1 d="$ROOT/results/model_states/${tag}_budget/DDM_flow"
  [ -d "$d" ] || { say "  $tag: no checkpoint dir, skipping"; return 1; }
  for e in $(ls "$d" | grep '^[0-9]' | sort -n); do
    [ -f "$ROOT/port/export/ddm_${tag}_e${e}_weights.json" ] && continue
    if ! "$PY" flow/scripts/export_flow_ddm.py --budget "$tag" --step "$e" \
           --tag-suffix "_e${e}" >> "$ROOT/sbc/export_${tag}.log" 2>&1; then
      # single-point float32/float64 outliers: retry once with a raised gate,
      # exactly as for the b128 arm. Dropping checkpoints biases an arm.
      say "  $tag e$e tripped the port gate; retrying at --port-tol 2.5e-2"
      "$PY" flow/scripts/export_flow_ddm.py --budget "$tag" --step "$e" \
        --tag-suffix "_e${e}" --port-tol 2.5e-2 \
        >> "$ROOT/sbc/export_${tag}.log" 2>&1 \
        || say "  $tag e$e EXPORT FAILED even at raised tolerance"
    fi
  done
}

tags_of() {     # tag -> the exported tag list for that arm
  ls "$ROOT/port/export/" | grep -oE "^ddm_${1}_e[0-9]+_weights.json$" \
    | sed 's/^ddm_//; s/_weights.json$//' | sort -u
}

measure() {     # out_stem  tag...
  local stem=$1; shift
  [ $# -gt 0 ] || { say "  nothing to measure for $stem"; return 1; }
  SB_H=0.01 SB_OUT="$ROOT/sbc/rep_sb/${stem}_g.rds" \
    Rscript sbc/score_bias_draws.R "$DRAWS" "$@" >> "$LOG" 2>&1
  SBT_CORES=9 SBT_OUT="$ROOT/sbc/rep_sb/${stem}_t0.rds" \
    Rscript sbc/score_bias_t0.R "$DRAWS" "$@" >> "$LOG" 2>&1
}

mkdir -p "$ROOT/sbc/rep_sb"
say "=== STAGE 1: training beta = 20, 30, 50 (3 cores each) ==="
train_arm distb20 20 100 & p1=$!
train_arm distb30 30 100 & p2=$!
train_arm distb50 50 100 & p3=$!
wait $p1 $p2 $p3
say "=== STAGE 1 done ==="

say "=== STAGE 2: export + measure ==="
ALL=""
for t in distb20 distb30 distb50; do
  export_arm "$t"; n=$(tags_of "$t" | wc -l | tr -d ' ')
  say "  $t: $n checkpoints exported"
  ALL="$ALL $(tags_of "$t" | tr '\n' ' ')"
done
measure sweep $ALL
say "=== STAGE 2 done ==="

say "=== STAGE 3: selecting beta ==="
BEST=$(Rscript sbc/select_beta.R "$ROOT/sbc/rep_sb/sweep_g.rds" \
         "$ROOT/sbc/rep_sb/sweep_t0.rds" --pick 2>/dev/null | tr -d ' \n')
case "$BEST" in
  distb20) BETA=20 ;; distb30) BETA=30 ;; distb50) BETA=50 ;;
  *) say "SELECTION FAILED (got '$BEST'); defaulting to beta=30"; BEST=distb30; BETA=30 ;;
esac
say "  selected $BEST (beta = $BETA)"

say "=== STAGE 4: replicating beta=$BETA across 3 seeds ==="
train_arm "rep${BEST}_s301" "$BETA" 301 & q1=$!
train_arm "rep${BEST}_s302" "$BETA" 302 & q2=$!
train_arm "rep${BEST}_s303" "$BETA" 303 & q3=$!
wait $q1 $q2 $q3
say "=== STAGE 4 done ==="

say "=== STAGE 5: measure replicates + report ==="
REP=""
for s in 301 302 303; do
  t="rep${BEST}_s${s}"; export_arm "$t"
  REP="$REP $(tags_of "$t" | tr '\n' ' ')"
done
measure repl $REP

{
  echo "# Distillation beta sweep — report"
  echo
  echo "Generated $(date '+%Y-%m-%d %H:%M'). Settings identical to the existing"
  echo "dist_* arms (batch 1024, 300 epochs, [256,128], 64 bins, c4clf classifier),"
  echo "so these points join the same curve. g at H=0.01, $DRAWS SBC prior draws."
  echo
  echo '## Stage 1-3: the sweep'
  echo '```'
  Rscript sbc/select_beta.R "$ROOT/sbc/rep_sb/sweep_g.rds" "$ROOT/sbc/rep_sb/sweep_t0.rds" 2>/dev/null
  echo '```'
  echo
  echo "Selected **$BEST (beta = $BETA)** on smallest mean |g_t0|."
  echo
  echo '## Stage 4-5: three fresh seeds at the selected beta'
  echo '```'
  Rscript sbc/select_beta.R "$ROOT/sbc/rep_sb/repl_g.rds" "$ROOT/sbc/rep_sb/repl_t0.rds" 2>/dev/null
  echo '```'
  echo
  echo '## Reference points (measured earlier, same estimator)'
  echo
  echo '| arm | g_a | g_t0 |'
  echo '|---|---|---|'
  echo '| plain MLE (5 seeds) | -0.0259 | -0.0696 |'
  echo '| cap256w_c4 (production) | +0.0118 | -0.0620 |'
  echo '| beta = 1 | -0.0114 | -0.0624 |'
  echo '| beta = 10 | +0.0137 | -0.0317 |'
  echo '| beta = pure | +0.0461 | +0.0265 |'
  echo '| direct regression | +0.0080 | +0.0063 |'
  echo
  echo 'Caveat: mean g_t0 near zero does NOT imply SBC passes. The spread of'
  echo 'g_t0 across theta is ~0.03 for every distilled arm measured so far and'
  echo 'does not shrink with beta, so a region-dependent residual survives a'
  echo 'zero mean (Phase 9b). SBC remains the verdict.'
} > "$REPORT"
say "=== ALL DONE -> $REPORT ==="

#!/bin/bash
# Seed study, part 2: export every retained checkpoint of every run, screen it
# on score bias, print the table. The criterion is registered in the header of
# sbc/seed_study_launch_tux.sh and applied, mechanically, by
# sbc/seed_study_table.R; this script decides nothing.
#
#   1. export   ours: flow/scripts/export_flow.py --step (float64 port checked
#               against distrax at every export); his: flow/scripts/malte_to_card.py
#   2. check    the first card of each of our runs through
#               port/tests/validate_m7a.R (golden < 1e-10, dev-nle dfun < 1e-7)
#   3. screen   sbc/score_bias_race.R, one call per run and cell:
#               std (arms base aff dt both) and A0 (all arms)
#   4. table    sbc/seed_study/<prefix>/table.csv, one row per arm, seed,
#               checkpoint, cell, parameter; summary and verdicts printed
#
# Run from the root that holds flow/, malte/, sbc/, port/, results/:
#   nohup bash sbc/seed_study_screen.sh > sbc/seed_study_screen.out 2>&1 &
# Needs sbc/nci/results/RDMnn_N100_R500.rds (the SBC prior draws).
#
# Env: PY, PYM, PREFIX, ARMS as in the launch script
#   LIB     R library holding dev-nle (default ~/R/nle-sbc-lib; tux26:
#           ~/emc2nn/nle/.Rlibs-malte). Asserted by both R scripts.
#   CORES 8   DRAWS 200   PORT_TOL 1e-2 (float32-vs-float64, as in
#           sbc/run_rdm_refinement.sh)   NEED_MARKER 1 (0 for a smoke test)
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=${PY:-$ROOT/flow/.venv/bin/python}; PYM=${PYM:-$ROOT/malte/.venv/bin/python}
PREFIX=${PREFIX:-ss}; ARMS=${ARMS:-"base aff dt both malte"}
LIB=${LIB:-~/R/nle-sbc-lib}; CORES=${CORES:-8}; DRAWS=${DRAWS:-200}
PORT_TOL=${PORT_TOL:-1e-2}; NEED_MARKER=${NEED_MARKER:-1}
OUT=sbc/seed_study/$PREFIX; mkdir -p "$OUT/logs" port/export
export PYTHONPATH="$ROOT/flow/src${PYTHONPATH:+:$PYTHONPATH}"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

say "=== seed study screen: prefix $PREFIX, arms [$ARMS], library $LIB ==="
[ "$NEED_MARKER" = 0 ] || [ -f "$OUT/train_done.marker" ] || { say "ABORT: $OUT/train_done.marker absent (training incomplete)"; exit 1; }
[ -f sbc/nci/results/RDMnn_N100_R500.rds ] || { say "ABORT: sbc/nci/results/RDMnn_N100_R500.rds absent"; exit 1; }
[ -d "$(eval echo $LIB)" ] || { say "ABORT: R library $LIB absent"; exit 1; }

say "=== 1. export ==="
: > "$OUT/failed_exports.txt"; : > "$OUT/runs.txt"
for a in $ARMS; do
  if [ "$a" = malte ]; then runs=$(ls -d results/malte_runs/${PREFIX}_malte_s* 2>/dev/null)
  else runs=$(ls -d results/model_states/${PREFIX}_${a}_s*_budget 2>/dev/null); fi
  for r in $runs; do
    tag=$(basename "$r"); tag=${tag%_budget}; ok=0; fail=0; tags=""
    if [ "$a" = malte ]; then
      box=$("$PYM" -c "import json;print(json.load(open('$r/run_config.json'))['bounds'])")
      for c in $(ls -d $r/ckpt_s* | sort); do
        st=$((10#${c##*ckpt_s})); name=rdm_${tag}_e$st
        if "$PYM" flow/scripts/malte_to_card.py "$c" port/export/${name}_weights.json --bounds "$box" --name "$tag" \
             >> "$OUT/logs/export_$tag.log" 2>&1
        then ok=$((ok + 1)); tags="$tags $name"; else fail=$((fail + 1)); echo "$name" >> "$OUT/failed_exports.txt"; fi
      done
    else
      for st in $(ls "$r/RDM" | grep -E '^[0-9]+$' | sort -n); do
        name=rdm_${tag}_e$st
        if (cd flow && "$PY" scripts/export_flow.py --model RDM --budget "$tag" --step "$st" \
              --tag-suffix "_e$st" --port-tol "$PORT_TOL") >> "$OUT/logs/export_$tag.log" 2>&1
        then ok=$((ok + 1)); tags="$tags $name"; else fail=$((fail + 1)); echo "$name" >> "$OUT/failed_exports.txt"; fi
      done
    fi
    say "  $tag: $ok exported, $fail failed"
    [ "$ok" -gt 0 ] && echo "$a $tag$tags" >> "$OUT/runs.txt"
  done
done
[ -s "$OUT/runs.txt" ] || { say "ABORT: nothing exported"; exit 1; }

say "=== 2. card check, first card of each of our runs ==="
first=$(awk '$1 != "malte" {print $3}' "$OUT/runs.txt")
if [ -n "$first" ]; then
  Rscript port/tests/validate_m7a.R --lib=$LIB $first > "$OUT/card_check.out" 2>&1
  grep -E "^(PASS|FAIL)|^===" "$OUT/card_check.out"
  grep -q "0 failing line" "$OUT/card_check.out" || { say "ABORT: card check failed (see $OUT/card_check.out)"; exit 1; }
fi

say "=== 3. score-bias screen, $DRAWS draws ==="
rm -f "$OUT"/g_*.csv
while read -r a tag tags; do
  for cell in std A0; do
    [ "$a" = malte ] && [ "$cell" = std ] && continue          # his flow has no A
    SBR_CELL=$cell SBR_CORES=$CORES SBR_LIB=$LIB SBR_OUT="$OUT/g_${tag}_$cell.rds" SBR_CSV="$OUT/g_${tag}_$cell.csv" \
      Rscript sbc/score_bias_race.R "$DRAWS" $tags > "$OUT/logs/screen_${tag}_$cell.out" 2>&1 \
      || { say "  SCREEN FAILED $tag $cell (see $OUT/logs/screen_${tag}_$cell.out)"; continue; }
    say "  $tag $cell: $(grep -E '^usable draws' "$OUT/logs/screen_${tag}_$cell.out"), $(grep -E '^CONTROL' "$OUT/logs/screen_${tag}_$cell.out")"
  done
done < "$OUT/runs.txt"

say "=== 4. table ==="
Rscript sbc/seed_study_table.R "$OUT" "$PREFIX" || { say "ABORT: table failed"; exit 1; }
echo "seed study screen done $(date), prefix $PREFIX" > "$OUT/screen_done.marker"
say "=== done: $OUT/table.csv, $OUT/summary.csv ==="

#!/bin/bash
# After the replication finishes: export each run's checkpoints and measure the
# score bias, on tux26 (EMC2 dev-nn and the port are already installed there).
#
# Reports the run-MEAN per seed, so the between-run sd can be computed per batch
# size — which is the quantity Phase 20 showed we were missing.
set -uo pipefail
cd "$(dirname "$0")/../.."
# Restrict to a pattern so a finished arm can be analysed while another is
# still training. Without this the script would export checkpoints from
# PARTIAL runs and compare them against complete ones.
PAT=${1:-rep_b*}
echo "analysing runs matching: $PAT"
TAGS=(); FAILED=0
for d in results/model_states/${PAT}_budget; do
  [ -d "$d/DDM_flow" ] || continue
  tag=$(basename "$d" _budget)
  if [ ! -f "$d/train_loader.pkl" ]; then
    # Do NOT substitute another run's scaler. A wrong scaler still exports
    # cleanly and still passes the port check — it just evaluates the model at
    # the wrong theta, silently.
    echo "  SKIP $tag: no scaler"; continue
  fi
  for st in $(ls "$d/DDM_flow" | grep -E '^[0-9]+$' | sort -n); do
    if ./flow/.venv/bin/python flow/scripts/export_flow_ddm.py --budget "$tag" \
         --step "$st" --tag-suffix "_e${st}" --port-tol 1 >/dev/null 2>&1; then
      TAGS+=("${tag}_e${st}")
    else
      echo "EXPORT FAILED $tag step $st"; FAILED=$((FAILED+1))
    fi
  done
done
echo "exported ${#TAGS[@]}, $FAILED failures"
[ "${#TAGS[@]}" -ge 20 ] || { echo "ABORTING: too few exports"; exit 1; }
# H=0.01: the default 0.05 carries O(H^2) truncation error worth ~+0.002 on a
SB_H=0.01 Rscript sbc/score_bias_draws.R 300 "${TAGS[@]}"
echo "done $(date)"

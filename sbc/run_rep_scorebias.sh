#!/bin/bash
# Score bias for the batch-replication arms, H = 0.01, c4 classifier spliced.
#
# score_bias_draws.R is serial over tags, so tags are split across processes.
# The estimator is deterministic per tag (same prior draws, same grid), so
# chunking is exactly equivalent to one invocation -- only faster.
#
#   bash sbc/run_rep_scorebias.sh <arm> <n_draws> <n_chunks>
# e.g. bash sbc/run_rep_scorebias.sh rep_b1024 200 5
set -uo pipefail
cd "$(dirname "$0")/.."
ARM=${1:?arm prefix, e.g. rep_b1024}
NDRAW=${2:-200}
NCHUNK=${3:-5}

# macOS ships bash 3.2, which has no mapfile
TAGS=()
while IFS= read -r line; do TAGS+=("$line"); done < <(
  ls port/export/ | grep -oE "${ARM}_s[0-9]+_e[0-9]+_c4" | sort -u)
N=${#TAGS[@]}
echo "$ARM: $N tags, $NDRAW draws, $NCHUNK chunks"
[ "$N" -gt 0 ] || { echo "ABORT: no spliced (_c4) exports for $ARM"; exit 1; }

per=$(( (N + NCHUNK - 1) / NCHUNK ))
mkdir -p sbc/rep_sb
for ((k=0; k<NCHUNK; k++)); do
  chunk=( "${TAGS[@]:$((k*per)):$per}" )
  [ ${#chunk[@]} -eq 0 ] && continue
  SB_H=0.01 SB_OUT="sbc/rep_sb/${ARM}_chunk${k}.rds" \
    Rscript sbc/score_bias_draws.R "$NDRAW" "${chunk[@]}" \
    > "sbc/rep_sb/${ARM}_chunk${k}.log" 2>&1 &
done
wait
echo "$ARM done: $(ls sbc/rep_sb/${ARM}_chunk*.rds 2>/dev/null | wc -l) chunk files"

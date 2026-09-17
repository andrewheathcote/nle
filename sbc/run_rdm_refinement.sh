#!/bin/bash
# RDM refinement: simulate at scale, train with the DDM's learned settings,
# screen checkpoints on score bias. Waits for any local training to finish.
#
# Everything here is smoke-tested end to end. The DDM search is NOT repeated:
# no ensembling, no weight averaging, no score penalty, no reparameterisation,
# no edge weighting — all are recorded failures in sbc/SBC_PLAN.md.
#
#   bash sbc/run_rdm_refinement.sh            # full run
#   RDM_N=2e6 bash sbc/run_rdm_refinement.sh  # smaller, for a dry run
set -uo pipefail
cd /Users/andrew/Downloads/julian/nle
N=${RDM_N:-16e6}
TAG=${RDM_TAG:-rdm_b1k}

# Wait for TRAINING and for any queued ANALYSIS. The batch-sweep analysis runs
# automatically after its training ends (export + score bias), so waiting only
# on training would start the RDM simulation on top of it and slow both.
echo "=== waiting for current local work to finish ==="
# Anchor each pattern to its interpreter. A bare `pgrep -f queue_batchsweep`
# also matches any shell whose command line merely MENTIONS that name — the
# watcher, an editor, another tool call — and the script would then wait on a
# phantom forever.
busy() {
  pgrep -f "[Pp]ython.*(train_ddm|train_race|export_flow)\.py" >/dev/null && return 0
  pgrep -f "Rscript.*(score_bias_draws|score_bias_t0|score_bias_race)\.R" >/dev/null && return 0
  return 1
}
while busy; do sleep 180; done
echo "=== started $(date) ==="

if [ ! -d "data/train/rdm16m/RDM/trials" ]; then
  echo "--- simulating $N rows ---"
  RDM_N="$N" Rscript simulation/sim_rdm_16m.R || { echo "SIMULATION FAILED"; exit 1; }
else
  echo "--- training data already present, skipping simulation ---"
fi

echo "--- training $TAG ---"
(cd flow && ./.venv/bin/python scripts/train_race.py --model RDM --tag "$TAG" \
   --train-data data/train/rdm16m/RDM/trials) || { echo "TRAINING FAILED"; exit 1; }

echo "--- exporting checkpoints ---"
TAGS=(); FAILED=0
steps=$(ls "results/model_states/${TAG}_budget/RDM" | grep -E '^[0-9]+$' | sort -n)
for st in $steps; do
  if (cd flow && ./.venv/bin/python scripts/export_flow.py --model RDM \
        --budget "$TAG" --step "$st" --tag-suffix "_e${st}" --port-tol 1e-2 >/dev/null 2>&1); then
    TAGS+=("rdm_${TAG}_e${st}")
  else
    echo "   EXPORT FAILED step $st"; FAILED=$((FAILED+1))
  fi
done
echo "exported ${#TAGS[@]}, $FAILED failures"
# abort rather than analyse an empty set — that mistake cost 8 h on the DDM
if [ "${#TAGS[@]}" -lt 5 ] || [ "$FAILED" -gt 0 ]; then
  echo "ABORTING: exports incomplete"; exit 1
fi

echo "--- score bias across checkpoints (plus rdm_large for reference) ---"
SBR_CORES=8 Rscript sbc/score_bias_race.R 200 "${TAGS[@]}" rdm_large
echo "=== done $(date) ==="

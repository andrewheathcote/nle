#!/bin/bash
# Does batch size shift the MEAN of the score bias, or only its spread?
#
# Phase 20 found the four-arm sweep inconclusive on the mean: each arm was ONE
# run, and the spread of the arm means (0.053) was 0.79 of the run-to-run sd
# measured over five seeds (0.067). Only the WITHIN-run spread effect survived
# (variance ratio 10.3x, F(9,9) crit 5.35). Settling the mean needs replication.
#
# Design: SEEDS seeds at each of batch 128 and 1024, 10 plateau checkpoints per
# run. Comparing run-MEANS across seeds within a batch isolates the batch effect
# from run-to-run noise, which a single run per arm cannot.
#
# Each run uses ~4 cores whatever the core count (batch 128 exposes too little
# parallelism per step — see sbc/bench_batch.log), so 10 runs fit in 48 cores.
#
#   bash run_seed_batch_replication.sh          # 5 seeds x 2 batches
#   SEEDS=3 bash run_seed_batch_replication.sh
set -uo pipefail
cd "$(dirname "$0")"
NSEED=${SEEDS:-5}
EPOCHS=${EPOCHS:-300}
# tux26 measured 137.5 s/epoch at batch 1024 (vs 54 s on the laptop), so batch
# 128 would be ~34 h per run here — impractical. Default to the batch-1024 arm
# only; the batch-128 between-run sd is already known from five laptop seeds
# (0.059, deconvolved in Phase 14). Between-run sd comes from SGD noise, which
# is a property of the algorithm and data rather than the hardware, so the
# cross-machine comparison is sound.
BATCHES=${BATCHES:-1024}
cd ../..                                  # repo root of the transferred tree

nb=$(echo $BATCHES | wc -w)
echo "launching $((NSEED*nb)) runs: seeds 1..$NSEED x batch {$BATCHES}, $EPOCHS epochs"
echo "at 137.5 s/epoch measured on tux26, batch 1024 x 300 epochs is ~11.5 h per run"
for s in $(seq 1 "$NSEED"); do
  for b in $BATCHES; do
    tag="rep_b${b}_s${s}"
    nohup ./flow/.venv/bin/python flow/scripts/train_ddm.py --tag "$tag" \
      --batch "$b" --seed "$((100 + s))" --keep-checkpoints 10 \
      --num-bins 64 --widths 256,128 --epochs "$EPOCHS" \
      --train-data data/train/tpt40p400k/DDM/trials \
      > "sbc/train_${tag}.log" 2>&1 &
    echo "  $tag (seed $((100+s)), batch $b) -> pid $!"
    sleep 2
  done
done
echo
echo "track:  grep -h 'Epoch' sbc/train_rep_*.log | tail"
echo "        ps -o %cpu= -p \$(pgrep -f train_ddm.py | tr '\\n' ',' | sed 's/,\$//')"

#!/bin/bash
# Driver for the scaled-DDM SBC batch on one EC2 machine (handover/SCALED_DDM_PLAN.md; AWS rules in
# ~/.claude/plans/aws/nle_sbc_handover.md). Runs every (card, regime) cell concurrently, syncs results and logs
# to S3 every 5 minutes, and TERMINATES THE MACHINE when all cells end (or setup fails).
#   nohup bash run_all.sh > run_all.log 2>&1 &        env: CPC (data sets in parallel per cell), CFC (3), REPS (500)
cd "$(dirname "$0")"
S3=s3://emc2-results-434352151599/nle/scaled_sbc/${RUN:-run1}
CPC=${CPC:-2}; CFC=${CFC:-3}; REPS=${REPS:-500}; TRIALS=${TRIALS:-400}
CARDS=${CARDS:-"sc_s1106_e245 sc_s1101_e276 sc_s1103_e272"}
REGIMES=${REGIMES:-"full sv0 SZ0 st00 st0only SZonly"}
mkdir -p logs results
say() { echo "[$(date '+%F %T')] $*" | tee -a logs/driver.log; }
sync_up() { aws s3 sync --quiet results "$S3/results"; aws s3 sync --quiet logs "$S3/logs"; }
finish() { say "finishing: $1"; sync_up; sync_up; [ "${NO_SHUTDOWN:-0}" = 1 ] || sudo shutdown -h now; exit 0; }
LIB=$(sudo bash setup.sh 2> logs/setup.err | tail -1) || finish "setup failed"
[ -f "$LIB/EMC2_SHA.txt" ] || finish "setup gave no library ($LIB)"
say "EMC2 $(cat $LIB/EMC2_SHA.txt) at $LIB; $(nproc) cores; cells: $CARDS x $REGIMES; ${CPC}x${CFC} cores each; $REPS reps"
( while true; do sleep 300; sync_up; done ) & SYNC=$!
# Cost cap (on-demand c7a.48xlarge $10.55/h): stop everything after MAX_SECONDS. Replicates already on disk
# are kept and synced; nn_sbc_cell resumes them if the batch is relaunched.
( sleep ${MAX_SECONDS:-16200}; say "MAX_SECONDS (default 4.5 h) reached: stopping cells"; pkill -x R; pkill -x Rscript ) & CAP=$!
pids=""
for c in $CARDS; do for r in $REGIMES; do
  LIB=$LIB CPC=$CPC CFC=$CFC OUT=results R_PROFILE_USER=/dev/null \
    Rscript sbc/scaled_cell.R $r port/export/ddm_${c}_weights.json $REPS $TRIALS > logs/${c}_${r}.log 2>&1 &
  pids="$pids $!"; say "started $c $r (pid $!)"
done; done
fail=0; for p in $pids; do wait $p || fail=$((fail + 1)); done
kill $SYNC $CAP 2>/dev/null
say "all cells ended; $fail failed"
for f in logs/sc_*.log; do grep -q "=== cell .* done" $f || say "  NOT DONE: $f"; done
finish "batch complete"

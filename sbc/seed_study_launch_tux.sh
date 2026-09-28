#!/bin/bash
# Seed study (MALTE_CRDM_PLAN.md section 3.2 step B8; MALTE_STAGES.md stage M7):
# five arms x four seeds of the RDM flow, 30 retained checkpoints per run.
#
#   arm    what                                         trained by
#   base   our pipeline as is                           flow/scripts/train_race.py
#   aff    + affine stage on log time                   ... --affine
#   dt     + decision time (t0 an exact shift)          ... --decision-time
#   both   + both                                       ... --affine --decision-time
#   malte  Malte Lueken's configuration as published    malte/train_conditioner_ckpt.py
#          (slurm/train_conditioner_rdm_affine_log_deep_clip_box.sh: affine, log
#          inputs, two hidden layers of 128, 12 bins, clipping, box v 0:8,
#          s and b 0.25:3.5, 100k steps of 64 x 500 simulated online)
# Arms 1-4 share data (rdm16m), architecture ([256,128], 64 bins), batch 1024,
# 300 epochs and context order (v, B, [t0,] s, A on the log scale, scaler), so
# each differs from the baseline in one thing. Arm 5 differs in everything.
#
# ---------------------------------------------------------------------------
# CRITERION, REGISTERED HERE BEFORE ANY NUMBER EXISTS (2026-09-28, stage M7a)
# ---------------------------------------------------------------------------
# Unit: one retained checkpoint. Its score bias g_p is measured by
# sbc/score_bias_race.R for every parameter p of the cell, over 200 draws of the
# SBC prior, H = 0.01; t0 by the restricted-region estimator, relative to the
# analytic value. The analytic control of each screen must be below 1e-3.
# Retained checkpoints: ours, the 30 of lowest validation loss among 300 epochs
# (the checkpoint manager's rule, as in every run before); his, every 1000th
# step of the last 30 000 (he has no validation set and no epochs).
# A checkpoint whose export fails counts with |g| = Inf.
#
# Cells. "std": the standard RDM cell (v, v_lMTRUE, B, A, t0 sampled).
#        "A0": the comparison cell, the standard cell minus A (A = 0 exactly in
#        the truth; our flows are evaluated at A = 1e-4, the edge of their box).
# Arms aff, dt, both are judged against base in the std cell. Arm malte has no
# A, so it is judged against base in the A0 cell.
#
# Primary statistic, per checkpoint c:  G_c = mean over p of |g_pc| / m_p,
#   m_p = the baseline arm's median |g_p| in that cell (so base has median G ~ 1
#   and every parameter weighs the same).
# AN ARM HELPS if the median of G over its 120 checkpoints is below the lower
# quartile (25th percentile) of G over the baseline's 120 checkpoints.
# Secondary, reported per parameter with the same rule on |g_p|: median of the
# arm below the baseline's lower quartile. Secondary lines describe where a
# difference sits; they do not change the primary verdict.
# The distribution is the result (medians, quartiles, range per arm), not a
# best checkpoint.
#
# Known before the numbers, and to be read with them:
#  * A0 handicaps our flows. rdm_b1k_e242, |g| <= 0.004 in the std cell, has
#    |g| 0.09-0.33 in the A0 cell, because A is sampled uniformly on [1e-4, 2]
#    and conditioned on its log: 0.5% of the training draws have A < 0.01, and
#    the density error there is 0.066 against 0.014 at A = 0.3 (measured in
#    M7a). The malte-vs-base verdict therefore measures "his flow at A = 0
#    against ours at the edge of its box", not the tweaks.
#  * His four seeds differ in initialisation AND in every simulated batch; ours
#    share one stored data set and differ in initialisation and batch order.
# ---------------------------------------------------------------------------
#
# Run from the root that holds flow/, malte/, sbc/, data/ (tux26: ~/emc2nn/nle-malte):
#   setsid nohup bash sbc/seed_study_launch_tux.sh > sbc/seed_study_launch.out 2>&1 &
# Then sbc/seed_study_screen.sh. Nothing here needs R.
#
# Env (defaults are the study; the smoke test overrides them):
#   PY      python with our flow environment  (default flow/.venv/bin/python)
#   PYM     python with Malte's environment   (default malte/.venv/bin/python)
#   JOBS    runs in parallel (8)      THREADS  cores per run (9)     -> 72 cores
#   CORE0   first core used when taskset is available (0)
#   PREFIX  tag prefix (ss)           ARMS     "base aff dt both malte"
#   SEEDS   ours "801 802 803 804"    MSEEDS   his "2026 2027 2028 2029"
#   EPOCHS 300  KEEP 30  TRAIN data/train/rdm16m/RDM/trials  VAL (script default)
#   MSTEPS 100000  MEVERY 1000
#   DRY=1   print the commands and the preflight, start nothing
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=${PY:-$ROOT/flow/.venv/bin/python}; PYM=${PYM:-$ROOT/malte/.venv/bin/python}
JOBS=${JOBS:-8}; THREADS=${THREADS:-9}; CORE0=${CORE0:-0}
PREFIX=${PREFIX:-ss}; ARMS=${ARMS:-"base aff dt both malte"}
SEEDS=${SEEDS:-"801 802 803 804"}; MSEEDS=${MSEEDS:-"2026 2027 2028 2029"}
EPOCHS=${EPOCHS:-300}; KEEP=${KEEP:-30}; TRAIN=${TRAIN:-data/train/rdm16m/RDM/trials}; VAL=${VAL:-}
MSTEPS=${MSTEPS:-100000}; MEVERY=${MEVERY:-1000}; DRY=${DRY:-0}
OUT=sbc/seed_study/$PREFIX; mkdir -p "$OUT/logs"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
export PYTHONPATH="$ROOT/flow/src${PYTHONPATH:+:$PYTHONPATH}"

say "=== seed study launch: prefix $PREFIX, arms [$ARMS], root $ROOT ==="
# ---- preflight: name what is missing, start nothing if anything is ----------
bad=0; need() { say "  MISSING: $*"; bad=1; }
ours=0; his=0
for a in $ARMS; do [ "$a" = malte ] && his=1 || ours=1; done
if [ $ours = 1 ]; then
  [ -x "$PY" ] || need "our python $PY"
  case "$TRAIN" in /*) T="$TRAIN";; *) T="$ROOT/$TRAIN";; esac
  ls "$T"/*.parquet >/dev/null 2>&1 || need "training data $T/*.parquet"
  V="${VAL:-data/val/RDM_single/trials}"; case "$V" in /*) ;; *) V="$ROOT/$V";; esac
  ls "$V"/*.parquet >/dev/null 2>&1 || need "validation data $V/*.parquet"
  # the flow package must be THIS root's, with the affine option (an installed
  # copy of an older flow/src trains a plain flow under --affine's name)
  [ -x "$PY" ] && { "$PY" - "$ROOT" <<'PYEOF' || need "flow.config under $ROOT/flow/src with the affine option"
import inspect, sys, pathlib
import flow.config as c
root = pathlib.Path(sys.argv[1]).resolve()
assert root in pathlib.Path(c.__file__).resolve().parents, c.__file__
assert "affine" in inspect.signature(c.make_mlp_conditioner).parameters
PYEOF
  }
fi
if [ $his = 1 ]; then
  [ -x "$PYM" ] || need "Malte's python $PYM (malte/README.md, plus hydra-core)"
  [ -x "$PYM" ] && { "$PYM" -c "import hydra, eamax, confrdm_jax.flows_affine" 2>/dev/null \
      || need "hydra-core, eamax and confrdm_jax importable by $PYM"; }
  [ -d "$ROOT/malte/neural-race-model-estimation/conf_jax" ] || need "malte/neural-race-model-estimation/conf_jax"
fi
[ $bad = 0 ] || { say "ABORT: preflight failed, nothing started"; exit 1; }
say "  preflight passed"

# ---- the job list ------------------------------------------------------------
JOB_TAG=(); JOB_CMD=()
for a in $ARMS; do
  if [ "$a" = malte ]; then
    for s in $MSEEDS; do
      JOB_TAG+=("${PREFIX}_malte_s$s")
      JOB_CMD+=("$PYM malte/train_conditioner_ckpt.py --out results/malte_runs/${PREFIX}_malte_s$s --seed $s --steps $MSTEPS --save-every $MEVERY --keep $KEEP")
    done
  else
    case $a in base) f="";; aff) f="--affine";; dt) f="--decision-time";; both) f="--affine --decision-time";;
      *) say "ABORT: unknown arm $a"; exit 1;; esac
    for s in $SEEDS; do
      JOB_TAG+=("${PREFIX}_${a}_s$s")
      JOB_CMD+=("$PY flow/scripts/train_race.py --model RDM --tag ${PREFIX}_${a}_s$s --train-data $TRAIN ${VAL:+--val-data $VAL} --epochs $EPOCHS --keep-checkpoints $KEEP --seed $s --shuffle-seed $s $f")
    done
  fi
done
N=${#JOB_TAG[@]}; say "  $N runs, $JOBS at a time, $THREADS cores each"

# A finished run is never restarted: train_race.py wipes its checkpoint
# directory when it starts.
finished() {
  local tag=$1 f n
  case $tag in
    *_malte_s*) f=results/malte_runs/$tag/training_log.json
       [ -f $f ] && n=$("$PYM" -c "import json;l=json.load(open('$f'));print(len(l['saved']) if 'total_time_s' in l else 0)") || n=0
       [ "$n" -ge "$KEEP" ];;
    *) f=results/model_states/${tag}_budget/RDM/training_log.json
       [ -f $f ] && n=$("$PY" -c "import json;print(len(json.load(open('$f'))['train_losses']))") || n=0
       [ "$n" -ge "$EPOCHS" ];;
  esac
}
worker() {            # slot k runs jobs k, k + JOBS, k + 2 JOBS, ...
  local k=$1 i pin=""
  if command -v taskset >/dev/null 2>&1; then
    pin="taskset -c $((CORE0 + k * THREADS))-$((CORE0 + (k + 1) * THREADS - 1))"; fi
  for ((i = k; i < N; i += JOBS)); do
    local tag=${JOB_TAG[$i]}
    if finished "$tag"; then say "  $tag already complete, skipped"; continue; fi
    if [ "$DRY" = 1 ]; then echo "DRY slot $k: $pin ${JOB_CMD[$i]}"; continue; fi
    say "  start $tag (slot $k)"
    if XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$THREADS" \
       OMP_NUM_THREADS=$THREADS $pin ${JOB_CMD[$i]} > "$OUT/logs/$tag.log" 2>&1 && finished "$tag"
    then say "  done  $tag"; else say "  FAILED $tag (see $OUT/logs/$tag.log)"; fi
  done
}
for ((k = 0; k < JOBS; k++)); do worker $k & done
wait
[ "$DRY" = 1 ] && exit 0

# ---- the table the watcher reads ----------------------------------------------
nok=0
printf "\n%-24s %-9s %s\n" run status "last loss (ours: val; his: train window)"
for ((i = 0; i < N; i++)); do
  tag=${JOB_TAG[$i]}
  if finished "$tag"; then st=complete; nok=$((nok + 1)); else st=INCOMPLETE; fi
  case $tag in
    *_malte_s*) l=$("$PYM" -c "import json;print('%.5f' % json.load(open('results/malte_runs/$tag/training_log.json'))['loss'][-1])" 2>/dev/null);;
    *) l=$("$PY" -c "import json;print('%.5f' % json.load(open('results/model_states/${tag}_budget/RDM/training_log.json'))['val_losses'][-1])" 2>/dev/null);;
  esac
  printf "%-24s %-9s %s\n" "$tag" "$st" "${l:-NA}"
done
say "=== $nok of $N runs complete ==="
if [ "$nok" = "$N" ]; then echo "seed study training done $(date), prefix $PREFIX, $N runs" > "$OUT/train_done.marker"; else exit 1; fi

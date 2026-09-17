#!/bin/bash
# Reproducibility test: run SBC on SEVERAL FLOW SEEDS concurrently.
#
# The question this answers is the one that decides whether we have a recipe:
# does a flow trained by the same procedure with a different seed still pass
# calibration? Score bias cannot answer it — the displacement is multivariate
# (I^-1 g), so a parameter's score bias does not map onto its own SBC bias, and
# in the one matched comparison we have the two even have opposite signs.
#
# DDMnn reads a HARDCODED filename inside its EMC2 install, so seeds cannot
# share one library. This clones the dev-nn library per seed and installs that
# seed's bundle into the clone, so the runs are genuinely independent.
#
#   bash run_seed_sbc_local.sh 400 s101 s102 s103
#   bash run_seed_sbc_local.sh 400 s101              (just one)
#
# Cores are split evenly across seeds; override the total with SBC_TOTAL_CORES.
set -e
N=${1:?usage: run_seed_sbc_local.sh <trials> <seed> [seed ...]}
shift
SEEDS=("$@")
[ ${#SEEDS[@]} -ge 1 ] || { echo "give at least one seed, e.g. s101"; exit 1; }

BASE=${EMC2NN_LIB:-$HOME/emc2nn/Rlib-devnn}
[ -d "$BASE/EMC2" ] || { echo "ABORT: no dev-nn library at $BASE"; exit 1; }
TOTAL=${SBC_TOTAL_CORES:-72}
PER=$(( TOTAL / ${#SEEDS[@]} ))
[ "$PER" -ge 1 ] || { echo "ABORT: not enough cores"; exit 1; }
REPS=${SBC_REPS:-500}

echo "seeds: ${SEEDS[*]}   N=$N   reps=$REPS   cores: $PER each of $TOTAL"
echo

for s in "${SEEDS[@]}"; do
  SRC="ddm_ens5_single_${s}_c4.rds"
  [ -f "$SRC" ] || { echo "ABORT: $SRC not found"; exit 1; }
  LIB="${BASE}-${s}"
  if [ ! -d "$LIB/EMC2" ]; then
    echo "cloning library for $s -> $LIB"
    mkdir -p "$LIB"; cp -r "$BASE/EMC2" "$LIB/"
  fi
  DST="$LIB/EMC2/extdata/flownn/ddm_ens5.rds"
  cp "$SRC" "$DST"
  # confirm the clone really carries THIS seed before launching anything
  EMC2NN_LIB="$LIB" Rscript -e '
    .libPaths(c(Sys.getenv("EMC2NN_LIB"), .libPaths()))
    b <- readRDS(system.file("extdata","flownn","ddm_ens5.rds",package="EMC2",mustWork=TRUE))
    cat("  ", Sys.getenv("EMC2NN_LIB"), "->", paste(b$member_tags, collapse=","), "\n")'
done

echo
for s in "${SEEDS[@]}"; do
  LIB="${BASE}-${s}"
  OUT="results_seed_${s}"
  LOG="SBCseed_${s}_N${N}.log"
  nohup env EMC2NN_LIB="$LIB" SBC_CORES="$PER" SBC_REPS="$REPS" \
    Rscript SBCc4.R "$OUT" "$N" > "$LOG" 2>&1 &
  echo "launched $s on $PER cores -> $OUT (pid $!, log $LOG)"
done
echo
echo "each log prints its EMC2 path and member tags at startup — check them before"
echo "trusting any result, since a mis-cloned library would run the wrong flow."

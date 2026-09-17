#!/bin/bash
# Run the c4 SBC campaign on an ordinary Linux box (no scheduler).
# Gadi equivalent: launch_sbc_c4.sh
#
#   bash run_sbc_c4_local.sh single    # cap256w_c4 alone (recommended)
#   bash run_sbc_c4_local.sh ens       # 5-member c4 ensemble
#
# Cells run CONCURRENTLY, with cores split by cost rather than evenly. Cost
# scales with N, so equal shares would leave N=100 idle while N=400 ran on. Using
# Gadi's measured 55/78/119 min the work splits 21/32/47%, i.e. 16/22/34 of 72
# cores, and all three then finish together (~3 h) instead of being paced by
# N=400 (~4.3 h at 24 each).
#
#   SBC_TOTAL_CORES=72              total to divide (default 72)
#   SBC_CORES_100/200/400=..        override a single cell explicitly
#   SBC_SEQUENTIAL=1                run cells one at a time instead
#
# CRITICAL, same as on Gadi: DDMnn reads <EMC2>/extdata/flownn/ddm_ens5.rds and
# that name is hardcoded, so single and ens CANNOT run at the same time against
# one install.
set -e
WHICH=${1:?usage: run_sbc_c4_local.sh single|ens}
case "$WHICH" in
  single) SRC=ddm_ens5_single_c4.rds; OUT=results_c4single ;;
  single5) SRC=ddm_ens5_single_c5.rds; OUT=results_c5single ;;
  ens)    SRC=ddm_ens5_c4.rds;        OUT=results_c4ens ;;
  *) echo "unknown: $WHICH (expected single|ens)"; exit 1 ;;
esac
export EMC2NN_LIB=${EMC2NN_LIB:-$HOME/emc2nn/Rlib-devnn}
[ -d "$EMC2NN_LIB" ] || { echo "ABORT: no dev-nn library at $EMC2NN_LIB"
  echo "  run:  bash install_devnn_local.sh"; exit 1; }
[ -f "$SRC" ] || { echo "ABORT: $SRC not found in $(pwd)"; exit 1; }

if pgrep -f "SBCc4\.R" >/dev/null; then
  echo "REFUSING: an SBCc4.R run is already active - swapping the bundle now"
  echo "would change the weights under it."; pgrep -af "SBCc4\.R"; exit 1
fi

Rscript -e '.libPaths(c(Sys.getenv("EMC2NN_LIB"), .libPaths()));
  q(status = if (isTRUE(requireNamespace("EMC2", quietly=TRUE)) &&
    "ddm_ens_eval_trials_cpp" %in% ls(getNamespace("EMC2"))) 0 else 1)' || {
  echo "ABORT: EMC2 at $EMC2NN_LIB is not the dev-nn build."
  echo "  run:  bash install_devnn_local.sh"; exit 1; }

DST=$(Rscript -e '.libPaths(c(Sys.getenv("EMC2NN_LIB"), .libPaths()));
  cat(system.file("extdata","flownn","ddm_ens5.rds",package="EMC2",mustWork=TRUE))')
[ -n "$DST" ] || { echo "could not locate EMC2 extdata"; exit 1; }
[ -f "$DST.orig" ] || { cp "$DST" "$DST.orig"; echo "backed up -> $DST.orig"; }
cp "$SRC" "$DST"
echo "installed $SRC -> $DST"
Rscript -e '.libPaths(c(Sys.getenv("EMC2NN_LIB"), .libPaths()));
  b <- readRDS(system.file("extdata","flownn","ddm_ens5.rds",package="EMC2"));
  cat("verify:", length(if (is.null(b$members)) list(b) else b$members),
      "member(s):", paste(b$member_tags, collapse=", "), "\n")'

TOTAL=${SBC_TOTAL_CORES:-72}
# weights from Gadi's measured per-cell times (55 / 78 / 119 min)
C100=${SBC_CORES_100:-$(( TOTAL * 55 / 252 ))}
C200=${SBC_CORES_200:-$(( TOTAL * 78 / 252 ))}
C400=${SBC_CORES_400:-$(( TOTAL - C100 - C200 ))}

if [ -n "${SBC_SEQUENTIAL:-}" ]; then
  LOG="SBCc4_${OUT}.log"
  echo "running cells sequentially on $TOTAL cores -> $OUT (log: $LOG)"
  nohup env EMC2NN_LIB="$EMC2NN_LIB" SBC_CORES="$TOTAL" \
    Rscript SBCc4.R "$OUT" 100 200 400 > "$LOG" 2>&1 &
  echo "started pid $! — track with:  tail -f $LOG"
else
  echo "running 3 cells concurrently on $TOTAL cores: N100=$C100 N200=$C200 N400=$C400"
  for spec in "100 $C100" "200 $C200" "400 $C400"; do
    set -- $spec; N=$1; C=$2
    LOG="SBCc4_${OUT}_N${N}.log"
    nohup env EMC2NN_LIB="$EMC2NN_LIB" SBC_CORES="$C" \
      Rscript SBCc4.R "$OUT" "$N" > "$LOG" 2>&1 &
    echo "  N=$N on $C cores -> pid $! (log: $LOG)"
  done
  echo
  echo "track with:  tail -f SBCc4_${OUT}_N*.log"
  echo "all cells write to ./$OUT/ under distinct filenames, so they cannot clash"
fi

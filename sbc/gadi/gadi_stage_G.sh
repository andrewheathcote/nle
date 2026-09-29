#!/bin/bash
# Stage G: dev-nle on Gadi. Run from the laptop; every step goes over ssh with the key.
#
#   bash sbc/gadi/gadi_stage_G.sh setup     copy the bundle to scratch, install the library
#   bash sbc/gadi/gadi_stage_G.sh submit    qsub the check job (prints the job id)
#   bash sbc/gadi/gadi_stage_G.sh verify    bring the logs down, print GADI CHECK PASS / FAIL
#   bash sbc/gadi/gadi_stage_G.sh status    qstat and the tail of the check log; changes nothing
#
# Rules (AH, 2026-09-29):
#  * Nothing before 1 October 2026 (the allocation starts then): setup and submit refuse.
#  * EVERYTHING uploaded to or run on Gadi lives under /scratch/tg53/ajh575: the home
#    directory is small. That includes the bundle, the build directory, R's temporary files,
#    every cell and every log. The library is installed in scratch too,
#    /scratch/tg53/ajh575/R/nle-lib, and ~/R/nle-lib is a symlink to it, so the agreed name
#    and the cells' assertion (find.package("EMC2") under ~/R/nle-lib, both sides through
#    normalizePath) hold while home holds nothing. Scratch expires: if the library has been
#    purged, `setup` rebuilds it from the tarball in a few minutes.
#  * PBS only; the install is the one thing that runs on a login node (a compile of a few
#    minutes, as for ~/R/ss-lib). ~/R/ss-lib, ~/R/ssd-lib and the default library are not touched.
#  * No password is ever asked for: BatchMode. If the key stops working this script fails at
#    its first line and AH runs the same commands by hand (they are printed with DRY=1).
set -eu
HOST=ajh575@gadi.nci.org.au
SCR=/scratch/tg53/ajh575
BUNDLE_LOCAL=${BUNDLE_LOCAL:-$HOME/Documents/Projects/EMCpackage/SBC-pending/2026-10-01_gadi_dev-nle}
BUNDLE=$SCR/2026-10-01_gadi_dev-nle
LIBREAL=$SCR/R/nle-lib
BACK=${BACK:-$(cd "$(dirname "$0")/../.." && pwd)/sbc/gadi/stage_G}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 $HOST"
run() { if [ "${DRY:-0}" = 1 ]; then echo "+ $*"; else "$@"; fi; }
guard() {
  today=$($SSH date +%Y%m%d)
  [ "$today" -ge 20261001 ] || { echo "REFUSED: Gadi's date is $today; nothing is copied, installed or submitted before 20261001"; exit 2; }
}
node=$($SSH hostname) || { echo "NO KEY LOGIN: do not retry, do not ask for a password; AH runs the commands (DRY=1 prints them)"; exit 3; }
case "$node" in gadi-login-*) ;; *) echo "unexpected host: $node"; exit 3;; esac

case "${1:-}" in
setup)
  guard
  (cd "$BUNDLE_LOCAL/cards" && shasum -a 256 -c SHA256)
  run rsync -a --delete "$BUNDLE_LOCAL/" "$HOST:$BUNDLE/"
  run $SSH "bash -lc 'set -e
    mkdir -p $LIBREAL $SCR/tmp \$HOME/R
    if [ -e \$HOME/R/nle-lib ] && [ ! -L \$HOME/R/nle-lib ]; then echo \"~/R/nle-lib exists and is not a symlink: stop\"; exit 1; fi
    ln -sfn $LIBREAL \$HOME/R/nle-lib
    cd $BUNDLE && export TMPDIR=$SCR/tmp NLE_LIB=\$HOME/R/nle-lib
    bash install_nle_lib.sh
    du -sh $LIBREAL; ls -la \$HOME/R/nle-lib'"
  ;;
submit)
  guard
  run $SSH "bash -lc 'cd $BUNDLE && ls \$HOME/R/nle-lib/EMC2/DESCRIPTION && export TMPDIR=$SCR/tmp && qsub -v NLE_LIB=/home/575/ajh575/R/nle-lib,TMPDIR=$SCR/tmp check_nle_lib.pbs'"
  ;;
status)
  $SSH "bash -lc 'qstat -u ajh575; tail -5 $BUNDLE/check_nle_lib.log 2>/dev/null'"
  ;;
verify)
  mkdir -p "$BACK"
  rsync -a "$HOST:$BUNDLE/install_nle_lib.log" "$HOST:$BUNDLE/check_nle_lib.log" "$BACK/"
  L="$BACK/check_nle_lib.log"; ok=1
  grep -q "TESTS expectations 783 failed 0 errors 0" "$L" || ok=0
  for v in "hybrid .* mean ll 131.286" "CRDM dt 5e-4 .* mean ll 131.325" "CRDM dt 2e-3 .* mean ll 131.359"; do grep -q "$v" "$L" || ok=0; done
  grep -q "EMC2 .* at .*nle-lib" "$L" || ok=0
  grep -E "^EMC2|^BLAS|TESTS expectations|ms per likelihood" "$L" || true
  if [ $ok = 1 ]; then echo "GADI CHECK PASS: cells may be submitted"; else echo "GADI CHECK FAIL: submit no cell"; exit 1; fi
  ;;
*) sed -n 2,8p "$0"; exit 1;;
esac

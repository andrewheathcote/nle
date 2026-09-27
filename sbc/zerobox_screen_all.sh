#!/bin/bash
# Zero-box S3, part 2, on the laptop (8 cores, dev-nle at ~/R/nle-sbc-lib):
#   1. fetch every exported checkpoint from tux26;
#   2. score-bias screen, all checkpoints, seven regimes (5 zero + 2 gap),
#      cap256w_c4 alongside in the full regime as the Q2 reference;
#   3. apply the declared selection rule (sbc/zerobox_select.R);
#   4. posterior-shift screen for the top 3 in the five zero regimes.
# Run from the NLE repo root:  nohup bash sbc/zerobox_screen_all.sh > sbc/zerobox_screen_all.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
export LIB=~/R/nle-sbc-lib SB_CORES=8 PS_CORES=8
DRAWS=${DRAWS:-300}; TOP=${TOP:-3}; ND=${ND:-48}
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

say "=== 1. fetch exports ==="
rsync -aq --include='ddm_zb_s90?_e*_weights.json' --exclude='*' \
  andrew@tux26psy.fmg.uva.nl:~/emc2nn/nle/handover/nle-methods/port/export/ port/export/ \
  || { say "ABORT: rsync failed"; exit 1; }
TAGS=$(ls port/export | grep -oE '^ddm_zb_s90[0-9]_e[0-9]+' | sed 's/^ddm_//' | sort -t e -k2 -n | tr '\n' ' ')
N=$(echo $TAGS | wc -w); say "  $N checkpoints"
[ "$N" -ge 60 ] || { say "ABORT: expected ~90 exports"; exit 1; }

say "=== 2. score-bias screen, $DRAWS draws, seven regimes ==="
for r in full sv0 SZ0 st00 all0 gap1 gap3; do
  extra=""; [ "$r" = full ] && extra="cap256w_c4"
  rm -f sbc/zerobox_screen/$r.rds
  Rscript sbc/zerobox_screen.R $r $DRAWS $extra $TAGS > sbc/zerobox_screen/$r.out 2>&1 \
    || { say "ABORT: screen failed in regime $r (see sbc/zerobox_screen/$r.out)"; exit 1; }
  say "  $r: $(grep -E '^CONTROL max' sbc/zerobox_screen/$r.out)"
done

say "=== 3. selection ==="
Rscript sbc/zerobox_select.R $TOP | tee sbc/zerobox_screen/selection.out || { say "ABORT: selection failed"; exit 1; }
SEL=$(Rscript -e 'cat(readRDS("sbc/zerobox_screen/ranking.rds")$top)')

say "=== 4. posterior-shift screen, top $TOP x five regimes, $ND datasets ==="
for t in $SEL; do for r in full sv0 SZ0 st00 all0; do
  Rscript sbc/zerobox_postshift.R $t $r $ND > sbc/zerobox_postshift/${t}_${r}.out 2>&1 \
    || { say "  FAILED $t $r"; continue; }
  say "  $t $r: $(awk '/^t0 |^Z |^a /{printf "%s flow %s  ", $1, $3}' sbc/zerobox_postshift/${t}_${r}.out)"
done; done
say "=== S3 done: read sbc/zerobox_screen/selection.out and sbc/zerobox_postshift/*.out ==="
echo "zerobox screen done $(date)" > sbc/zerobox_screen/done.marker

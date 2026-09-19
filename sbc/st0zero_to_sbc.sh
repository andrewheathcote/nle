#!/bin/bash
# After the st0 = 0 flows finish: export -> screen -> select -> bundle ->
# transfer -> launch SBC on tux26 at 24 cores.
#
# Waits for training rather than assuming it is done, so this can be started
# at any time. Selection is on TOTAL |g| across the seven parameters, the same
# criterion used for the RDM: score bias ranks checkpoints reliably WITHIN an
# architecture (cor -0.989), which is exactly this case, while it does not rank
# across architectures (Phase 25).
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd); PY="$ROOT/flow/.venv/bin/python"
LOG="$ROOT/sbc/st0zero_sbc.log"
TUX=andrew@tux26psy.fmg.uva.nl
CORES=${SBC_CORES:-8}          # x3 chains = 24
DRAWS=200
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "waiting for the st0=0 flow training to finish..."
while pgrep -f "[t]rain_ddm[.]py --tag z0_s" > /dev/null; do sleep 300; done
say "training finished"

SEEDS="z0_s501 z0_s502 z0_s503"
say "STAGE 1: exporting"
for t in $SEEDS; do
  d="$ROOT/results/model_states/${t}_budget/DDM_flow"
  [ -d "$d" ] || { say "  $t: no checkpoints, skipping"; continue; }
  for e in $(ls "$d" | grep '^[0-9]' | sort -n); do
    [ -f "$ROOT/port/export/ddm_${t}_e${e}_weights.json" ] && continue
    "$PY" flow/scripts/export_flow_ddm.py --budget "$t" --st0-zero --step "$e" \
      --tag-suffix "_e${e}" >> "$ROOT/sbc/export_${t}.log" 2>&1 \
     || "$PY" flow/scripts/export_flow_ddm.py --budget "$t" --st0-zero --step "$e" \
          --tag-suffix "_e${e}" --port-tol 2.5e-2 >> "$ROOT/sbc/export_${t}.log" 2>&1 \
     || say "  $t e$e EXPORT FAILED"
  done
  say "  $t: $(ls "$ROOT/port/export/" | grep -cE "^ddm_${t}_e[0-9]+_weights.json$") exported"
done

TAGS=""
for t in $SEEDS; do
  TAGS="$TAGS $(ls "$ROOT/port/export/" | grep -oE "^ddm_${t}_e[0-9]+_weights.json$" \
        | sed 's/^ddm_//; s/_weights.json$//' | sort -u | tr '\n' ' ')"
done
N=$(echo $TAGS | wc -w | tr -d ' ')
say "STAGE 2: measuring score bias on $N checkpoints"
[ "$N" -gt 0 ] || { say "ABORT: nothing exported"; exit 1; }
SB_H=0.01 SB_OUT="$ROOT/sbc/rep_sb/z0_g.rds" \
  Rscript sbc/score_bias_draws.R "$DRAWS" $TAGS >> "$LOG" 2>&1
SBT_CORES=9 SBT_OUT="$ROOT/sbc/rep_sb/z0_t0.rds" \
  Rscript sbc/score_bias_t0.R "$DRAWS" $TAGS >> "$LOG" 2>&1

say "STAGE 3: selecting on total |g|"
BEST=$(Rscript sbc/select_st0zero.R "$ROOT/sbc/rep_sb/z0_g.rds" \
        "$ROOT/sbc/rep_sb/z0_t0.rds" --pick 2>/dev/null | tr -d ' \n')
[ -n "$BEST" ] || { say "ABORT: selection produced nothing"; exit 1; }
say "  selected $BEST"
Rscript sbc/select_st0zero.R "$ROOT/sbc/rep_sb/z0_g.rds" "$ROOT/sbc/rep_sb/z0_t0.rds" \
  2>/dev/null | tee -a "$LOG"

say "STAGE 4: bundle"
Rscript sbc/make_bundle_st0zero.R "$ROOT/sbc/ddm_st0zero.rds" "$BEST" 2>&1 | tee -a "$LOG"
[ -f "$ROOT/sbc/ddm_st0zero.rds" ] || { say "ABORT: bundle not written"; exit 1; }

say "STAGE 5: transfer to tux26"
scp -q "$ROOT/sbc/ddm_st0zero.rds" "$TUX:~/emc2nn/nle/sbc/" || { say "ABORT: scp bundle"; exit 1; }
scp -q "$ROOT/sbc/nci/SBCddm7.R" "$TUX:~/emc2nn/nle/sbc/nci/" || { say "ABORT: scp driver"; exit 1; }
ssh "$TUX" 'mkdir -p ~/emc2nn/nle/handover/st0_zero'
scp -q "$ROOT/handover/st0_zero/DDMnn7.R" "$TUX:~/emc2nn/nle/handover/st0_zero/" \
  || { say "ABORT: scp model wrapper"; exit 1; }

say "STAGE 6: install bundle + launch SBC at $((CORES*3)) cores"
ssh "$TUX" "cd ~/emc2nn/nle && \
  DST=\$(Rscript -e 'if (dir.exists(\".Rlibs\")) .libPaths(c(normalizePath(\".Rlibs\"), .libPaths())); cat(dirname(system.file(\"extdata\",\"flownn\",\"ddm_ens5.rds\", package=\"EMC2\", mustWork=TRUE)))' 2>/dev/null) && \
  cp sbc/ddm_st0zero.rds \"\$DST/ddm_st0zero.rds\" && \
  echo installed to \"\$DST/ddm_st0zero.rds\" && \
  setsid nohup env SBC_CORES=$CORES Rscript sbc/nci/SBCddm7.R sbc/nci/results_ddm7 400 500 \
    > sbc/ddm7_sbc.log 2>&1 < /dev/null & " 2>&1 | tee -a "$LOG"
sleep 60
ssh "$TUX" 'cd ~/emc2nn/nle && grep -v "LC_CTYPE\|^During startup" sbc/ddm7_sbc.log | head -12' \
  2>&1 | tee -a "$LOG"
say "=== SBC LAUNCHED on tux26 (N=400, 500 reps, $((CORES*3)) cores) ==="

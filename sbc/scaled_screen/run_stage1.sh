#!/bin/bash
# Stage 1 of the scaled selection (SCALED_DDM_PLAN.md section 5): all 180 checkpoints of
# sc_s1101..sc_s1106, every regime, 100 draws, laptop 6 cores. Stage 2 re-screens the leaders at 300.
cd /Users/andrew/Downloads/julian/nle
TAGS=$(ls handover/nle-methods/port/export | sed -nE 's/^ddm_(sc_s11[0-9]{2}_e[0-9]+)_weights.json$/sc:\1/p' | tr '\n' ' ')
for r in full sv0 SZ0 st00 st0only SZonly all0; do
  SB_CORES=6 SB_OUT=sbc/scaled_screen/stage1/$r.rds Rscript sbc/scaled_screen.R $r 100 sc:oracle $TAGS \
    > sbc/scaled_screen/stage1/$r.out 2>&1 || echo "$r FAILED $(date)"
  echo "$r done $(date)"
done
echo "stage1 done $(date)"

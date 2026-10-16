#!/bin/bash
# Pilot screen (laptop, 2 cores while N=1600 holds 6): scaled pilot checkpoints vs v1, v2, oracle.
cd /Users/andrew/Downloads/julian/nle
for r in full sv0 SZ0 st00 st0only SZonly all0; do
  SB_CORES=2 Rscript sbc/scaled_screen.R $r 300 sc:oracle sc:sc_pilot_e27 sc:sc_pilot_e29 zb_s903_e271 zb2_s901_e274 \
    > sbc/scaled_screen/pilot_$r.out 2>&1 || echo "$r FAILED"
done
echo "pilot screen done $(date)"

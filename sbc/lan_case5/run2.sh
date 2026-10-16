#!/bin/bash
# Rebalance (2026-10-16): the LAN's replicates run ~4x slower than the control's. When the control
# (pid 89480) finishes, restart the LAN on all 8 cores (4 data sets x 2 chain cores); run_sbc resumes
# from rep_<i>.rds and prior_samples.rds in sbc_nn_temp. Then the joint summary.
cd /Users/andrew/Downloads/julian/nle
export R_PROFILE_USER=/dev/null OUT=sbc/lan_case5/results
while kill -0 89480 2>/dev/null; do sleep 60; done
echo "control ended $(date)"
pkill -P 89479; kill 89479 2>/dev/null; sleep 5
echo "nn restarted $(date): $(ls sbc/lan_case5/results/results/sbc_nn_temp | grep -c '^rep_') reps on disk"
CPC=4 CFC=2 Rscript sbc/lan_case5/lan_cell.R nn >> sbc/lan_case5/nn.log 2>&1
Rscript sbc/lan_case5/lan_cell.R summary > sbc/lan_case5/summary.log 2>&1
echo "lan case5 done $(date)"

#!/bin/bash
# Test machine only: build, the cell's own checks (BENCH), then a short SBC to time replicates. No shutdown.
cd "$(dirname "$0")"; mkdir -p logs
LIB=$(sudo bash setup.sh 2> logs/setup.err | tail -1); echo "LIB $LIB"
LIB=$LIB BENCH=1 R_PROFILE_USER=/dev/null Rscript sbc/scaled_cell.R full port/export/ddm_sc_s1106_e245_weights.json 2>&1 | grep -vE "^Processing|WienR" | tail -4
LIB=$LIB ORACLE=1 BENCH=1 R_PROFILE_USER=/dev/null Rscript sbc/scaled_cell.R st0only port/export/ddm_sc_s1106_e245_weights.json 2>&1 | grep -E "LL vs|ms" | tail -2
t0=$(date +%s)
LIB=$LIB CPC=${CPC:-5} CFC=3 OUT=smoke_results R_PROFILE_USER=/dev/null Rscript sbc/scaled_cell.R full port/export/ddm_sc_s1106_e245_weights.json ${REPS:-10} 400 > logs/smoke_sbc.log 2>&1
echo "SBC smoke: ${REPS:-10} reps, CPC ${CPC:-5} x CFC 3, $(( $(date +%s) - t0 )) s; exit $?"
grep -E "=== cell|network" logs/smoke_sbc.log | tail -8

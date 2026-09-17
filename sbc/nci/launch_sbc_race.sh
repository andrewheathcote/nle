#!/bin/bash
# Phase 4, RACE CELLS ONLY (RDM and LNR). Run from
# /scratch/tg53/ajh575/emc2nn/nci.
#
# NO EMC2 REINSTALL NEEDED. These cells use LNRnn and RDMnn, which entered
# dev-nn at bd70c709 and are unchanged since; the Phase 3 install (>= 85dde2a6)
# already has them. Only the DDM cells (P1-P4, C3, C4) depend on the new
# score-penalised flow and must wait for that rebuild.
#
#   R1  RDMnn N=1000   A at parity (N*I/prior 0.96) - first real test of A,
#                      whose Phase 1 "PASS" at N=100 was ~content-free (0.10)
#   R2  RDM   N=1000   analytic control
#   L1  LNRnn N=400    does the Phase 1 s anomaly worsen with N?
#   L2  LNRnn N=1000   ditto, further out
#   L3  LNR   N=1000   analytic control
#
# Track: qstat -u $USER  (SBCidR1 ...).  Results -> ./results_id/
set -e
for c in R1 R2 L1 L2 L3; do
  id=$(qsub SBCid_$c.pbs); echo "submitted $c -> $id"
done

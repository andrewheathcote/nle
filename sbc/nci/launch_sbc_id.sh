#!/bin/bash
# Phase 4 (identification 2x2). Run from /scratch/tg53/ajh575/emc2nn/nci.
# Prereq: EMC2 dev-nn installed with the PRODUCTION penalised flow shipped as
# DDMnn, and sbc_specs.R + sbc_specs_id.R present.
# Cells: P1/P2/P3/P4 = DDMnn; C3/C4 = analytic controls at the new point.
set -e
for c in P1 P2 P3 P4 C3 C4 R1 R2 L1 L2 L3; do
  id=$(qsub SBCid_$c.pbs); echo "submitted $c -> $id"
done

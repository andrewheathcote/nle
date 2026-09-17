#!/bin/bash
# Launch the SBC job set on NCI (run from the directory holding these files).
#
# Prerequisites (once):
#   - dev-nn EMC2 installed into your R 4.5.0 library on Gadi, e.g.
#       module load gcc/12.2.0 intel-compiler/2021.10.0 R/4.5.0
#       R CMD INSTALL /path/to/EMC2   # the dev-nn checkout
#   - this directory contains: sbc_specs.R, SBCfast.R, SBCfast.pbs,
#     SBCddm.R, SBCddm.pbs
#
# Track with:  qstat -u $USER   (job names SBCfast, SBCddm)
# Results land in ./results/<model>_N<trials>_R500.rds; logs in ./SBC*.log

set -e
for job in SBCfast.pbs SBCddm.pbs; do
  id=$(qsub "$job")
  echo "submitted $job -> $id"
done

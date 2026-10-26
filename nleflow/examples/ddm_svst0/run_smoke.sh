#!/bin/sh
# The smoke chain on this machine (1 % of the data, 2 epochs): spec, simulate, the
# simulator vs WienR, train (flow + classifier), export, golden from the card alone,
# EMC2 (golden check with the card's fixed values as exceptions, compiled vs R path,
# card vs WienR, the EMC2 cases), nle_verify.R, ONNX both signatures and their checks.
set -e
cd "$(dirname "$0")"
export NLEFLOW_RLIB=${NLEFLOW_RLIB:-$HOME/R/nle-lib} R_PROFILE_USER=/dev/null
export R_LIBS=$NLEFLOW_RLIB
NF="uv run --project ../.. nleflow"
$NF spec spec_smoke.yaml > /dev/null && echo "spec_smoke.yaml validates"
$NF simulate --cores 8 spec_smoke.yaml
Rscript check_simulator.R data_smoke/val
$NF train --cores 8 spec_smoke.yaml
$NF export runs/ddm_svst0_smoke
E=runs/ddm_svst0_smoke/export
CARD=$E/ddm_svst0_smoke_e1_weights.json
$NF golden $CARD $E/regolden.csv && cmp $E/regolden.csv $E/ddm_svst0_smoke_e1_golden.csv && echo "golden from the card alone: identical"
Rscript -e 'library(EMC2); f <- commandArgs(TRUE)[1]
  m <- register_nn_model(f, twin = DDM, exceptions = unlist(jsonlite::fromJSON(f)$fixed)); print(nn_golden_check(m))' $CARD
Rscript check_in_emc2.R $CARD data_smoke/val $E/cases.csv
Rscript -e 'source(system.file("scripts", "nle_verify.R", package = "EMC2"))' $CARD
$NF onnx $CARD
for s in trialwise shared; do $NF onnx-check $E/ddm_svst0_smoke_e1_$s.onnx $CARD --cases $E/cases.csv; done

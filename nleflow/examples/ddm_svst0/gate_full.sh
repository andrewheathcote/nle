#!/bin/sh
# The gates of the full card (after run_full.sh): export the best checkpoint (the
# float64 gate), golden from the card alone, nn_golden_check (twin = DDM, the card's
# fixed values as exceptions), the EMC2 check (compiled vs R path, the network vs the
# analytic DDM, per trial vs dWDM on 1e5 held-out rows, the EMC2 cases), nle_verify.R,
# and the ONNX graphs against the golden rows and the EMC2 cases.
set -e
cd "$(dirname "$0")"
export NLEFLOW_RLIB=${NLEFLOW_RLIB:-$HOME/R/nle-m4-lib} R_PROFILE_USER=/dev/null
export R_LIBS=$NLEFLOW_RLIB
NF="uv run --project ../.. nleflow"
$NF steps runs/ddm_svst0 | tail -5
$NF export runs/ddm_svst0 ${STEP:+--step $STEP}
E=runs/ddm_svst0/export
CARD=$(ls -t $E/ddm_svst0_e*_weights.json | head -1); TAG=$(basename $CARD _weights.json)
echo "card $CARD"
$NF golden $CARD $E/regolden.csv && cmp $E/regolden.csv $E/${TAG}_golden.csv && echo "golden from the card alone: identical"
Rscript -e 'library(EMC2); f <- commandArgs(TRUE)[1]
  m <- register_nn_model(f, twin = DDM, exceptions = unlist(jsonlite::fromJSON(f)$fixed)); print(nn_golden_check(m))' $CARD
Rscript check_in_emc2.R $CARD data/val $E/cases.csv
Rscript -e 'source(system.file("scripts", "nle_verify.R", package = "EMC2"))' $CARD || echo "nle_verify.R: exit $? (see README)"
# with an EMC2 that reads the card's `fixed` (DMC2 main does), if one is given
if [ -n "${EMC2_FIXED_LIB:-}" ]; then
  echo "== nle_verify.R, EMC2 reading the card's fixed ($EMC2_FIXED_LIB)"
  R_LIBS=$EMC2_FIXED_LIB:$NLEFLOW_RLIB Rscript -e 'source(system.file("scripts", "nle_verify.R", package = "EMC2"))' $CARD
fi
$NF onnx $CARD
for s in trialwise shared; do $NF onnx-check $E/${TAG}_$s.onnx $CARD --cases $E/cases.csv; done
Rscript check_simulator.R data/val 100000

#!/bin/sh
# A published LAN into EMC2: download HSSM's ddm_uniform_st (pinned revision,
# checked by sha256), convert it to an mlp_joint card with --verify, register it
# with p_types and check it against its golden rows, nle_verify.R, and the card
# against EMC2's analytic DDM (check_in_emc2.R). LAN_ONNX=<file> skips the download.
set -e
cd "$(dirname "$0")"
export NLEFLOW_RLIB=${NLEFLOW_RLIB:-$HOME/R/nle-ondev-lib:$HOME/R/nle-m4-lib} R_PROFILE_USER=/dev/null
export R_LIBS=$NLEFLOW_RLIB
NF="uv run --project ../.. nleflow"
REV=40e0327c22a4
SHA=4330d72313b322ba9799bad1017180e62a82a7dfc58944826ac54efccab1d173
mkdir -p out
ONNX=${LAN_ONNX:-out/ddm_uniform_st.onnx}
if [ ! -f "$ONNX" ]; then
  curl -sSL -o "$ONNX" "https://huggingface.co/Eitanm/ddm-st-lans/resolve/$REV/ddm_uniform_st.onnx"
fi
got=$(uv run --project ../.. python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$ONNX")
if [ "$got" = "$SHA" ]; then echo "$ONNX: sha256 $got (revision $REV)"; else echo "$ONNX: sha256 $got is not revision $REV's; the preset will say whether it knows it"; fi
$NF convert --list-presets
$NF convert "$ONNX" --preset ddm_uniform_st --out out/lan_ddm_uniform_st_weights.json --verify 300
CARD=out/lan_ddm_uniform_st_weights.json
Rscript -e 'library(EMC2); f <- commandArgs(TRUE)[1]
  m <- register_nn_model(f, p_types = unlist(jsonlite::fromJSON(f)$p_types)); print(nn_golden_check(m))' $CARD
Rscript -e 'source(system.file("scripts", "nle_verify.R", package = "EMC2"))' $CARD
Rscript check_in_emc2.R $CARD

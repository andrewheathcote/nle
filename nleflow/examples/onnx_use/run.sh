#!/bin/sh
# A card's ONNX graphs used from Python with onnxruntime alone. Usage: ./run.sh [card]
# (default: the ddm_svst0 card committed in nleCards). The
# card and its golden file are copied to out/, `nleflow onnx` writes both graphs
# beside them, and use_onnx.py (numpy + onnxruntime only) does the rest.
set -e
cd "$(dirname "$0")"
NF="uv run --project ../.. nleflow"
CARD=${1:-$(ls ../../../nleCards/ddm_svst0_e*_weights.json | tail -1)}
TAG=$(basename "$CARD" _weights.json)
mkdir -p out
cp "$CARD" "$(dirname "$CARD")/${TAG}_golden.csv" out/
[ -f out/${TAG}_trialwise.onnx ] && [ -f out/${TAG}_shared.onnx ] || $NF onnx out/${TAG}_weights.json
uv run --project ../.. python use_onnx.py out/${TAG}_weights.json

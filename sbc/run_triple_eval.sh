#!/bin/bash
# Evaluate the (SZ0, SZ, SZ1) run against its cap256w baseline.
#
# Everything here is already smoke-tested end to end on an early checkpoint, so
# this should run clean; the only thing that changes overnight is which flow
# checkpoint is best. Run from the repo root.
#
#   bash sbc/run_triple_eval.sh [n_draws]     (default 300)
set -e
cd "$(dirname "$0")/.."
N=${1:-300}
echo "=== [$(date +%H:%M:%S)] exporting triple256w ==="
(cd flow && ./.venv/bin/python scripts/export_flow_ddm.py --budget triple256w --triple)
echo "=== [$(date +%H:%M:%S)] golden validation ==="
Rscript port/tests/validate_ddm.R triple256w
echo "=== [$(date +%H:%M:%S)] score bias over $N prior draws ==="
Rscript sbc/score_bias_draws.R "$N" triple256w cap256w_c3
echo "=== [$(date +%H:%M:%S)] done ==="

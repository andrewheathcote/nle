#!/bin/sh
# The full recipe on the laptop, 8 cores: simulate, train (300 epochs), log to full.log.
set -e
cd "$(dirname "$0")"
export NLEFLOW_RLIB=${NLEFLOW_RLIB:-$HOME/R/nle-m4-lib} R_PROFILE_USER=/dev/null
uv run --project ../.. nleflow simulate --cores 8 spec.yaml
uv run --project ../.. nleflow train --cores 8 spec.yaml

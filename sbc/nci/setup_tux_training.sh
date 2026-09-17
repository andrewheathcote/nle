#!/bin/bash
# One-time setup for TRAINING on tux26. The machine has only ever had the R/EMC2
# side; this adds the Python/JAX stack and the training data.
#
# Run from ~/emc2nn/nle (the transferred tree).
set -e
command -v uv >/dev/null || { echo "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
cd flow
uv sync --frozen || uv sync     # uv.lock pins every version
./.venv/bin/python -c "
import jax, flax, optax, distrax, pyarrow, sklearn
print('jax', jax.__version__, jax.devices())
print('flax', flax.__version__, '| optax', optax.__version__)
print('environment OK')"

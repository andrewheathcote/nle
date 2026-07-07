"""Smoke test for the flow package.

Verifies on CPU that:
  - the package imports cleanly,
  - a tiny spline-flow conditioner can be built,
  - a forward + backward pass runs without error.

Run with: pytest -q flow/tests/test_smoke.py
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from flax import nnx

import flow


def test_import():
    expected = {
        "spline_flow", "spline_flow_ddm",
        "make_mlp_conditioner", "make_choice_classifier",
        "batches", "sign_rts",
        "make_checkpoint_manager", "save_checkpoint", "restore_checkpoint",
        "train_model", "train_model_classifier",
    }
    missing = expected - set(dir(flow))
    assert not missing, f"missing from flow: {sorted(missing)}"


def test_tiny_flow_forward_backward():
    rngs = nnx.Rngs(0)
    K = 3                          # number of context features
    conditioner = flow.make_mlp_conditioner(
        rngs=rngs,
        dims=[K, 8, 8],            # tiny MLP
        num_bins=4,
        num_splines=2,
    )

    N = 16
    rt = jnp.linspace(0.1, 1.0, N)
    context = jnp.zeros((N, K), dtype=jnp.float32)

    def loss_fn(cond, x, ctx):
        log_probs, _ = flow.spline_flow(x, ctx, cond)
        return -jnp.mean(log_probs)

    loss, grads = nnx.value_and_grad(loss_fn)(conditioner, rt, context)
    assert jnp.isfinite(loss), f"loss not finite: {loss}"
    # Check grads exist for at least one leaf
    leaves = jax.tree_util.tree_leaves(grads)
    assert any(jnp.any(jnp.isfinite(leaf)) for leaf in leaves), "no finite grads"

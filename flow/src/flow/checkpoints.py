"""Orbax checkpoint manager + save/restore helpers.

Extracted from training.py; semantics unchanged.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx


def make_checkpoint_manager(path, clear_dir=False):
    """Manager that keeps the latest epoch + the best-val-loss epoch."""

    path = Path(path)
    if path.exists() and clear_dir:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    return ocp.CheckpointManager(
        os.path.abspath(path),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            best_fn=lambda m: m["val_loss"],
            best_mode="min",
        ),
    )


def save_checkpoint(mgr, epoch, conditioner, optimizer, val_loss):
    _, cond_state = nnx.split(conditioner)
    _, opt_state = nnx.split(optimizer)
    mgr.save(
        epoch,
        args=ocp.args.Composite(
            conditioner=ocp.args.StandardSave(cond_state),
            optimizer=ocp.args.StandardSave(opt_state),
        ),
        metrics={"val_loss": float(val_loss)},
    )


def restore_checkpoint(mgr, step, conditioner_factory, optimizer_factory):
    """Restore both conditioner and optimizer at `step`.

    *_factory: zero-arg callables that build fresh instances with the same
    architecture/config as the saved ones.
    """
    abstract_cond = nnx.eval_shape(conditioner_factory)
    cond_graphdef, abstract_cond_state = nnx.split(abstract_cond)

    abstract_opt = nnx.eval_shape(optimizer_factory)
    opt_graphdef, abstract_opt_state = nnx.split(abstract_opt)

    restored = mgr.restore(
        step,
        args=ocp.args.Composite(
            conditioner=ocp.args.StandardRestore(abstract_cond_state),
            optimizer=ocp.args.StandardRestore(abstract_opt_state),
        ),
    )
    conditioner = nnx.merge(cond_graphdef, restored["conditioner"])
    optimizer = nnx.merge(opt_graphdef, restored["optimizer"])
    return conditioner, optimizer

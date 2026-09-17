"""Orbax checkpoint manager + save/restore helpers.

Extracted from training.py; semantics unchanged.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx


def make_checkpoint_manager(path, clear_dir=False, max_to_keep=None,
                            save_interval_steps=1):
    """Manager that keeps the latest epoch + the best-val-loss epoch.

    max_to_keep defaults to 1 (the historical behaviour: only the best-val
    checkpoint survives). Raise it to retain a window of late epochs, which is
    what any within-run analysis needs — with one checkpoint per run there is no
    way to tell run-to-run variation from epoch-to-epoch variation inside a
    single run, nor to average weights across epochs. Set from train_ddm.py via
    --keep-checkpoints, or the FLOW_KEEP_CHECKPOINTS environment variable.

    Checkpoints are ~1.5 MB for a [256,128] conditioner plus its optimizer
    state, so keeping 30 costs well under 100 MB.
    """
    if max_to_keep is None:
        max_to_keep = int(os.environ.get("FLOW_KEEP_CHECKPOINTS", "1"))

    path = Path(path)
    if path.exists() and clear_dir:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    return ocp.CheckpointManager(
        os.path.abspath(path),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=save_interval_steps,
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

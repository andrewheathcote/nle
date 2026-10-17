"""`nleflow train <spec>`: the race-flow training of the NLE project's
train_race.py, driven by a spec.

The run directory `<runs>/<name>` holds spec.json (the resolved spec),
scaler.json, training_log.json and checkpoints/ (Orbax; the `keep` latest
epochs plus the best validation loss). The loss, optimiser, schedule and
checkpointing are train_race.py's: mean negative log density of the spline
flow over rows with a finite positive time, AdamW under a warmup-cosine
schedule, one checkpoint per epoch, no early stopping.
"""
from __future__ import annotations

import json
import os
import shutil
import time as _time
from pathlib import Path

import numpy as np

from .data import Batches, Scaler, encode_context, load_table, open_dataset
from .spec import dump_json, flow_context

MIN_LOG_PROB = 1e-12


def run_dir_for(spec, runs):
    return Path(runs).resolve() / spec["name"]


def build_conditioner(spec, seed):
    import jax
    from flax import nnx
    from .flow import make_mlp_conditioner
    dims = [len(flow_context(spec))] + list(spec["conditioner"]["widths"])
    rngs = nnx.Rngs(default=jax.random.key(seed), sampling=jax.random.key(seed + 1))
    return make_mlp_conditioner(rngs, dims=dims, num_bins=spec["spline"]["bins"], num_splines=1,
                                use_norm=spec["conditioner"]["norm"],
                                range_min=spec["spline"]["range"][0],
                                range_max=spec["spline"]["range"][1],
                                affine=spec["conditioner"]["affine"])


def _steps():
    import jax.numpy as jnp
    from flax import nnx

    def loss_fn(conditioner, data, context):
        data = data.squeeze()
        ok = jnp.isfinite(data) & (data > 0.0)
        safe = jnp.where(ok, data, 1.0)
        lp, _ = conditioner.spline_flow(safe, context, conditioner)
        return -jnp.mean(jnp.where(ok, lp, jnp.log(MIN_LOG_PROB)))

    @nnx.jit
    def train_step(conditioner, optimizer, metrics, data, context):
        loss, grads = nnx.value_and_grad(loss_fn)(conditioner, data, context)
        metrics.update(loss=loss)
        optimizer.update(grads)

    @nnx.jit
    def eval_step(conditioner, metrics, data, context):
        metrics.update(loss=loss_fn(conditioner, data, context))

    return train_step, eval_step


def train(spec, runs="runs", epochs=None, cores=None, quiet=False):
    """Train the spec; returns the run directory."""
    import jax
    import optax
    from flax import nnx
    from .flow import make_checkpoint_manager, save_checkpoint

    tr = dict(spec["training"])
    if epochs is not None:
        tr["epochs"] = int(epochs)
    run = run_dir_for(spec, runs)
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))

    t_start = _time.time()
    train_tbl = load_table(open_dataset(spec["data"]["train"]), spec)
    val_tbl = load_table(open_dataset(spec["data"]["val"]), spec)
    t_train, ctx_train = encode_context(train_tbl, spec)
    t_val, ctx_val = encode_context(val_tbl, spec)
    names = flow_context(spec)
    scaler = Scaler.fit(ctx_train, names)
    scaler.save(run / "scaler.json")
    rng = None if tr["shuffle_seed"] is None else np.random.default_rng(tr["shuffle_seed"])
    train_batches = Batches(t_train, scaler.transform(ctx_train), tr["batch"], shuffle=True, rng=rng)
    val_batches = Batches(t_val, scaler.transform(ctx_val), tr["val_batch"], shuffle=False)
    say(f"[{_time.strftime('%H:%M:%S')}] {spec['name']}: train {train_batches.n} rows, "
        f"val {val_batches.n} rows, context {names}, widths {spec['conditioner']['widths']}, "
        f"bins {spec['spline']['bins']}, range {spec['spline']['range']}, batch {tr['batch']}, "
        f"epochs {tr['epochs']}" + ("  affine" if spec["conditioner"]["affine"] else "")
        + (f"  decision time (pre = {spec['pre']})" if spec["pre"] else "")
        + f"  loaded in {_time.time() - t_start:.1f}s")

    conditioner = build_conditioner(spec, tr["seed"])
    total_steps = len(train_batches) * tr["epochs"]
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=tr["peak_lr"],
        warmup_steps=tr["warmup"] * total_steps, decay_steps=total_steps)
    optimizer = nnx.Optimizer(conditioner, optax.adamw(schedule), wrt=nnx.Param)
    train_metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    val_metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    train_step, eval_step = _steps()

    resolved = {**spec, "training": tr, "run": str(run), "flow_context": names}
    dump_json(resolved, run / "spec.json")
    (run / "schedule.json").write_text(json.dumps({
        "schedule_type": "warmup_cosine_decay", "init_value": 0, "peak_value": tr["peak_lr"],
        "warmup_steps": tr["warmup"] * total_steps, "decay_steps": total_steps}))

    os.environ["FLOW_KEEP_CHECKPOINTS"] = str(tr["keep"])
    mgr = make_checkpoint_manager(run / "checkpoints", clear_dir=True, max_to_keep=tr["keep"])
    log = {"train_losses": [], "val_losses": [], "train_times_s": [], "val_times_s": []}
    try:
        for epoch in range(tr["epochs"]):
            t0 = _time.perf_counter()
            conditioner.train(); train_metrics.reset()
            for data, context in train_batches:
                train_step(conditioner, optimizer, train_metrics, data, context)
            train_loss = float(train_metrics.compute()["loss"])
            t1 = _time.perf_counter()
            conditioner.eval(); val_metrics.reset()
            for data, context in val_batches:
                eval_step(conditioner, val_metrics, data, context)
            val_loss = float(val_metrics.compute()["loss"])
            t2 = _time.perf_counter()
            log["train_losses"].append(train_loss); log["val_losses"].append(val_loss)
            log["train_times_s"].append(t1 - t0); log["val_times_s"].append(t2 - t1)
            say(f"epoch {epoch + 1}/{tr['epochs']}  train {train_loss:.5f}  val {val_loss:.5f}  "
                f"({t1 - t0:.1f}s + {t2 - t1:.1f}s)")
            if not (np.isfinite(train_loss) and np.isfinite(val_loss)):
                say("loss is not finite; stopping")
                break
            save_checkpoint(mgr, epoch, conditioner, optimizer, val_loss)
            (run / "training_log.json").write_text(json.dumps(
                {**log, "total_time_s": sum(log["train_times_s"]) + sum(log["val_times_s"])}, indent=1))
    finally:
        mgr.wait_until_finished()
    say(f"[{_time.strftime('%H:%M:%S')}] done in {(_time.time() - t_start) / 60:.1f} min -> {run}")
    return run

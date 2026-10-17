"""`nleflow train <spec>`: the flow training of the NLE project's
train_race.py (race specs) and train_ddm.py (choice specs), driven by a spec.

The run directory `<runs>/<name>` holds spec.json (the resolved spec),
scaler.json, training_log.json and checkpoints/ (Orbax; the `keep` latest
epochs plus the best validation loss). The loss, optimiser, schedule and
checkpointing are the scripts': mean negative log density of the spline
flow over rows with a finite positive time, AdamW under a warmup-cosine
schedule, one checkpoint per epoch, no early stopping. A choice model's flow
is conditioned on (standardised context, R) with R appended raw (1.0 or 2.0);
its choice classifier (classifier.json, classifier_log.json) is trained
afterwards on the exact P(R = 2) targets of the choice sets, with the same
standardiser, as retrain_classifier.py's tuned configuration does.
"""
from __future__ import annotations

import json
import os
import shutil
import time as _time
from pathlib import Path

import numpy as np

from .data import Batches, Scaler, encode_context, load_choice_set, load_response, load_table, open_dataset
from .spec import dump_json, flow_context

MIN_LOG_PROB = 1e-12


def run_dir_for(spec, runs):
    return Path(runs).resolve() / spec["name"]


def build_conditioner(spec, seed):
    import jax
    from flax import nnx
    from .flow import make_mlp_conditioner
    dims = [len(flow_context(spec)) + (1 if spec["response"] == "choice" else 0)] + list(spec["conditioner"]["widths"])
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


def _flow_context_array(tbl, spec, scaler):
    """The flow's context: the standardised encoded context, plus R (raw)
    for a choice model."""
    time, ctx = encode_context(tbl, spec)
    std = scaler.transform(ctx)
    if spec["response"] == "choice":
        R = load_response(tbl).astype(np.float64)
        std = np.concatenate([std, R[:, None]], axis=1)
    return time, std


def train(spec, runs="runs", epochs=None, cores=None, quiet=False, classifier_from=None):
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
    choice = spec["response"] == "choice"

    t_start = _time.time()
    train_tbl = load_table(open_dataset(spec["data"]["train"]), spec)
    val_tbl = load_table(open_dataset(spec["data"]["val"]), spec)
    names = flow_context(spec)
    _, ctx_train = encode_context(train_tbl, spec)
    scaler = Scaler.fit(ctx_train, names)
    scaler.save(run / "scaler.json")
    t_train, c_train = _flow_context_array(train_tbl, spec, scaler)
    t_val, c_val = _flow_context_array(val_tbl, spec, scaler)
    rng = None if tr["shuffle_seed"] is None else np.random.default_rng(tr["shuffle_seed"])
    train_batches = Batches(t_train, c_train, tr["batch"], shuffle=True, rng=rng)
    val_batches = Batches(t_val, c_val, tr["val_batch"], shuffle=False)
    say(f"[{_time.strftime('%H:%M:%S')}] {spec['name']}: train {train_batches.n} rows, "
        f"val {val_batches.n} rows, context {names}{' + R' if choice else ''}, "
        f"widths {spec['conditioner']['widths']}, bins {spec['spline']['bins']}, "
        f"range {spec['spline']['range']}, batch {tr['batch']}, epochs {tr['epochs']}"
        + ("  affine" if spec["conditioner"]["affine"] else "")
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
    say(f"[{_time.strftime('%H:%M:%S')}] flow done in {(_time.time() - t_start) / 60:.1f} min -> {run}")

    if choice:
        if classifier_from:
            copy_classifier(classifier_from, run, scaler, say)
        else:
            train_classifier(spec, run, scaler, say)
    return run


# --- the choice classifier -----------------------------------------------------

def classifier_layers(clf):
    return [{"W": np.asarray(l.kernel.value, dtype=np.float64).tolist(),
             "b": np.asarray(l.bias.value, dtype=np.float64).tolist()} for l in clf.layers]


def copy_classifier(src_run, run, scaler, say=print):
    """Reuse another run's classifier: valid only when the two runs standardise
    alike (the classifier takes the flow's standardised context)."""
    src_run = Path(src_run).resolve()
    other = Scaler.load(src_run / "scaler.json")
    dm = float(np.max(np.abs(other.mean - scaler.mean))); ds = float(np.max(np.abs(other.scale - scaler.scale)))
    if other.names != scaler.names or dm > 1e-3 or ds > 1e-3:
        raise ValueError(f"{src_run.name}'s standardiser differs from this run's (max |dmean| {dm:.2e}, "
                         f"|dscale| {ds:.2e}): its classifier cannot be reused")
    shutil.copy(src_run / "classifier.json", run / "classifier.json")
    log = json.loads((src_run / "classifier_log.json").read_text())
    log["copied_from"] = str(src_run)
    (run / "classifier_log.json").write_text(json.dumps(log, indent=1))
    say(f"classifier copied from {src_run}")


def train_classifier(spec, run, scaler, say=print):
    """P(R = 2 | context) on the exact probabilities of the choice sets:
    retrain_classifier.py's recipe (sigmoid output, MSE on the probability or
    the hybrid MSE + 0.1 BCE, AdamW under warmup-cosine)."""
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from .flow import make_choice_classifier

    cl = spec["classifier"]
    t_start = _time.time()
    Xtr, ytr = load_choice_set(spec["data"]["choice_train"], spec)
    Xva, yva = load_choice_set(spec["data"]["choice_val"], spec)
    Xtr = scaler.transform(Xtr).astype(np.float32); Xva = scaler.transform(Xva).astype(np.float32)
    ytr = ytr.astype(np.float32); yva = yva.astype(np.float32)
    dims = [Xtr.shape[1]] + list(cl["widths"])
    clf = make_choice_classifier(rngs=nnx.Rngs(default=jax.random.key(cl["seed"])), dims=dims)
    n, bs = len(Xtr), cl["batch"]
    if n < bs:
        raise ValueError(f"the choice set has {n} rows, fewer than one classifier batch of {bs}")
    steps = cl["epochs"] * (n // bs)
    sched = optax.warmup_cosine_decay_schedule(0.0, cl["lr"], warmup_steps=max(1, steps // 10), decay_steps=steps)
    opt = nnx.Optimizer(clf, optax.adamw(sched), wrt=nnx.Param)
    Xs = jnp.asarray(Xtr); ys = jnp.asarray(ytr)
    loss_kind = cl["loss"]

    def loss_fn(m, xb, yb):
        logit = m(xb)
        p = jax.nn.sigmoid(logit)
        mse = jnp.mean((p - yb) ** 2)
        bce = optax.sigmoid_binary_cross_entropy(logit, yb).mean()
        if loss_kind == "mse":
            return mse
        if loss_kind == "hybrid":
            return mse + 0.1 * bce
        return bce

    @nnx.jit
    def step(m, o, xb, yb):
        l, g = nnx.value_and_grad(loss_fn)(m, xb, yb)
        o.update(g)
        return l

    rng = np.random.default_rng(cl["seed"])
    say(f"classifier: {n} rows, dims {dims}, loss {loss_kind}, batch {bs}, epochs {cl['epochs']}")
    hist = []
    for ep in range(cl["epochs"]):
        clf.train()
        idx = rng.permutation(n)
        for s0 in range(0, n - bs + 1, bs):
            sl = jnp.asarray(idx[s0:s0 + bs])
            step(clf, opt, Xs[sl], ys[sl])
        if (ep + 1) % 40 == 0 or ep + 1 == cl["epochs"]:
            clf.eval()
            pv = np.asarray(jax.nn.sigmoid(clf(jnp.asarray(Xva))))
            err = float(np.abs(pv - yva).mean())
            hist.append({"epoch": ep + 1, "val_mean_abs_err": err})
            say(f"  classifier epoch {ep + 1:4d}  val mean |P_hat - P| {err:.6f}")
    clf.eval()
    pv = np.asarray(jax.nn.sigmoid(clf(jnp.asarray(Xva))))
    err = pv - yva
    log = {"dims": dims, "loss": loss_kind, "epochs": cl["epochs"], "batch": bs, "lr": cl["lr"],
           "seed": cl["seed"], "n_train": int(n), "n_val": int(len(yva)),
           "val_mean_abs_err": float(np.abs(err).mean()), "val_signed_err": float(err.mean()),
           "val_max_abs_err": float(np.abs(err).max()), "history": hist,
           "time_s": _time.time() - t_start}
    (run / "classifier.json").write_text(json.dumps({"activation": "gelu_tanh", "layers": classifier_layers(clf)}))
    (run / "classifier_log.json").write_text(json.dumps(log, indent=1))
    say(f"classifier done: val mean |err| {log['val_mean_abs_err']:.6f}, max {log['val_max_abs_err']:.4f} "
        f"in {log['time_s']:.0f}s")

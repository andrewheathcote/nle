"""Optuna study for the DDM choice classifier.

Why this component. The classifier supplies P(R | theta) in the DDM
factorisation P(R|theta) * f(rt|theta,R). It has never been retrained since
July - every flow run copies the same checkpoint - while the flow itself
improved 2.5x. Measured against the analytic choice probability it carries a
systematic bias of +0.0089 in P(R=2), consistent across all quartiles of drawn
v. Because v governs choice proportions, over-predicting P(R=2) forces the
fitted v DOWN, which is the sign and roughly half the magnitude of the v
residual - now the only parameter outside the G1 gate.

Two changes are tested together:
  * TARGET. The shipped loss is binary cross-entropy on simulated R labels,
    i.e. learning a probability from Bernoulli draws (originally two per
    theta). P(R|theta) is available EXACTLY as the defective CDF at infinity
    (WienR::pWDM), so `mse`/`hybrid` regress on the true probability instead.
  * ARCHITECTURE + OPTIMISER, never tuned.

Objective is mean |P_hat - P_true| on held-out theta - the quantity that
biases v, not a proxy like cross-entropy. The classifier has ~833 parameters
against the flow's 60,000, so a trial costs seconds rather than hours; that is
what makes BO worthwhile here and not for the flow architecture.

  uv run python scripts/tune_classifier.py --trials 100
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import jax, jax.numpy as jnp, optax, optuna
from flax import nnx

from flow.config import make_choice_classifier

REPO = Path(__file__).resolve().parents[2]
CONTEXT = ["v", "a", "t0", "s", "Z", "SZ", "sv", "st0"]


def load(split):
    t = pq.read_table(REPO / "data" / "choice_prob" / f"{split}.parquet").to_pandas()
    X = t[CONTEXT].to_numpy(np.float32)
    y = t["p_R2"].to_numpy(np.float32)
    return X, y


def run(trial, Xtr, ytr, Xva, yva, mu, sd, seed=0):
    depth = trial.suggest_int("depth", 1, 4)
    width = trial.suggest_categorical("width", [16, 32, 64, 128, 256])
    taper = trial.suggest_categorical("taper", [True, False])
    lr = trial.suggest_float("lr", 1e-4, 3e-2, log=True)
    bs = trial.suggest_categorical("batch", [256, 1024, 4096, 16384])
    epochs = trial.suggest_int("epochs", 20, 200)
    loss_kind = trial.suggest_categorical("loss", ["bce", "mse", "hybrid"])

    dims = [len(CONTEXT)] + [max(8, width // (2 ** i) if taper else width)
                             for i in range(depth)]
    clf = make_choice_classifier(rngs=nnx.Rngs(default=jax.random.key(seed)), dims=dims)
    sched = optax.warmup_cosine_decay_schedule(
        0.0, lr, warmup_steps=max(1, int(0.1 * epochs * len(Xtr) // bs)),
        decay_steps=max(1, epochs * len(Xtr) // bs))
    opt = nnx.Optimizer(clf, optax.adamw(sched), wrt=nnx.Param)

    Xs = jnp.asarray((Xtr - mu) / sd); ys = jnp.asarray(ytr)

    def loss_fn(m, xb, yb):
        logit = m(xb)
        if loss_kind == "bce":                      # learn from Bernoulli draws
            draw = (yb > jax.random.uniform(jax.random.key(0), yb.shape)).astype(jnp.float32)
            return optax.sigmoid_binary_cross_entropy(logit, draw).mean()
        p = jax.nn.sigmoid(logit)
        mse = jnp.mean((p - yb) ** 2)
        if loss_kind == "mse":
            return mse
        return mse + 0.1 * optax.sigmoid_binary_cross_entropy(logit, yb).mean()

    @nnx.jit
    def step(m, o, xb, yb):
        l, g = nnx.value_and_grad(loss_fn)(m, xb, yb)
        o.update(g); return l

    rng = np.random.default_rng(seed)
    n = len(Xs)
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n - bs + 1, bs):
            sl = jnp.asarray(idx[s:s + bs])
            step(clf, opt, Xs[sl], ys[sl])

    clf.eval()
    pv = np.asarray(jax.nn.sigmoid(clf(jnp.asarray((Xva - mu) / sd))))
    err = pv - yva
    trial.set_user_attr("mean_signed_err", float(err.mean()))
    trial.set_user_attr("max_abs_err", float(np.abs(err).max()))
    trial.set_user_attr("n_params", int(sum(p.size for p in
                        jax.tree_util.tree_leaves(nnx.state(clf, nnx.Param)))))
    return float(np.abs(err).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--out", default="results/classifier_tuning")
    a = ap.parse_args()
    Xtr, ytr = load("train"); Xva, yva = load("val")
    # Standardise with the FLOW's scaler, not this set's own moments. The
    # classifier must consume context exactly as the flow does, or the tuned
    # weights cannot be swapped into an existing export (the exporter asserts
    # the two scalers match, and a 0.06% mismatch has already tripped it once).
    import sys; sys.path.insert(0, str(REPO / "flow" / "src"))
    from flow.training import load_obj
    sc = load_obj(str(REPO / "results/model_states/large_budget/DDM_flow/train_loader.pkl"))._scaler
    mu, sd = sc.mean_.astype(np.float32), sc.scale_.astype(np.float32)
    print("using flow scaler; scale =", np.round(sd, 4), flush=True)
    print(f"train {Xtr.shape}  val {Xva.shape}", flush=True)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(lambda t: run(t, Xtr, ytr, Xva, yva, mu, sd),
                   n_trials=a.trials, show_progress_bar=False)

    out = REPO / a.out; out.mkdir(parents=True, exist_ok=True)
    rows = [{"value": t.value, **t.params, **t.user_attrs} for t in study.trials
            if t.value is not None]
    json.dump({"best_value": study.best_value, "best_params": study.best_params,
               "best_attrs": study.best_trial.user_attrs, "trials": rows},
              open(out / "study.json", "w"), indent=2)
    print("\nBEST mean|P_hat - P_true| = %.6f" % study.best_value)
    print("  signed bias %+.6f   max|err| %.4f   params %d"
          % (study.best_trial.user_attrs["mean_signed_err"],
             study.best_trial.user_attrs["max_abs_err"],
             study.best_trial.user_attrs["n_params"]))
    print("  params:", study.best_params)


if __name__ == "__main__":
    main()

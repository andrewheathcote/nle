"""Direct-regression baseline: an MLP from (theta, R, rt) straight to log-density.

The obvious reviewer question about the flow+classifier machinery is "why not
just learn the likelihood directly?" This is that alternative, built so the
question can be answered with a measurement instead of an argument.

What it gives up, and why those matter:
  - NORMALISATION. A flow integrates to 1 by construction; this does not. A
    theta-DEPENDENT normalisation error Z(theta) makes the effective likelihood
    f_hat - log Z(theta), displacing the posterior by grad log Z — which is
    exactly a score bias, so our existing estimator detects it. A constant
    offset is harmless for inference; a varying one is not.
  - THE CDF. A flow yields Phi(z) alongside the density, so survival functions
    are free. Race models need them structurally (f_winner * prod S_loser), and
    censoring needs them too. A regression net would need a second network with
    nothing enforcing dF/drt = f.
  - TRAINING WITHOUT AN ANALYTIC. A flow learns by MLE from samples. This
    requires the analytic likelihood, so it can only ever compress something you
    already have — fine for a fast artefact, useless for an intractable model.

What it gains: no classifier at all (it predicts the JOINT density directly),
one forward pass at inference, and a plain regression fit.

Capacity is matched to what it replaces: the flow conditioner (~60k params) plus
the c4 classifier (~134k) is ~194k, so the default here is comparable.

  uv run python scripts/train_regress.py --tag reg_b1k
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import jax
import jax.numpy as jnp
import optax
from flax import nnx

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flow.training import save_obj

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTEXT = ["v", "a", "t0", "s", "Z", "SZ", "sv", "st0"]
# 7-parameter st0 = 0 build; see handover/ST0_ZERO_RETRAIN.md
CONTEXT_ST0ZERO = ["v", "a", "t0", "s", "Z", "SZ", "sv"]
LL_FLOOR = float(np.log(1e-10))


class MLP(nnx.Module):
    def __init__(self, dims, rngs):
        self.layers = [nnx.Linear(dims[i], dims[i + 1], rngs=rngs)
                       for i in range(len(dims) - 1)]

    def __call__(self, x):
        for i, lay in enumerate(self.layers):
            x = lay(x)
            if i < len(self.layers) - 1:
                x = nnx.gelu(x)
        return x.squeeze(-1)


def load_split(path, scaler=None, ctx=CONTEXT, target="ll"):
    """Features are [standardised theta, R, standardised log rt]; target is ll.

    Rows whose ll sits at the min_ll floor carry no information about the
    density — they mark where the analytic underflowed — and including them
    would drag the fit toward a constant. They are dropped.
    """
    t = ds.dataset(path, format="parquet").to_table(
        columns=ctx + ["R", "rt", target]).to_pandas()
    y = t[target].to_numpy()
    keep = np.isfinite(y) if target != "ll" else (np.isfinite(y) & (y > LL_FLOOR + 1e-9))
    t = t[keep]
    th = t[ctx].to_numpy(np.float32)
    lrt = np.log(t["rt"].to_numpy(np.float32))[:, None]
    R = t["R"].to_numpy(np.float32)[:, None]
    raw = np.concatenate([th, lrt], axis=1)
    if scaler is None:
        scaler = (raw.mean(0), raw.std(0))
    X = np.concatenate([(raw - scaler[0]) / scaler[1], R], axis=1)
    return X.astype(np.float32), t[target].to_numpy(np.float32), scaler, int(keep.sum()), int(len(keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train-data", default="data/train/tpt40p400k/DDM/trials")
    ap.add_argument("--val-data", default="data/val/DDM/trials")
    ap.add_argument("--widths", default="256,256,256")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--peak-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--st0-zero", action="store_true",
                    help="7-parameter build with st0 absent")
    ap.add_argument("--target", default="ll", choices=["ll", "cdf"],
                    help="ll = log density; cdf = defective CDF")
    a = ap.parse_args()

    CTX = CONTEXT_ST0ZERO if a.st0_zero else CONTEXT
    Xtr, ytr, scaler, kept, tot = load_split(REPO_ROOT / a.train_data, ctx=CTX, target=a.target)
    Xva, yva, _, _, _ = load_split(REPO_ROOT / a.val_data, scaler, ctx=CTX, target=a.target)
    print(f"[{time.strftime('%H:%M:%S')}] tag={a.tag}  train={kept:,}/{tot:,} rows "
          f"({100*(tot-kept)/tot:.2f}% at the ll floor, dropped)  "
          f"features={Xtr.shape[1]}  batch={a.batch}  epochs={a.epochs}", flush=True)

    widths = [int(w) for w in a.widths.split(",")]
    dims = [Xtr.shape[1]] + widths + [1]
    rngs = nnx.Rngs(default=jax.random.key(a.seed))
    net = MLP(dims, rngs)
    nparam = sum(int(np.prod(l.kernel.value.shape)) + l.bias.value.shape[0]
                 for l in net.layers)
    print(f"  dims {dims}  parameters {nparam:,} "
          f"(flow conditioner ~60k + c4 classifier ~134k = ~194k)", flush=True)

    steps = a.epochs * (len(Xtr) // a.batch)
    sched = optax.warmup_cosine_decay_schedule(0.0, a.peak_lr,
                                               warmup_steps=max(1, steps // 10),
                                               decay_steps=steps)
    opt = nnx.Optimizer(net, optax.adamw(sched), wrt=nnx.Param)

    def loss_fn(m, xb, yb):
        return jnp.mean((m(xb) - yb) ** 2)

    @nnx.jit
    def step(m, o, xb, yb):
        l, g = nnx.value_and_grad(loss_fn)(m, xb, yb)
        o.update(g)
        return l

    Xtr_j, ytr_j = jnp.asarray(Xtr), jnp.asarray(ytr)
    Xva_j, yva_j = jnp.asarray(Xva), jnp.asarray(yva)
    rng = np.random.default_rng(a.seed)
    out = REPO_ROOT / "results" / "model_states" / f"{a.tag}_budget"
    out.mkdir(parents=True, exist_ok=True)
    best = np.inf
    t0 = time.time()
    for ep in range(a.epochs):
        idx = rng.permutation(len(Xtr_j))
        for s in range(0, len(idx) - a.batch + 1, a.batch):
            sl = jnp.asarray(idx[s:s + a.batch])
            step(net, opt, Xtr_j[sl], ytr_j[sl])
        vl = float(jnp.mean((net(Xva_j) - yva_j) ** 2))
        if vl < best:
            best = vl
            _payload = {"dims": dims,
                      "layers": [{"W": np.asarray(l.kernel.value).tolist(),
                                  "b": np.asarray(l.bias.value).tolist()}
                                 for l in net.layers],
                      "scaler_mean": scaler[0].tolist(),
                      "scaler_scale": scaler[1].tolist(),
                      "context": CTX, "target": a.target, "epoch": ep + 1}
            save_obj(_payload, str(out / "regress.pkl"))
            # JSON too, so the R side can evaluate it with the SAME score-bias
            # estimator the flows use. A parallel Python implementation would
            # make the comparison a comparison of estimators as well as models.
            with open(out / "regress.json", "w") as _f:
                json.dump(_payload, _f)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  val MSE {vl:.6f}  (best {best:.6f})", flush=True)
    with open(out / "variant_config.json", "w") as f:
        json.dump(vars(a), f)
    print(f"[{time.strftime('%H:%M:%S')}] done in {(time.time()-t0)/3600:.2f} h  "
          f"best val MSE {best:.6f} -> {out}", flush=True)


if __name__ == "__main__":
    main()

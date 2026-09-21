"""Short-run sweep of the score-penalty weight.

Trains the DDM flow for a few epochs at one --score-lambda, then reports, on
held-out validation data, the two quantities that matter separately:

  val_nll     plain negative log-likelihood (loss_fn) — density fit
  val_score   the squared score-bias estimator (score_penalty) — the thing the
              penalty targets, and the thing that displaces posteriors

The training objective mixes them (nll + lam*score), so the logged loss is NOT
comparable across lambda; these two are. lam=0 is the control.

Single-threaded in practice (batch=128 matmuls are tiny), so run the arms
concurrently rather than limiting cores.

  uv run python scripts/sweep_score_lambda.py --score-lambda 1.0 --epochs 10
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import pyarrow.dataset as ds
import jax, jax.numpy as jnp, optax
from flax import nnx

from flow.config import make_mlp_conditioner
from flow.training import train_model, loss_fn, score_penalty
from flow.data import PreloadedBatches

REPO = Path(__file__).resolve().parents[2]
DATA, TARGET, RESPONSE = "rt", "ll", "R"
CONTEXT = ["v", "a", "t0", "s", "Z", "SZ", "sv", "st0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-lambda", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num-bins", type=int, default=64)
    ap.add_argument("--widths", default="64,32")
    ap.add_argument("--peak-lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", default="sweep_score_lambda.jsonl")
    a = ap.parse_args()
    lam = a.score_lambda
    use_pen = lam > 0

    dset_train = ds.dataset(REPO / "data/train/large/DDM/trials", format="parquet")
    dset_val = ds.dataset(REPO / "data/val/DDM/trials", format="parquet")
    widths = [int(w) for w in a.widths.split(",")]

    rngs = nnx.Rngs(default=jax.random.key(a.seed),
                    sampling=jax.random.key(a.seed + 1))
    cond = make_mlp_conditioner(dims=[len(CONTEXT) + 1] + widths,
                                num_bins=a.num_bins, num_splines=1, rngs=rngs,
                                use_norm=False, range_min=-5.0, range_max=5.0)
    steps = (dset_train.count_rows() // a.batch) * a.epochs
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0., peak_value=a.peak_lr,
        warmup_steps=0.1 * steps, decay_steps=steps)
    opt = nnx.Optimizer(cond, optax.adamw(learning_rate=sched), wrt=nnx.Param)

    largs = dict(data_names=DATA, context_names=CONTEXT + [RESPONSE],
                 target_names=TARGET, response_name=RESPONSE, standardize=True)
    # pairs always: the evaluator needs them even when training without the penalty
    tl = PreloadedBatches(dset_train, batch_size=a.batch, pair_rows=True,
                          shuffle=True, **largs)
    vl = PreloadedBatches(dset_val, batch_size=256, pair_rows=True,
                          shuffle=False, scaler=tl._scaler, **largs)

    t0 = time.time()
    train_model(cond, opt, nnx.MultiMetric(loss=nnx.metrics.Average("loss")),
                nnx.MultiMetric(loss=nnx.metrics.Average("loss")),
                tl, vl, dset_train=dset_train, dset_val=dset_val,
                batch_size_train=a.batch, batch_size_val=256,
                num_epochs=a.epochs,
                score_lambda=(lam if use_pen else None),
                n_theta=len(CONTEXT),
                checkpoint_path=None, early_stopping=False)
    train_s = time.time() - t0

    cond.eval()
    nlls, pens, n = [], [], 0
    for data, ctx, _ in vl:
        nlls.append(float(loss_fn(cond, data, ctx)))
        pens.append(float(score_penalty(cond, data, ctx, len(CONTEXT))))
        n += 1
    rec = dict(score_lambda=lam, epochs=a.epochs, seed=a.seed,
               val_nll=sum(nlls) / n, val_score=sum(pens) / n,
               train_s=train_s, s_per_epoch=train_s / a.epochs, n_val_batches=n)
    print(json.dumps(rec), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()

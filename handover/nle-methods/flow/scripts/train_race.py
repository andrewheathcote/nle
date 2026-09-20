"""Train a race-model flow (RDM, LNR): one accumulator's first-passage density.

Unlike the DDM there is NO choice classifier. A race likelihood is assembled at
inference as (winner density) x (survivors), so the network models a single
accumulator's f(rt | theta) and nothing else. That also means none of the
classifier work from the DDM transfers — but the score-bias diagnostic, the
checkpoint-screening recipe and the batch-size finding all do.

Defaults encode what the DDM work established, so the RDM does not have to
repeat the search:
  --batch 1024       the only change found to shift the MEAN of the score bias
                     (Phase 16), and 3x faster per epoch than 128
  --widths 256,128   the RDM's existing flow is a single 5->32 layer; the DDM
                     needed far more conditioner capacity before spline bins
                     ever bound
  --num-bins 64      likewise
  --keep-checkpoints 30
                     roughly half the apparent "seed" variance is epoch-to-epoch
                     variance on the plateau (Phase 14), so retaining a window
                     and SELECTING on score bias is ~5x cheaper than training
                     multiple seeds. Never average them (Phase 14b).

  uv run python scripts/train_race.py --model RDM --tag rdm_b1k \
      --train-data data/train/rdm16m/RDM/trials
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pyarrow.dataset as ds
import jax
import optax
from flax import nnx

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flow.config import make_mlp_conditioner
from flow.data import PreloadedBatches
from flow.training import train_model, save_obj

REPO_ROOT = Path(__file__).resolve().parents[2]

# Context columns per race model, matching simulation/designs.R and the
# exporter's expectations (flow/scripts/export_flow.py).
# MUST match export_flow.py's MODEL_REGISTRY[model]["context"] exactly. The
# exporter writes its own order into context_names, and the port builds theta
# from that — so a mismatch silently permutes the inputs. Training with
# ["v","B","t0","A","s"] against an exporter declaring ["v","B","t0","s","A"]
# (A and s transposed) produced a score bias ~500x rdm_large's, uniform across
# checkpoints. The network was fine; only the metadata disagreed.
CONTEXT = {
    "RDM": ["v", "B", "t0", "s", "A"],
    "LNR": ["m", "s", "t0"],
}
DATA = "rt"
TARGET = "ll"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="RDM", choices=sorted(CONTEXT))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train-data", default="data/train/rdm16m/RDM/trials")
    ap.add_argument("--val-data", default=None,
                    help="defaults to data/val/<model>_single/trials")
    ap.add_argument("--num-bins", type=int, default=64)
    ap.add_argument("--widths", default="256,128")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--peak-lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--range-min", type=float, default=-5.0)
    ap.add_argument("--range-max", type=float, default=5.0)
    ap.add_argument("--keep-checkpoints", type=int, default=30)
    args = ap.parse_args()
    os.environ["FLOW_KEEP_CHECKPOINTS"] = str(args.keep_checkpoints)

    CTX = CONTEXT[args.model]
    val_data = args.val_data or f"data/val/{args.model}_single/trials"

    out_base = REPO_ROOT / "results" / "model_states" / f"{args.tag}_budget"
    # matches export_flow.py's checkpoint_head: <tag>_budget/<MODEL>/
    flow_dir = out_base / args.model
    out_base.mkdir(parents=True, exist_ok=True)

    dset_train = ds.dataset(REPO_ROOT / args.train_data, format="parquet")
    dset_val = ds.dataset(REPO_ROOT / val_data, format="parquet")
    n_train = dset_train.count_rows()
    widths = [int(w) for w in args.widths.split(",")]
    dims = [len(CTX)] + widths            # no +1: a race flow has no response code
    print(f"[{time.strftime('%H:%M:%S')}] {args.model} tag={args.tag}  "
          f"train={n_train} rows  context={CTX}  dims={dims}  "
          f"bins={args.num_bins}  batch={args.batch}  epochs={args.epochs}",
          flush=True)

    rngs = nnx.Rngs(default=jax.random.key(args.seed),
                    sampling=jax.random.key(args.seed + 1))
    conditioner = make_mlp_conditioner(dims=dims, num_bins=args.num_bins,
                                       num_splines=1, rngs=rngs, use_norm=False,
                                       range_min=args.range_min,
                                       range_max=args.range_max)

    total_steps = (n_train // args.batch) * args.epochs
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0., peak_value=args.peak_lr,
        warmup_steps=0.1 * total_steps, decay_steps=total_steps)
    optimizer = nnx.Optimizer(conditioner, optax.adamw(schedule), wrt=nnx.Param)
    train_metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    val_metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

    loader_args = dict(data_names=DATA, context_names=CTX,
                       target_names=TARGET, response_name=None,
                       standardize=True)
    train_loader = PreloadedBatches(dset_train, batch_size=args.batch,
                                    shuffle=True, **loader_args)
    val_loader = PreloadedBatches(dset_val, batch_size=256, shuffle=False,
                                  scaler=train_loader._scaler, **loader_args)

    # out_base, not flow_dir: the checkpointer WIPES flow_dir when train_model
    # starts, so a scaler written there beforehand is deleted and the run cannot
    # be exported afterwards.
    save_obj(train_loader, str(out_base / "train_loader.pkl"))
    with open(out_base / "schedule_config.json", "w") as f:
        json.dump({"schedule_type": "warmup_cosine_decay", "init_value": 0,
                   "peak_value": args.peak_lr,
                   "warmup_steps": 0.1 * total_steps,
                   "decay_steps": total_steps}, f)
    with open(out_base / "variant_config.json", "w") as f:
        json.dump(vars(args), f)

    t0 = time.time()
    train_model(conditioner, optimizer, train_metrics, val_metrics,
                train_loader, val_loader,
                batch_size_train=args.batch, batch_size_val=256,
                dset_train=dset_train, dset_val=dset_val,
                num_epochs=args.epochs,
                data_names=DATA, context_names=CTX,
                target_names=TARGET, response_name=None,
                checkpoint_path=str(flow_dir), early_stopping=False)
    print(f"[{time.strftime('%H:%M:%S')}] training done in "
          f"{(time.time() - t0) / 3600:.2f} h -> {out_base}", flush=True)


if __name__ == "__main__":
    main()

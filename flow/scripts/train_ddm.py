"""Unattended DDM flow training with configurable architecture.

Trains the rt spline flow (the DDM_flow head) on existing training data,
mirroring the notebook's large-budget DDM section exactly except for the
architecture knobs exposed here (spline bins, conditioner widths). The
choice classifier is architecture-independent of the flow; by default the
existing large-budget classifier checkpoint is COPIED into the output tag
so the manipulation is isolated to the flow (pass --train-classifier to
retrain it instead).

Outputs the canonical layout the exporter/timing/analysis code read:

    results/model_states/<tag>_budget/DDM_flow/        (+ train_loader.pkl,
                                                        schedule_config.json,
                                                        training_log.json)
    results/model_states/<tag>_budget/DDM_classifier/  (copied or trained)

Run from `flow/` (nohup-safe; prints one line per epoch via train_model):
  uv run python scripts/train_ddm.py --tag arch64 --num-bins 64
  uv run python scripts/train_ddm.py --tag arch64w --num-bins 64 --widths 64,32
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import pyarrow.dataset as ds
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from flow.config import make_mlp_conditioner
from flow.training import train_model, save_obj, load_obj
from flow.data import PreloadedBatches

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA = "rt"
CONTEXT = ["v", "a", "t0", "s", "Z", "SZ", "sv", "st0"]
# Andrew's (SZ0, SZ, SZ1) reparameterisation: a = SZ0+SZ+SZ1, w = (SZ0+SZ/2)/a,
# sw = SZ/a. Same parameter count, same data, same likelihoods - only the
# conditioning representation changes, so the conditioner no longer has to
# represent EMC2's Ttransform kink at Z = 0.5. Select with --triple.
CONTEXT_TRIPLE = ["v", "SZ0", "t0", "s", "SZw", "SZ1", "sv", "st0"]
# st0 = 0 build: a 7-parameter DDM with no non-decision-time variability.
# EMC2 gives st0 a default of -Inf on the sampled scale under an exp transform,
# so omitting st0 from the design formula sets it to exactly 0 (verified via
# mapped_pars). The flow's existing box cannot reach st0 = 0 -- its floor is
# 0.01, since log(0) = -Inf -- so this needs its own training data and its own
# network. See handover/ST0_ZERO_RETRAIN.md.
CONTEXT_ST0ZERO = ["v", "a", "t0", "s", "Z", "SZ", "sv"]
TARGET = "ll"
RESPONSE = "R"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="output tag: results/model_states/<tag>_budget/")
    ap.add_argument("--train-data", default="data/train/large/DDM/trials",
                    help="training trials dir, relative to repo root")
    ap.add_argument("--val-data", default="data/val/DDM/trials")
    ap.add_argument("--num-bins", type=int, default=32)
    ap.add_argument("--widths", default="32,16",
                    help="hidden widths of the flow conditioner MLP")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--peak-lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--score-lambda", type=float, default=None,
                    help="weight on the score-bias penalty (see "
                         "flow.training.score_penalty). Omit for plain MLE. "
                         "Requires 2 trials per parameter draw in the training "
                         "data; the loader is switched to pair-preserving "
                         "shuffling automatically. Start with 1.0 and sweep.")
    ap.add_argument("--distill-beta", type=float, default=None,
                    help="weight on the analytic-log-density MSE term (see "
                         "flow.training.loss_fn_distill). BENCHMARK ONLY - it "
                         "uses the analytic likelihood, so it is not a method "
                         "for models without one; it bounds what this "
                         "architecture can achieve. Omit for plain MLE.")
    ap.add_argument("--edge-amp", type=float, default=None,
                    help="upweight the leading edge by 1 + amp*exp(-u/scale), "
                         "u = (rt-t0)/st0. Phase 13: the flow over-predicts "
                         "density inside the st0 smear by up to +0.12 log, and "
                         "that region carries ~6%% of the mass but ~0.1%% of the "
                         "log-likelihood, so plain MLE cannot prioritise it. "
                         "Try 20-50.")
    ap.add_argument("--edge-scale", type=float, default=1.0,
                    help="decay of the edge weight in units of st0 (default 1)")
    ap.add_argument("--keep-checkpoints", type=int, default=None,
                    help="how many checkpoints to retain (default 1 = best-val "
                         "only). Raise it to study within-run variation or to "
                         "average weights across late epochs; a [256,128] "
                         "checkpoint is ~1.5 MB.")
    ap.add_argument("--st0-zero", action="store_true",
                    help="7-parameter build with st0 absent (see "
                         "handover/ST0_ZERO_RETRAIN.md); requires training data "
                         "simulated with st0 = 0")
    ap.add_argument("--triple", action="store_true",
                    help="train on the (SZ0, SZ, SZ1) reparameterisation "
                         "instead of (a, Z, SZ); requires a dataset written "
                         "by simulation/reencode_triple.py")
    ap.add_argument("--geometry", action="store_true",
                    help="append a*Z and a*(1-Z) to the context - the start "
                         "point's distance to each boundary. Tests whether the "
                         "shared a/Z residual is the conditioner failing to "
                         "form that multiplicative interaction.")
    ap.add_argument("--score-match", type=float, default=None,
                    help="weight on the MSE between the flow's score and the "
                         "TRUE score (see flow.training.loss_fn_scorematch). "
                         "Requires training data with score_<par> columns "
                         "(simulation/sim_score_data.R).")
    ap.add_argument("--huber-delta", type=float, default=1.0,
                    help="Huber threshold for --score-match, in z units. "
                         "Scores of t0/st0 reach ~250 near the support "
                         "boundary against a median ~1, so squared error is "
                         "dominated by <1%% of rows.")
    ap.add_argument("--grid-data", default=None,
                    help="extra OFF-SAMPLE rows with analytic ll, concatenated "
                         "with --train-data. These are not model draws, so "
                         "they feed the distillation MSE only, never the NLL. "
                         "Use with --distill-beta.")
    ap.add_argument("--scaler-from", default=None,
                    help="tag whose context standardiser to reuse (e.g. "
                         "'large'). Needed when --train-data differs from the "
                         "data the COPIED classifier was trained on: the flow "
                         "would otherwise fit its own scaler and standardise "
                         "context differently from the classifier that "
                         "supplies P(R|theta). The exporter asserts they "
                         "match and will refuse to export otherwise.")
    ap.add_argument("--nll-weight", type=float, default=1.0,
                    help="weight on the maximum-likelihood term. 0 with "
                         "--distill-beta gives PURE regression onto the "
                         "analytic log-density (no MLE term at all).")
    ap.add_argument("--init-from", default=None,
                    help="tag of an existing run to warm-start the flow from "
                         "(reads results/model_states/<tag>_budget/DDM_flow). "
                         "Use to FINE-TUNE a converged MLE flow with a small "
                         "score penalty, rather than co-training the two "
                         "objectives from step 1 - co-training cost kappa "
                         "0.46 and a 150x t0 regression.")
    ap.add_argument("--score-weights", default=None,
                    help="per-parameter weights for the score penalty: a "
                         "comma list of len(CONTEXT) floats, or 'displacement' "
                         "for 1/(sigma_p^2 * (1/psd_p^2 + N*I_p)) at N=400 "
                         "(protects low-information params like sv). Omit for "
                         "equal weights.")
    ap.add_argument("--range-min", type=float, default=-5.0,
                    help="spline support lower bound (log-rt)")
    ap.add_argument("--range-max", type=float, default=5.0,
                    help="spline support upper bound (log-rt)")
    ap.add_argument("--censor", type=float, default=None,
                    help="upper censor (s): rts >= censor contribute the "
                         "survivor mass at the censor instead of a density "
                         "term (censored MLE; see sbc/censor_design.md)")
    ap.add_argument("--classifier-from", default="large",
                    help="budget tag whose trained classifier to copy")
    ap.add_argument("--train-classifier", action="store_true")
    args = ap.parse_args()
    if args.keep_checkpoints is not None:
        os.environ["FLOW_KEEP_CHECKPOINTS"] = str(args.keep_checkpoints)

    # (SZ0, SZ, SZ1) swaps which columns condition the flow; everything
    # downstream is driven by this list, so nothing else needs to change.
    if args.st0_zero and args.triple:
        raise SystemExit("--st0-zero and --triple are mutually exclusive")
    CTX = (CONTEXT_ST0ZERO if args.st0_zero
           else CONTEXT_TRIPLE if args.triple else CONTEXT)

    out_base = REPO_ROOT / "results" / "model_states" / f"{args.tag}_budget"
    flow_dir = out_base / "DDM_flow"
    clf_dir = out_base / "DDM_classifier"
    out_base.mkdir(parents=True, exist_ok=True)

    if args.grid_data:
        # ds.dataset accepts ONE directory or a list of FILES, not a list of
        # directories - so expand both trees to parquet files.
        _files = sorted(str(f) for d in (args.train_data, args.grid_data)
                        for f in (REPO_ROOT / d).rglob("*.parquet"))
        if not _files:
            raise SystemExit("no parquet files under --train-data/--grid-data")
        dset_train = ds.dataset(_files, format="parquet")
        print(f"train set = {len(_files)} files across "
              f"{args.train_data} + {args.grid_data}", flush=True)
    else:
        dset_train = ds.dataset(REPO_ROOT / args.train_data, format="parquet")
    dset_val = ds.dataset(REPO_ROOT / args.val_data, format="parquet")
    n_train = dset_train.count_rows()
    widths = [int(w) for w in args.widths.split(",")]
    dims = [len(CTX) + 1] + widths          # +1: response code in context
    if args.geometry:                           # + a*Z and a*(1-Z)
        dims[0] += 2
        print(f"geometry features on: conditioner input dim {dims[0]}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] tag={args.tag}  train={n_train} trials  "
          f"dims={dims}  bins={args.num_bins}  epochs={args.epochs}  "
          f"censor={args.censor}", flush=True)

    # --- classifier: copy (default) or retrain ---
    if not args.train_classifier:
        src = REPO_ROOT / "results" / "model_states" / \
            f"{args.classifier_from}_budget" / "DDM_classifier"
        if clf_dir.exists():
            shutil.rmtree(clf_dir)
        shutil.copytree(src, clf_dir)
        print(f"classifier copied from {src}", flush=True)
    else:
        raise NotImplementedError(
            "classifier retraining not wired here yet; use the notebook "
            "section or extend this script when the classifier becomes the "
            "manipulated component")

    # --- flow (mirrors the notebook's large-budget DDM flow section) ---
    train_key = jax.random.key(args.seed)
    conditioner_key, sampling_key = jax.random.split(train_key, 2)
    rngs = nnx.Rngs(default=conditioner_key, sampling=sampling_key)

    conditioner = make_mlp_conditioner(dims=dims, num_bins=args.num_bins,
                                       num_splines=1, rngs=rngs, use_norm=False,
                                       range_min=args.range_min,
                                       range_max=args.range_max)

    if args.init_from:
        import orbax.checkpoint as ocp
        src_dir = (REPO_ROOT / "results" / "model_states" /
                   f"{args.init_from}_budget" / "DDM_flow")
        # Raw restore, matching read_mlp() in export_flow_ddm.py: a typed
        # StandardRestore against the nnx state tree fails here, so pull the
        # nested dict and assign the arrays in directly.
        mgr0 = ocp.CheckpointManager(str(src_dir.resolve()))
        step0 = mgr0.best_step()
        raw = mgr0.restore(step0, args=ocp.args.Composite(
            conditioner=ocp.args.StandardRestore()))["conditioner"]
        lin = raw.get("linears", {})
        assert len(lin) == len(conditioner.linears), (
            f"warm start shape mismatch: checkpoint has {len(lin)} hidden "
            f"layers, this architecture has {len(conditioner.linears)}")
        for i, layer in enumerate(conditioner.linears):
            src = lin[i] if i in lin else lin[str(i)]
            layer.kernel.value = jnp.asarray(src["kernel"]["value"])
            layer.bias.value = jnp.asarray(src["bias"]["value"])
        conditioner.out.kernel.value = jnp.asarray(raw["out"]["kernel"]["value"])
        conditioner.out.bias.value = jnp.asarray(raw["out"]["bias"]["value"])
        print(f"warm-started flow from {args.init_from} step {step0}", flush=True)

    total_steps = (n_train // args.batch) * args.epochs
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0., peak_value=args.peak_lr,
        warmup_steps=0.1 * total_steps, decay_steps=total_steps)
    optimizer = nnx.Optimizer(conditioner, optax.adamw(learning_rate=schedule),
                              wrt=nnx.Param)
    train_metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    val_metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

    # target columns: ll alone normally; ll + the 8 true scores for
    # --score-match; ll + param_id for --grid-data (grid rows carry a param_id
    # offset of 1e7, which the loss uses to withhold them from the NLL).
    tnames = TARGET
    if args.score_match is not None:
        tnames = [TARGET] + [f"score_{p}" for p in CTX]
    elif args.grid_data is not None:
        tnames = [TARGET, "param_id"]
    loader_args = dict(data_names=DATA, context_names=CTX + [RESPONSE],
                       target_names=tnames, response_name=RESPONSE,
                       standardize=True, geometry=args.geometry)
    # The score penalty needs both trials of a parameter draw in the same
    # batch at positions 2k/2k+1, so pairs are shuffled instead of rows.
    pair_rows = args.score_lambda is not None
    ext_scaler = None
    if args.scaler_from:
        ext_scaler = load_obj(str(REPO_ROOT / "results" / "model_states" /
                                  f"{args.scaler_from}_budget" / "DDM_flow" /
                                  "train_loader.pkl"))._scaler
        print(f"reusing context scaler from {args.scaler_from}", flush=True)
    train_loader = PreloadedBatches(dset_train, batch_size=args.batch,
                                    pair_rows=pair_rows, scaler=ext_scaler,
                                    shuffle=True, **loader_args)
    val_loader = PreloadedBatches(dset_val, batch_size=256, shuffle=False,
                                  pair_rows=pair_rows,
                                  scaler=train_loader._scaler, **loader_args)

    DISPLACEMENT_W = [0.07833, 0.05211, 0.43132, 0.00000,
                      0.20252, 2.57573, 2.40020, 1.25979]   # v a t0 s Z SZ sv st0
    if args.score_weights is None:
        score_w = None
    elif args.score_weights == "displacement":
        score_w = DISPLACEMENT_W
    else:
        score_w = [float(x) for x in args.score_weights.split(",")]
    if score_w is not None and len(score_w) != len(CTX):
        raise SystemExit("--score-weights needs %d values" % len(CTX))

    # frozen choice classifier, needed to convert the parquet's JOINT `ll`
    # into the CONDITIONAL target the flow models: log f = ll - log P(R|theta)
    clf = None
    if args.distill_beta is not None or args.score_match is not None:
        import orbax.checkpoint as ocp
        from flow.config import make_choice_classifier
        mgr_c = ocp.CheckpointManager(str(clf_dir.resolve()))
        raw_c = mgr_c.restore(mgr_c.best_step(), args=ocp.args.Composite(
            conditioner=ocp.args.StandardRestore()))["conditioner"]
        keys = sorted(raw_c["layers"], key=int)
        widths = [int(raw_c["layers"][k]["kernel"]["value"].shape[0]) for k in keys]
        clf = make_choice_classifier(rngs=nnx.Rngs(default=jax.random.key(0)),
                                     dims=widths)
        for k, layer in zip(keys, clf.layers):
            layer.kernel.value = jnp.asarray(raw_c["layers"][k]["kernel"]["value"])
            layer.bias.value = jnp.asarray(raw_c["layers"][k]["bias"]["value"])
        clf.eval()
        print(f"loaded frozen classifier, widths {widths} (+1 logit)", flush=True)

    # Persist the scaler and configs BEFORE training, not after. These are
    # fully determined by the arguments and the training data, and writing them
    # late means a run that is killed (or simply not taken to the nominal epoch
    # count) leaves no scaler and therefore cannot be exported at all. Borrowing
    # another run's scaler is not a fallback whenever the context columns
    # differ, as they do for --triple.
    flow_dir.mkdir(parents=True, exist_ok=True)   # normally made by the
    out_base.mkdir(parents=True, exist_ok=True)   # checkpointer inside train_model
    # out_base, NOT flow_dir: the checkpointer WIPES flow_dir when train_model
    # starts, so anything written there beforehand is deleted. The exporter
    # falls back to this location.
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
                num_epochs=args.epochs, censor=args.censor,
                score_lambda=args.score_lambda, n_theta=len(CTX),
                score_weights=score_w,
                distill_beta=args.distill_beta, classifier=clf,
                nll_weight=args.nll_weight,
                score_match=args.score_match,
                sigma=(train_loader._scaler.scale_ if args.score_match is not None else None),
                huber_delta=args.huber_delta,
                edge_amp=args.edge_amp, edge_scale=args.edge_scale,
                edge_stats=(None if args.edge_amp is None else (
                    float(train_loader._scaler.mean_[CTX.index("t0")]),
                    float(train_loader._scaler.scale_[CTX.index("t0")]),
                    float(train_loader._scaler.mean_[CTX.index("st0")]),
                    float(train_loader._scaler.scale_[CTX.index("st0")]))),
                data_names=DATA, context_names=CTX + [RESPONSE],
                target_names=TARGET, response_name=RESPONSE,
                checkpoint_path=str(flow_dir), early_stopping=False)
    print(f"[{time.strftime('%H:%M:%S')}] training done in "
          f"{(time.time() - t0) / 3600:.2f} h", flush=True)

    print(f"all outputs -> {out_base}", flush=True)


if __name__ == "__main__":
    main()

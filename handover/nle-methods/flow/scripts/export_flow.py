"""Export a trained race-model flow to a self-contained JSON + golden vectors.

Purpose: make flows evaluable outside Python (R reference implementation and
the C++/Rcpp port destined for EMC2). Model-agnostic for "race character"
models (LNR, RDM, ...): a single spline flow over rt conditioned on the model
parameters; pdf/cdf/survivor all come from one spline inversion. The DDM
(flow + choice classifier) will get a separate exporter.

The network architecture (hidden widths, LayerNorm or not, bin count) is
INFERRED from the checkpoint. Model-specific facts (context column order,
training-region bounds, natural->sampled transforms) come from MODEL_REGISTRY
and match simulation/designs.R and the training notebook.

Writes to `port/export/`:
  - <model>_<budget>_weights.json        — scaler, MLP, spline config, bounds
  - <model>_<budget>_golden.csv          — float64 golden vectors (z, log_pdf,
                                           cdf, log_sf) from a NumPy port of
                                           the exact distrax math
  - <model>_<budget>_intermediates.json  — per-layer values, first 3 param sets

The NumPy float64 port is cross-checked against the authoritative float32
distrax/JAX pipeline (`flow.config.spline_flow`) before anything is written.

Run from `flow/`:  uv run python scripts/export_flow.py --model LNR --budget small
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy import stats as sstats
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from flow.config import make_mlp_conditioner, spline_flow
from flow.training import load_obj

_FORCE_STEP = [None]

REPO_ROOT = Path(__file__).resolve().parents[2]

RANGE_MIN, RANGE_MAX = -5.0, 5.0     # spline range used by flow.config.spline_flow
MIN_BIN_SIZE = 1e-4
MIN_KNOT_SLOPE = 1e-4

# Model-specific facts. Context order must match the training notebook's
# CONTEXT list; bounds/transforms come from simulation/designs.R (the
# *_single designs) and how sim_code.R sampled the training data.
MODEL_REGISTRY = {
    "LNR": {
        # sim_code.R: transforms = transformsLNR_single (s, t0 log-sampled)
        "context": ["m", "s", "t0"],
        "transforms": {"m": "identity", "s": "log", "t0": "log"},
        "bounds_natural": {"lower": [-3.0, 0.1, 0.05], "upper": [3.0, 2.0, 1.0]},
        "checkpoint_head": "LNR",
    },
    "RDM": {
        # sim_code.R passes transforms = NULL to param_matrix, whose default
        # is rep("log", dim) (sim_functions.R:34, "For RDM"): sampled
        # uniformly on the natural scale, stored/conditioned on the log
        # scale for EVERY parameter. Verified against the train/val/test
        # parquet ranges (all equal log of the designs.R bounds).
        "context": ["v", "B", "t0", "s", "A"],
        "transforms": {p: "log" for p in ["v", "B", "t0", "s", "A"]},
        "bounds_natural": {"lower": [1e-3, 0.1, 0.05, 0.1, 1e-4],
                           "upper": [5.0, 3.0, 1.0, 2.0, 2.0]},
        "checkpoint_head": "RDM",
    },
}


def sampled_bounds(spec):
    def tf(v, t):
        return float(np.log(v)) if t == "log" else float(v)
    tfs = [spec["transforms"][p] for p in spec["context"]]
    return {
        "lower": [tf(v, t) for v, t in zip(spec["bounds_natural"]["lower"], tfs)],
        "upper": [tf(v, t) for v, t in zip(spec["bounds_natural"]["upper"], tfs)],
    }


# ----------------------------------------------------------------------------
# NumPy float64 reimplementation of the distrax pipeline (the port spec).
# Ported from distrax/_src/bijectors/rational_quadratic_spline.py and
# flow/src/flow/config.py. Keep in exact sync with the R / C++ ports.
# ----------------------------------------------------------------------------

def gelu_tanh(x):
    # jax.nn.gelu default (approximate=True)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def layernorm(x, scale, bias, eps=1e-6):
    # flax nnx.LayerNorm defaults (normalize over last axis)
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * scale + bias


def normalize_bin_sizes(u, total_size, min_bin_size):
    k = u.shape[-1]
    e = np.exp(u - u.max(axis=-1, keepdims=True))
    sm = e / e.sum(axis=-1, keepdims=True)
    return sm * (total_size - k * min_bin_size) + min_bin_size


def normalize_knot_slopes(u, min_knot_slope):
    offset = np.log(np.exp(1.0 - min_knot_slope) - 1.0)
    return softplus(u + offset) + min_knot_slope


def mlp_forward(ctx_std, layers, norms):
    """layers: list of (W, b), last one is the linear output head."""
    h = ctx_std
    for i, (W, b) in enumerate(layers[:-1]):
        h = h @ W + b
        if norms is not None:
            h = layernorm(h, norms[i][0], norms[i][1])
        h = gelu_tanh(h)
    W, b = layers[-1]
    return h @ W + b


def build_knots(raw, num_bins, range_min=RANGE_MIN, range_max=RANGE_MAX):
    """raw: (n, 3K+1) conditioner output -> x_pos, y_pos, slopes (n, K+1)."""
    k = num_bins
    widths = normalize_bin_sizes(raw[:, :k], range_max - range_min, MIN_BIN_SIZE)
    heights = normalize_bin_sizes(raw[:, k:2 * k], range_max - range_min, MIN_BIN_SIZE)
    slopes = normalize_knot_slopes(raw[:, 2 * k:], MIN_KNOT_SLOPE)
    n = raw.shape[0]
    pad_lo = np.full((n, 1), range_min)
    pad_hi = np.full((n, 1), range_max)
    x_pos = np.concatenate([pad_lo, range_min + np.cumsum(widths[:, :-1], axis=1), pad_hi], axis=1)
    y_pos = np.concatenate([pad_lo, range_min + np.cumsum(heights[:, :-1], axis=1), pad_hi], axis=1)
    # boundary_slopes="identity"
    slopes[:, 0] = 1.0
    slopes[:, -1] = 1.0
    return x_pos, y_pos, slopes


def rqs_inverse(y, x_pos, y_pos, slopes):
    """Inverse spline + log|d inverse/dy| for one knot set, vectorized over y.

    Mirrors distrax._rational_quadratic_spline_inv (incl. stable quadratic
    root and identity tails).
    """
    below = y <= y_pos[0]
    above = y >= y_pos[-1]
    idx = np.clip(np.searchsorted(y_pos, y, side="right") - 1, 0, len(y_pos) - 2)

    x_lo, x_hi = x_pos[idx], x_pos[idx + 1]
    y_lo, y_hi = y_pos[idx], y_pos[idx + 1]
    d_lo, d_hi = slopes[idx], slopes[idx + 1]

    bin_width = x_hi - x_lo
    bin_height = y_hi - y_lo
    bin_slope = bin_height / bin_width

    w = np.clip((y - y_lo) / bin_height, 0.0, 1.0)
    slopes_term = d_hi + d_lo - 2.0 * bin_slope
    c = -bin_slope * w
    b = d_lo - slopes_term * w
    a = bin_slope - b

    sqrt_diff = b**2 - 4.0 * a * c
    safe_sqrt = np.sqrt(np.clip(sqrt_diff, np.finfo(np.float64).tiny, None))
    safe_sqrt = np.where(sqrt_diff > 0.0, safe_sqrt, 0.0)
    num = np.where(b >= 0.0, 2.0 * c, -b + safe_sqrt)
    den = np.where(b >= 0.0, -b - safe_sqrt, 2.0 * a)
    z = np.clip(num / den, 0.0, 1.0)

    x = x_lo + bin_width * z

    sq_z = z * z
    z1mz = z - sq_z
    sq_1mz = (1.0 - z) ** 2
    denominator = bin_slope + slopes_term * z1mz
    logdet = (-2.0 * np.log(bin_slope)
              - np.log(d_hi * sq_z + 2.0 * bin_slope * z1mz + d_lo * sq_1mz)
              + 2.0 * np.log(denominator))

    x = np.where(below, y - y_pos[0] + x_pos[0], x)
    x = np.where(above, y - y_pos[-1] + x_pos[-1], x)
    logdet = np.where(below | above, 0.0, logdet)
    return x, logdet


def flow_eval_f64(theta, rt, layers, norms, scaler_mean, scaler_scale, num_bins):
    """Full float64 evaluation. theta: (n, K) sampled scale; rt: (m,)."""
    ctx_std = (theta - scaler_mean) / scaler_scale
    raw = mlp_forward(ctx_std, layers, norms)
    x_pos, y_pos, slopes = build_knots(raw, num_bins)
    u = np.log(rt)
    n, m = theta.shape[0], rt.shape[0]
    z = np.empty((n, m)); logdet = np.empty((n, m))
    for i in range(n):
        z[i], logdet[i] = rqs_inverse(u, x_pos[i], y_pos[i], slopes[i])
    log_pdf = sstats.norm.logpdf(z) + logdet - u[None, :]
    cdf = sstats.norm.cdf(z)
    log_sf = sstats.norm.logsf(z)
    return z, log_pdf, cdf, log_sf, (ctx_std, raw, x_pos, y_pos, slopes)


# ----------------------------------------------------------------------------
# Checkpoint reading (architecture inferred, no factories needed)
# ----------------------------------------------------------------------------

def read_conditioner(ckpt_dir):
    mgr = ocp.CheckpointManager(str(ckpt_dir.resolve()))
    step = _FORCE_STEP[0] if _FORCE_STEP[0] is not None else mgr.best_step()
    if step is None:
        step = mgr.latest_step()
    raw = mgr.restore(step, args=ocp.args.Composite(
        conditioner=ocp.args.StandardRestore()))["conditioner"]

    hidden = raw.get("linears", {})
    layers = []
    for i in sorted(hidden, key=int):
        layers.append((np.asarray(hidden[i]["kernel"]["value"], dtype=np.float64),
                       np.asarray(hidden[i]["bias"]["value"], dtype=np.float64)))
    layers.append((np.asarray(raw["out"]["kernel"]["value"], dtype=np.float64),
                   np.asarray(raw["out"]["bias"]["value"], dtype=np.float64)))

    norms = None
    if "norms" in raw:
        norms = [(np.asarray(raw["norms"][i]["scale"]["value"], dtype=np.float64),
                  np.asarray(raw["norms"][i]["bias"]["value"], dtype=np.float64))
                 for i in sorted(raw["norms"], key=int)]

    dout = layers[-1][0].shape[1]
    if (dout - 1) % 3 != 0:
        raise ValueError(f"Conditioner output dim {dout} is not 3K+1: "
                         f"multi-spline chains are not supported by this exporter.")
    num_bins = (dout - 1) // 3
    return step, layers, norms, num_bins


def build_jax_conditioner(layers, norms, num_bins):
    """Fresh nnx conditioner with the checkpoint weights assigned (float32)."""
    dims = [layers[0][0].shape[0]] + [W.shape[1] for W, _ in layers[:-1]]
    cond = make_mlp_conditioner(nnx.Rngs(0), dims=dims, num_bins=num_bins,
                                num_splines=1, use_norm=norms is not None)
    for i, (W, b) in enumerate(layers[:-1]):
        cond.linears[i].kernel.value = jnp.asarray(W, dtype=jnp.float32)
        cond.linears[i].bias.value = jnp.asarray(b, dtype=jnp.float32)
    if norms is not None:
        for i, (scale, bias) in enumerate(norms):
            cond.norms[i].scale.value = jnp.asarray(scale, dtype=jnp.float32)
            cond.norms[i].bias.value = jnp.asarray(bias, dtype=jnp.float32)
    W, b = layers[-1]
    cond.out.kernel.value = jnp.asarray(W, dtype=jnp.float32)
    cond.out.bias.value = jnp.asarray(b, dtype=jnp.float32)
    return cond


# ----------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LNR", choices=sorted(MODEL_REGISTRY))
    # free-form: budgets are now arbitrary run tags, not the original three
    ap.add_argument("--budget", default="small",
                    help="run tag under results/model_states/<budget>_budget/")
    ap.add_argument("--step", type=int, default=None,
                    help="export this checkpoint step instead of the best-val "
                         "one — needed to screen a window of plateau "
                         "checkpoints, which is the recipe (see SBC_PLAN "
                         "Phase 14)")
    ap.add_argument("--tag-suffix", default="",
                    help="appended to the output tag, e.g. _e207")
    ap.add_argument("--context", default=None,
                    help="comma list overriding the registry's context order, "
                         "for a run trained with a different column order. The "
                         "network is unchanged; this only records what its "
                         "inputs actually are.")
    ap.add_argument("--n-param-sets", type=int, default=25)
    ap.add_argument("--port-tol", type=float, default=5e-3,
                    help="max |dlog_pdf| tolerated between the float64 port and "
                         "distrax. Raise only after checking the printed "
                         "distribution — a single outlier is not a port bug.")
    args = ap.parse_args()

    spec = dict(MODEL_REGISTRY[args.model])
    if args.context:
        spec["context"] = args.context.split(",")
        print(f"context order overridden: {spec['context']}", flush=True)
    ckpt_dir = (REPO_ROOT / "results" / "model_states"
                / f"{args.budget}_budget" / spec["checkpoint_head"])
    out_dir = REPO_ROOT / "port" / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    _FORCE_STEP[0] = args.step
    tag = f"{args.model.lower()}_{args.budget}{args.tag_suffix}"

    step, layers, norms, num_bins = read_conditioner(ckpt_dir)
    n_par = sum(W.size + b.size for W, b in layers) + \
        (sum(s.size + b.size for s, b in norms) if norms else 0)
    widths = [layers[0][0].shape[0]] + [W.shape[1] for W, _ in layers]
    print(f"Restored step {step} from {ckpt_dir}")
    print(f"Inferred architecture: widths {widths}, layernorm={norms is not None}, "
          f"{num_bins} spline bins, {n_par} parameters")
    if layers[0][0].shape[0] != len(spec["context"]):
        raise ValueError(
            f"Checkpoint input dim {layers[0][0].shape[0]} != "
            f"{len(spec['context'])} context params for {args.model}.")

    # The checkpointer wipes the checkpoint dir when training starts, so a

    # scaler written there beforehand is gone; train_race.py persists it to

    # the run root instead. Prefer the checkpoint dir, fall back to the root.

    _sp = ckpt_dir  / "train_loader.pkl"

    if not _sp.exists():

        _sp = ckpt_dir.parent / "train_loader.pkl"

        print(f"scaler from {_sp} (checkpoint-dir copy absent)", flush=True)

    loader = load_obj(str(_sp))
    scaler = loader._scaler
    scaler_mean = np.asarray(scaler.mean_, dtype=np.float64)
    scaler_scale = np.asarray(scaler.scale_, dtype=np.float64)

    # --- golden grid: params inside the training box, rts incl. tails ---
    bounds_s = sampled_bounds(spec)
    rng = np.random.default_rng(20260707)
    lo = np.array(bounds_s["lower"]); hi = np.array(bounds_s["upper"])
    theta = rng.uniform(lo, hi, size=(args.n_param_sets, len(lo)))
    theta = theta.astype(np.float32).astype(np.float64)  # identical inputs everywhere
    rt = np.geomspace(0.005, 20.0, 23).astype(np.float32).astype(np.float64)

    z, log_pdf, cdf, log_sf, inter = flow_eval_f64(
        theta, rt, layers, norms, scaler_mean, scaler_scale, num_bins)

    # --- cross-check float64 port against the authoritative distrax path ---
    conditioner = build_jax_conditioner(layers, norms, num_bins)
    ctx32 = ((theta - scaler_mean) / scaler_scale).astype(np.float32)
    max_lp, max_z = 0.0, 0.0
    all_lp = []
    for i in range(theta.shape[0]):
        rt32 = jnp.asarray(rt, dtype=jnp.float32)
        ctx_rep = jnp.tile(jnp.asarray(ctx32[i]), (len(rt), 1))
        lp32, flow = spline_flow(rt32, ctx_rep, conditioner)
        z32 = flow.bijector.inverse(rt32)
        d_lp = np.abs(np.asarray(lp32) - log_pdf[i])
        max_lp = max(max_lp, float(np.max(d_lp)))
        max_z = max(max_z, float(np.max(np.abs(np.asarray(z32) - z[i]))))
        all_lp.extend(d_lp.ravel().tolist())
    # Report the DISTRIBUTION, not just the max. This guard is a max over ~575
    # sampled points and has a fat-tailed failure mode: one unlucky draw near a
    # spline knot or the range boundary can trip it while the model is
    # indistinguishable from one that passed. Without percentiles there is no
    # way to tell a real port bug (everything shifted) from a single outlier.
    a = np.array(all_lp)
    print(f"float64 port vs distrax float32: max |dlog_pdf| = {max_lp:.3e}, "
          f"max |dz| = {max_z:.3e}")
    print(f"  |dlog_pdf| over {a.size} points: median {np.median(a):.2e}  "
          f"p90 {np.percentile(a,90):.2e}  p99 {np.percentile(a,99):.2e}  "
          f"max {a.max():.2e} ; frac>1e-3 = {(a>1e-3).mean():.4f}")
    # Report BOTH values and name the one that failed. The two conditions mean
    # different things: dlog_pdf drifts with float32 accumulation and scales with
    # network size (raise --port-tol knowingly), whereas dz measures the spline
    # INVERSION and a large value means an ill-conditioned knot layout at some
    # theta — a real defect in that checkpoint, not a tolerance question.
    _fail = []
    if max_lp >= args.port_tol: _fail.append(f"dlog_pdf {max_lp:.3e} >= {args.port_tol:.1e}")
    if max_z >= 5e-4: _fail.append(f"dz {max_z:.3e} >= 5.0e-04 (spline inversion, NOT a tolerance issue)")
    assert not _fail, (
        "float64 port disagrees with distrax: " + "; ".join(_fail) +
        f"  [max |dlog_pdf| = {max_lp:.3e}, max |dz| = {max_z:.3e}]. "
        "Check the distribution line above.")

    # --- weights JSON ---
    weights = {
        "model": args.model, "budget": args.budget,
        "checkpoint_step": int(step), "exported": str(date.today()),
        "context_names": spec["context"],
        "context_transforms": spec["transforms"],
        "parameter_scale": "sampled",
        "bounds_sampled": bounds_s,
        "bounds_natural": spec["bounds_natural"],
        "scaler": {"mean": scaler_mean.tolist(), "scale": scaler_scale.tolist()},
        "mlp": {
            "activation": "gelu_tanh",
            "use_norm": norms is not None,
            "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in layers],
            "norms": ([{"scale": s.tolist(), "bias": b.tolist(), "eps": 1e-6}
                       for s, b in norms] if norms else None),
        },
        "spline": {
            "num_bins": num_bins, "num_splines": 1,
            "range_min": RANGE_MIN, "range_max": RANGE_MAX,
            "min_bin_size": MIN_BIN_SIZE, "min_knot_slope": MIN_KNOT_SLOPE,
            "boundary_slopes": "identity", "output_transform": "exp",
            "base_distribution": "standard_normal",
        },
        "ll_floor_log": float(np.log(1e-10)),
    }
    wpath = out_dir / f"{tag}_weights.json"
    wpath.write_text(json.dumps(weights))
    print(f"Wrote {wpath}")

    # --- golden CSV (long format) ---
    gpath = out_dir / f"{tag}_golden.csv"
    ctx_cols = ",".join(spec["context"])
    with open(gpath, "w") as f:
        f.write(f"param_id,{ctx_cols},rt,z,log_pdf,cdf,log_sf\n")
        for i in range(theta.shape[0]):
            th = ",".join(f"{v:.17g}" for v in theta[i])
            for j in range(len(rt)):
                f.write(f"{i + 1},{th},{rt[j]:.17g},{z[i, j]:.17g},"
                        f"{log_pdf[i, j]:.17g},{cdf[i, j]:.17g},{log_sf[i, j]:.17g}\n")
    print(f"Wrote {gpath} ({theta.shape[0] * len(rt)} rows)")

    # --- intermediates for the first 3 parameter sets ---
    ctx_std, raw, x_pos, y_pos, slopes = inter
    ipath = out_dir / f"{tag}_intermediates.json"
    ipath.write_text(json.dumps({
        "theta": theta[:3].tolist(),
        "ctx_std": ctx_std[:3].tolist(),
        "conditioner_raw": raw[:3].tolist(),
        "x_pos": x_pos[:3].tolist(),
        "y_pos": y_pos[:3].tolist(),
        "knot_slopes": slopes[:3].tolist(),
    }))
    print(f"Wrote {ipath}")


if __name__ == "__main__":
    main()

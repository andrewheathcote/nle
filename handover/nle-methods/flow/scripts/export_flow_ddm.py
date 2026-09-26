"""Export the trained DDM flow + choice classifier to JSON + golden vectors.

DDM-character models use TWO networks (see flow/src/flow/config.py
`joint_log_prob` and the DDM sections of the training notebook):

  - a spline flow over rt whose conditioner sees [params (standardized), R]
    with R the RAW response code (1.0 or 2.0) appended unstandardized;
  - a choice classifier MLP mapping standardized params -> logit of P(R=2),
    with log P(R) = -softplus(-logit) for R=2 and -softplus(logit) for R=1.

The joint (defective) density is
    pdf(rt, R | params) = flow_pdf(rt | params, R) * P(R | params)
and the defective CDF is  P(R | params) * Phi(z),  z = spline^{-1}(log rt).

As with export_flow.py: architectures are inferred from the checkpoints, a
NumPy float64 port computes the golden vectors, and the exporter aborts if
that port disagrees with the authoritative float32 JAX pipeline.

Run from `flow/`:  uv run python scripts/export_flow_ddm.py --budget large
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy import stats as sstats
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from flow.config import make_mlp_conditioner, make_choice_classifier, joint_log_prob
from flow.training import load_obj

# reuse the validated race-flow float64 machinery
from export_flow import (gelu_tanh, softplus, layernorm, mlp_forward,  # noqa: F401
                         build_knots, rqs_inverse, RANGE_MIN, RANGE_MAX,
                         MIN_BIN_SIZE, MIN_KNOT_SLOPE)

REPO_ROOT = Path(__file__).resolve().parents[2]

CONTEXT = ["v", "a", "t0", "s", "Z", "SZ", "sv", "st0"]
# --triple: the flow is conditioned on (SZ0, SZ, SZ1) instead of (a, Z, SZ).
# The CLASSIFIER is unchanged and still takes the EMC2 parameters, so the two
# heads carry different context encodings and different scalers. The port keeps
# the EMC2 parameterisation as its public interface and re-encodes for the flow.
CONTEXT_TRIPLE = ["v", "SZ0", "t0", "s", "SZw", "SZ1", "sv", "st0"]
# st0 = 0 build: a 7-parameter DDM with no non-decision-time variability.
# EMC2 gives st0 a default of -Inf on the sampled scale under an exp transform,
# so omitting st0 from the design formula sets it to exactly 0 (verified via
# mapped_pars). The flow's existing box cannot reach st0 = 0 -- its floor is
# 0.01, since log(0) = -Inf -- so this needs its own training data and its own
# network. See handover/ST0_ZERO_RETRAIN.md.
CONTEXT_ST0ZERO = ["v", "a", "t0", "s", "Z", "SZ", "sv"]

# --step exports a NAMED checkpoint instead of the best-val one. Needed to study
# variation WITHIN a single training run: with one checkpoint per run there is
# no way to separate run-to-run variance from epoch-to-epoch variance on the
# converged plateau, and that distinction decides whether seed screening is the
# right frame at all.
_FORCE_STEP = [None]


# --zerobox: sv, SZ, st0 are conditioned on asinh(x / c) so that exact zeros
# (stored as -Inf on the EMC2 scale) are representable. Constants match
# simulation/zerobox.R and flow/src/flow/data.py; they are written into the export.
ZEROBOX_C = {"sv": 0.01, "SZ": 0.05, "st0": 0.002}
ZEROBOX_BOUNDS_NAT = {"sv": (0.0, 4.0), "SZ": (0.0, 0.99), "st0": (0.0, 0.49)}


def emc2_to_zerobox(theta, ctx):
    """EMC2 sampled-scale theta (n, k) -> the zero-box network context."""
    out = np.array(theta, dtype=np.float64, copy=True)
    for name, c in ZEROBOX_C.items():
        if name not in ctx:
            continue
        j = ctx.index(name)
        nat = sstats.norm.cdf(out[:, j]) if name == "SZ" else np.exp(out[:, j])
        out[:, j] = np.arcsinh(nat / c)
    return out


def emc2_to_triple(theta):
    """(v, log a, log t0, log s, qnorm Z, qnorm SZ, log sv, log st0)
    -> (v, log SZ0, log t0, log s, log SZ, log SZ1, log sv, log st0).

    a = exp(theta_a); w = Phi(theta_Z); sw = 2*Phi(theta_SZ)*min(w, 1-w) is
    EMC2's Ttransform. Then SZ = sw*a, SZ0 = a*w - SZ/2, SZ1 = a - SZ0 - SZ.
    SZ0 and SZ1 are non-negative by construction (sw <= 2*min(w, 1-w) always);
    the clip only guards float cancellation."""
    from scipy.stats import norm as _norm
    th = np.asarray(theta, dtype=np.float64).copy()
    a = np.exp(th[:, 1])
    w = _norm.cdf(th[:, 4])
    sw = 2.0 * _norm.cdf(th[:, 5]) * np.minimum(w, 1.0 - w)
    SZ = sw * a
    SZ0 = a * w - SZ / 2.0
    SZ1 = a - SZ0 - SZ
    eps = 1e-12
    th[:, 1] = np.log(np.maximum(SZ0, eps))
    th[:, 4] = np.log(np.maximum(SZ, eps))
    th[:, 5] = np.log(np.maximum(SZ1, eps))
    return th
TRANSFORMS = {"v": "identity", "a": "log", "t0": "log", "s": "log",
              "Z": "probit", "SZ": "probit", "sv": "log", "st0": "log"}
# Keyed by parameter, not positional: a positional list silently mis-pairs
# bounds with columns as soon as the context is a subset (the st0 = 0 build).
BOUNDS_NAT = {"v": (-6.0, 6.0), "a": (0.05, 5.0), "t0": (0.05, 1.0),
              "s": (0.1, 2.0), "Z": (0.1, 0.9), "SZ": (0.05, 0.99),
              "sv": (0.01, 4.0), "st0": (0.01, 0.49)}
BOUNDS_NATURAL = {"lower": [BOUNDS_NAT[p][0] for p in CONTEXT],
                  "upper": [BOUNDS_NAT[p][1] for p in CONTEXT]}


def tf_one(v, t):
    if t == "log":
        return float(np.log(v))
    if t == "probit":
        return float(sstats.norm.ppf(v))
    return float(v)


def sampled_bounds(ctx=None, zerobox=False):
    ctx = ctx if ctx is not None else CONTEXT
    def one(p, k):
        if zerobox and p in ZEROBOX_C:          # network scale: asinh(x / c)
            return float(np.arcsinh(ZEROBOX_BOUNDS_NAT[p][k] / ZEROBOX_C[p]))
        return tf_one(BOUNDS_NAT[p][k], TRANSFORMS[p])
    return {"lower": [one(p, 0) for p in ctx], "upper": [one(p, 1) for p in ctx]}


def read_mlp(ckpt_dir, kind):
    """Raw-restore an MLP checkpoint. kind: 'conditioner' (MLP_no_norm/MLP)
    or 'classifier' (ChoiceClassifier: uniform `layers` list)."""
    mgr = ocp.CheckpointManager(str(Path(ckpt_dir).resolve()))
    # --step selects a flow checkpoint only; the frozen classifier has its own
    # step numbering (it is copied in, not trained here) and must keep best_step
    force = _FORCE_STEP[0] if kind == "conditioner" else None
    step = force if force is not None else mgr.best_step()
    raw = mgr.restore(step, args=ocp.args.Composite(
        conditioner=ocp.args.StandardRestore()))["conditioner"]

    layers, norms = [], None
    if kind == "classifier":
        for i in sorted(raw["layers"], key=int):
            l = raw["layers"][i]
            layers.append((np.asarray(l["kernel"]["value"], dtype=np.float64),
                           np.asarray(l["bias"]["value"], dtype=np.float64)))
    else:
        for i in sorted(raw.get("linears", {}), key=int):
            l = raw["linears"][i]
            layers.append((np.asarray(l["kernel"]["value"], dtype=np.float64),
                           np.asarray(l["bias"]["value"], dtype=np.float64)))
        layers.append((np.asarray(raw["out"]["kernel"]["value"], dtype=np.float64),
                       np.asarray(raw["out"]["bias"]["value"], dtype=np.float64)))
        if "norms" in raw:
            norms = [(np.asarray(raw["norms"][i]["scale"]["value"], dtype=np.float64),
                      np.asarray(raw["norms"][i]["bias"]["value"], dtype=np.float64))
                     for i in sorted(raw["norms"], key=int)]
    return int(step), layers, norms


def ddm_eval_f64(theta, rt, R, flow_layers, flow_norms, clf_layers,
                 scaler_mean, scaler_scale, num_bins,
                 range_min=RANGE_MIN, range_max=RANGE_MAX,
                 theta_clf=None, clf_scaler_mean=None, clf_scaler_scale=None):
    """Float64 joint evaluation. theta: (n, 8) sampled scale; rt: (m,);
    R: scalar response code (1 or 2). Returns (n, m) arrays.

    Under --triple the flow and the classifier take DIFFERENT encodings of the
    same parameter draw: theta is the flow's (SZ0, SZ, SZ1) form, theta_clf the
    EMC2 (a, Z, SZ) form the frozen classifier was trained on. They default to
    the same thing, which is the ordinary single-encoding case."""
    ctx_std = (theta - scaler_mean) / scaler_scale
    full_ctx = np.concatenate(
        [ctx_std, np.full((theta.shape[0], 1), float(R))], axis=1)
    raw = mlp_forward(full_ctx, flow_layers, flow_norms)
    x_pos, y_pos, slopes = build_knots(raw, num_bins, range_min, range_max)

    if theta_clf is None:
        clf_ctx_std = ctx_std
    else:
        clf_ctx_std = (theta_clf - clf_scaler_mean) / clf_scaler_scale
    logit = mlp_forward(clf_ctx_std, clf_layers, None).squeeze(-1)   # (n,)
    log_p_R = -softplus(-logit) if R == 2 else -softplus(logit)

    u = np.log(rt)
    n, m = theta.shape[0], rt.shape[0]
    z = np.empty((n, m)); logdet = np.empty((n, m))
    for i in range(n):
        z[i], logdet[i] = rqs_inverse(u, x_pos[i], y_pos[i], slopes[i])
    log_pdf_rt = sstats.norm.logpdf(z) + logdet - u[None, :]
    log_pdf_joint = log_pdf_rt + log_p_R[:, None]
    cdf_defective = sstats.norm.cdf(z) * np.exp(log_p_R)[:, None]
    return z, log_pdf_joint, cdf_defective, log_p_R, (ctx_std, raw, x_pos, y_pos, slopes)


def CTX_SEL(args):
    """Context column list for this build. The exporter writes this into
    context_names and the R/C++ port builds theta from it, so it MUST match
    what training used -- a mismatch silently permutes the inputs."""
    if getattr(args, "st0_zero", False):
        return CONTEXT_ST0ZERO
    return CONTEXT_TRIPLE if args.triple else CONTEXT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="large",
                    help="checkpoint tag: reads results/model_states/"
                         "<budget>_budget/ (a training budget or an "
                         "architecture-variant tag from train_ddm.py)")
    ap.add_argument("--step", type=int, default=None,
                    help="export this checkpoint step instead of the best-val one")
    ap.add_argument("--tag-suffix", default="",
                    help="appended to the output tag, e.g. _e207")
    ap.add_argument("--zerobox", action="store_true",
                    help="flow was trained with train_ddm.py --zerobox")
    ap.add_argument("--st0-zero", action="store_true",
                    help="7-parameter build with st0 absent (see "
                         "handover/ST0_ZERO_RETRAIN.md); requires training data "
                         "simulated with st0 = 0")
    ap.add_argument("--triple", action="store_true",
                    help="flow was trained with train_ddm.py --triple")
    ap.add_argument("--port-tol", type=float, default=5e-3,
                    help="max |dlog_pdf| allowed between the float64 port and "
                         "JAX float32 in the gated region. Raise only for "
                         "diagnostic exports, and only when the percentile "
                         "distribution matches a known-good flow.")
    ap.add_argument("--n-param-sets", type=int, default=25)
    args = ap.parse_args()

    base = REPO_ROOT / "results" / "model_states" / f"{args.budget}_budget"
    out_dir = REPO_ROOT / "port" / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    _FORCE_STEP[0] = args.step
    tag = f"ddm_{args.budget}{args.tag_suffix}"

    f_step, flow_layers, flow_norms = read_mlp(base / "DDM_flow", "conditioner")
    c_step, clf_layers, _ = read_mlp(base / "DDM_classifier", "classifier")

    # spline range: variants trained by train_ddm.py record it; the original
    # notebook-trained checkpoints used the [-5, 5] default
    range_min, range_max = RANGE_MIN, RANGE_MAX
    vc_path = base / "variant_config.json"
    if vc_path.exists():
        vc = json.loads(vc_path.read_text())
        range_min = float(vc.get("range_min", RANGE_MIN))
        range_max = float(vc.get("range_max", RANGE_MAX))
    print(f"Spline range: [{range_min}, {range_max}]")

    dout = flow_layers[-1][0].shape[1]
    assert (dout - 1) % 3 == 0, f"flow output dim {dout} is not 3K+1"
    num_bins = (dout - 1) // 3
    f_widths = [flow_layers[0][0].shape[0]] + [W.shape[1] for W, _ in flow_layers]
    c_widths = [clf_layers[0][0].shape[0]] + [W.shape[1] for W, _ in clf_layers]
    n_par = sum(W.size + b.size for W, b in flow_layers + clf_layers)
    print(f"Flow: step {f_step}, widths {f_widths}, {num_bins} bins, "
          f"layernorm={flow_norms is not None}")
    print(f"Classifier: step {c_step}, widths {c_widths}")
    print(f"Total parameters: {n_par}")
    CTX = CTX_SEL(args)
    assert f_widths[0] == len(CTX) + 1, "flow input dim != n_params + 1 (R)"
    assert c_widths[0] == len(CTX), "classifier input dim != n_params"

    # scaler: the flow head's pkl (the notebook saved the classifier's loader
    # there, but both were fitted on the same data -> verify they agree)
    # The checkpointer wipes DDM_flow when training starts, so a run that was
    # stopped before its nominal epoch count has no scaler there; fall back to
    # out_base, where it is safe to persist.
    _fp = base / "DDM_flow" / "train_loader.pkl"
    if not _fp.exists():
        _fp = base / "train_loader.pkl"
        print(f"flow scaler from {_fp} (DDM_flow copy absent)", flush=True)
    sc_f = load_obj(str(_fp))._scaler
    sc_c = load_obj(str(base / "DDM_classifier" / "train_loader.pkl"))._scaler
    if args.triple:
        # Different encodings, so the scalers MUST differ; both are exported.
        assert not np.allclose(sc_f.mean_, sc_c.mean_), \
            "--triple given but the flow scaler matches the classifier's"
    else:
        # Tolerance, not exact equality: two StandardScalers fitted on the SAME
        # data in float32 differ by ~1e-5 from accumulation order alone (measured
        # 9.13e-06 between cap256w and b1k_amp0, both fitted on tpt40p400k).
        # The assertion exists to catch a genuine mismatch — a flow and
        # classifier standardised for different data or different encodings —
        # and those differ by orders of magnitude, not 1e-5.
        _dm = float(np.max(np.abs(sc_f.mean_ - sc_c.mean_)))
        _ds = float(np.max(np.abs(sc_f.scale_ - sc_c.scale_)))
        assert _dm < 1e-3 and _ds < 1e-3, (
            f"flow and classifier scalers differ by max {_dm:.2e} (mean) / "
            f"{_ds:.2e} (scale) — resolve before exporting")
    scaler_mean = np.asarray(sc_f.mean_, dtype=np.float64)
    scaler_scale = np.asarray(sc_f.scale_, dtype=np.float64)
    clf_scaler_mean = np.asarray(sc_c.mean_, dtype=np.float64)
    clf_scaler_scale = np.asarray(sc_c.scale_, dtype=np.float64)
    assert scaler_mean.shape[0] == len(CTX), "scaler covers params only"

    # --- golden grid ---
    bounds_s = sampled_bounds(CTX, zerobox=args.zerobox)
    rng = np.random.default_rng(20260708)
    lo = np.array(bounds_s["lower"]); hi = np.array(bounds_s["upper"])
    theta = rng.uniform(lo, hi, size=(args.n_param_sets, len(lo)))
    theta = theta.astype(np.float32).astype(np.float64)
    rt = np.geomspace(0.005, 20.0, 23).astype(np.float32).astype(np.float64)

    # theta is always the EMC2 draw - that is the port's public interface and
    # what the golden CSV records. Under --triple the FLOW additionally needs
    # the re-encoded form; the classifier keeps the EMC2 one.
    theta_flow, theta_clf = theta, None
    if args.triple:
        theta_flow = emc2_to_triple(theta)
        theta_clf = theta
    if args.zerobox:
        # bounds_s is on the network (asinh) scale, so theta was drawn there.
        # Force exact zeros into a quarter of the rows, one variability
        # parameter at a time; both heads take the network scale, and the
        # golden CSV records the EMC2-scale theta (-Inf at zeros).
        n12 = theta.shape[0] // 12
        for i, name in enumerate(("sv", "SZ", "st0")):
            theta[i * n12:(i + 1) * n12, CTX.index(name)] = 0.0
        theta_flow = theta.copy()
        theta = theta_flow.copy()
        for name, c in ZEROBOX_C.items():
            j = CTX.index(name); nat = np.sinh(theta_flow[:, j]) * c
            with np.errstate(divide="ignore"):
                theta[:, j] = sstats.norm.ppf(nat) if name == "SZ" else np.log(nat)

    res = {R: ddm_eval_f64(theta_flow, rt, R, flow_layers, flow_norms, clf_layers,
                           scaler_mean, scaler_scale, num_bins,
                           range_min, range_max,
                           theta_clf=theta_clf,
                           clf_scaler_mean=clf_scaler_mean,
                           clf_scaler_scale=clf_scaler_scale) for R in (1, 2)}

    # --- cross-check against the authoritative JAX joint_log_prob ---
    dims_flow = f_widths[:-1]
    cond = make_mlp_conditioner(nnx.Rngs(0), dims=dims_flow, num_bins=num_bins,
                                num_splines=1, use_norm=flow_norms is not None,
                                range_min=range_min, range_max=range_max)
    for i, (W, b) in enumerate(flow_layers[:-1]):
        cond.linears[i].kernel.value = jnp.asarray(W, dtype=jnp.float32)
        cond.linears[i].bias.value = jnp.asarray(b, dtype=jnp.float32)
    cond.out.kernel.value = jnp.asarray(flow_layers[-1][0], dtype=jnp.float32)
    cond.out.bias.value = jnp.asarray(flow_layers[-1][1], dtype=jnp.float32)
    clf = make_choice_classifier(nnx.Rngs(0), dims=c_widths[:-1])
    for i, (W, b) in enumerate(clf_layers):
        clf.layers[i].kernel.value = jnp.asarray(W, dtype=jnp.float32)
        clf.layers[i].bias.value = jnp.asarray(b, dtype=jnp.float32)

    ctx32 = ((theta_flow - scaler_mean) / scaler_scale).astype(np.float32)
    clf_ctx32 = (None if theta_clf is None else
                 ((theta_clf - clf_scaler_mean) / clf_scaler_scale).astype(np.float32))
    ll_floor = float(np.log(1e-10))
    _ALL_D = []
    max_lp = max_lp_used = 0.0
    for R in (1, 2):
        for i in range(theta.shape[0]):
            lp = joint_log_prob(cond, clf,
                                jnp.asarray(rt, dtype=jnp.float32),
                                jnp.tile(jnp.asarray(ctx32[i]), (len(rt), 1)),
                                jnp.full((len(rt),), R, dtype=jnp.int32),
                                clf_params=(None if clf_ctx32 is None else
                                            jnp.tile(jnp.asarray(clf_ctx32[i]),
                                                     (len(rt), 1))))
            d = np.abs(np.asarray(lp) - res[R][1][i])
            max_lp = max(max_lp, float(np.max(d)))
            # gate on the usable region: above the min_ll floor AND away from
            # the spline range boundaries, where float32 can flip between the
            # last bin and the identity tail (a ~1e-2 log-density branch
            # artifact at points of negligible density)
            u = np.log(rt)
            interior = (np.abs(u - range_max) > 1e-2) & (np.abs(u - range_min) > 1e-2)
            used = (res[R][1][i] > ll_floor) & interior
            if used.any():
                max_lp_used = max(max_lp_used, float(np.max(d[used])))
                _ALL_D.extend(np.asarray(d[used]).ravel().tolist())
    if _ALL_D:
        _a = np.array(_ALL_D)
        print("  |dlog_pdf| distribution over %d gated points: "
              "median %.2e  p90 %.2e  p99 %.2e  max %.2e ; frac>1e-3 = %.4f"
              % (_a.size, np.median(_a), np.percentile(_a,90),
                 np.percentile(_a,99), _a.max(), float((_a>1e-3).mean())))
    print(f"float64 port vs JAX float32 joint_log_prob: "
          f"max |dlog_pdf| = {max_lp:.3e} overall, {max_lp_used:.3e} in the "
          f"gated region (above min_ll floor, away from range boundaries)")
    # The guard is a MAX over ~1080 sampled points, so it has a fat-tailed
    # failure mode: a single unlucky draw can trip it while the distribution is
    # indistinguishable from a flow that passed (measured: tpt40 median 9.6e-6 /
    # p90 1.4e-4 / p99 7.7e-4 vs arch64w 1.1e-5 / 1.6e-4 / 7.6e-4, and distill1
    # PASSED with 3x tpt40's fraction above 1e-3). Keep the default strict for
    # ship candidates; --port-tol raises it explicitly for diagnostic exports.
    if max_lp_used >= args.port_tol:
        raise AssertionError(
            "float64 port disagrees with JAX beyond float32 tolerance in the "
            f"gated region ({max_lp_used:.3e} >= {args.port_tol:.1e}). Check the "
            "distribution line above: if only the max is an outlier and the "
            "percentiles match a known-good flow, re-run with --port-tol.")
    if args.port_tol > 5e-3:
        print(f"  NOTE: exported with a relaxed port tolerance "
              f"({args.port_tol:.1e}); default is 5.0e-3")

    # --- weights JSON ---
    weights = {
        "model": "DDM", "budget": args.budget,
        "checkpoint_step_flow": f_step, "checkpoint_step_classifier": c_step,
        "exported": str(date.today()),
        "context_names": CTX,
        "classifier_context_names": CTX,
        "context_encoding": ("zerobox" if args.zerobox
                             else "st0zero" if args.st0_zero
                             else "triple" if args.triple else "emc2"),
        "zerobox_c": ZEROBOX_C if args.zerobox else None,
        "classifier_scaler": {"mean": clf_scaler_mean.tolist(),
                              "scale": clf_scaler_scale.tolist()},
        "context_transforms": ({**TRANSFORMS, **{k: f"asinh(x/{c})" for k, c in ZEROBOX_C.items()}}
                               if args.zerobox else TRANSFORMS),
        "parameter_scale": "sampled",
        "response_coding": "flow context gets raw R (1.0 or 2.0) appended after "
                           "standardized params; classifier logit is for P(R=2)",
        "bounds_sampled": bounds_s,
        "bounds_natural": BOUNDS_NATURAL,
        "scaler": {"mean": scaler_mean.tolist(), "scale": scaler_scale.tolist()},
        "flow_mlp": {
            "activation": "gelu_tanh",
            "use_norm": flow_norms is not None,
            "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in flow_layers],
            "norms": ([{"scale": s.tolist(), "bias": b.tolist(), "eps": 1e-6}
                       for s, b in flow_norms] if flow_norms else None),
        },
        "classifier_mlp": {
            "activation": "gelu_tanh",
            "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in clf_layers],
        },
        "spline": {
            "num_bins": num_bins, "num_splines": 1,
            "range_min": range_min, "range_max": range_max,
            "min_bin_size": MIN_BIN_SIZE, "min_knot_slope": MIN_KNOT_SLOPE,
            "boundary_slopes": "identity", "output_transform": "exp",
            "base_distribution": "standard_normal",
        },
        "ll_floor_log": float(np.log(1e-10)),
    }
    wpath = out_dir / f"{tag}_weights.json"
    wpath.write_text(json.dumps(weights))
    print(f"Wrote {wpath}")

    # --- golden CSV (long format, both responses) ---
    gpath = out_dir / f"{tag}_golden.csv"
    ctx_cols = ",".join(CTX)
    with open(gpath, "w") as f:
        f.write(f"param_id,{ctx_cols},R,rt,z,log_pdf_joint,cdf_defective,log_p_R\n")
        for i in range(theta.shape[0]):
            th = ",".join(f"{v:.17g}" for v in theta[i])
            for R in (1, 2):
                z, lp, cdfd, lpr, _ = res[R]
                for j in range(len(rt)):
                    f.write(f"{i + 1},{th},{R},{rt[j]:.17g},{z[i, j]:.17g},"
                            f"{lp[i, j]:.17g},{cdfd[i, j]:.17g},{lpr[i]:.17g}\n")
    n_rows = theta.shape[0] * 2 * len(rt)
    print(f"Wrote {gpath} ({n_rows} rows)")

    # --- intermediates (first 3 param sets, R = 1) ---
    ctx_std, raw, x_pos, y_pos, slopes = res[1][4]
    ipath = out_dir / f"{tag}_intermediates.json"
    ipath.write_text(json.dumps({
        "theta": theta[:3].tolist(),
        "ctx_std": ctx_std[:3].tolist(),
        "conditioner_raw_R1": raw[:3].tolist(),
        "x_pos_R1": x_pos[:3].tolist(),
        "y_pos_R1": y_pos[:3].tolist(),
        "knot_slopes_R1": slopes[:3].tolist(),
        # the classifier takes the EMC2 encoding and its own scaler, which
        # differs from the flow's ctx_std whenever --triple is set
        "classifier_logit": mlp_forward(
            ctx_std[:3] if theta_clf is None
            else ((theta_clf[:3] - clf_scaler_mean) / clf_scaler_scale),
            clf_layers, None).squeeze(-1).tolist(),
    }))
    print(f"Wrote {ipath}")


if __name__ == "__main__":
    main()

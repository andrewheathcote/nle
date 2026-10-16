"""Which side of the export's port-vs-JAX gap is wrong? (SCALED_DDM_PLAN.md, 2026-10-14)

The exporter compares the float64 NumPy port with the float32 JAX joint_log_prob and gates
on max |dlog_pdf|. The scaled pilot's gap is about 10x broader than the zero-box flows'
(p90 1.4e-3). Here the same network is evaluated three ways on the exporter's golden grid
(same theta seed, same rt grid, same exact zeros):
    port  float64 NumPy (export_flow_ddm.ddm_eval_f64, what the C++ port matches to 1e-12)
    j32   JAX in float32 (the exporter's reference)
    j64   JAX in float64 (jax_enable_x64; same code, same weights)
If j64 agrees with the port, the gap is float32 rounding in JAX; otherwise the port differs.

    python flow/scripts/port_x64_check.py --budget sc_pilot --step 28 --scaled [--n-param-sets 200]
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_flow_ddm as E
from flow.config import make_mlp_conditioner, make_choice_classifier, joint_log_prob
from flow.training import load_obj


def build(flow_layers, flow_norms, clf_layers, f_widths, c_widths, num_bins, rmin, rmax, dt):
    cond = make_mlp_conditioner(nnx.Rngs(0), dims=f_widths[:-1], num_bins=num_bins,
                                num_splines=1, use_norm=flow_norms is not None,
                                range_min=rmin, range_max=rmax)
    for i, (W, b) in enumerate(flow_layers[:-1]):
        cond.linears[i].kernel.value = jnp.asarray(W, dtype=dt)
        cond.linears[i].bias.value = jnp.asarray(b, dtype=dt)
    cond.out.kernel.value = jnp.asarray(flow_layers[-1][0], dtype=dt)
    cond.out.bias.value = jnp.asarray(flow_layers[-1][1], dtype=dt)
    if flow_norms is not None:
        for i, (s, b) in enumerate(flow_norms):
            cond.norms[i].scale.value = jnp.asarray(s, dtype=dt)
            cond.norms[i].bias.value = jnp.asarray(b, dtype=dt)
    clf = make_choice_classifier(nnx.Rngs(0), dims=c_widths[:-1])
    for i, (W, b) in enumerate(clf_layers):
        clf.layers[i].kernel.value = jnp.asarray(W, dtype=dt)
        clf.layers[i].bias.value = jnp.asarray(b, dtype=dt)
    return cond, clf


def q(a):
    return "median %.2e  p90 %.2e  p99 %.2e  max %.2e  frac>1e-3 %.4f" % (
        np.median(a), np.percentile(a, 90), np.percentile(a, 99), a.max(), (a > 1e-3).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--scaled", action="store_true")
    ap.add_argument("--n-param-sets", type=int, default=25)
    args = ap.parse_args()
    args.zerobox, args.triple, args.st0_zero = True, False, False
    assert args.scaled, "written for the scaled build"

    base = E.REPO_ROOT / "results" / "model_states" / f"{args.budget}_budget"
    E._FORCE_STEP[0] = args.step
    f_step, flow_layers, flow_norms = E.read_mlp(base / "DDM_flow", "conditioner")
    c_step, clf_layers, _ = E.read_mlp(base / "DDM_classifier", "classifier")
    import json
    vc = json.loads((base / "variant_config.json").read_text())
    rmin, rmax = float(vc["range_min"]), float(vc["range_max"])
    num_bins = (flow_layers[-1][0].shape[1] - 1) // 3
    f_widths = [flow_layers[0][0].shape[0]] + [W.shape[1] for W, _ in flow_layers]
    c_widths = [clf_layers[0][0].shape[0]] + [W.shape[1] for W, _ in clf_layers]
    CTX = E.CTX_SEL(args)
    _fp = base / "DDM_flow" / "train_loader.pkl"
    if not _fp.exists():
        _fp = base / "train_loader.pkl"
    sc = load_obj(str(_fp))._scaler
    mean, scale = np.asarray(sc.mean_, np.float64), np.asarray(sc.scale_, np.float64)
    print(f"{args.budget} step {f_step}, range [{rmin}, {rmax}], {num_bins} bins, ctx {CTX}")

    # the exporter's golden grid, verbatim (seed, float32 rounding, exact zeros)
    bs = E.sampled_bounds(CTX, zerobox=True)
    rng = np.random.default_rng(20260708)
    lo, hi = np.array(bs["lower"]), np.array(bs["upper"])
    theta = rng.uniform(lo, hi, size=(args.n_param_sets, len(lo))).astype(np.float32).astype(np.float64)
    rt = np.geomspace(1e-4, 4.0, 23).astype(np.float32).astype(np.float64)
    n12 = theta.shape[0] // 12
    for i, name in enumerate([p for p in ("sv", "SZ", "st0", "eta") if p in CTX]):
        theta[i * n12:(i + 1) * n12, CTX.index(name)] = 0.0

    port = {R: E.ddm_eval_f64(theta, rt, R, flow_layers, flow_norms, clf_layers, mean, scale,
                              num_bins, rmin, rmax)[1] for R in (1, 2)}
    ctx = (theta - mean) / scale
    out = {}
    for dt, name in ((jnp.float32, "j32"), (jnp.float64, "j64")):
        cond, clf = build(flow_layers, flow_norms, clf_layers, f_widths, c_widths, num_bins, rmin, rmax, dt)
        res = {}
        for R in (1, 2):
            rows = []
            for i in range(theta.shape[0]):
                lp = joint_log_prob(cond, clf, jnp.asarray(rt, dtype=dt),
                                    jnp.tile(jnp.asarray(ctx[i], dtype=dt), (len(rt), 1)),
                                    jnp.full((len(rt),), R, dtype=jnp.int32))
                assert lp.dtype == dt, (lp.dtype, dt)
                rows.append(np.asarray(lp, np.float64))
            res[R] = np.stack(rows)
        out[name] = res

    # the exporter's gate: above the ll floor, away from the spline range ends
    u = np.log(rt)
    interior = (np.abs(u - rmax) > 1e-2) & (np.abs(u - rmin) > 1e-2)
    recs = []
    for R in (1, 2):
        for i in range(theta.shape[0]):
            used = (port[R][i] > np.log(1e-10)) & interior
            for j in np.where(used)[0]:
                recs.append((R, i, j, port[R][i, j], out["j32"][R][i, j], out["j64"][R][i, j]))
    r = np.array(recs)
    p, a32, a64 = r[:, 3], r[:, 4], r[:, 5]
    print(f"{len(r)} gated points ({args.n_param_sets} theta x 23 rt x 2 R)")
    print("  |j32 - port| ", q(np.abs(a32 - p)))
    print("  |j64 - port| ", q(np.abs(a64 - p)))
    print("  |j32 - j64|  ", q(np.abs(a32 - a64)))

    d = np.abs(a32 - p)
    top = np.argsort(-d)[:12]
    print("worst |j32 - port| points: R, kappa, asinh-eta, probit-Z, asinh-SZ, log u, port, j32-port, j64-port")
    for k in top:
        R, i, j = int(r[k, 0]), int(r[k, 1]), int(r[k, 2])
        print("  %d  %+6.2f %6.2f %+5.2f %6.2f  %+6.2f  %+9.3f  %+.2e  %+.2e" % (
            R, *theta[i], u[j], p[k], a32[k] - p[k], a64[k] - p[k]))
    # where does the float32 error live: by log u band and by |kappa|
    lu = u[r[:, 2].astype(int)]
    kap = np.abs(theta[r[:, 1].astype(int), CTX.index("kappa")])
    print("|j32 - port| p90 by log u band:")
    for a_, b_ in ((-12, -8), (-8, -4), (-4, -1), (-1, 2)):
        m = (lu >= a_) & (lu < b_)
        if m.any():
            print("  [%3d, %3d)  n %4d  p90 %.2e  max %.2e" % (a_, b_, m.sum(), np.percentile(d[m], 90), d[m].max()))
    print("|j32 - port| p90 by |kappa|:")
    for a_, b_ in ((0, 3), (3, 6), (6, 10.01)):
        m = (kap >= a_) & (kap < b_)
        if m.any():
            print("  [%4.1f, %4.1f)  n %4d  p90 %.2e  max %.2e" % (a_, b_, m.sum(), np.percentile(d[m], 90), d[m].max()))


if __name__ == "__main__":
    main()

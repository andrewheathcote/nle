"""Golden vectors: a card's fixed inputs and the outputs the network must return.

A golden file is a fixed set of inputs with the outputs the network must
return, computed once in float64 from the trained weights; every port
(compiled EMC2, the R reference, an ONNX graph) is tested against it.
`nleflow export` writes it beside the card from this code, and
`nleflow golden <card>` rewrites the same file from the card alone, so that a
card whose golden file was not kept can be checked without the checkpoint.
numpy and scipy only (no JAX): the JAX float64 gate the exporter applies is
not repeated here.

    nleflow golden <card>_weights.json [<out>_golden.csv]

The grid is the exporters': 25 parameter sets drawn uniformly on the
network-scale box (seed 20260708 for joint/DDM cards, 20260707 for race
cards) with exact zeros forced into zero-box inputs, rounded through float32,
and 23 log-spaced times (1e-4..4 for a scaled DDM card, 0.005..20 otherwise).
Joint cards record the inputs on the exporter's storage scale (log, probit for
SZ, -Inf at an exact zero); race cards record the network scale.

This file is the NLE project's handover/nle-methods/flow/scripts/golden.py
(milestone M0) moved into the package; that script now calls this module.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sstats

MIN_BIN_SIZE = 1e-4
MIN_KNOT_SLOPE = 1e-4
AFFINE_MIN_SCALE = 1e-3
AFFINE_OFFSET = float(np.log(np.expm1(1.0 - 1e-3)))


# --- the float64 port (export_flow.py, kept in sync by the golden files) -----
def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def layernorm(x, scale, bias, eps=1e-6):
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
    h = ctx_std
    for i, (W, b) in enumerate(layers[:-1]):
        h = h @ W + b
        if norms is not None:
            h = layernorm(h, norms[i][0], norms[i][1], norms[i][2])
        h = gelu_tanh(h)
    W, b = layers[-1]
    return h @ W + b


def build_knots(raw, num_bins, range_min, range_max, min_bin_size, min_knot_slope):
    k = num_bins
    raw = raw[:, :3 * k + 1]
    widths = normalize_bin_sizes(raw[:, :k], range_max - range_min, min_bin_size)
    heights = normalize_bin_sizes(raw[:, k:2 * k], range_max - range_min, min_bin_size)
    slopes = normalize_knot_slopes(raw[:, 2 * k:], min_knot_slope)
    n = raw.shape[0]
    pad_lo = np.full((n, 1), range_min)
    pad_hi = np.full((n, 1), range_max)
    x_pos = np.concatenate([pad_lo, range_min + np.cumsum(widths[:, :-1], axis=1), pad_hi], axis=1)
    y_pos = np.concatenate([pad_lo, range_min + np.cumsum(heights[:, :-1], axis=1), pad_hi], axis=1)
    slopes[:, 0] = 1.0
    slopes[:, -1] = 1.0
    return x_pos, y_pos, slopes


def rqs_inverse(y, x_pos, y_pos, slopes):
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


# --- the card ---------------------------------------------------------------
def read_layers(mlp):
    layers = [(np.asarray(l["W"], dtype=np.float64), np.asarray(l["b"], dtype=np.float64))
              for l in mlp["layers"]]
    norms = None
    if mlp.get("use_norm") and mlp.get("norms"):
        norms = [(np.asarray(n["scale"], dtype=np.float64), np.asarray(n["bias"], dtype=np.float64),
                  float(n.get("eps", 1e-6))) for n in mlp["norms"]]
    return layers, norms


def spline_settings(sp):
    assert sp.get("num_splines", 1) == 1 and sp.get("boundary_slopes", "identity") == "identity"
    assert sp.get("output_transform", "exp") == "exp" and sp.get("base_distribution", "standard_normal") == "standard_normal"
    return (int(sp["num_bins"]), float(sp["range_min"]), float(sp["range_max"]),
            float(sp.get("min_bin_size", MIN_BIN_SIZE)), float(sp.get("min_knot_slope", MIN_KNOT_SLOPE)))


def fmt(v):
    if np.isfinite(v):
        return f"{v:.17g}"
    return "-Inf" if v < 0 else "Inf"


def standardise(card, theta):
    if card.get("input_scaling"):
        isc = card["input_scaling"]
        return (np.log(theta + np.asarray(isc["eps"])) - np.asarray(isc["loc"])) / np.asarray(isc["scale"])
    if card.get("scaler"):
        return (theta - np.asarray(card["scaler"]["mean"])) / np.asarray(card["scaler"]["scale"])
    return theta


def zerobox_inputs(card):
    ctx = list(card["context_names"])
    tr = card.get("context_transforms") or {}
    zc = card.get("zerobox_c") or {}
    return [(p, float(zc[p])) for p in ctx if tr.get(p) == "zerobox"]


# --- race cards -------------------------------------------------------------
def golden_race(card, n_sets=25, arrays=False):
    """Header and rows of a race card's golden file; with arrays=True also a
    dict of the grid and the float64 outputs (theta on the network scale,
    t_flow, z, log_pdf, cdf, log_sf) and the intermediates of the first three
    parameter sets (the exporter's <tag>_intermediates.json)."""
    ctx = list(card["context_names"])
    layers, norms = read_layers(card["mlp"])
    K, rmin, rmax, mbs, mks = spline_settings(card["spline"])
    affine = bool(card.get("affine"))
    lo = np.asarray(card["bounds_sampled"]["lower"], dtype=np.float64)
    hi = np.asarray(card["bounds_sampled"]["upper"], dtype=np.float64)
    rng = np.random.default_rng(20260707)
    theta = rng.uniform(lo, hi, size=(n_sets, len(lo)))
    for p, _ in zerobox_inputs(card):
        theta[:min(5, n_sets), ctx.index(p)] = 0.0
    theta = theta.astype(np.float32).astype(np.float64)
    rt = np.geomspace(0.005, 20.0, 23).astype(np.float32).astype(np.float64)
    pre = card.get("pre")
    t0_nat = None
    if pre:
        pb = card.get("pre_bounds_natural")
        if not pb:
            raise SystemExit("a decision-time card needs pre_bounds_natural to draw its t0 grid")
        # the exporters write {"lower": .., "upper": ..}; a plain [lower, upper] is read too
        lo0, hi0 = (pb["lower"], pb["upper"]) if isinstance(pb, dict) else (pb[0], pb[1])
        t0_s = rng.uniform(np.log(lo0), np.log(hi0), size=n_sets).astype(np.float32).astype(np.float64)
        t0_nat = np.exp(t0_s)
        rt_gold = t0_nat[:, None] + rt[None, :]
        t_flow = rt_gold - t0_nat[:, None]
    else:
        rt_gold = np.broadcast_to(rt, (n_sets, len(rt)))
        t_flow = rt_gold
    ctx_std = standardise(card, theta)
    raw = mlp_forward(ctx_std, layers, norms)
    x_pos, y_pos, slopes = build_knots(raw, K, rmin, rmax, mbs, mks)
    if affine:
        loc = raw[:, 3 * K + 1]
        asc = card.get("affine_scale") or {}
        scale = softplus(raw[:, 3 * K + 2] + float(asc.get("offset", AFFINE_OFFSET))) + float(asc.get("min_scale", AFFINE_MIN_SCALE))
    u = np.log(t_flow)
    rows = []
    header = ["param_id"] + ctx + (["t0"] if pre else []) + ["rt", "z", "log_pdf", "cdf", "log_sf"]
    Z = np.empty((n_sets, len(rt))); LP = np.empty_like(Z); CDF = np.empty_like(Z); LSF = np.empty_like(Z)
    for i in range(n_sets):
        if affine:
            z, logdet = rqs_inverse((u[i] - loc[i]) / scale[i], x_pos[i], y_pos[i], slopes[i])
            logdet = logdet - np.log(scale[i])
        else:
            z, logdet = rqs_inverse(u[i], x_pos[i], y_pos[i], slopes[i])
        lp = sstats.norm.logpdf(z) + logdet - u[i]
        cdf = sstats.norm.cdf(z)
        lsf = sstats.norm.logsf(z)
        Z[i], LP[i], CDF[i], LSF[i] = z, lp, cdf, lsf
        for j in range(len(rt)):
            row = [str(i + 1)] + [fmt(v) for v in theta[i]]
            if pre:
                row.append(fmt(t0_nat[i]))
            row += [fmt(rt_gold[i, j]), fmt(z[j]), fmt(lp[j]), fmt(cdf[j]), fmt(lsf[j])]
            rows.append(row)
    if not arrays:
        return header, rows
    inter = {"theta": theta[:3].tolist(), "ctx_std": ctx_std[:3].tolist(),
             "conditioner_raw": raw[:3].tolist(), "x_pos": x_pos[:3].tolist(),
             "y_pos": y_pos[:3].tolist(), "knot_slopes": slopes[:3].tolist(),
             **({"affine_loc": loc[:3].tolist(), "affine_scale": scale[:3].tolist()} if affine else {})}
    return header, rows, {"theta": theta, "ctx_std": ctx_std, "rt": rt_gold, "t_flow": t_flow,
                          "t0_nat": t0_nat, "z": Z, "log_pdf": LP, "cdf": CDF, "log_sf": LSF,
                          "intermediates": inter}


# --- joint cards (DDM flows, scaled or not) -----------------------------------
def golden_joint(card, n_sets=25, arrays=False):
    """Header and rows of a joint card's golden file (both responses); with
    arrays=True also the grid and the float64 outputs: theta_net (network
    scale), stored (the storage scale, -Inf at zeros), ctx_std, rt, and per
    response z, log_pdf (joint), cdf (defective), log_p_R, plus the st0 = 0
    support mask `hard` and the intermediates of the first three sets."""
    ctx = list(card["context_names"])
    if card.get("context_encoding") == "triple":
        raise SystemExit("the (SZ0, SZ, SZ1) encoding is not supported by golden.py")
    fl, fn = read_layers(card["flow_mlp"])
    cl, _ = read_layers(card["classifier_mlp"])
    K, rmin, rmax, mbs, mks = spline_settings(card["spline"])
    scaled = card.get("scaling") is not None
    zb = zerobox_inputs(card)
    lo = np.asarray(card["bounds_sampled"]["lower"], dtype=np.float64)
    hi = np.asarray(card["bounds_sampled"]["upper"], dtype=np.float64)
    rng = np.random.default_rng(20260708)
    theta = rng.uniform(lo, hi, size=(n_sets, len(lo))).astype(np.float32).astype(np.float64)
    rt = (np.geomspace(1e-4, 4.0, 23) if scaled else np.geomspace(0.005, 20.0, 23)).astype(np.float32).astype(np.float64)
    n12 = n_sets // 12
    zb_names = [p for p in ("sv", "SZ", "st0", "eta") if p in dict(zb)]
    for i, name in enumerate(zb_names):
        theta[i * n12:(i + 1) * n12, ctx.index(name)] = 0.0
    theta_net = theta.copy()
    stored = theta_net.copy()           # the exporter's storage scale: EMC2 sampled, -Inf at zeros
    for name, c in zb:
        j = ctx.index(name)
        nat = np.sinh(theta_net[:, j]) * c
        with np.errstate(divide="ignore"):
            stored[:, j] = sstats.norm.ppf(nat) if name == "SZ" else np.log(nat)
    ctx_std = standardise(card, theta_net)
    u = np.log(rt)
    # the st0 = 0 support rule of a zero-box DDM card: rt <= t0 has density 0
    hard = None
    if "st0" in dict(zb) and "t0" in ctx:
        j_st0, j_t0 = ctx.index("st0"), ctx.index("t0")
        t0_nat = np.exp(stored[:, j_t0]) if (card.get("context_transforms") or {}).get("t0") == "log" else stored[:, j_t0]
        hard = np.isneginf(stored[:, j_st0])[:, None] & (rt[None, :] <= t0_nat[:, None])
    header = ["param_id"] + ctx + ["R", "rt", "z", "log_pdf_joint", "cdf_defective", "log_p_R"]
    logit = mlp_forward(ctx_std, cl, None).squeeze(-1)
    res = {}; inter = {}
    for R in (1, 2):
        full_ctx = np.concatenate([ctx_std, np.full((n_sets, 1), float(R))], axis=1)
        raw = mlp_forward(full_ctx, fl, fn)
        x_pos, y_pos, slopes = build_knots(raw, K, rmin, rmax, mbs, mks)
        inter[R] = {"raw": raw, "x_pos": x_pos, "y_pos": y_pos, "slopes": slopes}
        log_p_R = -softplus(-logit) if R == 2 else -softplus(logit)
        z = np.empty((n_sets, len(rt))); lp = np.empty_like(z); cdf = np.empty_like(z)
        for i in range(n_sets):
            z[i], logdet = rqs_inverse(u, x_pos[i], y_pos[i], slopes[i])
            lp[i] = sstats.norm.logpdf(z[i]) + logdet - u + log_p_R[i]
            cdf[i] = sstats.norm.cdf(z[i]) * np.exp(log_p_R[i])
        if hard is not None:
            lp = np.where(hard, -np.inf, lp)
            cdf = np.where(hard, 0.0, cdf)
        res[R] = (z, lp, cdf, log_p_R)
    # the exporter's row order: parameter set, then response, then time
    rows = []
    for i in range(n_sets):
        th = [fmt(v) for v in stored[i]]
        for R in (1, 2):
            z, lp, cdf, log_p_R = res[R]
            for j in range(len(rt)):
                rows.append([str(i + 1)] + th + [str(R), fmt(rt[j]), fmt(z[i, j]), fmt(lp[i, j]), fmt(cdf[i, j]),
                                                 fmt(log_p_R[i])])
    if not arrays:
        return header, rows
    inter1 = inter[1]
    intermediates = {"theta": [[v if np.isfinite(v) else ("-Inf" if v < 0 else "Inf") for v in row]
                               for row in stored[:3].tolist()],
                     "ctx_std": ctx_std[:3].tolist(), "conditioner_raw_R1": inter1["raw"][:3].tolist(),
                     "x_pos_R1": inter1["x_pos"][:3].tolist(), "y_pos_R1": inter1["y_pos"][:3].tolist(),
                     "knot_slopes_R1": inter1["slopes"][:3].tolist(),
                     "classifier_logit": logit[:3].tolist()}
    return header, rows, {"theta_net": theta_net, "stored": stored, "ctx_std": ctx_std, "rt": rt,
                          "hard": hard, "res": res, "intermediates": intermediates}


def card_kind(card):
    if "members" in card:
        raise SystemExit("an ensemble bundle has one golden file per member; export the members separately")
    if "flow_mlp" in card and "classifier_mlp" in card:
        return "flow_scaled" if card.get("scaling") is not None else "flow_joint"
    if "mlp" in card and "spline" in card:
        return "flow_race"
    raise SystemExit("not a flow card (neither flow_mlp + classifier_mlp nor mlp + spline)")


def golden_rows(card):
    kind = card_kind(card)
    header, rows = golden_joint(card) if kind != "flow_race" else golden_race(card)
    return kind, header, rows


def golden_arrays(card):
    """(kind, header, rows, arrays) for either kind of flow card."""
    kind = card_kind(card)
    if kind == "flow_race":
        header, rows, arr = golden_race(card, arrays=True)
    else:
        header, rows, arr = golden_joint(card, arrays=True)
    return kind, header, rows, arr


def write_golden(header, rows, dst):
    with open(dst, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")


def golden_path_for(card_path):
    p = Path(card_path)
    name = p.name
    for suf in ("_weights.json", ".json"):
        if name.endswith(suf):
            name = name[:-len(suf)]
            break
    return p.with_name(name + "_golden.csv")


def main(src: str, dst: str | None = None) -> None:
    card = json.load(open(src))
    kind, header, rows = golden_rows(card)
    if dst is None:
        dst = str(golden_path_for(src))
    write_golden(header, rows, dst)
    print(f"wrote {dst}: {kind}, {len(rows)} rows, {len(header)} columns")


if __name__ == "__main__":
    main(*sys.argv[1:])

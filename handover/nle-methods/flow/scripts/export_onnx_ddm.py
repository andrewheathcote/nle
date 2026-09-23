"""Export our DDM flow as a single ONNX graph, LAN-style.

The published LANs (Fengler et al.) are ONNX MLPs mapping
(parameters, rt, choice) -> log-likelihood, which is what HSSM consumes. Our
flow is a conditioner MLP feeding a rational-quadratic spline, plus a choice
classifier, so it is a bigger graph -- but the interface can be identical, and
this script builds it with the `onnx` helper API (no torch).

INPUT  float64 [N, 10]:
    columns 0-7  the context on the SAMPLED (unconstrained) scale, in the
                 export's own order: v, log a, log t0, log s, probit Z,
                 probit SZ, log sv, log st0
    column  8    rt in seconds
    column  9    choice, -1 = lower boundary, +1 = upper (the LAN convention;
                 our R = 1 and 2 respectively)
OUTPUT float64 [N]: log p(rt, choice | theta), the JOINT defective log-density,
                 the same quantity a LAN returns.

The sampled scale is deliberate: the probit transform has no exact ONNX op, so
taking natural-scale Z and SZ would mean baking in an approximation. Callers
apply log() and qnorm() outside, where they are exact -- and must, because
silently mis-wiring a context is this project's most expensive recurring bug.

Outside the training box the graph returns `ll_floor_log` (-23.03), matching the
R port's refusal to extrapolate rather than inventing a value.

    python flow/scripts/export_onnx_ddm.py <weights.json> <out.onnx>
"""

import json
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FT = TensorProto.DOUBLE
NP = np.float64


def main(src: str, dst: str, mode: str = "trialwise", dtype: str = "float64") -> None:
    global FT, NP
    FT = TensorProto.DOUBLE if dtype == "float64" else TensorProto.FLOAT
    NP = np.float64 if dtype == "float64" else np.float32
    fl = json.load(open(src))
    sp = fl["spline"]
    K = int(sp["num_bins"])
    rng_min, rng_max = float(sp["range_min"]), float(sp["range_max"])
    rng_size = rng_max - rng_min
    assert sp["boundary_slopes"] == "identity" and sp["base_distribution"] == "standard_normal"
    assert fl["parameter_scale"] == "sampled" and not fl["flow_mlp"].get("use_norm")

    init, nodes = [], []

    def const(name, arr):
        init.append(numpy_helper.from_array(np.asarray(arr, dtype=NP), name))
        return name

    def iconst(name, arr):
        init.append(numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name))
        return name

    def node(op, ins, outs, **kw):
        nodes.append(helper.make_node(op, ins, outs, **kw))
        return outs[0]

    def mlp(x, spec, tag):
        """gelu_tanh on every layer but the last, exactly as .mlp_fwd does."""
        h = x
        L = spec["layers"]
        for i, lyr in enumerate(L):
            W = const(f"{tag}_W{i}", np.array(lyr["W"], dtype=NP))
            b = const(f"{tag}_b{i}", np.array(lyr["b"], dtype=NP))
            z = node("MatMul", [h, W], [f"{tag}_mm{i}"])
            h = node("Add", [z, b], [f"{tag}_add{i}"])
            if i < len(L) - 1:                      # 0.5x(1+tanh(c(x+0.044715x^3)))
                x3 = node("Mul", [h, h], [f"{tag}_sq{i}"])
                x3 = node("Mul", [x3, h], [f"{tag}_cu{i}"])
                s = node("Mul", [x3, const(f"{tag}_k{i}", 0.044715)], [f"{tag}_ks{i}"])
                s = node("Add", [h, s], [f"{tag}_in{i}"])
                s = node("Mul", [s, const(f"{tag}_c{i}", np.sqrt(2.0 / np.pi))], [f"{tag}_sc{i}"])
                t = node("Tanh", [s], [f"{tag}_th{i}"])
                t = node("Add", [t, const(f"{tag}_one{i}", 1.0)], [f"{tag}_1p{i}"])
                t = node("Mul", [t, h], [f"{tag}_xt{i}"])
                h = node("Mul", [t, const(f"{tag}_half{i}", 0.5)], [f"{tag}_gelu{i}"])
        return h

    def slice_cols(x, lo, hi, tag):
        return node("Slice", [x, iconst(f"{tag}_s", [lo]), iconst(f"{tag}_e", [hi]),
                              iconst(f"{tag}_a", [1])], [f"{tag}_out"])

    if mode == "trialwise":
        inputs = [helper.make_tensor_value_info("input", FT, ["N", 10])]
        theta = slice_cols("input", 0, 8, "theta")        # [N,8] sampled scale
        rt = slice_cols("input", 8, 9, "rt")              # [N,1]
        choice = slice_cols("input", 9, 10, "choice")     # [N,1]
    else:
        # shared-theta: one parameter vector, many trials. The conditioner and
        # the classifier then run ONCE (twice for the flow, once per response)
        # instead of once per trial -- which is what the compiled port does, and
        # why it is ~8x faster than the trial-wise graph.
        inputs = [helper.make_tensor_value_info("theta", FT, [8]),
                  helper.make_tensor_value_info("rt", FT, ["N"]),
                  helper.make_tensor_value_info("choice", FT, ["N"])]
        theta = node("Reshape", ["theta", iconst("th_shape", [1, 8])], ["theta_row"])
        rt = node("Reshape", ["rt", iconst("rt_shape", [-1, 1])], ["rt_col"])
        choice = node("Reshape", ["choice", iconst("ch_shape", [-1, 1])], ["choice_col"])
    out = helper.make_tensor_value_info("log_likelihood", FT, ["N"])

    # ---- standardise, append raw R (1 or 2), run the conditioner -------------
    mu = const("scaler_mean", np.array(fl["scaler"]["mean"], dtype=NP))
    sd = const("scaler_scale", np.array(fl["scaler"]["scale"], dtype=NP))
    ctx = node("Sub", [theta, mu], ["ctx_c"])
    ctx = node("Div", [ctx, sd], ["ctx_std"])
    if mode == "trialwise":
        R = node("Add", [choice, const("three", 3.0)], ["R_a"])   # -1 -> 1, +1 -> 2
        R = node("Mul", [R, const("half", 0.5)], ["R"])
        flow_in = node("Concat", [ctx, R], ["flow_in"], axis=1)
    else:
        ctx2 = node("Tile", [ctx, iconst("tile2", [2, 1])], ["ctx2"])
        flow_in = node("Concat", [ctx2, const("R_rows", np.array([[1.0], [2.0]]))],
                       ["flow_in"], axis=1)                       # [2, 9]
    raw = mlp(flow_in, fl["flow_mlp"], "flow")                    # [rows, 3K+1]

    # ---- spline knots --------------------------------------------------------
    def knots(raw_slice, tag):
        sm = node("Softmax", [raw_slice], [f"{tag}_sm"], axis=1)
        w = node("Mul", [sm, const(f"{tag}_span", rng_size - K * sp["min_bin_size"])],
                 [f"{tag}_span_m"])
        w = node("Add", [w, const(f"{tag}_min", float(sp["min_bin_size"]))], [f"{tag}_w"])
        cw = node("CumSum", [w, iconst(f"{tag}_ax", 1)], [f"{tag}_cw"])
        head = node("Slice", [cw, iconst(f"{tag}_hs", [0]), iconst(f"{tag}_he", [K - 1]),
                              iconst(f"{tag}_ha", [1])], [f"{tag}_head"])
        zero = node("Mul", [node("Slice", [cw, iconst(f"{tag}_zs", [0]), iconst(f"{tag}_ze", [1]),
                                           iconst(f"{tag}_za", [1])], [f"{tag}_z0"]),
                            const(f"{tag}_zero", 0.0)], [f"{tag}_zc"])
        full = node("Add", [zero, const(f"{tag}_size", rng_size)], [f"{tag}_full"])
        pos = node("Concat", [zero, head, full], [f"{tag}_cat"], axis=1)   # [N, K+1]
        return node("Add", [pos, const(f"{tag}_rmin", rng_min)], [f"{tag}_pos"])

    x_pos = knots(slice_cols(raw, 0, K, "wraw"), "xk")
    y_pos = knots(slice_cols(raw, K, 2 * K, "hraw"), "yk")
    sraw = slice_cols(raw, 2 * K, 3 * K + 1, "sraw")
    off = float(np.log(np.exp(1.0 - sp["min_knot_slope"]) - 1.0))
    sl = node("Add", [sraw, const("slope_off", off)], ["sl_off"])
    sl = node("Softplus", [sl], ["sl_sp"])
    sl = node("Add", [sl, const("slope_min", float(sp["min_knot_slope"]))], ["sl_all"])
    mid = node("Slice", [sl, iconst("sl_ms", [1]), iconst("sl_me", [K]),
                         iconst("sl_ma", [1])], ["sl_mid"])
    one_col = node("Add", [node("Mul", [node("Slice", [sl, iconst("sl_1s", [0]), iconst("sl_1e", [1]),
                                                       iconst("sl_1a", [1])], ["sl_first"]),
                                        const("zero_m", 0.0)], ["sl_zero"]),
                           const("one_c", 1.0)], ["ones_col"])
    slopes = node("Concat", [one_col, mid, one_col], ["slopes_rows"], axis=1)

    if mode == "shared":
        # pick each trial's response row: choice -1 -> row 0, +1 -> row 1
        sel = node("Cast", [node("Greater", [choice, const("zero_sel", 0.0)], ["is_up"])],
                   ["sel_i"], to=TensorProto.INT64)
        sel = node("Reshape", [sel, iconst("sel_shape", [-1])], ["sel_flat"])
        x_pos = node("Gather", [x_pos, sel], ["x_pos_t"], axis=0)
        y_pos = node("Gather", [y_pos, sel], ["y_pos_t"], axis=0)
        slopes = node("Gather", ["slopes_rows", sel], ["slopes"], axis=0)
    else:
        slopes = node("Identity", ["slopes_rows"], ["slopes"])

    # ---- locate the bin, then invert the RQS --------------------------------
    u = node("Log", [rt], ["u"])                                   # [N,1]
    le = node("LessOrEqual", [y_pos, u], ["le"])
    cnt = node("ReduceSum", [node("Cast", [le], ["le_d"], to=TensorProto.INT64),
                             iconst("cnt_ax", [1])], ["cnt"], keepdims=1)
    idx = node("Sub", [cnt, iconst("one_i", 1)], ["idx_raw"])
    idx = node("Clip", [idx, iconst("idx_lo", 0), iconst("idx_hi", K - 1)], ["idx"])
    idx1 = node("Add", [idx, iconst("one_i2", 1)], ["idx1"])

    def g(src, ix, tag):
        return node("GatherElements", [src, ix], [tag], axis=1)

    x_lo, x_hi = g(x_pos, idx, "x_lo"), g(x_pos, idx1, "x_hi")
    y_lo, y_hi = g(y_pos, idx, "y_lo"), g(y_pos, idx1, "y_hi")
    d_lo, d_hi = g(slopes, idx, "d_lo"), g(slopes, idx1, "d_hi")

    bw = node("Sub", [x_hi, x_lo], ["bw"])
    bh = node("Sub", [y_hi, y_lo], ["bh"])
    bs = node("Div", [bh, bw], ["bs"])
    w = node("Div", [node("Sub", [u, y_lo], ["u_ylo"]), bh], ["w_raw"])
    w = node("Clip", [w, const("w_lo", 0.0), const("w_hi", 1.0)], ["w"])
    st = node("Sub", [node("Add", [d_hi, d_lo], ["d_sum"]),
                      node("Mul", [bs, const("two", 2.0)], ["bs2"])], ["slopes_term"])
    cc = node("Mul", [node("Neg", [bs], ["nbs"]), w], ["cc"])
    bb = node("Sub", [d_lo, node("Mul", [st, w], ["stw"])], ["bb"])
    aa = node("Sub", [bs, bb], ["aa"])
    disc = node("Sub", [node("Mul", [bb, bb], ["bb2"]),
                        node("Mul", [node("Mul", [aa, cc], ["ac"]), const("four", 4.0)], ["ac4"])],
                ["disc"])
    pos_disc = node("Greater", [disc, const("zero_d", 0.0)], ["pos_disc"])
    safe = node("Sqrt", [node("Where", [pos_disc, disc, const("tiny", np.finfo(NP).tiny)],
                              ["disc_safe"])], ["sqrt_raw"])
    safe = node("Where", [pos_disc, safe, const("zero_s", 0.0)], ["safe_sqrt"])
    bb_nonneg = node("GreaterOrEqual", [bb, const("zero_b", 0.0)], ["bb_nonneg"])
    num = node("Where", [bb_nonneg, node("Mul", [cc, const("two_n", 2.0)], ["cc2"]),
                         node("Add", [node("Neg", [bb], ["nbb"]), safe], ["nbb_s"])], ["num"])
    den = node("Where", [bb_nonneg, node("Sub", [node("Neg", [bb], ["nbb2"]), safe], ["nbb_m"]),
                         node("Mul", [aa, const("two_d", 2.0)], ["aa2"])], ["den"])
    zz = node("Clip", [node("Div", [num, den], ["z_raw"]), const("z_lo", 0.0), const("z_hi", 1.0)],
              ["zz"])
    x_spline = node("Add", [x_lo, node("Mul", [bw, zz], ["bwz"])], ["x_spline"])

    sq = node("Mul", [zz, zz], ["sq"])
    z1mz = node("Sub", [zz, sq], ["z1mz"])
    omz = node("Sub", [const("one_z", 1.0), zz], ["omz"])
    sq1 = node("Mul", [omz, omz], ["sq1"])
    denom = node("Add", [bs, node("Mul", [st, z1mz], ["st_z"])], ["denominator"])
    term = node("Add", [node("Add", [node("Mul", [d_hi, sq], ["dhi_sq"]),
                                     node("Mul", [node("Mul", [bs, z1mz], ["bs_z"]),
                                                  const("two_t", 2.0)], ["bs_z2"])], ["t1"]),
                        node("Mul", [d_lo, sq1], ["dlo_sq1"])], ["term"])
    ld = node("Add", [node("Sub", [node("Mul", [node("Log", [bs], ["log_bs"]),
                                                const("neg_two", -2.0)], ["m2logbs"]),
                                   node("Log", [term], ["log_term"])], ["ld1"]),
                      node("Mul", [node("Log", [denom], ["log_den"]), const("two_l", 2.0)],
                           ["2logden"])], ["logdet_spline"])

    y0 = node("Slice", [y_pos, iconst("y0s", [0]), iconst("y0e", [1]), iconst("y0a", [1])], ["y0"])
    yK = node("Slice", [y_pos, iconst("yKs", [K]), iconst("yKe", [K + 1]), iconst("yKa", [1])], ["yK"])
    x0 = node("Slice", [x_pos, iconst("x0s", [0]), iconst("x0e", [1]), iconst("x0a", [1])], ["x0"])
    xK = node("Slice", [x_pos, iconst("xKs", [K]), iconst("xKe", [K + 1]), iconst("xKa", [1])], ["xK"])
    below = node("LessOrEqual", [u, y0], ["below"])
    above = node("GreaterOrEqual", [u, yK], ["above"])
    x_bel = node("Add", [node("Sub", [u, y0], ["u_y0"]), x0], ["x_bel"])
    x_abo = node("Add", [node("Sub", [u, yK], ["u_yK"]), xK], ["x_abo"])
    x = node("Where", [below, x_bel, node("Where", [above, x_abo, x_spline], ["x_in"])], ["x"])
    tail = node("Or", [below, above], ["tail"])
    logdet = node("Where", [tail, const("zero_ld", 0.0), ld], ["logdet"])

    # ---- classifier and assembly --------------------------------------------
    logit = mlp(ctx, fl["classifier_mlp"], "clf")                  # [N,1], logit for R=2
    up = node("Greater", [choice, const("zero_c", 0.0)], ["is_upper"])
    lp_up = node("Neg", [node("Softplus", [node("Neg", [logit], ["neg_logit"])], ["sp_neg"])],
                 ["lp_upper"])
    lp_lo = node("Neg", [node("Softplus", [logit], ["sp_pos"])], ["lp_lower"])
    log_p_R = node("Where", [up, lp_up, lp_lo], ["log_p_R"])

    quad = node("Mul", [node("Mul", [x, x], ["x2"]), const("neg_half", -0.5)], ["quad"])
    norm = node("Add", [quad, const("log_norm", -0.5 * np.log(2 * np.pi))], ["log_phi"])
    ll = node("Add", [node("Sub", [node("Add", [norm, logdet], ["a1"]), u], ["a2"]), log_p_R],
              ["ll_raw"])

    # ---- box check: refuse to extrapolate ------------------------------------
    lo = const("box_lo", np.array(fl["bounds_sampled"]["lower"], dtype=NP))
    hi = const("box_hi", np.array(fl["bounds_sampled"]["upper"], dtype=NP))
    okl = node("GreaterOrEqual", [theta, lo], ["ok_lo"])
    okh = node("LessOrEqual", [theta, hi], ["ok_hi"])
    okb = node("And", [okl, okh], ["ok_both"])
    okn = node("ReduceMin", [node("Cast", [okb], ["ok_d"], to=FT), iconst("ok_ax", [1])],
               ["ok_all"], keepdims=1)
    in_box = node("Greater", [okn, const("half_ok", 0.5)], ["in_box"])
    ll = node("Where", [in_box, ll, const("floor", float(fl["ll_floor_log"]))], ["ll_boxed"])
    node("Squeeze", [ll, iconst("sq_ax", [1])], ["log_likelihood"])

    graph = helper.make_graph(nodes, "ddm_flow", inputs, [out], initializer=init)
    model = helper.make_model(graph, producer_name="nle-methods",
                              opset_imports=[helper.make_opsetid("", 18)])
    # onnxruntime 1.30 caps IR at 10; onnx 1.23 defaults to 14. Pin it so the
    # artefact loads in the runtime HSSM actually uses.
    model.ir_version = 10
    iface = ("Input [N,10]: 8 context values on the sampled scale (v, log a, "
             "log t0, log s, probit Z, probit SZ, log sv, log st0), rt, choice in "
             "{-1,+1}." if mode == "trialwise" else
             "Inputs: theta [8] on the sampled scale (v, log a, log t0, log s, "
             "probit Z, probit SZ, log sv, log st0), rt [N], choice [N] in {-1,+1}. "
             "The conditioner and classifier run once, not per trial.")
    model.doc_string = (
        "DDM neural likelihood (spline flow + choice classifier). " + iface +
        " Output: joint log-density log p(rt, choice | theta). Outside the "
        "training box the output is the floor " + str(fl["ll_floor_log"]) + "."
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, dst)
    print(f"wrote {dst}  ({len(nodes)} nodes, {len(init)} initialisers)")


if __name__ == "__main__":
    main(*sys.argv[1:])

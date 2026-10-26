"""`nleflow onnx <card>`: a DDM flow card as one ONNX graph, LAN-style.

Two card kinds, two signatures each (NLEFLOW_PLAN.md section 7.3):

  flow_joint   the eight-parameter DDM flows (export_onnx_ddm.py's graph):
               INPUT float64 [N, k + 2]: the k context values on the card's
               stored (sampled) scale in context_names order (a zero-box
               input as log x, or probit x for SZ, -Inf at an exact zero),
               rt in seconds, choice (-1 lower, +1 upper).
  flow_scaled  the scaled DDM card: INPUT float64 [N, 10]: the natural
               DDM parameters v, a, t0, s, Z, SZ (the start-point
               proportion), sv, st0, then rt and choice. The graph computes
               kappa = v a/s^2, eta = sv a/s^2, the zero-box encodings, the
               probit of Z, c = (s/a)^2, u = (rt - t0) c, and the st0 branch:
               c g(u) when st0 = 0 (or the window is narrower than 1e-7 d),
               [G(u1) - G(max(u0, 0))] / st0 otherwise, with G the network's
               defective CDF p_R Phi(z(u)); zero density (-Inf) for rt <= t0.
               Exactly EMC2's nn_eval_scaled() / nn_trial_ll_scaled().
  OUTPUT float64 [N]: log p(rt, choice | theta), the joint defective log
               density a LAN returns. Outside the training box (the network-
               scale inputs against bounds_sampled) the output is the card's
               ll_floor_log, as the eight-parameter graph does.

  --signature trialwise   one input [N, k + 2] (the LAN shape; HSSM's)
  --signature shared      theta [k], rt [N], choice [N]: the conditioner runs
                          twice (once per response) and the classifier once,
                          not once per trial

Everything is float64 (float32 was rejected at 1.9e-4; EMC2_INTEGRATION.md).
onnxruntime has no double-precision Erf or Asinh kernels, so the normal CDF
is Cody's algorithm and the probit Wichura's AS 241, as R computes them
(rmath.py, the same code path EMC2's R::pnorm takes), and asinh(x) is
log(x + sqrt(x^2 + 1)). Opset 18, IR version 10 (onnxruntime <= 1.30).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import rmath

SCALED_INPUTS = ["v", "a", "t0", "s", "Z", "SZ", "sv", "st0"]
NEG_INF = -np.inf


class Graph:
    """A thin builder over onnx.helper: every op appends a node and returns
    the output name; constants become initialisers."""

    def __init__(self):
        from onnx import TensorProto
        self.nodes, self.init, self.n = [], [], 0
        self.FT = TensorProto.DOUBLE

    def name(self, tag):
        self.n += 1
        return f"{tag}_{self.n}"

    def const(self, arr, tag="c"):
        from onnx import numpy_helper
        nm = self.name(tag)
        self.init.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float64), nm))
        return nm

    def iconst(self, arr, tag="i"):
        from onnx import numpy_helper
        nm = self.name(tag)
        self.init.append(numpy_helper.from_array(np.asarray(arr, dtype=np.int64), nm))
        return nm

    def op(self, kind, ins, tag=None, **attrs):
        from onnx import helper
        out = self.name(tag or kind.lower())
        self.nodes.append(helper.make_node(kind, list(ins), [out], **attrs))
        return out

    # arithmetic on tensors or Python floats
    def _t(self, x):
        return x if isinstance(x, str) else self.const(float(x))

    def add(self, a, b): return self.op("Add", [self._t(a), self._t(b)])
    def sub(self, a, b): return self.op("Sub", [self._t(a), self._t(b)])
    def mul(self, a, b): return self.op("Mul", [self._t(a), self._t(b)])
    def div(self, a, b): return self.op("Div", [self._t(a), self._t(b)])
    def neg(self, a): return self.op("Neg", [a])
    def exp(self, a): return self.op("Exp", [a])
    def log(self, a): return self.op("Log", [a])
    def sqrt(self, a): return self.op("Sqrt", [a])
    def abs(self, a): return self.op("Abs", [a])
    def floor(self, a): return self.op("Floor", [a])
    def sign(self, a): return self.op("Sign", [a])
    def tanh(self, a): return self.op("Tanh", [a])
    def softplus(self, a): return self.op("Softplus", [a])
    def maximum(self, a, b): return self.op("Max", [self._t(a), self._t(b)])
    def minimum(self, a, b): return self.op("Min", [self._t(a), self._t(b)])
    def gt(self, a, b): return self.op("Greater", [self._t(a), self._t(b)])
    def ge(self, a, b): return self.op("GreaterOrEqual", [self._t(a), self._t(b)])
    def lt(self, a, b): return self.op("Less", [self._t(a), self._t(b)])
    def le(self, a, b): return self.op("LessOrEqual", [self._t(a), self._t(b)])
    def and_(self, a, b): return self.op("And", [a, b])
    def or_(self, a, b): return self.op("Or", [a, b])
    def not_(self, a): return self.op("Not", [a])
    def where(self, c, a, b): return self.op("Where", [c, self._t(a), self._t(b)])
    def clip(self, a, lo, hi): return self.op("Clip", [a, self.const(lo), self.const(hi)])
    def concat(self, xs, axis=1): return self.op("Concat", xs, axis=axis)
    def slice_cols(self, x, lo, hi): return self.op("Slice", [x, self.iconst([lo]), self.iconst([hi]), self.iconst([1])])

    def poly(self, x, coefs):
        """Horner: ((c0 x + c1) x + c2) ... for coefs in that order."""
        acc = self.const(coefs[0])
        for c in coefs[1:]:
            acc = self.add(self.mul(acc, x), c)
        return acc

    # --- the elementary functions the runtime lacks in double --------------------
    def asinh(self, x):
        """asinh(x) = log(x + sqrt(x^2 + 1)), for x >= 0 (the zero-box inputs)."""
        return self.log(self.add(x, self.sqrt(self.add(self.mul(x, x), 1.0))))

    def trunc16(self, x):
        return self.div(self.mul(self.sign(x), self.floor(self.mul(self.abs(x), 16.0))), 16.0)

    def pnorm(self, x):
        """R's pnorm (Cody), rmath.pnorm transliterated."""
        A, B, C, D, P, Q = rmath._A, rmath._B, rmath._C, rmath._D, rmath._P, rmath._Q
        y = self.abs(x)
        xsq = self.mul(x, x)
        xnum = self.mul(self.const(A[4]), xsq); xden = xsq
        for i in range(3):
            xnum = self.mul(self.add(xnum, A[i]), xsq); xden = self.mul(self.add(xden, B[i]), xsq)
        cum1 = self.add(0.5, self.div(self.mul(x, self.add(xnum, A[3])), self.add(xden, B[3])))
        xnum = self.mul(self.const(C[8]), y); xden = y
        for i in range(7):
            xnum = self.mul(self.add(xnum, C[i]), y); xden = self.mul(self.add(xden, D[i]), y)
        temp2 = self.div(self.add(xnum, C[7]), self.add(xden, D[7]))
        xsq2 = self.trunc16(y)
        del2 = self.mul(self.sub(y, xsq2), self.add(y, xsq2))
        upper2 = self.mul(self.mul(self.exp(self.neg(self.mul(xsq2, self.div(xsq2, 2.0)))),
                                   self.exp(self.neg(self.div(del2, 2.0)))), temp2)
        pos = self.gt(x, 0.0)
        cum2 = self.where(pos, self.sub(1.0, upper2), upper2)
        safe_x = self.where(self.gt(y, 1e-300), x, 1.0)
        xsq3 = self.div(1.0, self.mul(safe_x, safe_x))
        xnum = self.mul(self.const(P[5]), xsq3); xden = xsq3
        for i in range(4):
            xnum = self.mul(self.add(xnum, P[i]), xsq3); xden = self.mul(self.add(xden, Q[i]), xsq3)
        temp3 = self.mul(xsq3, self.div(self.add(xnum, P[4]), self.add(xden, Q[4])))
        temp3 = self.div(self.sub(rmath.M_1_SQRT_2PI, temp3), self.where(self.gt(y, 1e-300), y, 1.0))
        xsq3b = self.trunc16(x)
        del3 = self.mul(self.sub(x, xsq3b), self.add(x, xsq3b))
        upper3 = self.mul(self.mul(self.exp(self.neg(self.mul(xsq3b, self.div(xsq3b, 2.0)))),
                                   self.exp(self.neg(self.div(del3, 2.0)))), temp3)
        cum3 = self.where(pos, self.sub(1.0, upper3), upper3)
        far = self.where(pos, 1.0, 0.0)
        in3 = self.and_(self.gt(x, -38.4674), self.lt(x, 8.2924))
        return self.where(self.le(y, 0.67448975), cum1,
                          self.where(self.le(y, rmath.M_SQRT_32), cum2, self.where(in3, cum3, far)))

    def qnorm(self, p):
        """R's qnorm (AS 241) for p in (0, 1), rmath.qnorm transliterated."""
        q = self.sub(p, 0.5)
        r = self.sub(0.180625, self.mul(q, q))
        num = self.poly(r, [2509.0809287301226727, 33430.575583588128105, 67265.770927008700853, 45921.953931549871457,
                            13731.693765509461125, 1971.5909503065514427, 133.14166789178437745, 3.387132872796366608])
        den = self.poly(r, [5226.495278852854561, 28729.085735721942674, 39307.89580009271061, 21213.794301586595867,
                            5394.1960214247511077, 687.1870074920579083, 42.313330701600911252, 1.0])
        val1 = self.div(self.mul(q, num), den)
        pm = self.where(self.gt(q, 0.0), self.sub(1.0, p), p)
        pm = self.where(self.gt(pm, 0.0), pm, 0.5)
        rr = self.sqrt(self.neg(self.log(pm)))
        r5 = self.sub(rr, 1.6)
        val2 = self.div(self.poly(r5, [7.7454501427834140764e-4, 0.0227238449892691845833, 0.24178072517745061177,
                                       1.27045825245236838258, 3.64784832476320460504, 5.7694972214606914055,
                                       4.6303378461565452959, 1.42343711074968357734]),
                        self.poly(r5, [1.05075007164441684324e-9, 5.475938084995344946e-4, 0.0151986665636164571966,
                                       0.14810397642748007459, 0.68976733498510000455, 1.6763848301838038494,
                                       2.05319162663775882187, 1.0]))
        r27 = self.sub(rr, 5.0)
        val3 = self.div(self.poly(r27, [2.01033439929228813265e-7, 2.71155556874348757815e-5, 0.0012426609473880784386,
                                        0.026532189526576123093, 0.29656057182850489123, 1.7848265399172913358,
                                        5.4637849111641143699, 6.6579046435011037772]),
                        self.poly(r27, [2.04426310338993978564e-15, 1.4215117583164458887e-7, 1.8463183175100546818e-5,
                                        7.868691311456132591e-4, 0.0148753612908506148525, 0.13692988092273580531,
                                        0.59983220655588793769, 1.0]))
        tail = self.where(self.le(rr, 5.0), val2, val3)
        tail = self.where(self.lt(q, 0.0), self.neg(tail), tail)
        return self.where(self.le(self.abs(q), 0.425), val1, tail)

    # --- the networks ----------------------------------------------------------------
    def gelu(self, h):
        inner = self.mul(self.add(h, self.mul(self.mul(self.mul(h, h), h), 0.044715)), float(np.sqrt(2.0 / np.pi)))
        return self.mul(self.mul(self.add(self.tanh(inner), 1.0), h), 0.5)

    def mlp(self, x, spec):
        """gelu_tanh on every layer but the last (the cards' MLPs)."""
        h = x
        L = spec["layers"]
        for i, lyr in enumerate(L):
            h = self.add(self.op("MatMul", [h, self.const(np.array(lyr["W"]), "W")]), self.const(np.array(lyr["b"]), "b"))
            if i < len(L) - 1:
                h = self.gelu(h)
        return h

    def knots(self, raw, sp):
        """x_pos, y_pos, slopes [rows, K+1] from the conditioner output [rows, 3K+1]."""
        K = int(sp["num_bins"]); rmin, rmax = float(sp["range_min"]), float(sp["range_max"])
        size = rmax - rmin

        def positions(sl):
            sm = self.op("Softmax", [sl], axis=1)
            w = self.add(self.mul(sm, size - K * float(sp["min_bin_size"])), float(sp["min_bin_size"]))
            cw = self.op("CumSum", [w, self.iconst(1)])
            head = self.slice_cols(cw, 0, K - 1)
            first = self.slice_cols(cw, 0, 1)
            zero = self.mul(first, 0.0)
            full = self.add(zero, size)
            return self.add(self.concat([zero, head, full]), rmin)

        x_pos = positions(self.slice_cols(raw, 0, K))
        y_pos = positions(self.slice_cols(raw, K, 2 * K))
        sraw = self.slice_cols(raw, 2 * K, 3 * K + 1)
        off = float(np.log(np.exp(1.0 - float(sp["min_knot_slope"])) - 1.0))
        sl = self.add(self.softplus(self.add(sraw, off)), float(sp["min_knot_slope"]))
        mid = self.slice_cols(sl, 1, K)
        one = self.add(self.mul(self.slice_cols(sl, 0, 1), 0.0), 1.0)
        return x_pos, y_pos, self.concat([one, mid, one]), K

    def rqs_inverse(self, u, x_pos, y_pos, slopes, K):
        """z and log|dz/du| of the spline inverse at u [N,1] with per-row knots
        [N,K+1] (distrax's _rational_quadratic_spline_inv, identity tails)."""
        from onnx import TensorProto
        le = self.le(y_pos, u)
        cnt = self.op("ReduceSum", [self.op("Cast", [le], to=TensorProto.INT64), self.iconst([1])], keepdims=1)
        idx = self.op("Clip", [self.op("Sub", [cnt, self.iconst(1)]), self.iconst(0), self.iconst(K - 1)])
        idx1 = self.op("Add", [idx, self.iconst(1)])

        def g(src, ix):
            return self.op("GatherElements", [src, ix], axis=1)
        x_lo, x_hi = g(x_pos, idx), g(x_pos, idx1)
        y_lo, y_hi = g(y_pos, idx), g(y_pos, idx1)
        d_lo, d_hi = g(slopes, idx), g(slopes, idx1)
        bw = self.sub(x_hi, x_lo); bh = self.sub(y_hi, y_lo); bs = self.div(bh, bw)
        w = self.clip(self.div(self.sub(u, y_lo), bh), 0.0, 1.0)
        st = self.sub(self.add(d_hi, d_lo), self.mul(bs, 2.0))
        cc = self.mul(self.neg(bs), w)
        bb = self.sub(d_lo, self.mul(st, w))
        aa = self.sub(bs, bb)
        disc = self.sub(self.mul(bb, bb), self.mul(self.mul(aa, cc), 4.0))
        pos_disc = self.gt(disc, 0.0)
        safe = self.sqrt(self.where(pos_disc, disc, float(np.finfo(np.float64).tiny)))
        safe = self.where(pos_disc, safe, 0.0)
        bb_nonneg = self.ge(bb, 0.0)
        num = self.where(bb_nonneg, self.mul(cc, 2.0), self.add(self.neg(bb), safe))
        den = self.where(bb_nonneg, self.sub(self.neg(bb), safe), self.mul(aa, 2.0))
        zz = self.clip(self.div(num, den), 0.0, 1.0)
        x_spline = self.add(x_lo, self.mul(bw, zz))
        sq = self.mul(zz, zz); z1mz = self.sub(zz, sq); omz = self.sub(1.0, zz); sq1 = self.mul(omz, omz)
        denom = self.add(bs, self.mul(st, z1mz))
        term = self.add(self.add(self.mul(d_hi, sq), self.mul(self.mul(bs, z1mz), 2.0)), self.mul(d_lo, sq1))
        ld = self.add(self.sub(self.mul(self.log(bs), -2.0), self.log(term)), self.mul(self.log(denom), 2.0))
        y0 = self.slice_cols(y_pos, 0, 1); yK = self.slice_cols(y_pos, K, K + 1)
        x0 = self.slice_cols(x_pos, 0, 1); xK = self.slice_cols(x_pos, K, K + 1)
        below = self.le(u, y0); above = self.ge(u, yK)
        x = self.where(below, self.add(self.sub(u, y0), x0), self.where(above, self.add(self.sub(u, yK), xK), x_spline))
        logdet = self.where(self.or_(below, above), 0.0, ld)
        return x, logdet

    def log_phi(self, z):
        return self.add(self.mul(self.mul(z, z), -0.5), float(-0.5 * np.log(2 * np.pi)))


def build(card, signature="trialwise"):
    """The ONNX model of a flow_joint or flow_scaled card."""
    import onnx
    from onnx import TensorProto, helper
    from .golden import card_kind
    kind = card_kind(card)
    if kind not in ("flow_joint", "flow_scaled"):
        raise ValueError(f"ONNX export is for DDM flow cards (flow_joint, flow_scaled), not {kind}")
    if signature not in ("trialwise", "shared"):
        raise ValueError("signature must be trialwise or shared")
    sp = card["spline"]
    assert sp.get("boundary_slopes", "identity") == "identity" and sp.get("base_distribution", "standard_normal") == "standard_normal"
    assert card.get("parameter_scale", "sampled") == "sampled" and not card["flow_mlp"].get("use_norm")
    ctx = list(card["context_names"])
    k = len(ctx)
    tr = card.get("context_transforms") or {}
    zc = card.get("zerobox_c") or {}
    scaled = kind == "flow_scaled"
    n_in = len(SCALED_INPUTS) if scaled else k
    g = Graph()
    FT = TensorProto.DOUBLE

    # --- inputs -------------------------------------------------------------------
    if signature == "trialwise":
        inputs = [helper.make_tensor_value_info("input", FT, ["N", n_in + 2])]
        theta = g.slice_cols("input", 0, n_in)           # [N, n_in]
        rt = g.slice_cols("input", n_in, n_in + 1)       # [N, 1]
        choice = g.slice_cols("input", n_in + 1, n_in + 2)
    else:
        inputs = [helper.make_tensor_value_info("theta", FT, [n_in]),
                  helper.make_tensor_value_info("rt", FT, ["N"]),
                  helper.make_tensor_value_info("choice", FT, ["N"])]
        theta = g.op("Reshape", ["theta", g.iconst([1, n_in])])
        rt = g.op("Reshape", ["rt", g.iconst([-1, 1])])
        choice = g.op("Reshape", ["choice", g.iconst([-1, 1])])
    out = helper.make_tensor_value_info("log_likelihood", FT, ["N"])

    # --- the network-scale context [rows, k] --------------------------------------
    def col(j):
        return g.slice_cols(theta, j, j + 1)

    if scaled:
        v, a, t0, s, Z, SZ, sv, st0 = (col(j) for j in range(8))
        s2 = g.mul(s, s)
        kappa = g.div(g.mul(v, a), s2)
        eta = g.div(g.mul(sv, a), s2)
        SZp = g.maximum(SZ, 0.0)
        nat = {"kappa": kappa, "eta": eta, "Z": Z, "SZ": SZp}
        cols = []
        for name in ctx:
            t = tr[name]
            x = nat[name]
            if t == "identity":
                cols.append(x)
            elif t == "log":
                cols.append(g.log(x))
            elif t == "probit":
                cols.append(g.qnorm(x))
            elif t == "zerobox":
                cols.append(g.asinh(g.div(x, float(zc[name]))))
            else:
                raise ValueError(f"transform {t!r} of {name}")
        theta_net = g.concat(cols)
    else:
        # stored scale in: a zero-box input arrives as log x (probit x for SZ), -Inf at 0
        cols = []
        for j, name in enumerate(ctx):
            x = col(j)
            if tr.get(name) == "zerobox":
                natv = g.pnorm(x) if name == "SZ" else g.exp(x)
                cols.append(g.asinh(g.div(natv, float(zc[name]))))
            else:
                cols.append(x)
        theta_net = g.concat(cols) if k > 1 else cols[0]
    mu = g.const(np.array(card["scaler"]["mean"]), "scaler_mean")
    sd = g.const(np.array(card["scaler"]["scale"]), "scaler_scale")
    ctx_std = g.div(g.sub(theta_net, mu), sd)                     # [rows, k]
    csc = card.get("classifier_scaler")
    if csc and (not np.allclose(csc["mean"], card["scaler"]["mean"]) or not np.allclose(csc["scale"], card["scaler"]["scale"])):
        cmu = g.const(np.array(csc["mean"]), "clf_mean"); csd = g.const(np.array(csc["scale"]), "clf_scale")
        clf_in = g.div(g.sub(theta_net, cmu), csd)
    else:
        clf_in = ctx_std

    # --- conditioner and knots -----------------------------------------------------------
    if signature == "trialwise":
        R = g.mul(g.add(choice, 3.0), 0.5)                        # -1 -> 1, +1 -> 2
        flow_in = g.concat([ctx_std, R])
    else:
        ctx2 = g.op("Tile", [ctx_std, g.iconst([2, 1])])
        flow_in = g.concat([ctx2, g.const(np.array([[1.0], [2.0]]), "R_rows")])
    raw = g.mlp(flow_in, card["flow_mlp"])
    x_pos, y_pos, slopes, K = g.knots(raw, sp)
    if signature == "shared":
        sel = g.op("Reshape", [g.op("Cast", [g.gt(choice, 0.0)], to=TensorProto.INT64), g.iconst([-1])])
        x_pos = g.op("Gather", [x_pos, sel], axis=0)
        y_pos = g.op("Gather", [y_pos, sel], axis=0)
        slopes = g.op("Gather", [slopes, sel], axis=0)

    # --- classifier ------------------------------------------------------------------------
    logit = g.mlp(clf_in, card["classifier_mlp"])                 # [rows, 1]
    up = g.gt(choice, 0.0)
    lp_up = g.neg(g.softplus(g.neg(logit)))
    lp_lo = g.neg(g.softplus(logit))
    log_p_R = g.where(up, lp_up, lp_lo)                           # [N, 1] (shared: broadcasts [1,1] vs [N,1])

    def log_density(u):
        """log g(u) (joint, with log p_R) and G(u) = p_R Phi(z) at u [N,1] > 0."""
        lu = g.log(u)
        z, logdet = g.rqs_inverse(lu, x_pos, y_pos, slopes, K)
        lg = g.add(g.sub(g.add(g.log_phi(z), logdet), lu), log_p_R)
        G = g.mul(g.pnorm(z), g.exp(log_p_R))
        return lg, G

    if scaled:
        c = g.div(s2, g.mul(a, a))
        d = g.sub(rt, t0)
        live = g.gt(d, 0.0)
        d_safe = g.where(live, d, 1.0)
        narrow = g.or_(g.le(st0, 0.0), g.lt(st0, g.mul(d_safe, 1e-7)))
        um = g.mul(g.sub(d_safe, g.div(g.maximum(st0, 0.0), 2.0)), c)
        u1 = g.mul(d_safe, c)
        u0 = g.mul(g.sub(d_safe, st0), c)
        ua = g.where(narrow, um, u1)
        lg_a, G_a = log_density(g.where(g.gt(ua, 0.0), ua, 1.0))
        has0 = g.gt(u0, 0.0)
        _, G_b = log_density(g.where(has0, u0, 1.0))
        G0 = g.where(has0, G_b, 0.0)
        ll_narrow = g.add(g.log(c), lg_a)
        ll_wide = g.log(g.div(g.maximum(g.sub(G_a, G0), 0.0), g.where(narrow, 1.0, st0)))
        ll = g.where(narrow, ll_narrow, ll_wide)
        ll = g.where(live, ll, g.const(NEG_INF, "neg_inf"))
    else:
        u_ok = g.gt(rt, 0.0)
        lg, _ = log_density(g.where(u_ok, rt, 1.0))
        ll = g.where(u_ok, lg, g.const(NEG_INF, "neg_inf"))
        # the st0 = 0 support rule: rt <= t0 has density 0 (a zero-box card at
        # st0 = 0, and an st0zero card always; EMC2 applies both)
        if "t0" in ctx and (tr.get("st0") == "zerobox" or card.get("context_encoding") == "st0zero"):
            t0_nat = g.exp(col(ctx.index("t0"))) if tr.get("t0") == "log" else col(ctx.index("t0"))
            hard = g.le(rt, t0_nat)
            if tr.get("st0") == "zerobox":
                hard = g.and_(g.le(g.exp(col(ctx.index("st0"))), 0.0), hard)
            ll = g.where(hard, g.const(NEG_INF, "neg_inf2"), ll)

    # --- the training box: refuse to extrapolate ---------------------------------------------
    lo = g.const(np.array(card["bounds_sampled"]["lower"]), "box_lo")
    hi = g.const(np.array(card["bounds_sampled"]["upper"]), "box_hi")
    okb = g.and_(g.ge(theta_net, lo), g.le(theta_net, hi))
    okn = g.op("ReduceMin", [g.op("Cast", [okb], to=FT), g.iconst([1])], keepdims=1)
    in_box = g.gt(okn, 0.5)
    ll = g.where(in_box, ll, float(card["ll_floor_log"]))
    g.nodes.append(helper.make_node("Squeeze", [ll, g.iconst([1])], ["log_likelihood"]))

    graph = helper.make_graph(g.nodes, "ddm_flow", inputs, [out], initializer=g.init)
    model = helper.make_model(graph, producer_name="nleflow", opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    if scaled:
        iface = ("the natural DDM parameters v, a, t0, s, Z, SZ (start-point proportion), sv, st0, "
                 "then rt (s) and choice in {-1, +1}")
    else:
        iface = (f"the {k} context values on the stored scale in the order {ctx} (a zero-box input as "
                 f"log x, probit x for SZ, -Inf at 0), then rt (s) and choice in {{-1, +1}}")
    model.doc_string = (
        f"{card.get('label', card.get('budget', ''))}: {kind} neural likelihood (spline flow + choice classifier) "
        f"exported by nleflow. " + (f"Input [N, {n_in + 2}]: " if signature == "trialwise" else
                                    f"Inputs theta [{n_in}], rt [N], choice [N]: ") + iface +
        ". Output [N]: log p(rt, choice | theta), the joint defective log density; -Inf below the "
        f"onset; the floor {card['ll_floor_log']} outside the training box.")
    onnx.checker.check_model(model, full_check=True)
    return model


def export_onnx(card_path, out=None, signature="trialwise", say=print):
    import onnx
    card = json.load(open(card_path))
    model = build(card, signature)
    if out is None:
        p = Path(card_path)
        name = p.name[:-len("_weights.json")] if p.name.endswith("_weights.json") else p.stem
        out = p.with_name(f"{name}_{signature}.onnx")
    onnx.save(model, str(out))
    say(f"wrote {out}  ({len(model.graph.node)} nodes, {len(model.graph.initializer)} initialisers, "
        f"opset 18, IR {model.ir_version}, {signature})")
    return Path(out)


# --- running and checking ---------------------------------------------------------------
def session(path):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def run(sess, theta, rt, choice):
    """log likelihood for rows of theta [N, n_in] (trialwise) or one theta
    [n_in] applied to every trial (shared), rt [N], choice [N] in {-1, +1}."""
    theta = np.asarray(theta, dtype=np.float64); rt = np.asarray(rt, dtype=np.float64)
    choice = np.asarray(choice, dtype=np.float64)
    if len(sess.get_inputs()) == 3:
        if theta.ndim == 2:
            if not np.all(theta == theta[0]):
                raise ValueError("the shared signature takes one theta for all trials")
            theta = theta[0]
        return sess.run(None, {"theta": theta, "rt": rt, "choice": choice})[0]
    if theta.ndim == 1:
        theta = np.tile(theta, (len(rt), 1))
    return sess.run(None, {"input": np.column_stack([theta, rt, choice])})[0]


def golden_cases(card, golden_csv):
    """(theta rows, rt, choice, reference log density) from a card's golden
    file: for a scaled card the natural DDM parameters that reproduce each
    golden row at a = s = 1, t0 = st0 = 0."""
    from scipy.special import ndtr
    from .golden import card_kind
    g = np.genfromtxt(golden_csv, delimiter=",", names=True)
    ctx = list(card["context_names"])
    R = g["R"]; rt = g["rt"]; ref = g["log_pdf_joint"]
    choice = np.where(R == 2, 1.0, -1.0)
    if card_kind(card) == "flow_scaled":
        eta = np.exp(g["eta"]); SZ = ndtr(g["SZ"]); Z = ndtr(g["Z"])
        theta = np.column_stack([g["kappa"], np.ones_like(rt), np.zeros_like(rt), np.ones_like(rt), Z, SZ, eta,
                                 np.zeros_like(rt)])
    else:
        theta = np.column_stack([g[c] for c in ctx])
        # the golden file of an st0zero card holds the raw network below t0;
        # the graph, like EMC2, returns -Inf there
        if card.get("context_encoding") == "st0zero" and "t0" in ctx:
            t0 = np.exp(g["t0"]) if (card.get("context_transforms") or {}).get("t0") == "log" else g["t0"]
            ref = np.where(rt <= t0, -np.inf, ref)
    return theta, rt, choice, ref


def check_against(sess, theta, rt, choice, ref, floor, label, say=print, tol=1e-11):
    """Compare the graph with reference log densities; rows at or below the
    floor (density 1e-10) and -Inf rows are compared for agreement of kind."""
    got = np.empty(len(rt))
    if len(sess.get_inputs()) == 3:
        # group by identical theta rows, in order
        keys = np.ascontiguousarray(theta).view(np.dtype((np.void, theta.dtype.itemsize * theta.shape[1]))).ravel()
        _, first, inv = np.unique(keys, return_index=True, return_inverse=True)
        for gi in range(len(first)):
            m = inv == gi
            got[m] = run(sess, theta[first[gi]], rt[m], choice[m])
    else:
        got = run(sess, theta, rt, choice)
    fin = np.isfinite(ref) & (ref > floor + 1e-9)
    err = np.abs(got[fin] - ref[fin])
    inf_ok = np.all(np.isneginf(got[np.isneginf(ref)])) if np.any(np.isneginf(ref)) else True
    floor_rows = np.isfinite(ref) & ~fin
    floor_ok = np.all(got[floor_rows] <= ref[floor_rows] + 1e-6) if floor_rows.any() else True
    say(f"{label}: {len(rt)} rows, {int(fin.sum())} above the floor: max |dlog p| {err.max():.2e}, "
        f"median {np.median(err):.2e}; -Inf rows agree: {inf_ok}; floor rows agree: {floor_ok}")
    ok = err.max() < tol and inf_ok and floor_ok
    return ok, float(err.max())


def check_onnx(model_path, card_path, golden=None, cases=None, tol_density=1e-10, tol_window=1e-8, say=print):
    """`nleflow onnx-check`: the graph against the card's golden rows and,
    when given, EMC2 cases: for a scaled card those of
    scripts/onnx_cases_scaled.R (the eight natural DDM parameters), for a
    flow_joint card rows of its context on the stored scale (as golden_cases
    reads them), rt, choice and logp. Returns True when every comparison is
    within tolerance."""
    from .golden import card_kind, golden_path_for
    card = json.load(open(card_path))
    sess = session(model_path)
    floor = float(card["ll_floor_log"])
    ok_all = True
    golden = Path(golden) if golden else golden_path_for(card_path)
    if golden.exists():
        theta, rt, choice, ref = golden_cases(card, golden)
        ok, worst = check_against(sess, theta, rt, choice, ref, floor, f"golden rows of {golden.name}", say=say,
                                  tol=tol_density)
        say(f"  {'PASS' if ok else 'FAIL'} against {tol_density:.0e}")
        ok_all &= ok
    else:
        say(f"no golden file at {golden}: skipped")
    if cases and card_kind(card) != "flow_scaled":
        # a flow_joint card: the cases carry the card's own inputs, the context on
        # the stored scale (-Inf at an exact zero), and every row is a density
        d = np.genfromtxt(cases, delimiter=",", names=True, dtype=None, encoding=None)
        ctx = list(card["context_names"])
        missing = [c for c in ctx + ["rt", "choice", "logp"] if c not in d.dtype.names]
        if missing:
            raise ValueError(f"the cases {cases} lack the columns {missing} (a flow_joint card's cases carry "
                             f"its context {ctx} on the stored scale, rt, choice and logp)")
        theta = np.column_stack([d[c] for c in ctx]).astype(np.float64)
        rt = d["rt"].astype(np.float64); choice = d["choice"].astype(np.float64); ref = d["logp"].astype(np.float64)
        ok, worst = check_against(sess, theta, rt, choice, ref, floor, "EMC2 cases (the card's inputs)", say=say,
                                  tol=tol_density)
        say(f"  {'PASS' if ok else 'FAIL'} against {tol_density:.0e}")
        ok_all &= ok
    elif cases:
        d = np.genfromtxt(cases, delimiter=",", names=True, dtype=None, encoding=None)
        theta = np.column_stack([d[c] for c in SCALED_INPUTS]).astype(np.float64)
        rt = d["rt"].astype(np.float64); choice = d["choice"].astype(np.float64); ref = d["logp"].astype(np.float64)
        wide = (theta[:, SCALED_INPUTS.index("st0")] > 0) & \
               (theta[:, SCALED_INPUTS.index("st0")] >= 1e-7 * (rt - theta[:, SCALED_INPUTS.index("t0")]))
        for name, m, tol in (("st0 = 0 (a density)", ~wide, tol_density), ("st0 > 0 (a CDF difference)", wide, tol_window)):
            if not m.any():
                continue
            ok, worst = check_against(sess, theta[m], rt[m], choice[m], ref[m], floor, f"EMC2 cases, {name}", say=say,
                                      tol=tol)
            say(f"  {'PASS' if ok else 'FAIL'} against {tol:.0e}")
            ok_all &= ok
    return ok_all

"""`nleflow convert <model.onnx>`: an MLP likelihood network in ONNX format
(a LAN, e.g. HSSM's) to an EMC2 card of kind mlp_joint (card_version 1).

EMC2 does not run ONNX: a published LAN is a plain multilayer perceptron and
the .onnx file is only the container for its weights. The converter reads the
container, refuses anything that is not a plain MLP EMC2 can evaluate (naming
the first node it cannot take), and writes the card EMC2's compiled MLP
evaluator (src/mlp_lik.cpp) reads. What the graph does not record (what the
inputs mean, their scale, the training box, the label floor, the response
coding, the mapping onto an EMC2 model) comes from a preset
(nleflow/presets/*.yaml, `--list-presets`) or from the flags.

The graph: one non-initializer input, rank 1 `[k]` (one trial per call, the
published LANs) or rank 2 `[N, k]` (batched, as HSSM's converted graphs);
layers MatMul + Add or Gemm; hidden activations Tanh or Gelu; one linear
output, the log density. Identity, and Reshape / Flatten / Squeeze / Unsqueeze
/ Cast before the first layer or after the last, are passed over. EMC2's
evaluator (load_mlp, src/nle_flow.h) runs tanh and gelu_tanh hidden layers
only, so a Relu network, an exact (erf) Gelu or a Sigmoid is refused by name:
each needs a C++ change in EMC2 first.

`--verify N` checks the card against onnxruntime before it is trusted: the
card's float64 forward pass against a float64 copy of the graph (the gate,
1e-10: this is the test that the card holds the network; a Gelu node is
expanded into Mul, Add and Tanh in that copy, because onnxruntime's double
Gelu kernel is accurate to about 6e-9 only) and against the
graph as published (float32: reported, gated at 1e-4, the float32 round-off),
each with one-trial calls and a batched call, and writes the onnxruntime
outputs of the published graph as the card's golden file (<tag>_golden.csv:
the parameters, rt, choice, logp_ort), which EMC2's nn_golden_check() reads.
`--rows FILE` takes the verification rows from a CSV with those columns
instead of drawing them.

Converting does not validate: a LAN's calibration is unknown until it has
been checked (parameter recovery, SBC); the card says so in `validation`.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from . import __version__

PRESET_DIR = Path(__file__).resolve().parent / "presets"
EMC2_ACTIVATIONS = ("tanh", "gelu_tanh")      # what load_mlp (src/nle_flow.h) evaluates
PASS_THROUGH_EDGE = ("Reshape", "Flatten", "Squeeze", "Unsqueeze", "Cast")
CALIBRATION = "unknown; needs its own SBC cell before inference"
F32_TOL = 1e-4          # a float32 graph against the float64 card (EMC2's test-nn-mlp.R uses the same)
F64_TOL = 1e-10         # the card against the float64 copy of the graph: the gate


class ConvertError(ValueError):
    pass


# --- presets ----------------------------------------------------------------------------

def list_presets():
    return sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


def load_preset(name):
    p = PRESET_DIR / f"{name}.yaml"
    if not p.exists():
        raise ConvertError(f"no preset '{name}'; known presets: {', '.join(list_presets()) or 'none'}")
    d = yaml.safe_load(p.read_text())
    d["box"] = {k: [float(v[0]), float(v[1])] for k, v in d["box"].items()}
    d["response_values"] = [float(x) for x in d["response_values"]]
    d["ll_floor"] = float(d["ll_floor"])
    for v in d.get("versions") or []:
        v["box"] = {k: [float(x[0]), float(x[1])] for k, x in (v.get("box") or {}).items()}
        bad = set(v["box"]) - set(d["params"])
        if bad:
            raise ConvertError(f"preset '{name}', version {v.get('revision')}: box names non-parameters {sorted(bad)}")
    return d


def preset_version(p, sha):
    """The published file a preset knows by this sha256 (its `versions`), or None.
    A source can replace its network under the same name (HSSM's ddm_uniform_st
    did on 2026-10-05); each version records its own revision and box."""
    for v in p.get("versions") or []:
        if v["sha256"] == sha:
            return v
    return None


# --- reading the graph ---------------------------------------------------------------------

def _node_label(i, n):
    return f"node {i} ({n.op_type}, name '{n.name or '<unnamed>'}', output '{n.output[0] if n.output else ''}')"


def read_graph(path):
    """The MLP in an ONNX file: dict(layers [(W n_in x n_out, b)], activation,
    input name, input rank, n_in, elem_type, opset). Raises ConvertError naming
    the first node it cannot take."""
    import onnx
    from onnx import numpy_helper, helper

    model = onnx.load(str(path))
    g = model.graph
    init = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    init[n.output[0]] = numpy_helper.to_array(a.t).astype(np.float64)
    inputs = [i for i in g.input if i.name not in init]
    if len(inputs) != 1:
        raise ConvertError(f"expected exactly one network input; the graph has {len(inputs)} "
                           f"({[i.name for i in inputs]})")
    inp = inputs[0]
    shape = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in inp.type.tensor_type.shape.dim]
    if len(shape) not in (1, 2):
        raise ConvertError(f"the input '{inp.name}' has rank {len(shape)}; a LAN takes [k] or [N, k]")
    cur = inp.name
    layers, acts, pend, edge = [], [], None, []

    def weight(name, i, n):
        if name not in init:
            raise ConvertError(f"{_node_label(i, n)}: its operand '{name}' is computed, not a stored weight; "
                               "not a plain MLP")
        return init[name]

    for i, n in enumerate(g.node):
        op = n.op_type
        if op == "Constant":
            continue
        if cur not in n.input:
            raise ConvertError(f"{_node_label(i, n)} does not continue the layer chain (it does not take "
                               f"'{cur}'); a plain MLP is one chain of nodes")
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        if op == "MatMul" and pend is None and n.input[0] == cur:
            W = weight(n.input[1], i, n)
            if W.ndim != 2:
                raise ConvertError(f"{_node_label(i, n)}: the weight is not a matrix")
            pend = [W, None]
        elif op == "Add" and pend is not None and pend[1] is None:
            other = [x for x in n.input if x != cur]
            if len(other) != 1:
                raise ConvertError(f"{_node_label(i, n)}: an Add that is not a bias")
            pend[1] = weight(other[0], i, n).reshape(-1)
            layers.append(tuple(pend)); pend = None
        elif op == "Gemm" and pend is None and n.input[0] == cur:
            if attrs.get("transA", 0):
                raise ConvertError(f"{_node_label(i, n)}: Gemm with transA is not supported")
            W = weight(n.input[1], i, n)
            if attrs.get("transB", 0):
                W = W.T
            b = weight(n.input[2], i, n).reshape(-1) if len(n.input) > 2 and n.input[2] else np.zeros(W.shape[1])
            layers.append((float(attrs.get("alpha", 1.0)) * W, float(attrs.get("beta", 1.0)) * b))
        elif op in ("Tanh", "Gelu", "Relu") and pend is None and layers:
            if op == "Tanh":
                act = "tanh"
            elif op == "Gelu":
                appr = attrs.get("approximate", "none")
                appr = appr.decode() if isinstance(appr, bytes) else appr
                act = "gelu_tanh" if appr == "tanh" else "gelu_erf"
            else:
                act = "relu"
            if act not in EMC2_ACTIVATIONS:
                raise ConvertError(
                    f"{_node_label(i, n)}: activation '{act}' is not one EMC2's evaluator runs "
                    f"(load_mlp in src/nle_flow.h takes {', '.join(EMC2_ACTIVATIONS)}); converting this network "
                    "needs that support in EMC2 first")
            acts.append(act)
        elif op == "Identity":
            pass
        elif op in PASS_THROUGH_EDGE and pend is None:
            # allowed before the first layer or after the last; which one is
            # known only when the chain has been read (checked below)
            if op == "Cast" and attrs.get("to") not in (1, 11):          # FLOAT, DOUBLE
                raise ConvertError(f"{_node_label(i, n)}: a Cast to a non-float type")
            edge.append((i, n, len(layers), len(acts)))
        else:
            raise ConvertError(f"{_node_label(i, n)}: unsupported operation '{op}' here. The converter takes "
                               "MatMul + Add or Gemm layers, Tanh or Gelu(tanh) hidden activations and one linear "
                               "output; anything else needs a new evaluator in EMC2 first")
        cur = n.output[0]
    if pend is not None:
        raise ConvertError("a MatMul without its bias Add")
    if not layers:
        raise ConvertError("no layers found")
    if cur != g.output[0].name:
        raise ConvertError(f"the layer chain ends at '{cur}', not at the graph output '{g.output[0].name}'")
    if len(acts) != len(layers) - 1:
        raise ConvertError(f"{len(layers)} layers need {len(layers) - 1} hidden activations; found {len(acts)} "
                           "(the output layer must be linear)")
    for i, n, n_layers, n_acts in edge:
        if not (n_layers == 0 or (n_layers == len(layers) and n_acts == len(acts))):
            raise ConvertError(f"{_node_label(i, n)}: a {n.op_type} between layers is not a plain MLP "
                               "(reshapes are passed over only before the first layer or after the output)")
    if len(set(acts)) > 1:
        raise ConvertError(f"mixed hidden activations {sorted(set(acts))}: EMC2's evaluator takes one")
    for j in range(1, len(layers)):
        if layers[j][0].shape[0] != layers[j - 1][0].shape[1]:
            raise ConvertError(f"layer {j + 1}'s input width does not match layer {j}'s output width")
    for W, b in layers:
        if b.shape[0] != W.shape[1]:
            raise ConvertError("a bias does not match its layer's output width")
    if layers[-1][0].shape[1] != 1:
        raise ConvertError("the network must have one output (the log density)")
    if shape[-1] not in ("?",) and isinstance(shape[-1], int) and shape[-1] != layers[0][0].shape[0]:
        raise ConvertError(f"the input width {shape[-1]} does not match the first layer's {layers[0][0].shape[0]}")
    return {"model": model, "layers": layers, "activation": acts[0] if acts else "tanh",
            "input": inp.name, "rank": len(shape), "n_in": layers[0][0].shape[0],
            "elem_type": inp.type.tensor_type.elem_type,
            "opset": max((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=0)}


def forward(layers, activation, X):
    """The card's forward pass in float64; X is n x n_in."""
    H = np.asarray(X, dtype=np.float64)
    for i, (W, b) in enumerate(layers):
        H = H @ W + b
        if i < len(layers) - 1:
            H = np.tanh(H) if activation == "tanh" else \
                0.5 * H * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (H + 0.044715 * H ** 3)))
    return H[:, 0]


# --- the card -------------------------------------------------------------------------------

def resolve_settings(preset=None, params=None, box=None, ll_floor=None, response_values=None, layout=None,
                     transforms=None, sha=None):
    p = load_preset(preset) if preset else {}
    ver = preset_version(p, sha) if p else None
    s = {
        "params": list(params) if params else p.get("params"),
        "box": dict(box) if box else ({**p["box"], **ver["box"]} if ver else p.get("box")),
        "preset_version": ver,
        "ll_floor": ll_floor if ll_floor is not None else p.get("ll_floor"),
        "response_values": list(response_values) if response_values else p.get("response_values"),
        "preset": p,
    }
    for what in ("params", "box", "ll_floor", "response_values"):
        if s[what] is None:
            raise ConvertError(f"--{what.replace('_', '-')} is required (or use --preset): an ONNX graph does not "
                               "record it")
    if set(s["box"]) != set(s["params"]):
        raise ConvertError("--box must give a range for every parameter and no others")
    if len(s["response_values"]) != 2:
        raise ConvertError("--response-values needs two values (responses 1 and 2)")
    for k, (lo, hi) in s["box"].items():
        if not lo < hi:
            raise ConvertError(f"the box of {k} is empty")
    tf = {k: "identity" for k in s["params"]}
    tf.update(p.get("transforms") or {})
    tf.update(transforms or {})
    bad = [k for k, v in tf.items() if k not in s["params"] or v not in ("identity", "log", "probit")]
    if bad:
        raise ConvertError(f"bad transforms for {bad}")
    s["transforms"] = {k: tf[k] for k in s["params"]}
    s["layout"] = list(layout) if layout else list(p.get("input_layout") or (s["params"] + ["rt", "R"]))
    return s


def build_card(graph, settings, source_path):
    params, layout = settings["params"], settings["layout"]
    n_in = graph["n_in"]
    if len(layout) != n_in:
        raise ConvertError(f"the network has {n_in} inputs but the input layout has {len(layout)} ({layout})")
    if not set(params) <= set(layout) or len(set(layout)) != len(layout) or \
            not set(layout) - set(params) <= {"rt", "log_rt", "R"}:
        raise ConvertError("the layout must list every parameter once, plus only rt / log_rt / R")
    sha = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    p = settings["preset"]
    card = {
        "card_version": 1,
        "kind": "mlp_joint",
        "source": Path(source_path).name,
        "source_sha256": sha,
        "context_names": params,
        "context_transforms": settings["transforms"],
        "bounds_natural": {"lower": [settings["box"][k][0] for k in params],
                           "upper": [settings["box"][k][1] for k in params]},
        "input_layout": layout,
        "response_values": settings["response_values"],
        "ll_floor_log": settings["ll_floor"],
        "mlp": {"activation": graph["activation"], "use_norm": False,
                "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in graph["layers"]]},
        "converted_with": f"nleflow {__version__} convert",
    }
    for k in ("source_model", "encoding", "map_to_emc2_ddm"):
        if p.get(k):
            card[k] = p[k]
    # the defaults a card with no analytic twin registers with (sampled scale), as
    # EMC2's nle_verify.R reads them from the card
    if p.get("emc2_p_types"):
        card["p_types"] = {k: float(p["emc2_p_types"][k]) for k in params}
    ver = settings.get("preset_version")
    if ver:     # which published file (EMC2 reads card fields by exact name, so an extra field is inert)
        card["source_revision"] = ver["revision"]
    # no field of its own for the preset: EMC2 reads the card with R's `$`,
    # which partially matches names, so a field "preset" would be read as `pre`
    # (a decision-time shift). The card carries the vignette's fields and the
    # converter's provenance notes only.
    if p:
        card["converted_with"] += f" --preset {p['name']}"
    card["validation"] = {"calibration": CALIBRATION}
    warn = None
    known = [v["sha256"] for v in p.get("versions") or []] or ([p["source_sha256"]] if p.get("source_sha256") else [])
    if known and sha not in known:
        warn = (f"the preset '{p['name']}' was checked on a file with sha256 "
                f"{' or '.join(k[:16] + '...' for k in known)}, "
                f"this one is {sha[:16]}...: check that it is the same network before trusting the mapping")
    return card, warn


# --- onnxruntime --------------------------------------------------------------------------------

def _variant(model, dtype=None, batched=False):
    """A copy of the graph in float64 (dtype='double') and/or with a batch
    dimension added to a rank-1 input and output."""
    from onnx import TensorProto, numpy_helper
    m = copy.deepcopy(model)
    g = m.graph
    if dtype == "double":
        for t in g.initializer:
            a = numpy_helper.to_array(t)
            if a.dtype.kind == "f":
                t.CopyFrom(numpy_helper.from_array(a.astype(np.float64), t.name))
        for n in g.node:
            if n.op_type == "Constant":
                for a in n.attribute:
                    if a.name == "value":
                        arr = numpy_helper.to_array(a.t)
                        if arr.dtype.kind == "f":
                            a.t.CopyFrom(numpy_helper.from_array(arr.astype(np.float64), a.t.name))
            if n.op_type == "Cast":
                for a in n.attribute:
                    if a.name == "to" and a.i == TensorProto.FLOAT:
                        a.i = TensorProto.DOUBLE
        for vi in list(g.input) + list(g.output) + list(g.value_info):
            if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
                vi.type.tensor_type.elem_type = TensorProto.DOUBLE
        _expand_gelu(g)
    if batched:
        for vi in list(g.input) + list(g.output):
            dims = vi.type.tensor_type.shape.dim
            if len(dims) == 1:
                new = vi.type.tensor_type.shape.dim.add()
                new.CopyFrom(dims[0])
                dims[0].Clear(); dims[0].dim_param = "N"
        del g.value_info[:]
    if m.ir_version > 10:
        m.ir_version = 10
    return m


def _expand_gelu(g):
    """Replace each Gelu(approximate='tanh') node by its formula in Mul, Add
    and Tanh: onnxruntime's double-precision Gelu kernel is accurate to about
    6e-9 only (its Tanh to 6e-16), so the float64 reference copy computes the
    graph's mathematics with the exact kernels."""
    from onnx import helper, numpy_helper
    k = 0
    # constants only for a graph that uses them: onnxruntime warns about unused
    # initializers at every session (noise in a tanh network's --verify)
    if not any(n.op_type == "Gelu" for n in g.node):
        return
    for c, v in (("_gelu_half", 0.5), ("_gelu_one", 1.0), ("_gelu_cubic", 0.044715), ("_gelu_s2pi", np.sqrt(2.0 / np.pi))):
        g.initializer.append(numpy_helper.from_array(np.asarray(v, dtype=np.float64), c))
    new_nodes = []
    for n in g.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        appr = attrs.get("approximate", b"none")
        appr = appr.decode() if isinstance(appr, bytes) else appr
        if n.op_type != "Gelu" or appr != "tanh":
            new_nodes.append(n)
            continue
        x, y, t = n.input[0], n.output[0], f"_gelu{k}_"
        k += 1
        new_nodes += [helper.make_node("Mul", [x, x], [t + "x2"]),
                      helper.make_node("Mul", [t + "x2", x], [t + "x3"]),
                      helper.make_node("Mul", [t + "x3", "_gelu_cubic"], [t + "cx3"]),
                      helper.make_node("Add", [x, t + "cx3"], [t + "in"]),
                      helper.make_node("Mul", [t + "in", "_gelu_s2pi"], [t + "arg"]),
                      helper.make_node("Tanh", [t + "arg"], [t + "th"]),
                      helper.make_node("Add", [t + "th", "_gelu_one"], [t + "1p"]),
                      helper.make_node("Mul", [t + "1p", x], [t + "x1p"]),
                      helper.make_node("Mul", [t + "x1p", "_gelu_half"], [y])]
    del g.node[:]
    g.node.extend(new_nodes)


def _session(model):
    import onnxruntime as ort
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), o, providers=["CPUExecutionProvider"])


def run_onnx(model, X, rank, mode, np_dtype):
    """Outputs of a graph whose input has the given rank, one trial per call
    (mode 'single') or all at once (mode 'batched'; a rank-1 graph is given a
    batch dimension first)."""
    X = np.asarray(X, dtype=np_dtype)
    if mode == "single":
        s = _session(model); name = s.get_inputs()[0].name
        if rank == 1:
            return np.array([np.asarray(s.run(None, {name: X[i]})[0]).reshape(-1)[0] for i in range(len(X))],
                            dtype=np.float64)
        return np.array([np.asarray(s.run(None, {name: X[i:i + 1]})[0]).reshape(-1)[0] for i in range(len(X))],
                        dtype=np.float64)
    m = _variant(model, batched=True) if rank == 1 else model
    s = _session(m); name = s.get_inputs()[0].name
    return np.asarray(s.run(None, {name: X})[0], dtype=np.float64).reshape(-1)


def design_matrix(card, rows):
    """The network's input matrix for rows given as a dict of the parameters,
    rt and choice (the response input value)."""
    cols = []
    for name in card["input_layout"]:
        if name == "rt":
            cols.append(rows["rt"])
        elif name == "log_rt":
            cols.append(np.log(rows["rt"]))
        elif name == "R":
            cols.append(rows["choice"])
        else:
            x = rows[name]
            t = card["context_transforms"].get(name, "identity")
            if t == "log":
                x = np.log(x)
            elif t == "probit":
                from scipy.special import ndtri
                x = ndtri(x)
            cols.append(x)
    return np.column_stack(cols)


def verify_rows(card, n, seed=1, rows_csv=None):
    names = card["context_names"]
    if rows_csv is not None:
        d = np.genfromtxt(rows_csv, delimiter=",", names=True)
        missing = [c for c in names + ["rt", "choice"] if c not in d.dtype.names]
        if missing:
            raise ConvertError(f"{rows_csv} lacks columns {missing}")
        return {c: np.asarray(d[c], dtype=np.float64) for c in names + ["rt", "choice"]}
    rng = np.random.default_rng(seed)
    lo, hi = card["bounds_natural"]["lower"], card["bounds_natural"]["upper"]
    rows = {k: rng.uniform(lo[j], hi[j], n) for j, k in enumerate(names)}
    rows["rt"] = rng.uniform(0.01, 5.0, n)
    rows["choice"] = rng.choice(np.asarray(card["response_values"], dtype=np.float64), n)
    return rows


def verify(card, graph, n=300, seed=1, rows_csv=None, say=print):
    """The card against onnxruntime; returns (ok, report, rows, ort outputs of
    the published graph)."""
    from onnx import TensorProto
    layers = [(np.asarray(l["W"], dtype=np.float64), np.asarray(l["b"], dtype=np.float64))
              for l in card["mlp"]["layers"]]
    rows = verify_rows(card, n, seed, rows_csv)
    X = design_matrix(card, rows)
    mine = forward(layers, card["mlp"]["activation"], X)
    model, rank = graph["model"], graph["rank"]
    src_f32 = graph["elem_type"] == TensorProto.FLOAT
    report, ok = {"n": int(len(X))}, True
    m64 = _variant(model, dtype="double")
    for mode in ("single", "batched"):
        o64 = run_onnx(m64, X, rank, mode, np.float64)
        d64 = float(np.max(np.abs(o64 - mine)))
        report[f"f64_{mode}"] = d64
        ok &= d64 <= F64_TOL
    pub = {}
    for mode in ("single", "batched"):
        pub[mode] = run_onnx(model, X, rank, mode, np.float32 if src_f32 else np.float64)
        report[f"src_{mode}"] = float(np.max(np.abs(pub[mode] - mine)))
    tol_src = F32_TOL if src_f32 else F64_TOL
    ok &= max(report["src_single"], report["src_batched"]) <= tol_src
    report["single_vs_batched"] = float(np.max(np.abs(pub["single"] - pub["batched"])))
    shape = "rank-1 [k], one trial per call" if rank == 1 else "batched [N, k]"
    say(f"--verify on {len(X)} rows ({'drawn in the box' if rows_csv is None else 'from ' + str(rows_csv)}); "
        f"the graph's input is {shape}")
    say(f"  card (float64) vs a float64 copy of the graph: one-trial calls {report['f64_single']:.2e}, "
        f"batched {report['f64_batched']:.2e}  (gate {F64_TOL:.0e})")
    say(f"  card (float64) vs the graph as published ({'float32' if src_f32 else 'float64'}): one-trial calls "
        f"{report['src_single']:.2e}, batched {report['src_batched']:.2e}  (gate {tol_src:.0e}"
        + (": float32 round-off" if src_f32 else "") + ")")
    say(f"  the published graph, one-trial calls vs batched: {report['single_vs_batched']:.2e}")
    say(f"  {'PASS' if ok else 'FAIL'}")
    return ok, report, rows, pub["single"]


def write_ort_golden(card, rows, logp, path):
    names = card["context_names"]
    with open(path, "w") as f:
        f.write(",".join(names + ["rt", "choice", "logp_ort"]) + "\n")
        for i in range(len(logp)):
            vals = [rows[k][i] for k in names] + [rows["rt"][i]]
            f.write(",".join(f"{v:.17g}" for v in vals) + f",{rows['choice'][i]:g},{logp[i]:.17g}\n")


def convert(onnx_path, out=None, preset=None, params=None, box=None, ll_floor=None, response_values=None,
            layout=None, transforms=None, verify_n=0, rows_csv=None, seed=1, say=print):
    """Convert; returns dict(card path, golden path or None, verify report)."""
    from .golden import golden_path_for
    sha = hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest()
    settings = resolve_settings(preset, params, box, ll_floor, response_values, layout, transforms, sha=sha)
    graph = read_graph(onnx_path)
    card, warn = build_card(graph, settings, onnx_path)
    if warn:
        say("WARNING: " + warn)
    out = Path(out) if out else Path.cwd() / f"{Path(onnx_path).stem}_weights.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card))
    dims = [graph["n_in"]] + [W.shape[1] for W, _ in graph["layers"]]
    say(f"wrote {out}: {'-'.join(map(str, dims))} {graph['activation']} MLP, inputs {card['input_layout']}, "
        f"{len(card['context_names'])} parameters, ll floor {card['ll_floor_log']}, source sha256 "
        f"{card['source_sha256'][:16]}, graph input rank {graph['rank']}, opset {graph['opset']}")
    res = {"card": out, "golden": None, "verify": None}
    if verify_n or rows_csv:
        back = json.loads(out.read_text())               # verify what was written
        ok, report, rows, logp = verify(back, graph, n=verify_n or 300, seed=seed, rows_csv=rows_csv, say=say)
        res["verify"] = report
        if not ok:
            raise ConvertError("--verify failed: the card does not reproduce the graph (see above); the golden file "
                               "was not written")
        gpath = golden_path_for(out)
        write_ort_golden(back, rows, logp, gpath)
        say(f"wrote {gpath} ({len(logp)} rows; onnxruntime on the published graph)")
        res["golden"] = gpath
        # the vignette's validation.golden record: the file and the tolerance it
        # holds at (a float32 source cannot meet nn_golden_check's default 1e-9)
        from onnx import TensorProto
        tol = F32_TOL if graph["elem_type"] == TensorProto.FLOAT else F64_TOL
        back["validation"]["golden"] = {"file": gpath.name, "tolerance": tol,
                                        "source": "onnxruntime on the published graph",
                                        "max_abs_card_vs_source": max(report["src_single"], report["src_batched"])}
        out.write_text(json.dumps(back))
    return res


def parse_box(text):
    box = {}
    for item in text.split(","):
        name, rng = item.split("=")
        lo, hi = rng.split(":")
        box[name.strip()] = (float(lo), float(hi))
    return box

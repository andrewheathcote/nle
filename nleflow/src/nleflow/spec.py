"""The model specification (NLEFLOW_PLAN.md section 7.2): what a user writes
to train a flow for a model. YAML or JSON; `load_spec()` validates it and
returns a dict with absolute paths, the form every command works from and the
form written into the run directory as spec.json.

    name: rdm_a0                  # the label of runs and cards made from it
    model: RDM                    # the analytic twin (an EMC2 model name)
    response: race                # race | choice (choice: milestone M2)
    context: [v, B, t0, s, A]     # the network's inputs, in its order
    transforms: {v: log, B: log, t0: log, s: log, A: zerobox}
    zerobox_c: {A: 0.05}          # asinh(x / c) inputs; exact at 0
    stored: {A: log}              # how the data store a zero-box input: log
                                  # (default) or probit; -Inf at an exact 0
    box_natural: {v: [1e-3, 5], B: [0.1, 3], t0: [0.05, 1], s: [0.1, 2], A: [0, 2]}
    sampling: {A: {uniform: 0.70, asinh_uniform: 0.15, zero: 0.15}}
    pre: null                     # t0: the flow models rt - t0 (unproven option)
    scaling: null                 # the scaled DDM (M2)
    spline: {bins: 64, range: [-5, 5]}
    conditioner: {widths: [256, 128], norm: false, affine: false}
    classifier: null              # choice models only
    training: {rows: 24e6, val_rows: 1e6, trials_per_draw: 1, batch: 1024,
               epochs: 300, keep: 30, peak_lr: 1e-4, seed: 801}
    data: {train: data/train, val: data/val}   # relative to the spec file
    simulator: Rscript simulate.R              # writes the data (see README)
    control_limits: {s: [0.1, 2], vb_over_s2_max: 354}   # race cards
    mass_tolerance: 1e-3
    validation: {}
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

TRANSFORMS = ("identity", "log", "probit", "zerobox")
STORED = ("log", "probit")
RESPONSES = ("race", "choice")
STRATA = ("uniform", "asinh_uniform", "zero")

TRAINING_DEFAULTS = {
    "rows": None,            # the simulator's row count (None: the simulator's own default)
    "val_rows": None,
    "trials_per_draw": 1,
    "batch": 1024,
    "val_batch": 256,
    "epochs": 300,
    "keep": 30,
    "peak_lr": 1e-4,
    "warmup": 0.1,           # fraction of the steps
    "seed": 100,
    "shuffle_seed": None,    # None: the batch order is unseeded (train_race.py's default)
}
CONDITIONER_DEFAULTS = {"widths": [256, 128], "norm": False, "affine": False}
SPLINE_DEFAULTS = {"bins": 64, "range": [-5.0, 5.0]}
BEYOND_SUPPORT = ("identity tails beyond [range_min, range_max] in log t carry the "
                  "remaining mass: the density integrates to 1 over t > 0")


class SpecError(ValueError):
    pass


def _need(cond, msg):
    if not cond:
        raise SpecError(msg)


def _num(x, what):
    _need(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x),
          f"{what} must be a finite number, got {x!r}")
    return float(x)


def read_spec_file(path):
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        return yaml.safe_load(text)
    if path.suffix.lower() == ".json":
        return json.loads(text)
    raise SpecError(f"{path}: a spec is a .yaml, .yml or .json file")


def load_spec(path):
    """Read, validate and resolve a spec file. Paths become absolute (relative
    to the spec file); defaults are filled in; the result is JSON-serialisable."""
    path = Path(path).resolve()
    raw = read_spec_file(path)
    _need(isinstance(raw, dict), f"{path}: the spec must be a mapping")
    return validate_spec(raw, base=path.parent, source=str(path))


def validate_spec(raw, base=None, source=None):
    base = Path(base or ".").resolve()
    s = dict(raw)
    for key in ("name", "model", "context", "transforms", "box_natural"):
        _need(key in s, f"spec needs '{key}'")
    _need(isinstance(s["name"], str) and s["name"].strip() and "/" not in s["name"],
          "'name' must be a non-empty string without '/'")
    _need(isinstance(s["model"], str) and s["model"].strip(), "'model' must be the twin's name")
    resp = s.get("response", "race")
    _need(resp in RESPONSES, f"'response' must be one of {RESPONSES}")
    if resp == "choice":
        raise SpecError("response: choice (the DDM's flow + classifier) is milestone M2; "
                        "nleflow 0.1 trains race flows only")

    ctx = s["context"]
    _need(isinstance(ctx, list) and ctx and all(isinstance(c, str) for c in ctx),
          "'context' must be a non-empty list of names")
    _need(len(set(ctx)) == len(ctx), "'context' names must be unique")
    for c in ctx:
        _need(c not in ("rt", "d", "ll", "R", "param_id"), f"'{c}' is a reserved column name")

    tr = s["transforms"]
    _need(isinstance(tr, dict), "'transforms' must map each context name to a transform")
    for c in ctx:
        _need(c in tr, f"transforms: no transform for context '{c}'")
        _need(tr[c] in TRANSFORMS, f"transforms: '{c}' is {tr[c]!r}, not one of {TRANSFORMS}")
    extra = [k for k in tr if k not in ctx]
    _need(not extra, f"transforms names {extra} that are not in the context")
    s["transforms"] = {c: tr[c] for c in ctx}

    zb = [c for c in ctx if tr[c] == "zerobox"]
    zc = dict(s.get("zerobox_c") or {})
    for c in zb:
        _need(c in zc, f"zerobox_c: no constant for the zero-box input '{c}'")
        _need(_num(zc[c], f"zerobox_c[{c}]") > 0, f"zerobox_c[{c}] must be positive")
    _need(not [k for k in zc if k not in zb], "zerobox_c names an input that is not zerobox")
    s["zerobox_c"] = {c: float(zc[c]) for c in zb}
    st = dict(s.get("stored") or {})
    for k in st:
        _need(k in zb, f"stored: '{k}' is not a zero-box input (only those have a storage scale)")
        _need(st[k] in STORED, f"stored[{k}] must be one of {STORED}")
    s["stored"] = {c: st.get(c, "probit" if c == "SZ" else "log") for c in zb}

    box = s["box_natural"]
    _need(isinstance(box, dict), "'box_natural' must map each context name to [lower, upper]")
    bn = {}
    for c in ctx:
        _need(c in box, f"box_natural: no box for '{c}'")
        b = box[c]
        _need(isinstance(b, (list, tuple)) and len(b) == 2, f"box_natural[{c}] must be [lower, upper]")
        lo, hi = _num(b[0], f"box_natural[{c}][0]"), _num(b[1], f"box_natural[{c}][1]")
        _need(lo < hi, f"box_natural[{c}]: lower must be below upper")
        if tr[c] == "log":
            _need(lo > 0, f"box_natural[{c}]: a log input needs a positive lower bound")
        if tr[c] == "probit":
            _need(0 < lo and hi < 1, f"box_natural[{c}]: a probit input lives in (0, 1)")
        if tr[c] == "zerobox":
            _need(lo >= 0, f"box_natural[{c}]: a zero-box input needs lower >= 0")
        bn[c] = [lo, hi]
    s["box_natural"] = bn

    samp = dict(s.get("sampling") or {})
    for c, strata in samp.items():
        _need(c in ctx, f"sampling: '{c}' is not a context name")
        _need(isinstance(strata, dict) and strata, f"sampling[{c}] must map strata to shares")
        for k, v in strata.items():
            _need(k in STRATA, f"sampling[{c}]: stratum {k!r} is not one of {STRATA}")
            _need(0 <= _num(v, f"sampling[{c}][{k}]") <= 1, f"sampling[{c}][{k}] must be a share in [0, 1]")
        _need(abs(sum(float(v) for v in strata.values()) - 1) < 1e-9, f"sampling[{c}]: shares must sum to 1")
        if "zero" in strata or "asinh_uniform" in strata:
            _need(tr[c] == "zerobox", f"sampling[{c}]: zero / asinh_uniform strata need a zerobox input")
    s["sampling"] = {c: {k: float(v) for k, v in strata.items()} for c, strata in samp.items()}

    pre = s.get("pre")
    if pre is not None:
        _need(pre in ctx, f"pre: '{pre}' is not a context name")
        _need(tr[pre] == "log", "pre: the shift parameter must be a log input (its box is natural)")
    s["pre"] = pre
    _need(s.get("scaling") is None, "scaling (the scaled DDM) is milestone M2")
    s["scaling"] = None
    _need(s.get("classifier") is None, "classifier: race flows have none (choice models: M2)")
    s["classifier"] = None

    sp = {**SPLINE_DEFAULTS, **(s.get("spline") or {})}
    _need(isinstance(sp["bins"], int) and sp["bins"] >= 2, "spline.bins must be an integer >= 2")
    rg = sp["range"]
    _need(isinstance(rg, (list, tuple)) and len(rg) == 2, "spline.range must be [min, max]")
    rg = [_num(rg[0], "spline.range[0]"), _num(rg[1], "spline.range[1]")]
    _need(rg[0] < rg[1], "spline.range: min must be below max")
    s["spline"] = {"bins": int(sp["bins"]), "range": rg}

    cond = s.get("conditioner", CONDITIONER_DEFAULTS["widths"])
    if isinstance(cond, (list, tuple)):
        cond = {"widths": list(cond)}
    cond = {**CONDITIONER_DEFAULTS, **cond}
    w = cond["widths"]
    _need(isinstance(w, (list, tuple)) and w and all(isinstance(x, int) and x > 0 for x in w),
          "conditioner.widths must be a non-empty list of positive integers")
    _need(isinstance(cond["norm"], bool) and isinstance(cond["affine"], bool),
          "conditioner.norm and conditioner.affine must be true or false")
    s["conditioner"] = {"widths": [int(x) for x in w], "norm": cond["norm"], "affine": cond["affine"]}

    tr_ = {**TRAINING_DEFAULTS, **(s.get("training") or {})}
    unknown = [k for k in tr_ if k not in TRAINING_DEFAULTS]
    _need(not unknown, f"training: unknown keys {unknown}")
    for k in ("batch", "val_batch", "epochs", "keep", "trials_per_draw", "seed"):
        _need(isinstance(tr_[k], int) and tr_[k] > 0, f"training.{k} must be a positive integer")
    for k in ("rows", "val_rows"):
        if tr_[k] is not None:
            tr_[k] = int(_num(tr_[k], f"training.{k}"))
            _need(tr_[k] > 0, f"training.{k} must be positive")
    _need(_num(tr_["peak_lr"], "training.peak_lr") > 0, "training.peak_lr must be positive")
    _need(0 <= _num(tr_["warmup"], "training.warmup") < 1, "training.warmup is a fraction of the steps")
    if tr_["shuffle_seed"] is not None:
        _need(isinstance(tr_["shuffle_seed"], int), "training.shuffle_seed must be an integer")
    s["training"] = tr_

    data = s.get("data") or {}
    _need(isinstance(data, dict), "'data' must map train/val to paths")
    d = {}
    for k in ("train", "val"):
        p = data.get(k, f"data/{k}")
        d[k] = str((base / p).resolve()) if not Path(p).is_absolute() else str(p)
    s["data"] = d

    sim = s.get("simulator")
    if sim is not None:
        _need(isinstance(sim, str) and sim.strip(), "'simulator' must be a command string")
    s["simulator"] = sim

    cl = s.get("control_limits")
    if cl is not None:
        _need(isinstance(cl, dict), "control_limits must be a mapping")
        out = {}
        if "s" in cl:
            v = cl["s"]
            _need(isinstance(v, (list, tuple)) and len(v) == 2, "control_limits.s must be [lower, upper]")
            out["s"] = [_num(v[0], "control_limits.s[0]"), _num(v[1], "control_limits.s[1]")]
        if "vb_over_s2_max" in cl:
            out["vb_over_s2_max"] = _num(cl["vb_over_s2_max"], "control_limits.vb_over_s2_max")
        _need(not [k for k in cl if k not in ("s", "vb_over_s2_max")], "control_limits: unknown key")
        cl = out
    s["control_limits"] = cl

    mt = s.get("mass_tolerance", 1e-3)
    _need(_num(mt, "mass_tolerance") > 0, "mass_tolerance must be positive")
    s["mass_tolerance"] = float(mt)
    s["beyond_support"] = str(s.get("beyond_support") or BEYOND_SUPPORT)
    val = s.get("validation")
    _need(val is None or isinstance(val, dict), "validation must be a mapping")
    s["validation"] = dict(val or {})
    s["response"] = resp
    if source is not None:
        s["spec_file"] = source
    known = {"name", "model", "response", "context", "transforms", "zerobox_c", "stored",
             "box_natural", "sampling", "pre", "scaling", "spline", "conditioner", "classifier",
             "training", "data", "simulator", "control_limits", "mass_tolerance",
             "beyond_support", "validation", "spec_file", "notes"}
    unknown = sorted(k for k in s if k not in known)
    _need(not unknown, f"unknown spec keys {unknown}")
    return s


# --- derived quantities every command needs ----------------------------------

def to_network_scale(spec, name, x):
    """A natural value of input `name` on the scale the network takes it."""
    t = spec["transforms"][name]
    if t == "identity":
        return float(x)
    if t == "log":
        return math.log(x)
    if t == "probit":
        from scipy.stats import norm
        return float(norm.ppf(x))
    return math.asinh(x / spec["zerobox_c"][name])


def flow_context(spec):
    """The context the flow is conditioned on: the spec's context minus `pre`."""
    return [c for c in spec["context"] if c != spec["pre"]]


def bounds(spec, names=None):
    names = spec["context"] if names is None else names
    nat = {"lower": [spec["box_natural"][c][0] for c in names],
           "upper": [spec["box_natural"][c][1] for c in names]}
    net = {"lower": [to_network_scale(spec, c, spec["box_natural"][c][0]) for c in names],
           "upper": [to_network_scale(spec, c, spec["box_natural"][c][1]) for c in names]}
    return nat, net


def dump_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=False))

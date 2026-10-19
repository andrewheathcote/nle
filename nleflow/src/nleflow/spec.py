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
    fixed: {}                     # twin parameters held at a natural value in training
                                  # (not inputs; e.g. {s: 1, SZ: 0}); `transforms` may
                                  # give their scale. The card records them (`fixed`)
                                  # for register_nn_model(exceptions = ...)
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

A choice model (the scaled DDM, DDMnle) differs in these keys:

    response: choice
    context: [kappa, eta, Z, SZ]
    transforms: {kappa: identity, eta: zerobox, Z: probit, SZ: zerobox}
    zerobox_c: {eta: 0.01, SZ: 0.05}
    stored: {eta: log, SZ: probit}
    box_natural: {kappa: [-10, 10], eta: [0, 8], Z: [0.1, 0.9], SZ: [0, 0.99]}
    sampling:                       # per-input strata of the FREE draws (long form)
      kappa: [{kind: uniform, share: 0.5}, {kind: uniform, share: 0.5, range: [-3, 3]}]
      eta: [{kind: log_uniform, share: 0.75, range: [0.01, 8]}, {kind: asinh_uniform, share: 0.25}]
      SZ: [{kind: asinh_uniform, share: 1, range: [0.05, 0.99]}]
    regimes: [{share: 0.55}, {share: 0.15, zero: [eta]}, {share: 0.15, zero: [SZ]},
              {share: 0.15, zero: [eta, SZ]}]      # which inputs are exactly 0, jointly
    scaling: {kind: ddm_unit, sz: proportion}      # the evaluator's rule (EMC2's vignette)
    classifier: {widths: [256, 256, 256], lr: 1.49e-3, batch: 256, epochs: 194, loss: mse}
    data: {train: data/train, val: data/val, choice_train: data/choice/train.parquet,
           choice_val: data/choice/val.parquet}     # context + p_R2 (exact P(R = 2))

The flow of a choice model is conditioned on (context, R) and the classifier
maps the context to the logit of P(R = 2), trained on exact probabilities
(retrain_classifier.py's `big` configuration, the DDMnle recipe).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

TRANSFORMS = ("identity", "log", "probit", "zerobox")
STORED = ("log", "probit")
RESPONSES = ("race", "choice")
STRATA = ("uniform", "log_uniform", "asinh_uniform", "zero")
SCALING_KINDS = ("ddm_unit",)
DDM_UNIT_CONTEXT = ["kappa", "eta", "Z", "SZ"]
CLASSIFIER_DEFAULTS = {"widths": [256, 256, 256], "lr": 0.001491898934850391, "batch": 256,
                       "epochs": 194, "loss": "mse", "seed": 0}
CLASSIFIER_LOSSES = ("mse", "hybrid", "bce")

TRAINING_DEFAULTS = {
    "rows": None,            # the simulator's row count (None: the simulator's own default)
    "val_rows": None,
    "choice_rows": None,     # choice models: rows of the exact-probability sets
    "choice_val_rows": None,
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

    ctx = s["context"]
    _need(isinstance(ctx, list) and ctx and all(isinstance(c, str) for c in ctx),
          "'context' must be a non-empty list of names")
    _need(len(set(ctx)) == len(ctx), "'context' names must be unique")
    for c in ctx:
        _need(c not in ("rt", "d", "ll", "R", "param_id"), f"'{c}' is a reserved column name")

    fx = s.get("fixed") or {}
    _need(isinstance(fx, dict), "'fixed' must map twin parameters to the natural value they were held at")
    for k, v in fx.items():
        _need(k not in ctx, f"fixed: '{k}' is an input of the network; only parameters outside the context are fixed")
        _need(k not in ("rt", "d", "ll", "R", "param_id"), f"fixed: '{k}' is a reserved column name")
        _num(v, f"fixed[{k}]")
    fx = {k: float(v) for k, v in fx.items()}
    s["fixed"] = fx

    tr = s["transforms"]
    _need(isinstance(tr, dict), "'transforms' must map each context name to a transform")
    for c in ctx:
        _need(c in tr, f"transforms: no transform for context '{c}'")
        _need(tr[c] in TRANSFORMS, f"transforms: '{c}' is {tr[c]!r}, not one of {TRANSFORMS}")
    for c in fx:
        if c in tr:
            _need(tr[c] in TRANSFORMS and tr[c] != "zerobox",
                  f"transforms: the fixed '{c}' is {tr[c]!r}; a fixed parameter's scale is one of "
                  f"identity, log, probit (zerobox is for network inputs)")
    extra = [k for k in tr if k not in ctx and k not in fx]
    _need(not extra, f"transforms names {extra} that are neither in the context nor fixed")
    # the context's transforms in network order, then those given for fixed parameters
    s["transforms"] = {**{c: tr[c] for c in ctx}, **{c: tr[c] for c in fx if c in tr}}

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
    out_samp = {}
    for c, strata in samp.items():
        _need(c in ctx, f"sampling: '{c}' is not a context name")
        if isinstance(strata, dict):            # short form {kind: share, ...}
            strata = [{"kind": k, "share": v} for k, v in strata.items()]
        _need(isinstance(strata, list) and strata, f"sampling[{c}] must map strata to shares or list them")
        rows = []
        for st in strata:
            _need(isinstance(st, dict) and "kind" in st and "share" in st,
                  f"sampling[{c}]: each stratum needs 'kind' and 'share'")
            _need(st["kind"] in STRATA, f"sampling[{c}]: stratum {st['kind']!r} is not one of {STRATA}")
            share = _num(st["share"], f"sampling[{c}].share")
            _need(0 <= share <= 1, f"sampling[{c}]: a share lies in [0, 1]")
            row = {"kind": st["kind"], "share": share}
            if st["kind"] in ("zero", "asinh_uniform"):
                _need(tr[c] == "zerobox", f"sampling[{c}]: zero / asinh_uniform strata need a zerobox input")
            if "range" in st:
                r = st["range"]
                _need(isinstance(r, (list, tuple)) and len(r) == 2, f"sampling[{c}].range must be [lower, upper]")
                r = [_num(r[0], f"sampling[{c}].range[0]"), _num(r[1], f"sampling[{c}].range[1]")]
                _need(bn[c][0] <= r[0] < r[1] <= bn[c][1], f"sampling[{c}].range must lie inside box_natural[{c}]")
                if st["kind"] == "log_uniform":
                    _need(r[0] > 0, f"sampling[{c}]: a log_uniform range needs a positive lower bound")
                row["range"] = r
            elif st["kind"] == "log_uniform":
                _need(bn[c][0] > 0, f"sampling[{c}]: log_uniform over a box reaching 0 needs a range")
            unknown = [k for k in st if k not in ("kind", "share", "range")]
            _need(not unknown, f"sampling[{c}]: unknown keys {unknown}")
            rows.append(row)
        _need(abs(sum(r["share"] for r in rows) - 1) < 1e-9, f"sampling[{c}]: shares must sum to 1")
        out_samp[c] = rows
    s["sampling"] = out_samp

    regs = s.get("regimes")
    if regs is not None:
        _need(isinstance(regs, list) and regs, "regimes must be a non-empty list of {share, zero: [...]}")
        rows = []
        for rg in regs:
            _need(isinstance(rg, dict) and "share" in rg, "regimes: each entry needs a share")
            share = _num(rg["share"], "regimes.share")
            zero = list(rg.get("zero") or [])
            for z in zero:
                _need(z in zb, f"regimes: '{z}' is not a zero-box input")
            unknown = [k for k in rg if k not in ("share", "zero")]
            _need(not unknown, f"regimes: unknown keys {unknown}")
            rows.append({"share": share, "zero": zero})
        _need(abs(sum(r["share"] for r in rows) - 1) < 1e-9, "regimes: shares must sum to 1")
        regs = rows
    s["regimes"] = regs

    pre = s.get("pre")
    if pre is not None:
        _need(pre in ctx, f"pre: '{pre}' is not a context name")
        _need(tr[pre] == "log", "pre: the shift parameter must be a log input (its box is natural)")
    s["pre"] = pre
    sc = s.get("scaling")
    if sc is not None:
        _need(resp == "choice", "scaling applies to choice models only")
        if isinstance(sc, str):
            sc = {"kind": sc}
        _need(isinstance(sc, dict) and sc.get("kind") in SCALING_KINDS,
              f"scaling.kind must be one of {SCALING_KINDS}")
        sc = {"kind": sc["kind"], "sz": sc.get("sz", "proportion"),
              "pars": list(sc.get("pars") or ["v", "a", "sv", "t0", "st0", "s", "Z", "SZ"])}
        _need(sc["sz"] == "proportion", "scaling.sz must be 'proportion' (EMC2's vignette)")
        _need(ctx == DDM_UNIT_CONTEXT, f"a ddm_unit scaling needs the context {DDM_UNIT_CONTEXT}")
        _need(s["transforms"] == {"kappa": "identity", "eta": "zerobox", "Z": "probit", "SZ": "zerobox"},
              "a ddm_unit scaling needs transforms kappa identity, eta zerobox, Z probit, SZ zerobox")
        _need(pre is None, "a scaled card applies t0 itself; pre does not apply")
        _need(not fx, "a scaled card takes every DDM parameter; fixed does not apply")
    s["scaling"] = sc
    cl = s.get("classifier")
    if resp == "choice":
        _need(cl is not None, "a choice model needs a classifier ({widths: [...]})")
        if isinstance(cl, (list, tuple)):
            cl = {"widths": list(cl)}
        cl = {**CLASSIFIER_DEFAULTS, **cl}
        w = cl["widths"]
        _need(isinstance(w, (list, tuple)) and w and all(isinstance(x, int) and x > 0 for x in w),
              "classifier.widths must be a non-empty list of positive integers")
        for k in ("batch", "epochs"):
            _need(isinstance(cl[k], int) and cl[k] > 0, f"classifier.{k} must be a positive integer")
        _need(isinstance(cl["seed"], int) and cl["seed"] >= 0, "classifier.seed must be a non-negative integer")
        _need(_num(cl["lr"], "classifier.lr") > 0, "classifier.lr must be positive")
        _need(cl["loss"] in CLASSIFIER_LOSSES, f"classifier.loss must be one of {CLASSIFIER_LOSSES}")
        unknown = [k for k in cl if k not in CLASSIFIER_DEFAULTS]
        _need(not unknown, f"classifier: unknown keys {unknown}")
        cl = {"widths": [int(x) for x in w], "lr": float(cl["lr"]), "batch": cl["batch"],
              "epochs": cl["epochs"], "loss": cl["loss"], "seed": cl["seed"]}
    else:
        _need(cl is None, "classifier: race flows have none")
    s["classifier"] = cl

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
    for k in ("rows", "val_rows", "choice_rows", "choice_val_rows"):
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
    keys = ("train", "val") + (("choice_train", "choice_val") if resp == "choice" else ())
    defaults = {"train": "data/train", "val": "data/val",
                "choice_train": "data/choice/train.parquet", "choice_val": "data/choice/val.parquet"}
    for k in keys:
        p = data.get(k, defaults[k])
        d[k] = str((base / p).resolve()) if not Path(p).is_absolute() else str(p)
    unknown = [k for k in data if k not in keys]
    _need(not unknown, f"data: unknown keys {unknown}")
    s["data"] = d

    sim = s.get("simulator")
    if sim is not None:
        _need(isinstance(sim, str) and sim.strip(), "'simulator' must be a command string")
    s["simulator"] = sim

    cl = s.get("control_limits")
    if cl is not None:
        _need(resp == "race", "control_limits are for race cards (the analytic Wald control)")
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
             "box_natural", "fixed", "sampling", "regimes", "pre", "scaling", "spline", "conditioner", "classifier",
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
    """The context the flow is conditioned on: the spec's context minus `pre`
    (a choice model's flow also sees R, appended after these, unstandardised)."""
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

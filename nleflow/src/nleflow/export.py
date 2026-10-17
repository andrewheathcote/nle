"""`nleflow export <run> [--step N]`: a trained checkpoint to a card
(EMC2's vignette "neural-likelihood-cards", version 1) with its golden
vectors and intermediates.

The card is built from the run's spec.json and scaler.json and the weights
of one checkpoint. Its golden vectors are the float64 port's (golden.py),
and before anything is written that port is gated against the flow itself
evaluated in float64 by JAX (`jax_enable_x64`): the two must agree on log
density to 1e-8 and on the spline inversion to 1e-8. The float32 gap (the
same flow as trained, in float32) is printed for information only; it is the
number the NLE project's export_flow.py used to gate on.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from . import __version__
from .data import Scaler
from .golden import golden_arrays, write_golden
from .spec import bounds, flow_context

GATE_TOL = 1e-8
GOLDEN_TOL = 1e-9       # what nn_golden_check() applies
MIN_BIN_SIZE = 1e-4
MIN_KNOT_SLOPE = 1e-4


def read_run(run):
    run = Path(run).resolve()
    spec_path = run / "spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"{run} is not a run directory (no spec.json)")
    spec = json.loads(spec_path.read_text())
    scaler = Scaler.load(run / "scaler.json")
    return run, spec, scaler


def checkpoint_steps(run):
    import orbax.checkpoint as ocp
    mgr = ocp.CheckpointManager(str(Path(run) / "checkpoints"))
    return mgr, sorted(mgr.all_steps()), mgr.best_step(), mgr.latest_step()


def read_conditioner(run, step=None):
    """(step, layers, norms, num_bins, affine) of a checkpoint; the
    architecture is read off the weights (export_flow.py's read_conditioner)."""
    import orbax.checkpoint as ocp
    mgr, steps, best, latest = checkpoint_steps(run)
    if not steps:
        raise FileNotFoundError(f"no checkpoints under {run}")
    if step is None:
        step = best if best is not None else latest
    if step not in steps:
        raise ValueError(f"step {step} is not among the kept checkpoints {steps}")
    raw = mgr.restore(step, args=ocp.args.Composite(
        conditioner=ocp.args.StandardRestore()))["conditioner"]
    hidden = raw.get("linears", {})
    layers = [(np.asarray(hidden[i]["kernel"]["value"], dtype=np.float64),
               np.asarray(hidden[i]["bias"]["value"], dtype=np.float64))
              for i in sorted(hidden, key=int)]
    layers.append((np.asarray(raw["out"]["kernel"]["value"], dtype=np.float64),
                   np.asarray(raw["out"]["bias"]["value"], dtype=np.float64)))
    norms = None
    if "norms" in raw:
        norms = [(np.asarray(raw["norms"][i]["scale"]["value"], dtype=np.float64),
                  np.asarray(raw["norms"][i]["bias"]["value"], dtype=np.float64))
                 for i in sorted(raw["norms"], key=int)]
    dout = layers[-1][0].shape[1]
    if dout % 3 == 1:
        num_bins, affine = (dout - 1) // 3, False
    elif dout % 3 == 0:
        num_bins, affine = (dout - 3) // 3, True
    else:
        raise ValueError(f"conditioner output width {dout} is neither 3K+1 nor 3K+3")
    return int(step), layers, norms, num_bins, affine, step_metrics(run, step).get("val_loss")


def step_metrics(run, step):
    """The metrics saved with a step (val_loss): orbax < 0.11.2 writes them as
    <step>/metrics/metrics (JSON) and its manager does not read them back."""
    p = Path(run) / "checkpoints" / str(step) / "metrics" / "metrics"
    if not p.exists():
        return {}
    try:
        return {k: float(v) for k, v in json.loads(p.read_text()).items()}
    except (ValueError, AttributeError):
        return {}


def read_classifier(run):
    """The choice classifier a run trained (classifier.json): float64 layers."""
    p = Path(run) / "classifier.json"
    if not p.exists():
        raise FileNotFoundError(f"{run} has no classifier.json: train the choice model first")
    c = json.loads(p.read_text())
    return [(np.asarray(l["W"], dtype=np.float64), np.asarray(l["b"], dtype=np.float64)) for l in c["layers"]]


def build_card(spec, scaler, step, layers, norms, num_bins, affine, val_loss=None, run_name=None,
               clf_layers=None):
    if spec["response"] == "choice":
        return build_card_joint(spec, scaler, step, layers, norms, num_bins, affine, clf_layers, val_loss, run_name)
    names = flow_context(spec)
    if layers[0][0].shape[0] != len(names):
        raise ValueError(f"checkpoint input width {layers[0][0].shape[0]} != {len(names)} flow inputs {names}")
    if scaler.names != names:
        raise ValueError(f"scaler.json names {scaler.names} != flow context {names}")
    if num_bins != spec["spline"]["bins"] or affine != spec["conditioner"]["affine"]:
        raise ValueError(f"checkpoint has {num_bins} bins, affine={affine}; the spec says "
                         f"{spec['spline']['bins']}, affine={spec['conditioner']['affine']}")
    nat, net = bounds(spec, names)
    rmin, rmax = spec["spline"]["range"]
    card = {
        "card_version": 1,
        "kind": "flow_race",
        "model": spec["model"],
        "label": spec["name"],
        "budget": run_name or spec["name"],
        "checkpoint_step": int(step),
        "exported": str(date.today()),
        "exporter_version": f"nleflow {__version__}",
        "source": f"{run_name or spec['name']} step {step}",
        "context_names": names,
        "context_transforms": {c: spec["transforms"][c] for c in names},
        "parameter_scale": "sampled",
        **({"zerobox_c": {c: spec["zerobox_c"][c] for c in names if c in spec["zerobox_c"]}}
           if any(c in spec["zerobox_c"] for c in names) else {}),
        "bounds_sampled": net,
        "bounds_natural": nat,
        "affine": bool(affine),
    }
    if affine:
        from .flow import AFFINE_MIN_SCALE, AFFINE_OFFSET
        card["affine_scale"] = {"min_scale": AFFINE_MIN_SCALE, "offset": AFFINE_OFFSET}
    if spec["pre"]:
        card["pre"] = spec["pre"]
        card["pre_bounds_natural"] = {"lower": spec["box_natural"][spec["pre"]][0],
                                      "upper": spec["box_natural"][spec["pre"]][1]}
    card.update({
        "scaler": {"mean": scaler.mean.tolist(), "scale": scaler.scale.tolist()},
        "mlp": {
            "activation": "gelu_tanh",
            "use_norm": norms is not None,
            "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in layers],
            "norms": ([{"scale": s.tolist(), "bias": b.tolist(), "eps": 1e-6} for s, b in norms]
                      if norms else None),
        },
        "spline": {
            "num_bins": int(num_bins), "num_splines": 1,
            "range_min": float(rmin), "range_max": float(rmax),
            "min_bin_size": MIN_BIN_SIZE, "min_knot_slope": MIN_KNOT_SLOPE,
            "boundary_slopes": "identity", "output_transform": "exp",
            "base_distribution": "standard_normal",
        },
        "ll_floor_log": float(np.log(1e-10)),
        # support and mass (vignette, v1): all of t > 0; "Inf" is what jsonlite
        # reads as R's Inf (JSON has no infinity)
        "support": {"time": [0.0, "Inf"]},
        "mass_tolerance": spec["mass_tolerance"],
        "beyond_support": spec["beyond_support"],
    })
    if spec.get("control_limits"):
        card["control_limits"] = spec["control_limits"]
    # the validation block is written only once it records something: EMC2
    # tells the user that a card without one has not been validated
    if spec.get("validation"):
        card["validation"] = dict(spec["validation"])
    card["training"] = {k: spec["training"][k] for k in ("batch", "epochs", "peak_lr", "seed", "keep")}
    if val_loss is not None:
        card["training"]["val_loss"] = val_loss
    return card


def build_card_joint(spec, scaler, step, layers, norms, num_bins, affine, clf_layers, val_loss=None, run_name=None):
    """A flow_joint or flow_scaled card (vignette, v1): the flow conditioned on
    (context, R), the classifier on the context, both standardised alike."""
    names = flow_context(spec)
    if layers[0][0].shape[0] != len(names) + 1:
        raise ValueError(f"flow input width {layers[0][0].shape[0]} != {len(names)} context inputs + R")
    if affine:
        raise ValueError("a choice model's flow has no affine stage")
    if clf_layers is None or clf_layers[0][0].shape[0] != len(names) or clf_layers[-1][0].shape[1] != 1:
        raise ValueError("the classifier must map the context to one logit")
    if scaler.names != names:
        raise ValueError(f"scaler.json names {scaler.names} != flow context {names}")
    if num_bins != spec["spline"]["bins"]:
        raise ValueError(f"checkpoint has {num_bins} bins; the spec says {spec['spline']['bins']}")
    nat, net = bounds(spec, names)
    rmin, rmax = spec["spline"]["range"]
    zb = {c: spec["zerobox_c"][c] for c in names if c in spec["zerobox_c"]}
    scaled = spec["scaling"] is not None
    card = {
        "card_version": 1,
        "kind": "flow_scaled" if scaled else "flow_joint",
        "model": "DDMsc" if scaled else spec["model"],
        "label": spec["name"],
        "budget": run_name or spec["name"],
        "checkpoint_step_flow": int(step),
        "checkpoint_step_classifier": "trained with the run (classifier.json)",
        "exported": str(date.today()),
        "exporter_version": f"nleflow {__version__}",
        "source": f"{run_name or spec['name']} step {step}",
        "context_names": names,
        "classifier_context_names": names,
        "context_encoding": "zerobox" if zb else "emc2",
        **({"zerobox_c": zb} if zb else {}),
        **({"scaling": dict(spec["scaling"])} if scaled else {}),
        "context_transforms": {c: spec["transforms"][c] for c in names},
        "parameter_scale": "sampled",
        "response_coding": ("flow context gets raw R (1.0 or 2.0) appended after standardized params; "
                            "classifier logit is for P(R=2)"),
        "bounds_sampled": net,
        "bounds_natural": nat,
        "scaler": {"mean": scaler.mean.tolist(), "scale": scaler.scale.tolist()},
        "classifier_scaler": {"mean": scaler.mean.tolist(), "scale": scaler.scale.tolist()},
        "flow_mlp": {
            "activation": "gelu_tanh",
            "use_norm": norms is not None,
            "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in layers],
            "norms": ([{"scale": s.tolist(), "bias": b.tolist(), "eps": 1e-6} for s, b in norms]
                      if norms else None),
        },
        "classifier_mlp": {
            "activation": "gelu_tanh",
            "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in clf_layers],
        },
        "spline": {
            "num_bins": int(num_bins), "num_splines": 1,
            "range_min": float(rmin), "range_max": float(rmax),
            "min_bin_size": MIN_BIN_SIZE, "min_knot_slope": MIN_KNOT_SLOPE,
            "boundary_slopes": "identity", "output_transform": "exp",
            "base_distribution": "standard_normal",
        },
        "ll_floor_log": float(np.log(1e-10)),
        # the network's time: u for a scaled card, rt otherwise; all of it
        "support": {"time": [0.0, "Inf"]},
        "mass_tolerance": spec["mass_tolerance"],
        "beyond_support": spec["beyond_support"],
    }
    if spec.get("validation"):
        card["validation"] = dict(spec["validation"])
    card["training"] = {k: spec["training"][k] for k in ("batch", "epochs", "peak_lr", "seed", "keep")}
    if val_loss is not None:
        card["training"]["val_loss"] = val_loss
    card["training"]["classifier"] = {k: spec["classifier"][k] for k in ("widths", "lr", "batch", "epochs", "loss", "seed")}
    return card


def _jax_conditioner(layers, norms, num_bins, affine, rmin, rmax, dtype):
    import jax.numpy as jnp
    from flax import nnx
    from .flow import make_mlp_conditioner
    dims = [layers[0][0].shape[0]] + [W.shape[1] for W, _ in layers[:-1]]
    cond = make_mlp_conditioner(nnx.Rngs(0), dims=dims, num_bins=num_bins, num_splines=1,
                                use_norm=norms is not None, range_min=rmin, range_max=rmax,
                                affine=affine)
    for i, (W, b) in enumerate(layers[:-1]):
        cond.linears[i].kernel.value = jnp.asarray(W, dtype=dtype)
        cond.linears[i].bias.value = jnp.asarray(b, dtype=dtype)
    if norms is not None:
        for i, (s, b) in enumerate(norms):
            cond.norms[i].scale.value = jnp.asarray(s, dtype=dtype)
            cond.norms[i].bias.value = jnp.asarray(b, dtype=dtype)
    W, b = layers[-1]
    cond.out.kernel.value = jnp.asarray(W, dtype=dtype)
    cond.out.bias.value = jnp.asarray(b, dtype=dtype)
    return cond


def _jax_eval(cond, ctx_std, t_flow, dtype):
    """log density and z of the JAX flow, one parameter set per row."""
    import jax.numpy as jnp
    from .flow import spline_flow
    lp = np.empty(t_flow.shape); z = np.empty(t_flow.shape)
    for i in range(ctx_std.shape[0]):
        rt = jnp.asarray(t_flow[i], dtype=dtype)
        ctx = jnp.tile(jnp.asarray(ctx_std[i], dtype=dtype), (t_flow.shape[1], 1))
        lp_i, flow = spline_flow(rt, ctx, cond)
        lp[i] = np.asarray(lp_i, dtype=np.float64)
        z[i] = np.asarray(flow.bijector.inverse(rt), dtype=np.float64)
    return lp, z


def _jax_classifier(clf_layers, dtype):
    import jax.numpy as jnp
    from flax import nnx
    from .flow import make_choice_classifier
    clf = make_choice_classifier(nnx.Rngs(0), dims=[clf_layers[0][0].shape[0]] + [W.shape[1] for W, _ in clf_layers[:-1]])
    for i, (W, b) in enumerate(clf_layers):
        clf.layers[i].kernel.value = jnp.asarray(W, dtype=dtype)
        clf.layers[i].bias.value = jnp.asarray(b, dtype=dtype)
    return clf


def _jax_eval_joint(cond, clf, ctx_std, rt, dtype):
    """log p(rt, R | theta) and z of the JAX joint flow per response: dicts
    keyed by R of (n_sets, n_rt) arrays."""
    import jax.numpy as jnp
    from .flow import joint_log_prob, spline_flow
    n, m = ctx_std.shape[0], rt.shape[0]
    out = {}
    for R in (1, 2):
        lp = np.empty((n, m)); z = np.empty((n, m))
        for i in range(n):
            rt_j = jnp.asarray(rt, dtype=dtype)
            ctx = jnp.tile(jnp.asarray(ctx_std[i], dtype=dtype), (m, 1))
            lp[i] = np.asarray(joint_log_prob(cond, clf, rt_j, ctx, jnp.full((m,), R, dtype=jnp.int32)), dtype=np.float64)
            full = jnp.concatenate([ctx, jnp.full((m, 1), float(R), dtype=dtype)], axis=1)
            _, flow = spline_flow(rt_j, full, cond)
            z[i] = np.asarray(flow.bijector.inverse(rt_j), dtype=np.float64)
        out[R] = (lp, z)
    return out


def gate(card, layers, norms, num_bins, affine, tol=GATE_TOL, say=print, clf_layers=None):
    """The float64 port (the golden vectors) against the JAX flow in float64,
    and the float32 gap for information. Returns (header, rows, arrays,
    report); raises if the gate fails."""
    import jax
    kind, header, rows, arr = golden_arrays(card)
    rmin, rmax = card["spline"]["range_min"], card["spline"]["range_max"]
    if kind == "flow_race":
        ctx_std, t_flow = arr["ctx_std"], arr["t_flow"]
        port_lp, port_z = arr["log_pdf"], arr["z"]

        def run(dtype):
            cond = _jax_conditioner(layers, norms, num_bins, affine, rmin, rmax, dtype)
            return _jax_eval(cond, ctx_std, t_flow, dtype)
    else:
        ctx_std, rt = arr["ctx_std"], arr["rt"]
        res = arr["res"]
        port_lp = np.concatenate([res[1][1], res[2][1]]); port_z = np.concatenate([res[1][0], res[2][0]])
        hard = arr["hard"]
        if hard is not None:              # the support rule is applied after the network
            port_lp = np.where(np.concatenate([hard, hard]), np.nan, port_lp)

        def run(dtype):
            cond = _jax_conditioner(layers, norms, num_bins, False, rmin, rmax, dtype)
            clf = _jax_classifier(clf_layers, dtype)
            o = _jax_eval_joint(cond, clf, ctx_std, rt, dtype)
            return np.concatenate([o[1][0], o[2][0]]), np.concatenate([o[1][1], o[2][1]])
    use = np.isfinite(port_lp)
    # float32 first: the flow as trained, with the default (float32) JAX config
    lp32, z32 = run("float32")
    gap32 = float(np.max(np.abs(lp32 - port_lp)[use]))
    # then float64: the gate
    jax.config.update("jax_enable_x64", True)
    lp64, z64 = run("float64")
    d_lp = float(np.max(np.abs(lp64 - port_lp)[use]))
    d_z = float(np.max(np.abs(z64 - port_z)[use]))
    report = {"gate_tol": tol, "max_abs_dlog_pdf_f64": d_lp, "max_abs_dz_f64": d_z,
              "max_abs_dlog_pdf_f32": gap32, "n_points": int(use.sum())}
    say(f"float64 gate: port vs JAX float64 over {int(use.sum())} points: max |dlog_pdf| {d_lp:.2e}, "
        f"max |dz| {d_z:.2e} (tolerance {tol:.0e})")
    say(f"float32 gap (information): port vs JAX float32 max |dlog_pdf| {gap32:.2e}")
    if not (d_lp <= tol and d_z <= tol):
        raise RuntimeError(f"the float64 gate failed: max |dlog_pdf| {d_lp:.2e}, max |dz| {d_z:.2e} "
                           f"> {tol:.0e}; the card was not written")
    return header, rows, arr, report


def export(run, step=None, out=None, tag=None, say=print):
    """Export one checkpoint of a run; returns the paths written."""
    run, spec, scaler = read_run(run)
    step, layers, norms, num_bins, affine, val_loss = read_conditioner(run, step)
    choice = spec["response"] == "choice"
    clf_layers = read_classifier(run) if choice else None
    card = build_card(spec, scaler, step, layers, norms, num_bins, affine, val_loss, run.name, clf_layers)
    say(f"run {run.name}: step {step}" + (f" (val loss {val_loss:.5f})" if val_loss is not None else "")
        + f", widths {[layers[0][0].shape[0]] + [W.shape[1] for W, _ in layers]}, "
          f"{num_bins} bins, layernorm {norms is not None}, affine {affine}, "
          f"context {card['context_names']}, kind {card['kind']}"
        + (f", classifier widths {[clf_layers[0][0].shape[0]] + [W.shape[1] for W, _ in clf_layers]}" if choice else ""))
    header, rows, arr, report = gate(card, layers, norms, num_bins, affine, say=say, clf_layers=clf_layers)
    card["export_gate"] = report
    tag = tag or f"{spec['name']}_e{step}"
    out = Path(out).resolve() if out else run / "export"
    out.mkdir(parents=True, exist_ok=True)
    wpath = out / f"{tag}_weights.json"
    gpath = out / f"{tag}_golden.csv"
    ipath = out / f"{tag}_intermediates.json"
    wpath.write_text(json.dumps(card))
    write_golden(header, rows, gpath)
    ipath.write_text(json.dumps(arr["intermediates"]))
    say(f"wrote {wpath}\nwrote {gpath} ({len(rows)} rows)\nwrote {ipath}")
    return {"weights": wpath, "golden": gpath, "intermediates": ipath, "step": step, "gate": report}

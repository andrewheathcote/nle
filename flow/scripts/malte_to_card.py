"""Convert one of Malte Lueken's Orbax conditioner checkpoints into our JSON card.

The card is the format port/R/flow_race.R loads. Run it in the `malte/` environment
(jax >= 0.9, flax 0.12, orbax), not in flow/.venv:

    malte/.venv/bin/python flow/scripts/malte_to_card.py CKPT_DIR OUT.json \
        --bounds "v=0:8,s=0.25:3.5,b=0.25:3.5"

CKPT_DIR holds the Orbax checkpoint and the `eamax_conditioner.json` sidecar written by
`confrdm_jax.flows_affine.save_conditioner`. The weights are read straight from the
checkpoint without importing his model code, so the export cannot silently inherit a
change to it; what the graph does with them is recorded in the card and evaluated by
port/R/flow_race.R.

`--bounds` is the uniform training box on the natural scale (nothing on disk records it).
When the sidecar carries a log-input scaling, its loc/scale are re-derived from the box
and eps in closed form and must agree, which catches a box that is not the one trained on.

Refuses (rather than exporting wrongly): boundary_slopes other than "identity", an
unknown layout, a sidecar without context names, hidden layers that are not the
linear1 / linear_mid* / linear2 chain, and any parameter that is not in the sidecar.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import numpy as np

MIN_SCALE = 1e-3
MIN_BIN_SIZE = 1e-4      # hard-coded in his _spline / eamax spline_flow
MIN_KNOT_SLOPE = 1e-4    # distrax default; neither module overrides it
SIDECAR = "eamax_conditioner.json"


def log_uniform_moments(low, high, eps):
    """Mean and SD of log(x + eps), x ~ U(low, high): the closed form his module uses."""
    a, b = low + eps, high + eps
    F = lambda y: y * math.log(y) - y
    G = lambda y: y * (math.log(y) ** 2 - 2.0 * math.log(y) + 2.0)
    mean = (F(b) - F(a)) / (b - a)
    return mean, math.sqrt(max((G(b) - G(a)) / (b - a) - mean**2, 0.0))


def parse_bounds(text, names):
    box = {}
    for item in text.split(","):
        key, rng = item.split("=")
        lo, hi = (float(x) for x in rng.split(":"))
        box[key.strip()] = (lo, hi)
    if set(box) != set(names):
        raise SystemExit(f"--bounds must name exactly {list(names)}, got {sorted(box)}")
    return box


def read_state(ckpt_dir):
    import orbax.checkpoint as ocp
    with ocp.CheckpointManager(str(Path(ckpt_dir).resolve())) as mgr:
        step = mgr.latest_step()
        if step is None:
            raise SystemExit(f"no checkpoint step under {ckpt_dir}")
        state = mgr.restore(step, args=ocp.args.StandardRestore())
    return step, state


def leaf(x):
    return np.asarray(x["value"] if isinstance(x, dict) and "value" in x else x, dtype=np.float64)


def build_card(ckpt_dir, bounds_text, name=None):
    ckpt_dir = Path(ckpt_dir)
    side = json.loads((ckpt_dir / SIDECAR).read_text())
    names = side.get("context_names")
    if not names:
        raise SystemExit("sidecar has no context_names; refusing to guess the input order")
    slopes = str(side.get("boundary_slopes", "identity"))
    if slopes != "identity":
        raise SystemExit(f'boundary_slopes = "{slopes}" is not supported (only "identity"); '
                         "exporting it would give wrong densities")
    spline_range = float(side.get("spline_range", 5.0))
    box = parse_bounds(bounds_text, names)

    step, state = read_state(ckpt_dir)
    n_mid = int(side.get("num_hidden", 1)) - 1
    order = ["linear1"] + [f"linear_mid{i}" for i in range(1, n_mid + 1)] + ["linear2"]
    extra = set(state) - set(order)
    if extra or any(k not in state for k in order):
        raise SystemExit(f"unexpected conditioner structure: have {sorted(state)}, expected {order}")
    layers = [{"W": leaf(state[k]["kernel"]).tolist(), "b": leaf(state[k]["bias"]).tolist()}
              for k in order]
    dout = len(layers[-1]["b"])
    if dout % 3 == 1:
        num_bins, affine = (dout - 1) // 3, False
    elif dout % 3 == 0:
        num_bins, affine = (dout - 3) // 3, True
    else:
        raise SystemExit(f"output width {dout} is neither 3K+1 nor 3K+3")
    if bool(side.get("affine", affine)) != affine:
        raise SystemExit("sidecar's affine flag disagrees with the output layer's width")
    if len(layers[0]["W"]) != len(names):
        raise SystemExit(f"input layer takes {len(layers[0]['W'])} inputs, sidecar names {len(names)}")
    if not affine and spline_range != 5.0:
        raise SystemExit("a plain flow with spline_range != 5 is not something his code builds")

    card = {
        "model": name or ckpt_dir.parent.name, "source": "malte", "checkpoint_step": int(step),
        "exported": str(date.today()),
        "context_names": list(names),
        "context_scale": "natural", "parameter_scale": "natural",
        "pre": "t0",
        "affine": affine,
        "bounds_natural": {"lower": [box[n][0] for n in names],
                           "upper": [box[n][1] for n in names]},
        "mlp": {"activation": "gelu_tanh", "use_norm": False, "layers": layers},
        "spline": {"num_bins": num_bins, "num_splines": 1,
                   "range_min": -spline_range, "range_max": spline_range,
                   "min_bin_size": MIN_BIN_SIZE, "min_knot_slope": MIN_KNOT_SLOPE,
                   "boundary_slopes": "identity", "output_transform": "exp",
                   "base_distribution": "standard_normal"},
    }
    if affine:
        card["affine_scale"] = {"min_scale": MIN_SCALE,
                                "offset": math.log(math.expm1(1.0 - MIN_SCALE))}
    sc = side.get("input_scaling")
    if sc is not None:
        for j, n in enumerate(names):
            mean, sd = log_uniform_moments(box[n][0], box[n][1], sc["eps"][j])
            if not (math.isclose(mean, sc["loc"][j], rel_tol=1e-9, abs_tol=1e-12)
                    and math.isclose(sd, sc["scale"][j], rel_tol=1e-9, abs_tol=1e-12)):
                raise SystemExit(f"--bounds do not reproduce the recorded input scaling for {n}: "
                                 f"closed form ({mean}, {sd}) vs sidecar ({sc['loc'][j]}, {sc['scale'][j]})")
        card["input_scaling"] = {k: [float(x) for x in sc[k]] for k in ("eps", "loc", "scale")}
    return card


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt_dir")
    ap.add_argument("out")
    ap.add_argument("--bounds", required=True, help='natural-scale training box, "v=0:8,s=0.25:3.5,..."')
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    card = build_card(a.ckpt_dir, a.bounds, a.name)
    Path(a.out).write_text(json.dumps(card))
    print(f"wrote {a.out}: {card['context_names']}, affine={card['affine']}, "
          f"{card['spline']['num_bins']} bins, {len(card['mlp']['layers']) - 1} hidden layer(s), "
          f"input_scaling={'input_scaling' in card}")


if __name__ == "__main__":
    main()

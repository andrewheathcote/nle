"""Use an nleflow card's ONNX graphs with onnxruntime alone (no nleflow, no EMC2).

    python use_onnx.py <card_weights.json>

expects <tag>_trialwise.onnx, <tag>_shared.onnx and <tag>_golden.csv beside the
card (`nleflow onnx <card>` writes the graphs; `nleflow export` the golden file).
It shows the input layout, evaluates a parameter vector given on the natural
scale, checks that the joint density integrates to 1, reproduces the golden rows
with both signatures, and times a batch. Reads the card's JSON only for the
input names and transforms; the graphs need nothing else.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from statistics import NormalDist

card_path = Path(sys.argv[1])
tag = card_path.name.removesuffix("_weights.json")
card = json.loads(card_path.read_text())
names, tf = card["context_names"], card["context_transforms"]
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
sess = {s: ort.InferenceSession(str(card_path.with_name(f"{tag}_{s}.onnx")), opts, providers=["CPUExecutionProvider"])
        for s in ("trialwise", "shared")}

# --- 1. the input layout ----------------------------------------------------------------------
print(f"{tag}: {card['kind']} card for EMC2's {card['model']}; inputs {names}"
      f"{', fixed ' + str(card['fixed']) if card.get('fixed') else ''}")
for s, ss in sess.items():
    print(f"  {s:9s} in  " + "; ".join(f"{i.name} {i.shape} {i.type}" for i in ss.get_inputs())
          + f"\n  {'':9s} out " + "; ".join(f"{o.name} {o.shape}" for o in ss.get_outputs()))
print("  " + sess["trialwise"].get_modelmeta().description)


def stored(natural):
    """Natural-scale parameters -> the graph's inputs: the card's stored scale.
    identity; log; probit; a zero-box input as log x (-Inf at an exact 0)."""
    out = []
    for n in names:
        x = float(natural[n])
        t = tf[n]
        if t == "identity":
            out.append(x)
        elif t in ("log", "zerobox"):
            out.append(np.log(x) if x > 0 else -np.inf)
        elif t == "probit":
            out.append(NormalDist().inv_cdf(x) if 0 < x < 1 else (-np.inf if x == 0 else np.inf))
        else:
            raise ValueError(f"transform {t} of {n}")
    return np.array(out)


def logp_trialwise(theta, rt, choice):
    theta = np.asarray(theta, np.float64)            # one vector [k] for every trial, or one row per trial [N, k]
    X = np.column_stack([np.broadcast_to(theta, (len(rt), theta.shape[-1])), rt, choice])
    return sess["trialwise"].run(None, {"input": X})[0]


def logp_shared(theta, rt, choice):
    return sess["shared"].run(None, {"theta": np.asarray(theta, np.float64), "rt": np.asarray(rt, np.float64),
                                     "choice": np.asarray(choice, np.float64)})[0]


# --- 2. one parameter vector on the natural scale ----------------------------------------------------
lo, hi = card["bounds_natural"]["lower"], card["bounds_natural"]["upper"]
nat = {n: min(max(x, l), h) for n, x, l, h in zip(names, (1.0, 1.2, 0.3, 0.5, 0.5, 0.1, 0, 0), lo, hi)}
for st0 in (nat.get("st0", 0.1), 0.0):
    p = {**nat, "st0": st0} if "st0" in nat else nat
    th = stored(p)
    grid = np.linspace(1e-4, 30.0, 300_001)               # rt in seconds, both responses
    dens = [np.exp(logp_shared(th, grid, np.full(grid.size, c))) for c in (-1.0, 1.0)]
    mass = [np.trapezoid(d, grid) for d in dens]
    below = logp_shared(th, np.array([p["t0"] - 0.01, p["t0"] + 1e-3]), np.array([1.0, 1.0]))
    print(f"natural {', '.join(f'{k}={v:g}' for k, v in p.items())} -> inputs {np.round(th, 4).tolist()}")
    print(f"  P(lower) {mass[0]:.4f} + P(upper) {mass[1]:.4f} = {sum(mass):.4f} (rt to 30 s; 1 for a trained card); "
          f"log p at rt = t0 - 0.01: {below[0]:.3f}, at t0 + 0.001: {below[1]:.3f}")

# --- 3. the golden rows, both signatures -------------------------------------------------------------
g = np.genfromtxt(card_path.with_name(f"{tag}_golden.csv"), delimiter=",", names=True)
theta = np.column_stack([g[n] for n in names])
choice = np.where(g["R"] == 2, 1.0, -1.0)
ref = g["log_pdf_joint"]
got = {"trialwise": logp_trialwise(theta, g["rt"], choice),
       "shared": np.concatenate([logp_shared(theta[g["param_id"] == i][0], g["rt"][g["param_id"] == i],
                                             choice[g["param_id"] == i]) for i in np.unique(g["param_id"])])}
order = np.concatenate([np.flatnonzero(g["param_id"] == i) for i in np.unique(g["param_id"])])
got["shared"] = got["shared"][np.argsort(order)]
floor = card["ll_floor_log"]
ok = True
for s, lp in got.items():
    fin = np.isfinite(ref) & (ref > floor)
    d = np.max(np.abs(lp[fin] - ref[fin]))
    same_inf = np.array_equal(np.isneginf(lp), np.isneginf(ref))
    same_floor = np.array_equal(lp == floor, ref == floor)
    passed = d < 1e-10 and same_inf and same_floor
    ok &= passed
    print(f"golden rows, {s}: {len(ref)} rows, {fin.sum()} above the floor: max |dlog p| {d:.2e}; -Inf rows agree "
          f"{same_inf}; floor rows agree {same_floor}: {'PASS' if passed else 'FAIL'} against 1e-10")

# --- 4. a batch -----------------------------------------------------------------------------------------
rng = np.random.default_rng(1)
n = 100_000
th = stored(nat)
rt = nat["t0"] + rng.exponential(0.6, n)
ch = rng.choice([-1.0, 1.0], n)
for s, f in (("trialwise", logp_trialwise), ("shared", logp_shared)):
    t = time.perf_counter(); f(th, rt, ch); t = time.perf_counter() - t
    print(f"{n} trials, one parameter vector, {s}: {t * 1e3:.0f} ms ({t / n * 1e6:.2f} us per trial, one thread)")
sys.exit(0 if ok else 1)

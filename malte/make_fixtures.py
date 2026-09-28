"""Random-weight conditioners in three of Malte Lueken's configurations, with float64
golden vectors from his own evaluate_pdf_sf, and the JSON card for each.

Random weights make the densities meaningless and the check exact; trained checkpoints
go through the same converter (flow/scripts/malte_to_card.py).

    malte/.venv/bin/python malte/make_fixtures.py

Writes malte/fixtures/<name>/{ckpt/, card.json, golden.csv}. ckpt/ (Orbax) is not tracked.
"""
import json
import subprocess
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from flax import nnx
from scipy.special import log_ndtr

from confrdm_jax import flows_affine as fa

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

CONFIGS = {
    "rdm_final": dict(
        context=["v", "s", "b"],
        box={"v": (0.0, 8.0), "s": (0.25, 3.5), "b": (0.25, 3.5)},
        eps={"v": 0.05, "s": 0.05, "b": 0.05},
        affine=True, log_inputs=True, num_hidden=2, num_bins=12, width=128, seed=11),
    "rdm_plain": dict(
        context=["v", "s", "b"],
        box={"v": (0.0, 8.0), "s": (0.25, 3.5), "b": (0.25, 3.5)},
        eps=None, affine=False, log_inputs=False, num_hidden=1, num_bins=12, width=128, seed=12),
    "crdm_final": dict(
        context=["v", "amp", "tau", "s", "b"],
        box={"v": (0.0, 8.0), "amp": (0.0, 1.0), "tau": (0.01, 0.5),
             "s": (0.25, 3.0), "b": (0.25, 3.0)},
        eps={"v": 0.05, "amp": 0.01, "tau": 0.005, "s": 0.05, "b": 0.05},
        affine=True, log_inputs=True, num_hidden=2, num_bins=12, width=128, seed=13),
}


def randomise(cond, key):
    """Weights with O(1) activations and a non-trivial output layer.

    The default zero-bias LeCun init leaves the spline almost at the identity, which would
    make the golden vectors test nothing.
    """
    rng = np.random.default_rng(key)
    names = ["linear1"] + [f"linear_mid{i}" for i in range(1, getattr(cond, "num_hidden", 1))] + ["linear2"]
    for k, name in enumerate(names):
        layer = getattr(cond, name)
        n_in, n_out = layer.kernel.get_value().shape
        gain = 2.0 if k < len(names) - 1 else 1.2
        layer.kernel.set_value(jnp.asarray(rng.normal(0, gain / np.sqrt(n_in), (n_in, n_out)), jnp.float32))
        bias = rng.normal(0, 0.4, n_out)
        if k == len(names) - 1 and n_out % 3 == 0:
            # affine layout [.., loc, raw_scale]: keep the scale in ~[0.2, 3], not 1e-3 (which
            # would put log densities at -1e7 and make an absolute 1e-10 check meaningless)
            kern = np.asarray(layer.kernel.get_value()).copy()
            kern[:, -2:] *= 0.3
            bias[-2:] *= 0.5
            layer.kernel.set_value(jnp.asarray(kern, jnp.float32))
        layer.bias.set_value(jnp.asarray(bias, jnp.float32))


def theta_design(cfg, rng, n_interior=40, n_edge=30):
    names, box = cfg["context"], cfg["box"]
    lo = np.array([box[n][0] for n in names]); hi = np.array([box[n][1] for n in names])
    d = len(names)
    corners = np.array([[(lo, hi)[(i >> j) & 1][j] for j in range(d)] for i in range(2**d)])
    interior = rng.uniform(lo, hi, (n_interior, d))
    edge = rng.uniform(lo, hi, (n_edge, d))
    for row in edge:                       # snap a random non-empty subset of coordinates to a face
        snap = rng.random(d) < 0.5
        snap[rng.integers(d)] = True
        row[snap] = np.where(rng.random(snap.sum()) < 0.5, lo[snap], hi[snap])
    return np.vstack([(lo + hi) / 2, corners, interior, edge])


def main():
    (ROOT / "fixtures").mkdir(exist_ok=True)
    dt = np.geomspace(1e-4, 1e4, 25)       # log dt in [-9.2, 9.2]: both spline tails and the identity regions
    for name, cfg in CONFIGS.items():
        out = ROOT / "fixtures" / name
        out.mkdir(parents=True, exist_ok=True)
        names = cfg["context"]
        options = dict(
            affine=cfg["affine"], num_hidden=cfg["num_hidden"],
            input_scaling=(fa.log_input_scaling(names, cfg["eps"], cfg["box"]) if cfg["log_inputs"] else None))
        cond = fa.make_mlp_conditioner(nnx.Rngs(cfg["seed"]), num_in=len(names), num_mid=cfg["width"],
                                       num_bins=cfg["num_bins"], **options)
        randomise(cond, cfg["seed"])
        fa.save_conditioner(cond, out / "ckpt", context_names=names, num_mid=cfg["width"], num_bins=cfg["num_bins"])

        rng = np.random.default_rng(cfg["seed"])
        theta = theta_design(cfg, rng)
        th = np.repeat(theta, len(dt), axis=0)
        t = np.tile(dt, len(theta))
        cond.eval()
        log_pdf, log_sf = fa.evaluate_pdf_sf(cond, jnp.asarray(t), jnp.asarray(th))
        _, flow = fa.spline_flow(jnp.asarray(t), jnp.asarray(th), cond)
        z = flow.bijector.inverse(jnp.asarray(t))
        log_pdf, log_sf, z = (np.asarray(a) for a in (log_pdf, log_sf, z))
        # jax's norm.logsf (log_ndtr) is only good to ~4e-9 for z >= 20 (float64), i.e. for
        # survival probabilities below e-87; scipy's is exact there. Keep his value as the golden
        # one and record the exact one beside it so the check can tell the two apart.
        log_sf_exact = log_ndtr(-z)
        band = z < 20.0
        assert np.abs(log_sf - log_sf_exact)[band].max() < 1e-10, name
        assert log_pdf.dtype == np.float64 and np.isfinite(log_pdf).all() and np.isfinite(log_sf).all(), name

        # coverage: which spline region does each point fall in?
        params = fa._conditioner_outputs(cond, jnp.asarray(th))
        y = np.log(t)
        if cfg["affine"]:
            loc, scale = (np.asarray(a) for a in fa._loc_scale(params))
            y = (y - loc) / scale
        lo_id, hi_id = int((y <= -5).sum()), int((y >= 5).sum())
        assert lo_id > 0 and hi_id > 0 and (np.abs(y) < 5).sum() > 0, (name, lo_id, hi_id)

        with open(out / "golden.csv", "w") as f:
            f.write("row," + ",".join(names) + ",dt,z,log_pdf,log_sf,log_sf_exact\n")
            for i in range(len(t)):
                f.write(f"{i + 1}," + ",".join(f"{v:.17g}" for v in th[i]) +
                        f",{t[i]:.17g},{z[i]:.17g},{log_pdf[i]:.17g},{log_sf[i]:.17g},{log_sf_exact[i]:.17g}\n")

        bounds = ",".join(f"{n}={cfg['box'][n][0]!r}:{cfg['box'][n][1]!r}" for n in names)
        subprocess.run([sys.executable, str(REPO / "flow/scripts/malte_to_card.py"),
                        str(out / "ckpt"), str(out / "card.json"), "--bounds", bounds, "--name", name],
                       check=True)
        print(f"{name}: {len(t)} points ({len(theta)} thetas x {len(dt)}), "
              f"y<=-5: {lo_id}, y>=5: {hi_id}, interior: {int((np.abs(y) < 5).sum())}; "
              f"log_pdf in [{log_pdf.min():.1f}, {log_pdf.max():.1f}]; "
              f"jax logsf vs exact: max {np.abs(log_sf - log_sf_exact).max():.1e} "
              f"({int((~band).sum())} points with z >= 20)")


if __name__ == "__main__":
    main()

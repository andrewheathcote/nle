"""Write a trained classifier from an export JSON back into an Orbax checkpoint.

`retrain_classifier.py` trains the choice classifier and splices its weights
straight into the export JSONs — it never saves a checkpoint. That is fine for
export, but `train_ddm.py` loads the FROZEN classifier from a checkpoint
directory, so the corrected (c4) classifier cannot currently be used during
training. It matters for distillation: the target is the analytic JOINT
log-density minus log P(R | theta), so an inaccurate classifier puts a
systematic offset straight into the regression target. The shipped classifier
is off by 0.0092 on average against the corrected target; c4 by 0.000262.

  python scripts/json_to_classifier_ckpt.py \
      --from-export ddm_cap256w_c4 --tag c4clf
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

import numpy as np
import jax
import orbax.checkpoint as ocp
from flax import nnx

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flow.config import make_choice_classifier

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-export", required=True,
                    help="export basename, e.g. ddm_cap256w_c4")
    ap.add_argument("--tag", required=True,
                    help="creates results/model_states/<tag>_budget/DDM_classifier")
    ap.add_argument("--scaler-from", default="large",
                    help="budget whose classifier train_loader.pkl to copy "
                         "(the scaler is unchanged by retraining)")
    a = ap.parse_args()

    w = json.load(open(REPO / "port" / "export" / f"{a.from_export}_weights.json"))
    layers = w["classifier_mlp"]["layers"]
    widths = [np.array(l["W"]).shape[0] for l in layers]      # input dims
    print(f"classifier dims {widths + [1]} from {a.from_export}")

    clf = make_choice_classifier(rngs=nnx.Rngs(default=jax.random.key(0)),
                                 dims=widths)
    assert len(clf.layers) == len(layers), (
        f"architecture mismatch: checkpoint has {len(clf.layers)} layers, "
        f"export has {len(layers)}")
    for lay, src in zip(clf.layers, layers):
        lay.kernel.value = jax.numpy.asarray(np.array(src["W"], dtype=np.float32))
        lay.bias.value = jax.numpy.asarray(np.array(src["b"], dtype=np.float32))

    out = REPO / "results" / "model_states" / f"{a.tag}_budget" / "DDM_classifier"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    mgr = ocp.CheckpointManager(str(out.resolve()),
                                options=ocp.CheckpointManagerOptions(max_to_keep=1))
    _, state = nnx.split(clf)
    mgr.save(0, args=ocp.args.Composite(conditioner=ocp.args.StandardSave(state)),
             metrics={"val_loss": 0.0})
    mgr.wait_until_finished()

    # train_ddm.py's copytree brings the scaler along with the checkpoint
    src_pkl = (REPO / "results" / "model_states" / f"{a.scaler_from}_budget"
               / "DDM_classifier" / "train_loader.pkl")
    if src_pkl.exists():
        shutil.copy(src_pkl, out / "train_loader.pkl")
    print(f"wrote {out}")

    # verify it round-trips through the same path train_ddm.py uses
    m2 = ocp.CheckpointManager(str(out.resolve()))
    raw = m2.restore(m2.best_step(),
                     args=ocp.args.Composite(conditioner=ocp.args.StandardRestore()))["conditioner"]
    keys = sorted(raw["layers"], key=int)
    d = max(float(np.max(np.abs(np.asarray(raw["layers"][k]["kernel"]["value"])
                                - np.array(layers[i]["W"]))))
            for i, k in enumerate(keys))
    print(f"round-trip max |weight diff| = {d:.2e}  {'OK' if d < 1e-6 else 'MISMATCH'}")


if __name__ == "__main__":
    main()

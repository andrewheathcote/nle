"""Run train_race.py from a given source tree for ONE epoch, with the batch
shuffle seeded (the script's own shuffle is np.random.default_rng(), unseeded).
usage: first_epoch.py <tree> <shuffle_seed|none> <train_race args...>
Schedule is built from --epochs as given; only the loop length is cut to 1."""
import sys, importlib.util, json
from pathlib import Path
import numpy as np
tree, sseed = Path(sys.argv[1]), sys.argv[2]
sys.argv = ["train_race.py"] + sys.argv[3:]
if sseed != "none":
    _orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: _orig(int(sseed)) if not a and not k else _orig(*a, **k)
spec = importlib.util.spec_from_file_location("train_race", tree / "flow/scripts/train_race.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
import flow.config as fc
assert str(tree.resolve()) in str(Path(fc.__file__).resolve()), fc.__file__
_tm = mod.train_model
def one_epoch(*a, **k):
    k["num_epochs"] = 1
    out = _tm(*a, **k)
    print("FIRST_EPOCH train %r val %r" % (float(out[0][0]), float(out[1][0])), flush=True)
    return out
mod.train_model = one_epoch
mod.main()

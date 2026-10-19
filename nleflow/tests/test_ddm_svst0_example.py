"""The DDM (sv, st0) example, the eight-parameter choice flow: its specs validate
(s and SZ fixed, sv and st0 zero-box inputs) and its simulator writes the data
contract, with the choice sets, from the resolved spec."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from nleflow.data import open_dataset
from nleflow.simulate import simulate
from nleflow.spec import load_spec

EX = Path(__file__).resolve().parents[1] / "examples" / "ddm_svst0"


@pytest.mark.parametrize("name", ["spec.yaml", "spec_smoke.yaml"])
def test_ddm_svst0_specs_validate(name):
    spec = load_spec(EX / name)
    assert spec["context"] == ["v", "a", "t0", "Z", "sv", "st0"] and spec["model"] == "DDM"
    assert spec["response"] == "choice" and spec["scaling"] is None
    assert spec["fixed"] == {"s": 1.0, "SZ": 0.0}
    assert spec["transforms"]["sv"] == spec["transforms"]["st0"] == "zerobox"
    assert spec["stored"] == {"sv": "log", "st0": "log"}


def test_ddm_svst0_smoke_simulator_writes_the_contract(tmp_path, monkeypatch):
    rlib = os.environ.get("NLEFLOW_RLIB", "")
    if not rlib or shutil.which("Rscript") is None:
        pytest.skip("set NLEFLOW_RLIB to a library holding EMC2 and WienR for the R simulator")
    work = tmp_path / "ddm_svst0"
    shutil.copytree(EX, work, ignore=shutil.ignore_patterns("data*", "runs", "*.log"))
    monkeypatch.setenv("R_PROFILE_USER", "/dev/null")
    spec = load_spec(work / "spec_smoke.yaml")
    simulate(spec, cores=2, say=lambda *_: None)
    t = open_dataset(spec["data"]["train"]).to_table()
    assert t.num_rows == spec["training"]["rows"]
    for c in ("sv", "st0"):               # 0.15 + 0.15 of draws exactly 0
        share0 = float(np.mean(np.isneginf(t.column(c).to_numpy())))
        assert 0.27 < share0 < 0.33, (c, share0)
    R = t.column("R").to_numpy()
    assert set(np.unique(R)) == {1, 2}
    rt, t0 = t.column("rt").to_numpy(), np.exp(t.column("t0").to_numpy())
    assert np.all(rt > t0) and np.all(np.isfinite(t.column("ll").to_numpy()))
    c = open_dataset(spec["data"]["choice_train"]).to_table()
    assert c.num_rows == spec["training"]["choice_rows"] and set(c.schema.names) == {*spec["context"], "p_R2"}
    p = c.column("p_R2").to_numpy()
    assert p.min() >= 0 and p.max() <= 1

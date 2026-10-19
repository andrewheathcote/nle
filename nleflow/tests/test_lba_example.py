"""The LBA example: its specs validate (an identity input whose box goes below 0, the
twin's own parameter names) and its simulator writes the data contract from the
resolved spec."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from nleflow.data import open_dataset
from nleflow.simulate import simulate
from nleflow.spec import load_spec

LBA = Path(__file__).resolve().parents[1] / "examples" / "lba"


@pytest.mark.parametrize("name", ["spec.yaml", "spec_smoke.yaml"])
def test_lba_specs_validate(name):
    spec = load_spec(LBA / name)
    assert spec["context"] == ["v", "sv", "B", "A", "t0"] and spec["model"] == "LBA"
    assert spec["transforms"]["v"] == "identity" and spec["box_natural"]["v"][0] < 0
    assert spec["control_limits"] is None


def test_lba_smoke_simulator_writes_the_contract(tmp_path, monkeypatch):
    rlib = os.environ.get("NLEFLOW_RLIB", "")
    if not rlib or shutil.which("Rscript") is None:
        pytest.skip("set NLEFLOW_RLIB to a library holding EMC2 for the R simulator")
    work = tmp_path / "lba"
    shutil.copytree(LBA, work, ignore=shutil.ignore_patterns("data*", "runs", "*.log"))
    monkeypatch.setenv("R_PROFILE_USER", "/dev/null")
    spec = load_spec(work / "spec_smoke.yaml")
    simulate(spec, cores=2, say=lambda *_: None)
    t = open_dataset(spec["data"]["train"]).to_table()
    assert t.num_rows == spec["training"]["rows"]
    share0 = float(np.mean(np.isneginf(t.column("A").to_numpy())))
    assert 0.12 < share0 < 0.18          # the spec asks for 0.15 exactly at A = 0
    v = t.column("v").to_numpy()          # identity: stored as drawn
    assert v.min() < 0 and -1 <= v.min() and v.max() <= 5
    rt, t0 = t.column("rt").to_numpy(), np.exp(t.column("t0").to_numpy())
    assert np.all(rt > t0) and np.all(np.isfinite(t.column("ll").to_numpy()))

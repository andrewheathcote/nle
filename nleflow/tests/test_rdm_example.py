"""The RDM A = 0 example's simulator reads the resolved spec: since M2 the strata of `sampling`
are (kind, share) rows, and simulate.R still read the short form (found writing the docs)."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from nleflow.data import open_dataset
from nleflow.simulate import simulate
from nleflow.spec import load_spec

RDM = Path(__file__).resolve().parents[1] / "examples" / "rdm_a0"


def test_rdm_smoke_simulator_runs_and_writes_the_a0_stratum(tmp_path, monkeypatch):
    rlib = os.environ.get("NLEFLOW_RLIB", "")
    if not rlib or shutil.which("Rscript") is None:
        pytest.skip("set NLEFLOW_RLIB to a library holding EMC2 for the R simulator")
    work = tmp_path / "rdm_a0"
    shutil.copytree(RDM, work, ignore=shutil.ignore_patterns("data*", "runs"))
    monkeypatch.setenv("R_PROFILE_USER", "/dev/null")
    spec = load_spec(work / "spec_smoke.yaml")
    simulate(spec, cores=2, say=lambda *_: None)
    t = open_dataset(spec["data"]["train"]).to_table()
    assert t.num_rows == spec["training"]["rows"]
    share0 = float(np.mean(np.isneginf(t.column("A").to_numpy())))
    assert 0.12 < share0 < 0.18          # the spec asks for 0.15 exactly at A = 0

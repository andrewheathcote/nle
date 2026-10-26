"""The choice-set names in a spec and the format the simulator writes agree.

The scaled DDM example's simulate.R wrote Parquet (when arrow is installed) to the names the
spec gave, `choice/train.csv`, and the reader chose the format from the suffix
and parsed Parquet bytes as CSV."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from nleflow.data import load_choice_set, open_dataset, write_parquet
from nleflow.simulate import simulate
from nleflow.spec import load_spec

DDM_SCALED = Path(__file__).resolve().parents[1] / "examples" / "ddm_scaled"
COLS = {"kappa": [0.5, 1.0], "eta": [0.1, 0.2], "Z": [0.0, 0.3], "SZ": [0.0, 0.1], "p_R2": [0.2, 0.7]}


@pytest.mark.parametrize("named,written", [("train.csv", "train.parquet"), ("train.parquet", "train.csv"),
                                           ("train.csv", "train.csv"), ("train.parquet", "train.parquet")])
def test_reader_finds_the_file_in_either_format(tmp_path, named, written):
    if written.endswith(".parquet"):
        write_parquet(tmp_path / written, COLS)
    else:
        (tmp_path / written).write_text("kappa,eta,Z,SZ,p_R2\n0.5,0.1,0,0,0.2\n1,0.2,0.3,0.1,0.7\n")
    assert open_dataset(tmp_path / named).count_rows() == 2


def test_missing_file_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_dataset(tmp_path / "train.csv")


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_ddm_scaled_smoke_simulator_names_agree_with_the_spec(tmp_path, fmt, monkeypatch):
    rlib = os.environ.get("NLEFLOW_RLIB", "")
    if not rlib or shutil.which("Rscript") is None:
        pytest.skip("set NLEFLOW_RLIB to a library holding EMC2 and WienR for the R simulator")
    if fmt == "parquet":
        r = os.popen(f'R_PROFILE_USER=/dev/null Rscript -e \'.libPaths(c("{rlib}", .libPaths())); '
                     f'cat(requireNamespace("arrow", quietly = TRUE))\'').read()
        if "TRUE" not in r:
            pytest.skip("R arrow is not installed")
    work = tmp_path / "ddm_scaled"
    shutil.copytree(DDM_SCALED, work, ignore=shutil.ignore_patterns("data*", "runs"))
    monkeypatch.setenv("NLEFLOW_FORMAT", fmt)
    monkeypatch.setenv("R_PROFILE_USER", "/dev/null")
    spec = load_spec(work / "spec_smoke.yaml")
    simulate(spec, cores=2, say=lambda *_: None)
    spec["context"] = ["kappa", "eta", "Z", "SZ"]
    for key in ("choice_train", "choice_val"):
        X, y = load_choice_set(spec["data"][key], spec)
        assert len(X) == len(y) > 0 and np.all((y >= 0) & (y <= 1))

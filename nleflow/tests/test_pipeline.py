"""End to end on synthetic data: simulate (in Python) -> train -> export ->
the golden file is reproduced from the card -> (with NLEFLOW_EMC2_LIB) the
card registers in EMC2 dev-nle-m0 with twin RDM and passes nn_golden_check().

The synthetic model is a shifted log-normal in the RDM's parameter names
(v, B, t0, s, A), not an RDM: the test is of the pipeline and the card
contract, not of the fit. The zero-box A with exact zeros and a `pre` variant
cover the encodings the exporter writes; the names let EMC2 register the card
with twin = RDM.
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from nleflow.data import write_parquet
from nleflow.export import export, read_run
from nleflow.golden import main as golden_main
from nleflow.spec import validate_spec
from nleflow.train import train


def synthetic(n, seed):
    rng = np.random.default_rng(seed)
    v = rng.uniform(0.5, 3.0, n)
    B = rng.uniform(0.5, 2.0, n)
    s = rng.uniform(0.5, 1.5, n)
    A = rng.uniform(0.0, 1.0, n)                   # zero-box input, stored as log A
    A[rng.random(n) < 0.2] = 0.0
    t0 = rng.uniform(0.1, 0.4, n)
    d = np.exp(np.log(B / v) + 0.3 * s * rng.standard_normal(n) + 0.1 * np.arcsinh(A / 0.05))
    with np.errstate(divide="ignore"):
        cols = {"param_id": np.arange(1, n + 1), "rt": t0 + d, "v": np.log(v), "B": np.log(B),
                "t0": np.log(t0), "s": np.log(s), "A": np.log(A)}
    return cols


def make_spec(tmp_path, pre=None, affine=False):
    spec = {
        "name": "synthetic", "model": "RDM", "context": ["v", "B", "t0", "s", "A"],
        "transforms": {"v": "log", "B": "log", "t0": "log", "s": "log", "A": "zerobox"},
        "zerobox_c": {"A": 0.05},
        "box_natural": {"v": [0.5, 3], "B": [0.5, 2], "t0": [0.1, 0.4], "s": [0.5, 1.5], "A": [0, 1]},
        "sampling": {"A": {"uniform": 0.8, "zero": 0.2}},
        "pre": pre,
        "spline": {"bins": 8, "range": [-5, 5]},
        "conditioner": {"widths": [16, 8], "norm": False, "affine": affine},
        "training": {"batch": 256, "epochs": 2, "keep": 2, "peak_lr": 1e-3, "seed": 3, "shuffle_seed": 1},
        "data": {"train": "train", "val": "val"},
        "control_limits": {"s": [0.5, 1.5], "vb_over_s2_max": 354},
    }
    write_parquet(tmp_path / "train" / "t.parquet", synthetic(4096, 1))
    write_parquet(tmp_path / "val" / "v.parquet", synthetic(512, 2))
    return validate_spec(spec, base=tmp_path, source=str(tmp_path / "spec.yaml"))


def read_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True)


@pytest.mark.parametrize("pre,affine", [(None, False), ("t0", True)])
def test_train_export_golden(tmp_path, pre, affine):
    spec = make_spec(tmp_path, pre=pre, affine=affine)
    run = train(spec, runs=tmp_path / "runs", quiet=True)
    assert (run / "scaler.json").exists() and (run / "spec.json").exists()
    log = json.loads((run / "training_log.json").read_text())
    assert len(log["val_losses"]) == 2 and all(np.isfinite(log["val_losses"]))

    res = export(run, say=lambda *a: None)
    card = json.loads(res["weights"].read_text())
    assert card["card_version"] == 1 and card["kind"] == "flow_race" and card["model"] == "RDM"
    expect_ctx = [c for c in ["v", "B", "t0", "s", "A"] if c != pre]
    assert card["context_names"] == expect_ctx
    assert card["context_transforms"] == {c: spec["transforms"][c] for c in expect_ctx}
    assert card["zerobox_c"] == {"A": 0.05}
    assert card["bounds_sampled"]["lower"][-1] == 0.0
    assert card["support"] == {"time": [0.0, "Inf"]} and card["mass_tolerance"] == 1e-3
    assert "validation" not in card                   # nothing recorded yet: EMC2 says so
    assert card["affine"] is affine
    assert ("pre" in card) == (pre is not None)
    if pre:
        assert card["pre"] == "t0" and card["pre_bounds_natural"] == {"lower": 0.1, "upper": 0.4}
    assert res["gate"]["max_abs_dlog_pdf_f64"] <= 1e-8 and res["gate"]["max_abs_dz_f64"] <= 1e-8
    assert res["gate"]["max_abs_dlog_pdf_f32"] > res["gate"]["max_abs_dlog_pdf_f64"]

    # the golden file is reproduced from the card alone
    again = tmp_path / "again.csv"
    golden_main(str(res["weights"]), str(again))
    a, b = read_csv(again), read_csv(res["golden"])
    assert a.dtype.names == b.dtype.names and len(a) == 575
    assert max(float(np.max(np.abs(a[c] - b[c]))) for c in a.dtype.names) < 1e-12
    assert np.sum(b["A"] == 0.0) == 5 * 23


def test_export_refuses_unknown_step(tmp_path):
    spec = make_spec(tmp_path)
    run = train(spec, runs=tmp_path / "runs", quiet=True)
    with pytest.raises(ValueError, match="not among the kept checkpoints"):
        export(run, step=99, say=lambda *a: None)


def test_card_registers_in_emc2(tmp_path, emc2_lib):
    """The R half of M1 acceptance (b): register with twin = RDM, golden check."""
    spec = make_spec(tmp_path)
    run = train(spec, runs=tmp_path / "runs", quiet=True)
    res = export(run, say=lambda *a: None)
    r_code = f"""
    .libPaths(c('{emc2_lib}', .libPaths())); suppressPackageStartupMessages(library(EMC2))
    m <- register_nn_model('{res["weights"]}', twin = RDM)
    reg <- m()$nn
    stopifnot(reg$kind == 'flow_race', reg$card_version == 1, identical(reg$pars, c('v','B','t0','s','A')))
    stopifnot(identical(reg$control_limits$s, c(0.5, 1.5)), reg$control_limits$vb_over_s2_max == 354)
    stopifnot(identical(EMC2:::nn_card_support(reg)$time, c(0, Inf)))
    g <- nn_golden_check(m, error = FALSE)
    cat(format(g), '\\n'); stopifnot(g$pass)
    """
    out = subprocess.run(["Rscript", "-e", r_code], capture_output=True, text=True,
                         env={**os.environ, "R_PROFILE_USER": "/dev/null"})
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PASS" in out.stdout

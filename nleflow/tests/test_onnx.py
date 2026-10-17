"""The ONNX export (M2): the graph of the committed DDMnle card reproduces
its golden rows in both signatures, the eight-parameter graph reproduces
the z0 card's golden rows, and (with NLEFLOW_EMC2_LIB) the scaled graph
agrees with EMC2's evaluation over prior draws in the six regimes of
sbc/scaled_cell.R: to 1e-10 where the likelihood is a density (st0 = 0) and
to 1e-8 on the st0 > 0 branch, a CDF difference whose conditioning limits
every float64 implementation alike (the README says how this was measured).
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from nleflow.onnx_export import SCALED_INPUTS, check_against, export_onnx, golden_cases, run, session

REPO = Path(__file__).resolve().parents[2]
EXPORT = REPO / "handover" / "nle-methods" / "port" / "export"
SC = "ddm_sc_s1106_e245"


@pytest.fixture(scope="module")
def sc_models(tmp_path_factory, export_dir):
    out = tmp_path_factory.mktemp("onnx")
    return {sig: export_onnx(export_dir / f"{SC}_weights.json", out=out / f"{sig}.onnx", signature=sig,
                             say=lambda *a: None) for sig in ("trialwise", "shared")}


@pytest.mark.parametrize("sig", ["trialwise", "shared"])
def test_scaled_graph_reproduces_golden_rows(sc_models, export_dir, sig):
    card = json.load(open(export_dir / f"{SC}_weights.json"))
    theta, rt, choice, ref = golden_cases(card, export_dir / f"{SC}_golden.csv")
    ok, worst = check_against(session(sc_models[sig]), theta, rt, choice, ref, card["ll_floor_log"], sig,
                              say=lambda *a: None, tol=1e-10)
    assert ok, worst
    assert worst < 1e-10


def test_shared_and_trialwise_agree(sc_models, export_dir):
    card = json.load(open(export_dir / f"{SC}_weights.json"))
    rng = np.random.default_rng(3)
    theta = np.array([1.2, 0.9, 0.3, 1.1, 0.45, 0.2, 0.3, 0.08])
    rt = theta[2] + rng.uniform(0.0, 2.0, 500); choice = rng.choice([-1.0, 1.0], 500)
    a = run(session(sc_models["trialwise"]), theta, rt, choice)
    b = run(session(sc_models["shared"]), theta, rt, choice)
    fin = np.isfinite(a)
    assert np.array_equal(np.isfinite(a), np.isfinite(b)) and np.abs(a[fin] - b[fin]).max() < 1e-12


def test_scaled_graph_branches(sc_models):
    """-Inf below the onset, the floor outside the box, finite otherwise."""
    s = session(sc_models["trialwise"])
    theta = np.array([[1.0, 0.8, 0.3, 1.0, 0.5, 0.2, 0.1, 0.05]] * 4)
    theta[3, 0] = 14.0                                   # kappa = 14 * 0.8 = 11.2 > 10: outside the box
    rt = np.array([0.2, 0.3, 0.9, 0.9]); choice = np.array([1.0, -1.0, 1.0, 1.0])
    got = run(s, theta, rt, choice)
    assert np.isneginf(got[0]) and np.isneginf(got[1]) and np.isfinite(got[2])
    assert got[3] == pytest.approx(np.log(1e-10))


def test_eight_parameter_graph_reproduces_z0_golden(export_dir, tmp_path):
    golden = REPO.parent / "EMC2" / "tests" / "testthat" / "golden" / "ddm_z0_s503_e282_golden.csv"
    if not golden.exists():
        pytest.skip("the z0 golden file lives in EMC2's test tree")
    card_path = export_dir / "ddm_z0_s503_e282_weights.json"
    card = json.load(open(card_path))
    for sig in ("trialwise", "shared"):
        p = export_onnx(card_path, out=tmp_path / f"z0_{sig}.onnx", signature=sig, say=lambda *a: None)
        theta, rt, choice, ref = golden_cases(card, golden)
        ok, worst = check_against(session(p), theta, rt, choice, ref, card["ll_floor_log"], sig,
                                  say=lambda *a: None, tol=1e-10)
        assert ok, (sig, worst)


def test_scaled_graph_against_emc2(sc_models, export_dir, emc2_lib, tmp_path):
    cases = tmp_path / "cases.csv"
    r = subprocess.run(["Rscript", str(Path(__file__).resolve().parents[1] / "scripts" / "onnx_cases_scaled.R"),
                        str(export_dir / f"{SC}_weights.json"), str(cases), "25", "60"],
                       capture_output=True, text=True, env={**os.environ, "NLEFLOW_RLIB": emc2_lib,
                                                            "R_PROFILE_USER": "/dev/null"})
    assert r.returncode == 0, r.stdout + r.stderr
    card = json.load(open(export_dir / f"{SC}_weights.json"))
    d = np.genfromtxt(cases, delimiter=",", names=True, dtype=None, encoding=None)
    theta = np.column_stack([d[c] for c in SCALED_INPUTS]).astype(np.float64)
    rt = d["rt"].astype(np.float64); choice = d["choice"].astype(np.float64); ref = d["logp"].astype(np.float64)
    st0, t0 = theta[:, SCALED_INPUTS.index("st0")], theta[:, SCALED_INPUTS.index("t0")]
    wide = (st0 > 0) & (st0 >= 1e-7 * (rt - t0))
    for sig in ("trialwise", "shared"):
        s = session(sc_models[sig])
        ok, worst = check_against(s, theta[~wide], rt[~wide], choice[~wide], ref[~wide], card["ll_floor_log"],
                                  f"{sig} density", say=lambda *a: None, tol=1e-10)
        assert ok, (sig, "density", worst)
        ok, worst = check_against(s, theta[wide], rt[wide], choice[wide], ref[wide], card["ll_floor_log"],
                                  f"{sig} window", say=lambda *a: None, tol=1e-8)
        assert ok, (sig, "window", worst)

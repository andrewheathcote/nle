"""The ONNX export (M2): the graph of the committed scaled DDM card reproduces
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


def numpy_reference(card, theta_net, rt, choice):
    """log joint density of the scaled card at a = s = 1, t0 = st0 = 0 from golden.py's
    numpy functions, with z and the width of the spline bin holding it."""
    import scipy.stats as ss
    from nleflow import golden as G
    fl, fn = G.read_layers(card["flow_mlp"]); cl, _ = G.read_layers(card["classifier_mlp"])
    K, rmin, rmax, mbs, mks = G.spline_settings(card["spline"])
    cs = G.standardise(card, theta_net[None, :])
    logit = G.mlp_forward(cs, cl, None).squeeze(-1)[0]
    lp = np.empty(len(rt)); z = np.empty(len(rt)); w = np.empty(len(rt))
    for R, cv in ((1, -1.0), (2, 1.0)):
        m = choice == cv
        if not m.any():
            continue
        raw = G.mlp_forward(np.concatenate([cs, [[float(R)]]], axis=1), fl, fn)
        xp, yp, sl = G.build_knots(raw, K, rmin, rmax, mbs, mks)
        u = np.log(rt[m]); zz, ld = G.rqs_inverse(u, xp[0], yp[0], sl[0])
        lpR = -G.softplus(-logit) if R == 2 else -G.softplus(logit)
        lp[m] = ss.norm.logpdf(zz) + ld - u + lpR; z[m] = zz
        w[m] = np.diff(xp[0])[np.clip(np.searchsorted(xp[0], zz) - 1, 0, len(xp[0]) - 2)]
    return lp, z, w


def natural_theta(card, theta_net):
    """The graph's eight natural inputs (a = s = 1, t0 = st0 = 0) for a network-scale row."""
    import scipy.stats as ss
    from scipy.special import ndtr
    from nleflow import golden as G
    ctx = list(card["context_names"]); st = theta_net.copy()
    for name, c in G.zerobox_inputs(card):
        j = ctx.index(name); nat = np.sinh(theta_net[j]) * c
        with np.errstate(divide="ignore"):
            st[j] = ss.norm.ppf(nat) if name == "SZ" else np.log(nat)
    d = dict(zip(ctx, st))
    return np.array([d["kappa"], 1.0, 0.0, 1.0, ndtr(d["Z"]), ndtr(d["SZ"]), np.exp(d["eta"]), 0.0])


def test_graphs_agree_with_the_numpy_reference_within_conditioning(sc_models, export_dir):
    """Both signatures against golden.py's numpy density over random parameter sets and times
    (to 4 s, so rows reach the top spline bin). An error is bounded by 1e-12 plus k rounding
    errors of the knots over the width of the bin holding z: k = 512 for either graph against
    numpy (both can miss numpy together on rare rows by up to ~200), k = 32 between the two."""
    card = json.load(open(export_dir / f"{SC}_weights.json"))
    lo = np.asarray(card["bounds_sampled"]["lower"]); hi = np.asarray(card["bounds_sampled"]["upper"])
    rng = np.random.default_rng(3)
    eps = np.finfo(float).eps
    n_checked = 0
    for theta_net in rng.uniform(lo, hi, size=(12, len(lo))):
        theta = natural_theta(card, theta_net)
        rt = np.concatenate([rng.uniform(0.01, 4.0, 450), np.geomspace(0.5, 4.0, 50)])
        choice = rng.choice([-1.0, 1.0], 500)
        ref, z, w = numpy_reference(card, theta_net, rt, choice)
        a = run(session(sc_models["trialwise"]), theta, rt, choice)
        b = run(session(sc_models["shared"]), theta, rt, choice)
        fin = np.isfinite(ref) & (ref > card["ll_floor_log"] + 1e-9)
        cond = eps * np.maximum(1.0, np.abs(z)) / w
        for name, got in (("trialwise", a), ("shared", b)):
            bad = fin & (np.abs(got - ref) >= 1e-12 + 512 * cond)
            assert not bad.any(), (name, np.abs(got - ref)[bad].max())
        assert not (fin & (np.abs(a - b) >= 1e-12 + 32 * cond)).any(), np.abs(a - b)[fin].max()
        n_checked += int(fin.sum())
    assert n_checked > 4000

    # the st0 > 0 branch with a != 1, s != 1, t0 != 0 has no numpy reference here; the two
    # graphs must still agree. The density is a CDF difference, so the graphs differ by an
    # absolute error of up to ~100 eps in p (measured, 3 seeds): bounded by 256 eps / p in
    # log p, plus the knot term for the narrowest bin and the largest |z| at K = 32
    worst_cond = eps * 2.0 / card["spline"]["min_bin_size"]
    n_st0 = 0
    for theta_net in rng.uniform(lo, hi, size=(12, len(lo))):
        theta = natural_theta(card, theta_net)
        theta[1], theta[3], theta[2], theta[7] = rng.uniform(0.6, 1.5), rng.uniform(0.7, 1.4), rng.uniform(0.05, 0.4), 0.08
        rt = theta[2] + np.concatenate([rng.uniform(0.01, 4.0, 450), np.geomspace(0.5, 4.0, 50)])
        choice = rng.choice([-1.0, 1.0], 500)
        a = run(session(sc_models["trialwise"]), theta, rt, choice)
        b = run(session(sc_models["shared"]), theta, rt, choice)
        fin = np.isfinite(a) & np.isfinite(b) & (a > card["ll_floor_log"] + 1e-9)
        assert np.array_equal(np.isfinite(a), np.isfinite(b))
        bound = 1e-12 + 32 * worst_cond + 256 * eps / np.exp(np.where(fin, a, 0.0))
        assert not (fin & (np.abs(a - b) >= bound)).any(), np.abs(a - b)[fin].max()
        n_st0 += int(fin.sum())
    assert n_st0 > 1000


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
    golden = EXPORT / "ddm_z0_s503_e282_golden.csv"
    if not golden.exists():
        pytest.skip("the z0 golden file is not in the reference exports")
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


def test_onnx_check_takes_cases_of_a_joint_card(export_dir, tmp_path):
    """`onnx-check --cases` on a flow_joint card: the cases carry the card's own
    inputs (the context on the stored scale), not the scaled card's eight natural
    parameters; here the z0 card's golden rows written as cases."""
    from nleflow.onnx_export import check_onnx
    golden = EXPORT / "ddm_z0_s503_e282_golden.csv"
    if not golden.exists():
        pytest.skip("the z0 golden file is not in the reference exports")
    card_path = export_dir / "ddm_z0_s503_e282_weights.json"
    card = json.load(open(card_path))
    theta, rt, choice, ref = golden_cases(card, golden)
    cases = tmp_path / "cases.csv"
    names = list(card["context_names"])
    with open(cases, "w") as f:
        f.write(",".join(["regime", "draw", *names, "rt", "choice", "logp"]) + "\n")
        for i in range(len(rt)):
            f.write(",".join(["full", "1", *(repr(float(x)) for x in theta[i]), repr(float(rt[i])),
                              repr(float(choice[i])), repr(float(ref[i]))]).replace("-inf", "-Inf") + "\n")
    p = export_onnx(card_path, out=tmp_path / "z0_shared.onnx", signature="shared", say=lambda *a: None)
    lines = []
    assert check_onnx(p, card_path, golden=tmp_path / "none.csv", cases=cases, say=lines.append), lines
    assert any(l.startswith("EMC2 cases") for l in lines) and sum("PASS" in l for l in lines) == 1, lines

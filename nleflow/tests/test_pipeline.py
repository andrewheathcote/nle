"""End to end on synthetic data: simulate (in Python) -> train -> export ->
the golden file is reproduced from the card -> (with NLEFLOW_EMC2_LIB) the
card registers in EMC2 with twin RDM and passes nn_golden_check().

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

from nleflow.data import Scaler, write_parquet
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


# --- choice models: the scaled DDM path (M2) -----------------------------------------

def synthetic_choice(n, seed):
    """Rows in the scaled DDM's columns: a shifted log-normal u with a
    kappa-dependent choice, not a DDM; the pipeline and the card contract are
    under test, not the fit."""
    rng = np.random.default_rng(seed)
    kappa = rng.uniform(-3.0, 3.0, n)
    eta = rng.uniform(0.0, 2.0, n); eta[rng.random(n) < 0.3] = 0.0
    Z = rng.uniform(0.2, 0.8, n)
    SZ = rng.uniform(0.0, 0.8, n); SZ[rng.random(n) < 0.3] = 0.0
    p2 = 1.0 / (1.0 + np.exp(-kappa))
    R = np.where(rng.random(n) < p2, 2, 1)
    u = np.exp(-1.0 + 0.3 * rng.standard_normal(n) - 0.1 * np.abs(kappa))
    with np.errstate(divide="ignore"):
        from scipy.special import ndtri
        rows = {"param_id": np.arange(1, n + 1), "R": R, "rt": u, "kappa": kappa, "eta": np.log(eta),
                "Z": ndtri(Z), "SZ": ndtri(SZ)}
    return rows, p2


def make_choice_spec(tmp_path):
    spec = {
        "name": "synthetic_sc", "model": "DDM", "response": "choice", "context": ["kappa", "eta", "Z", "SZ"],
        "transforms": {"kappa": "identity", "eta": "zerobox", "Z": "probit", "SZ": "zerobox"},
        "zerobox_c": {"eta": 0.01, "SZ": 0.05},
        "box_natural": {"kappa": [-3, 3], "eta": [0, 2], "Z": [0.2, 0.8], "SZ": [0, 0.8]},
        "regimes": [{"share": 0.5}, {"share": 0.25, "zero": ["eta"]}, {"share": 0.25, "zero": ["SZ"]}],
        "scaling": {"kind": "ddm_unit"},
        "spline": {"bins": 8, "range": [-6, 2]},
        "conditioner": {"widths": [16, 8]},
        "classifier": {"widths": [16, 8], "epochs": 5, "batch": 128},
        "training": {"batch": 256, "epochs": 2, "keep": 2, "peak_lr": 1e-3, "seed": 3, "shuffle_seed": 1},
        "data": {"train": "train", "val": "val", "choice_train": "choice/train.parquet",
                 "choice_val": "choice/val.parquet"},
    }
    rows, _ = synthetic_choice(4096, 1); write_parquet(tmp_path / "train" / "t.parquet", rows)
    rows, _ = synthetic_choice(512, 2); write_parquet(tmp_path / "val" / "v.parquet", rows)
    for name, n, seed in (("train", 1024, 5), ("val", 128, 6)):
        rows, p2 = synthetic_choice(n, seed)
        write_parquet(tmp_path / "choice" / f"{name}.parquet",
                      {k: rows[k] for k in ("kappa", "eta", "Z", "SZ")} | {"p_R2": p2})
    return validate_spec(spec, base=tmp_path, source=str(tmp_path / "spec.yaml"))


def test_choice_train_export_golden(tmp_path):
    spec = make_choice_spec(tmp_path)
    run = train(spec, runs=tmp_path / "runs", quiet=True)
    assert (run / "classifier.json").exists() and (run / "classifier_log.json").exists()
    res = export(run, say=lambda *a: None)
    card = json.loads(res["weights"].read_text())
    assert card["kind"] == "flow_scaled" and card["model"] == "DDMsc" and card["card_version"] == 1
    assert card["scaling"] == {"kind": "ddm_unit", "sz": "proportion",
                               "pars": ["v", "a", "sv", "t0", "st0", "s", "Z", "SZ"]}
    assert card["context_encoding"] == "zerobox" and card["zerobox_c"] == {"eta": 0.01, "SZ": 0.05}
    assert card["context_transforms"] == {"kappa": "identity", "eta": "zerobox", "Z": "probit", "SZ": "zerobox"}
    assert card["classifier_scaler"] == card["scaler"] and "control_limits" not in card
    assert len(card["flow_mlp"]["layers"][0]["W"]) == 5 and len(card["classifier_mlp"]["layers"][0]["W"]) == 4
    assert res["gate"]["max_abs_dlog_pdf_f64"] <= 1e-8
    g = read_csv(res["golden"])
    assert len(g) == 1150 and set(g.dtype.names) >= {"R", "rt", "log_pdf_joint", "cdf_defective", "log_p_R"}
    assert np.isneginf(g["eta"]).sum() > 0 and np.isneginf(g["SZ"]).sum() > 0   # exact zeros recorded as -Inf
    again = tmp_path / "again.csv"
    golden_main(str(res["weights"]), str(again))
    a = read_csv(again)
    for c in a.dtype.names:
        fa = np.isfinite(a[c]); assert np.array_equal(fa, np.isfinite(g[c]))
        assert np.abs(a[c][fa] - g[c][fa]).max() < 1e-12


def test_choice_classifier_reuse_needs_the_same_scaler(tmp_path):
    spec = make_choice_spec(tmp_path)
    run = train(spec, runs=tmp_path / "runs", quiet=True)
    run2 = train(spec, runs=tmp_path / "runs2", quiet=True, classifier_from=run)
    assert json.loads((run2 / "classifier_log.json").read_text())["copied_from"] == str(run)
    other = Scaler.load(run2 / "scaler.json"); other.mean[0] += 1.0; other.save(run2 / "scaler.json")
    with pytest.raises(ValueError, match="cannot be reused"):
        train(spec, runs=tmp_path / "runs3", quiet=True, classifier_from=run2)


def test_choice_card_registers_in_emc2(tmp_path, emc2_lib):
    """The scaled card registers with twin = DDM as flow_scaled and passes
    nn_golden_check() and nle_verify's compiled-vs-R check."""
    spec = make_choice_spec(tmp_path)
    run = train(spec, runs=tmp_path / "runs", quiet=True)
    res = export(run, say=lambda *a: None)
    r_code = f"""
    .libPaths(c('{emc2_lib}', .libPaths())); suppressPackageStartupMessages(library(EMC2))
    m <- register_nn_model('{res["weights"]}', twin = DDM)
    reg <- m()$nn
    stopifnot(reg$kind == 'flow_scaled', reg$card_version == 1, isTRUE(reg$scaled), reg$scaling$kind == 'ddm_unit')
    g <- nn_golden_check(m, error = FALSE); cat(format(g), '\\n'); stopifnot(g$pass)
    """
    out = subprocess.run(["Rscript", "-e", r_code], capture_output=True, text=True,
                         env={**os.environ, "R_PROFILE_USER": "/dev/null"})
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PASS" in out.stdout
    verify = subprocess.run(["Rscript", "-e", f".libPaths(c('{emc2_lib}', .libPaths())); "
                             "source(system.file('scripts', 'nle_verify.R', package = 'EMC2'))", str(res["weights"])],
                            capture_output=True, text=True, env={**os.environ, "R_PROFILE_USER": "/dev/null"})
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert "FAIL" not in verify.stdout and "compiled vs R path" in verify.stdout


# --- an eight-parameter DDM card (flow_joint, no scaling) with parameters fixed in training ---
def synthetic_joint(n, seed):
    """Rows in the DDM's names (v, a, t0, Z, sv, st0; s and SZ fixed): a
    shifted log-normal time and a v-dependent choice, not a DDM."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(-2.0, 2.0, n); a = rng.uniform(0.5, 2.0, n); t0 = rng.uniform(0.1, 0.4, n)
    Z = rng.uniform(0.3, 0.7, n)
    sv = rng.uniform(0.0, 2.0, n); sv[rng.random(n) < 0.3] = 0.0
    st0 = rng.uniform(0.0, 0.3, n); st0[rng.random(n) < 0.3] = 0.0
    p2 = 1.0 / (1.0 + np.exp(-v * a))
    R = np.where(rng.random(n) < p2, 2, 1)
    rt = t0 + rng.uniform(0, 1, n) * st0 + np.exp(np.log(a) - 1.0 + 0.3 * rng.standard_normal(n))
    from scipy.special import ndtri
    with np.errstate(divide="ignore"):
        ctx = {"v": v, "a": np.log(a), "t0": np.log(t0), "Z": ndtri(Z), "sv": np.log(sv), "st0": np.log(st0)}
    return {"param_id": np.arange(1, n + 1), "R": R, "rt": rt, **ctx}, ctx, p2


def make_joint_spec(tmp_path, fixed=True):
    spec = {
        "name": "synthetic_joint", "model": "DDM", "response": "choice",
        "context": ["v", "a", "t0", "Z", "sv", "st0"],
        "transforms": {"v": "identity", "a": "log", "t0": "log", "Z": "probit", "sv": "zerobox", "st0": "zerobox",
                       **({"s": "log", "SZ": "probit"} if fixed else {})},
        "zerobox_c": {"sv": 0.1, "st0": 0.01},
        "box_natural": {"v": [-2, 2], "a": [0.5, 2], "t0": [0.1, 0.4], "Z": [0.3, 0.7], "sv": [0, 2], "st0": [0, 0.3]},
        **({"fixed": {"s": 1, "SZ": 0}} if fixed else {}),
        "spline": {"bins": 8, "range": [-4, 3]},
        "conditioner": {"widths": [16, 8]},
        "classifier": {"widths": [16, 8], "epochs": 5, "batch": 128},
        "training": {"batch": 256, "epochs": 2, "keep": 2, "peak_lr": 1e-3, "seed": 3, "shuffle_seed": 1},
        "data": {"train": "train", "val": "val", "choice_train": "choice/train.parquet",
                 "choice_val": "choice/val.parquet"},
    }
    rows, _, _ = synthetic_joint(4096, 1); write_parquet(tmp_path / "train" / "t.parquet", rows)
    rows, _, _ = synthetic_joint(512, 2); write_parquet(tmp_path / "val" / "v.parquet", rows)
    for name, n, seed in (("train", 1024, 5), ("val", 128, 6)):
        _, ctx, p2 = synthetic_joint(n, seed)
        write_parquet(tmp_path / "choice" / f"{name}.parquet", ctx | {"p_R2": p2})
    return validate_spec(spec, base=tmp_path, source=str(tmp_path / "spec.yaml"))


def test_joint_card_records_the_fixed_parameters(tmp_path):
    """`fixed` (twin parameters held at a value in training: s = 1, SZ = 0)
    reaches the card as the vignette states it: names in context_transforms
    that are not in context_names, with their box in bounds_natural, and the
    values in `fixed` (what register_nn_model's exceptions need)."""
    spec = make_joint_spec(tmp_path)
    assert spec["fixed"] == {"s": 1.0, "SZ": 0.0}
    run = train(spec, runs=tmp_path / "runs", quiet=True)
    res = export(run, say=lambda *a: None)
    card = json.loads(res["weights"].read_text())
    assert card["kind"] == "flow_joint" and card["model"] == "DDM" and "scaling" not in card
    assert card["context_names"] == ["v", "a", "t0", "Z", "sv", "st0"]
    assert list(card["context_transforms"]) == ["v", "a", "t0", "Z", "sv", "st0", "s", "SZ"]
    assert card["context_transforms"]["s"] == "log" and card["context_transforms"]["SZ"] == "probit"
    assert card["fixed"] == {"s": 1.0, "SZ": 0.0}
    assert card["bounds_natural"]["lower"][6:] == [1.0, 0.0] and card["bounds_natural"]["upper"][6:] == [1.0, 0.0]
    assert len(card["bounds_sampled"]["lower"]) == 6
    assert res["gate"]["max_abs_dlog_pdf_f64"] <= 1e-8


def test_fixed_is_validated(tmp_path):
    from nleflow.spec import SpecError
    spec = make_joint_spec(tmp_path, fixed=False)
    raw = {k: v for k, v in spec.items() if k not in ("spec_file", "beyond_support")}
    with pytest.raises(SpecError, match="input of the network"):
        validate_spec({**raw, "fixed": {"v": 1}}, base=tmp_path)
    with pytest.raises(SpecError, match="finite number"):
        validate_spec({**raw, "fixed": {"s": "one"}}, base=tmp_path)
    with pytest.raises(SpecError, match="zerobox"):
        validate_spec({**raw, "fixed": {"SZ": 0}, "transforms": {**raw["transforms"], "SZ": "zerobox"}}, base=tmp_path)


def test_joint_card_with_fixed_parameters_registers_in_emc2(tmp_path, emc2_lib):
    """The card registers with twin = DDM and the card's `fixed` as exceptions,
    and passes nn_golden_check(). Without them, an EMC2 that reads the card's
    `fixed` (the current EMC2) registers it with those values; an older
    one names what is missing."""
    spec = make_joint_spec(tmp_path)
    run = train(spec, runs=tmp_path / "runs", quiet=True)
    res = export(run, say=lambda *a: None)
    r_code = f"""
    .libPaths(c('{emc2_lib}', .libPaths())); suppressPackageStartupMessages(library(EMC2))
    card <- jsonlite::fromJSON('{res["weights"]}')
    e <- tryCatch(register_nn_model('{res["weights"]}', twin = DDM), error = function(e) conditionMessage(e))
    if (is.character(e)) stopifnot(grepl('trained without', e)) else
      stopifnot(isTRUE(all.equal(e()$nn$fixed[names(card$fixed)], unlist(card$fixed))))
    m <- register_nn_model('{res["weights"]}', twin = DDM, exceptions = unlist(card$fixed))
    reg <- m()$nn
    stopifnot(reg$kind == 'flow_joint', reg$card_version == 1)
    g <- nn_golden_check(m, error = FALSE); cat(format(g), '\\n'); stopifnot(g$pass)
    """
    out = subprocess.run(["Rscript", "-e", r_code], capture_output=True, text=True,
                         env={**os.environ, "R_PROFILE_USER": "/dev/null"})
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PASS" in out.stdout


def test_race_card_records_the_fixed_parameters(tmp_path):
    spec = make_spec(tmp_path)
    raw = {k: v for k, v in spec.items() if k not in ("spec_file", "beyond_support")}
    spec = validate_spec({**raw, "fixed": {"w": 0.5}, "transforms": {**raw["transforms"], "w": "probit"}},
                         base=tmp_path, source=str(tmp_path / "spec.yaml"))
    run = train(spec, runs=tmp_path / "runs", quiet=True)
    card = json.loads(export(run, say=lambda *a: None)["weights"].read_text())
    assert card["kind"] == "flow_race" and card["fixed"] == {"w": 0.5}
    assert list(card["context_transforms"]) == ["v", "B", "t0", "s", "A", "w"] and card["context_transforms"]["w"] == "probit"
    assert card["bounds_natural"]["lower"][-1] == card["bounds_natural"]["upper"][-1] == 0.5

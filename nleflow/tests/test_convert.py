"""The ONNX -> card converter (M3): the published LAN converts to EMC2's
committed card, --verify reproduces the committed onnxruntime golden file,
rank-1, batched, Gemm and Gelu(tanh) graphs convert, every graph EMC2 cannot
evaluate is refused naming its first unsupported node, and (with
NLEFLOW_EMC2_LIB) the card registers in EMC2 as mlp_joint and passes
nn_golden_check()."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from conftest import EXPORT
from nleflow.convert import ConvertError, convert, list_presets, load_preset, read_graph

LAN_ONNX = EXPORT / "ddm_uniform_st.onnx"                 # the 3 x 100 published file
LAN_JSON = EXPORT / "lan_ddm_uniform_st.json"             # the NLE project's original export of it
LAN_ORT = EXPORT / "lan_ddm_uniform_st_ort.csv"           # its card and onnxruntime values, as EMC2's tests hold them
LAN_RDS = EXPORT / "lan_ddm_uniform_st.rds"
quiet = dict(say=lambda *a: None)


@pytest.fixture(scope="module")
def lan_onnx():
    if not LAN_ONNX.exists():
        pytest.skip(f"the published LAN is not at {LAN_ONNX}")
    return LAN_ONNX


def small_mlp(path, act="Tanh", dims=(4, 8, 6, 1), batched=False, gemm=False, act_attrs=None, extra=None,
              opset=18, seed=0):
    """A random MLP graph: input [k] or [N, k], hidden activations `act`."""
    rng = np.random.default_rng(seed)
    nodes, inits = [], []
    cur = "input"
    for i in range(len(dims) - 1):
        W = rng.normal(0, 1 / np.sqrt(dims[i]), (dims[i], dims[i + 1])).astype(np.float32)
        b = rng.normal(0, 0.1, dims[i + 1]).astype(np.float32)
        if gemm:
            inits += [numpy_helper.from_array(W.T.copy(), f"W{i}"), numpy_helper.from_array(b, f"b{i}")]
            nodes.append(helper.make_node("Gemm", [cur, f"W{i}", f"b{i}"], [f"lin{i}"], name=f"gemm{i}", transB=1))
        else:
            inits += [numpy_helper.from_array(W, f"W{i}"), numpy_helper.from_array(b, f"b{i}")]
            nodes.append(helper.make_node("MatMul", [cur, f"W{i}"], [f"mm{i}"], name=f"matmul{i}"))
            nodes.append(helper.make_node("Add", [f"mm{i}", f"b{i}"], [f"lin{i}"], name=f"add{i}"))
        cur = f"lin{i}"
        if i < len(dims) - 2:
            if extra == "mul_between" and i == 0:
                inits.append(numpy_helper.from_array(np.float32(2.0).reshape(()), "two"))
                nodes.append(helper.make_node("Mul", [cur, "two"], ["scaled0"], name="scale_hidden"))
                cur = "scaled0"
            nodes.append(helper.make_node(act, [cur], [f"act{i}"], name=f"{act.lower()}{i}", **(act_attrs or {})))
            cur = f"act{i}"
    shape_in = ["N", dims[0]] if batched else [dims[0]]
    shape_out = ["N", 1] if batched else [1]
    g = helper.make_graph(nodes, "mlp", [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape_in)],
                          [helper.make_tensor_value_info(cur, TensorProto.FLOAT, shape_out)], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 10
    onnx.checker.check_model(m)
    onnx.save(m, str(path))
    return path


SMALL = dict(params=["p1", "p2"], box={"p1": (-1.0, 1.0), "p2": (0.2, 2.0)}, ll_floor=-16.0,
             response_values=[-1.0, 1.0])


# --- the published LAN ----------------------------------------------------------------------------

def test_presets_listed_and_loaded():
    assert "ddm_uniform_st" in list_presets()
    p = load_preset("ddm_uniform_st")
    assert p["params"] == ["v", "a", "z", "t", "st"] and p["response_values"] == [-1.0, 1.0]
    assert p["mapping"]["a"] == "2 * a" and p["mapping"]["t0"] == "t - st" and p["twin"] == "DDM"
    with pytest.raises(ConvertError, match="no preset"):
        load_preset("nope")


def test_lan_converts_to_the_committed_card(lan_onnx, tmp_path):
    res = convert(lan_onnx, out=tmp_path / "lan_weights.json", preset="ddm_uniform_st", **quiet)
    card = json.loads(res["card"].read_text())
    ref = json.loads(LAN_JSON.read_text()) if LAN_JSON.exists() else None
    assert card["card_version"] == 1 and card["kind"] == "mlp_joint"
    assert card["context_names"] == ["v", "a", "z", "t", "st"]
    assert card["input_layout"] == ["v", "a", "z", "t", "st", "rt", "R"]
    assert card["response_values"] == [-1.0, 1.0] and card["ll_floor_log"] == pytest.approx(np.log(1e-7), abs=1e-14)
    assert card["source_sha256"] == "a9acbc4a238ff5df8ef53fda5bb0d89d20e4e23c3fc862ae25888739c99dc63f"
    assert card["bounds_natural"] == {"lower": [-3.0, 0.3, 0.3, 0.25, 0.001], "upper": [3.0, 2.5, 0.7, 2.25, 0.25]}
    assert card["mlp"]["activation"] == "tanh" and len(card["mlp"]["layers"]) == 4
    assert card["validation"]["calibration"].startswith("unknown")
    assert "preset" not in card and card["converted_with"].endswith("--preset ddm_uniform_st")
    if ref is not None:                       # the NLE project's original export of the same network
        assert [np.allclose(np.array(a["W"]), np.array(b["W"]), rtol=0, atol=0) and
                np.allclose(np.array(a["b"]), np.array(b["b"]), rtol=0, atol=0)
                for a, b in zip(card["mlp"]["layers"], ref["layers"])] == [True] * 4
        assert card["ll_floor_log"] == ref["label_floor"]
        assert {k: list(v) for k, v in zip(card["context_names"], zip(card["bounds_natural"]["lower"],
                card["bounds_natural"]["upper"]))} == {k: [float(x) for x in v] for k, v in ref["box"].items()}


def test_lan_card_equals_emc2s_committed_card(lan_onnx, tmp_path):
    if shutil.which("Rscript") is None:
        pytest.skip("R (Rscript, with jsonlite) is needed to read EMC2's committed .rds card")
    rds = LAN_RDS
    res = convert(lan_onnx, out=tmp_path / "lan_weights.json", preset="ddm_uniform_st", **quiet)
    fields = ["kind", "context_names", "context_transforms", "bounds_natural", "input_layout", "response_values",
              "ll_floor_log", "mlp", "source_sha256", "source", "source_model", "encoding", "map_to_emc2_ddm"]
    code = ("suppressPackageStartupMessages(library(jsonlite)); "
            f"new <- fromJSON('{res['card']}', simplifyDataFrame = FALSE); old <- readRDS('{rds}'); "
            f"f <- c({', '.join(repr(x) for x in fields)}); "
            "ok <- vapply(f, function(x) identical(new[[x]], old[[x]]), TRUE); print(ok); stopifnot(all(ok))")
    out = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True,
                         env={**os.environ, "R_PROFILE_USER": "/dev/null"})
    if "there is no package called" in out.stderr:
        pytest.skip("R with jsonlite is needed to read EMC2's committed .rds card")
    assert out.returncode == 0, out.stdout + out.stderr


def test_lan_verify_reproduces_the_committed_ort_csv(lan_onnx, tmp_path):
    res = convert(lan_onnx, out=tmp_path / "lan_weights.json", preset="ddm_uniform_st", rows_csv=LAN_ORT, **quiet)
    v = res["verify"]
    assert v["f64_single"] < 1e-10 and v["f64_batched"] < 1e-10           # the card holds the network
    assert v["src_single"] < 1e-4 and v["src_batched"] < 1e-4             # float32 round-off (observed 5e-6)
    assert v["single_vs_batched"] == 0.0
    new = np.genfromtxt(res["golden"], delimiter=",", names=True)
    old = np.genfromtxt(LAN_ORT, delimiter=",", names=True)
    assert new.dtype.names == old.dtype.names and len(new) == 300
    assert max(float(np.max(np.abs(new[c] - old[c]))) for c in new.dtype.names) < 1e-6
    card = json.loads(res["card"].read_text())
    assert card["validation"]["golden"]["tolerance"] == 1e-4 and card["validation"]["golden"]["file"] == res["golden"].name


def test_lan_batched_graph_converts_to_the_same_card(lan_onnx, tmp_path):
    from nleflow.convert import _variant
    m = _variant(onnx.load(str(lan_onnx)), batched=True)
    p = tmp_path / "lan_batched.onnx"; onnx.save(m, str(p))
    g = read_graph(p)
    assert g["rank"] == 2 and g["n_in"] == 7
    a = convert(lan_onnx, out=tmp_path / "a_weights.json", preset="ddm_uniform_st", **quiet)
    b = convert(p, out=tmp_path / "b_weights.json", preset="ddm_uniform_st", verify_n=200, **quiet)
    ca, cb = json.loads(a["card"].read_text()), json.loads(b["card"].read_text())
    assert ca["mlp"] == cb["mlp"]
    assert b["verify"]["f64_batched"] < 1e-10 and b["verify"]["f64_single"] < 1e-10


# --- other graph shapes -----------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [dict(), dict(batched=True), dict(gemm=True, batched=True)])   # Gemm takes rank 2
def test_small_tanh_graphs_convert_and_verify(tmp_path, kw):
    p = small_mlp(tmp_path / "m.onnx", **kw)
    res = convert(p, out=tmp_path / "m_weights.json", verify_n=200, **SMALL, **quiet)
    v = res["verify"]
    assert v["f64_single"] < 1e-10 and v["f64_batched"] < 1e-10 and v["src_single"] < 1e-4
    assert json.loads(res["card"].read_text())["mlp"]["activation"] == "tanh"


@pytest.mark.parametrize("batched", [False, True])
def test_gelu_tanh_graph_converts(tmp_path, batched):
    p = small_mlp(tmp_path / "g.onnx", act="Gelu", act_attrs={"approximate": "tanh"}, opset=20, batched=batched)
    res = convert(p, out=tmp_path / "g_weights.json", verify_n=200, **SMALL, **quiet)
    assert json.loads(res["card"].read_text())["mlp"]["activation"] == "gelu_tanh"
    assert res["verify"]["f64_single"] < 1e-10 and res["verify"]["src_single"] < 1e-4


@pytest.mark.parametrize("act,attrs,opset,what", [
    ("Sigmoid", None, 18, "unsupported operation 'Sigmoid'"),
    ("Relu", None, 18, "activation 'relu' is not one EMC2's evaluator runs"),
    ("Gelu", {"approximate": "none"}, 20, "activation 'gelu_erf' is not one EMC2's evaluator runs"),
])
def test_unsupported_activations_are_refused_by_name(tmp_path, act, attrs, opset, what):
    p = small_mlp(tmp_path / "bad.onnx", act=act, act_attrs=attrs, opset=opset)
    with pytest.raises(ConvertError) as e:
        convert(p, out=tmp_path / "bad_weights.json", **SMALL, **quiet)
    msg = str(e.value)
    assert what in msg
    assert f"node 2 ({act}, name '{act.lower()}0'" in msg               # the first such node, by position and name
    assert not (tmp_path / "bad_weights.json").exists()


def test_a_non_mlp_node_is_refused_by_name(tmp_path):
    p = small_mlp(tmp_path / "mul.onnx", extra="mul_between")
    with pytest.raises(ConvertError, match=r"node 2 \(Mul, name 'scale_hidden'.*unsupported operation 'Mul'"):
        convert(p, out=tmp_path / "x_weights.json", **SMALL, **quiet)


def test_what_the_graph_cannot_say_is_required(tmp_path):
    p = small_mlp(tmp_path / "m.onnx")
    with pytest.raises(ConvertError, match="--params is required"):
        convert(p, out=tmp_path / "x.json", **quiet)
    with pytest.raises(ConvertError, match="network has 4 inputs but the input layout has 5"):
        convert(p, out=tmp_path / "x.json", **{**SMALL, "params": ["p1", "p2", "p3"],
                                               "box": {"p1": (0, 1), "p2": (0, 1), "p3": (0, 1)}}, **quiet)


def test_preset_on_another_file_warns(tmp_path):
    p = small_mlp(tmp_path / "m.onnx", dims=(7, 8, 1))
    said = []
    convert(p, out=tmp_path / "w.json", preset="ddm_uniform_st", say=said.append)
    assert any("was checked on a file with sha256" in s for s in said)


# The published ddm_uniform_st was replaced on 2026-10-05 (huggingface.co/Eitanm/ddm-st-lans
# 40e0327c22a4: 5x100, st box to 0.35); the preset knows both files by sha256 and takes the
# box of the one it is given.
LAN_SHA_3X100 = "a9acbc4a238ff5df8ef53fda5bb0d89d20e4e23c3fc862ae25888739c99dc63f"
LAN_SHA_5X100 = "4330d72313b322ba9799bad1017180e62a82a7dfc58944826ac54efccab1d173"


def test_lan_preset_knows_both_published_files():
    from nleflow.convert import preset_version
    p = load_preset("ddm_uniform_st")
    new, old = preset_version(p, LAN_SHA_5X100), preset_version(p, LAN_SHA_3X100)
    assert new["revision"] == "40e0327c22a4" and new["box"]["st"] == [0.001, 0.35]
    assert old["revision"] == "8b1be7830fa5" and old["box"]["st"] == [0.001, 0.25]
    assert preset_version(p, "0" * 64) is None


def test_convert_takes_the_box_of_the_matching_file(tmp_path, monkeypatch):
    import hashlib
    import nleflow.convert as cv
    p = small_mlp(tmp_path / "m.onnx", dims=(7, 8, 1))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    base = load_preset("ddm_uniform_st")
    fake = {**base, "versions": [{"revision": "abc123", "sha256": sha, "box": {"st": [0.001, 0.35]}}]}
    monkeypatch.setattr(cv, "load_preset", lambda name: fake)
    said = []
    res = convert(p, out=tmp_path / "w.json", preset="ddm_uniform_st", say=said.append)
    card = json.loads(res["card"].read_text())
    assert card["bounds_natural"]["upper"][card["context_names"].index("st")] == 0.35
    assert card["source_revision"] == "abc123" and card["source_model"] == base["source_model"]
    assert not any("WARNING" in s for s in said)
    # a --box given by hand still wins over the preset's
    res = convert(p, out=tmp_path / "w2.json", preset="ddm_uniform_st", say=said.append,
                  box={**{k: tuple(v) for k, v in base["box"].items()}, "st": (0.01, 0.2)})
    card = json.loads(res["card"].read_text())
    assert card["bounds_natural"]["upper"][card["context_names"].index("st")] == 0.2


def test_card_carries_the_presets_p_types(tmp_path):
    # EMC2's nle_verify.R registers a card with no analytic twin with the card's own
    # `p_types`; the preset holds them (emc2_p_types)
    p = small_mlp(tmp_path / "m.onnx", dims=(7, 8, 1))
    card = json.loads(convert(p, out=tmp_path / "w.json", preset="ddm_uniform_st", **quiet)["card"].read_text())
    assert card["p_types"] == load_preset("ddm_uniform_st")["emc2_p_types"]
    q = small_mlp(tmp_path / "s.onnx")
    card = json.loads(convert(q, out=tmp_path / "x.json", **SMALL, **quiet)["card"].read_text())
    assert "p_types" not in card                                      # no preset, nothing to say


def test_lan_card_passes_nle_verify_from_the_card_alone(lan_onnx, tmp_path, emc2_lib):
    res = convert(lan_onnx, out=tmp_path / "lan_weights.json", preset="ddm_uniform_st", verify_n=300, **quiet)
    out = subprocess.run(["Rscript", "-e", f".libPaths(c('{emc2_lib}', .libPaths())); "
                          "source(system.file('scripts', 'nle_verify.R', package = 'EMC2'))", str(res["card"])],
                         capture_output=True, text=True, env={**os.environ, "R_PROFILE_USER": "/dev/null"})
    assert "all checks passed" in out.stdout, out.stdout + out.stderr


def test_float64_copy_of_a_tanh_graph_has_no_gelu_constants(tmp_path):
    from nleflow.convert import _variant
    m = onnx.load(str(small_mlp(tmp_path / "m.onnx")))
    assert not [t.name for t in _variant(m, dtype="double").graph.initializer if t.name.startswith("_gelu")]
    g = onnx.load(str(small_mlp(tmp_path / "g.onnx", act="Gelu", act_attrs={"approximate": "tanh"}, opset=20)))
    names = {t.name for t in _variant(g, dtype="double").graph.initializer}
    assert {"_gelu_half", "_gelu_one", "_gelu_cubic", "_gelu_s2pi"} <= names


# --- EMC2 -------------------------------------------------------------------------------------------

def test_lan_card_registers_in_emc2_and_passes_its_golden_check(lan_onnx, tmp_path, emc2_lib):
    res = convert(lan_onnx, out=tmp_path / "lan_weights.json", preset="ddm_uniform_st", verify_n=300, **quiet)
    code = f"""
    .libPaths(c('{emc2_lib}', .libPaths())); suppressPackageStartupMessages(library(EMC2))
    m <- register_nn_model('{res["card"]}', p_types = c(v = 0, a = 1, z = .5, t = .5, st = .05))
    reg <- m()$nn
    stopifnot(reg$kind == 'mlp_joint', reg$card_version == 1, m()$c_name == 'NN')
    g1 <- nn_golden_check(m, tol = 1e-4)
    g2 <- nn_golden_check(m, golden = '{LAN_ORT}', tol = 1e-4)
    cat(format(g1), '\\n', format(g2), '\\n')
    """
    out = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True,
                         env={**os.environ, "R_PROFILE_USER": "/dev/null"})
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.count("PASS") == 2


def test_gelu_card_registers_in_emc2(tmp_path, emc2_lib):
    """A converted gelu_tanh network runs in EMC2's evaluator (load_mlp takes it)."""
    p = small_mlp(tmp_path / "g.onnx", act="Gelu", act_attrs={"approximate": "tanh"}, opset=20, batched=True)
    res = convert(p, out=tmp_path / "g_weights.json", verify_n=300, **SMALL, **quiet)
    code = f"""
    .libPaths(c('{emc2_lib}', .libPaths())); suppressPackageStartupMessages(library(EMC2))
    m <- register_nn_model('{res["card"]}', p_types = c(p1 = 0, p2 = 1))
    stopifnot(m()$nn$kind == 'mlp_joint')
    cat(format(nn_golden_check(m, tol = 1e-4)), '\\n')
    """
    out = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True,
                         env={**os.environ, "R_PROFILE_USER": "/dev/null"})
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PASS" in out.stdout

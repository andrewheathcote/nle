"""The spec schema: the example specs load; malformed specs are refused by name."""
import copy
from pathlib import Path

import pytest
import yaml

from nleflow.spec import SpecError, bounds, flow_context, load_spec, validate_spec

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "rdm_a0"


@pytest.mark.parametrize("name", ["spec.yaml", "spec_smoke.yaml"])
def test_examples_load(name):
    s = load_spec(EXAMPLES / name)
    assert s["context"] == ["v", "B", "t0", "s", "A"]
    assert s["transforms"]["A"] == "zerobox" and s["zerobox_c"] == {"A": 0.05}
    assert s["stored"] == {"A": "log"}
    assert s["sampling"]["A"] == {"uniform": 0.70, "asinh_uniform": 0.15, "zero": 0.15}
    assert s["conditioner"] == {"widths": [256, 128], "norm": False, "affine": False}
    assert s["spline"] == {"bins": 64, "range": [-5.0, 5.0]}
    assert Path(s["data"]["train"]).is_absolute()
    assert s["control_limits"] == {"s": [0.1, 2.0], "vb_over_s2_max": 354.0}
    nat, net = bounds(s)
    assert nat["lower"] == [1e-3, 0.1, 0.05, 0.1, 0.0]
    assert net["lower"][-1] == 0.0 and abs(net["upper"][-1] - 4.382182848065498) < 1e-12


def base():
    return yaml.safe_load((EXAMPLES / "spec_smoke.yaml").read_text())


def refused(mutate, match):
    s = base()
    mutate(s)
    with pytest.raises(SpecError, match=match):
        validate_spec(s, base=EXAMPLES)


def test_refusals():
    refused(lambda s: s.pop("context"), "needs 'context'")
    refused(lambda s: s["transforms"].pop("A"), "no transform for context 'A'")
    refused(lambda s: s["transforms"].update(A="asinh"), "not one of")
    refused(lambda s: s["transforms"].update(A="log"), "zerobox_c names an input that is not zerobox")
    refused(lambda s: s.pop("zerobox_c"), "no constant for the zero-box input 'A'")
    refused(lambda s: s["box_natural"].update(v=[0, 5]), "positive lower bound")
    refused(lambda s: s["box_natural"].update(A=[2, 0]), "lower must be below upper")
    refused(lambda s: s["sampling"]["A"].update(zero=0.5), "shares must sum to 1")
    refused(lambda s: s.update(response="choice"), "milestone M2")
    refused(lambda s: s.update(scaling={"kind": "ddm_unit"}), "milestone M2")
    refused(lambda s: s.update(pre="Q"), "not a context name")
    refused(lambda s: s["training"].update(batch=0), "positive integer")
    refused(lambda s: s["training"].update(bogus=1), "unknown keys")
    refused(lambda s: s.update(conditioner={"widths": [256, 0]}), "positive integers")
    refused(lambda s: s.update(spline={"bins": 1}), "integer >= 2")
    refused(lambda s: s.update(extra_key=1), "unknown spec keys")
    refused(lambda s: s.update(context=["v", "B", "t0", "s", "A", "rt"]), "reserved")


def test_pre_drops_the_shift_from_the_flow_context():
    s = base(); s["pre"] = "t0"
    v = validate_spec(s, base=EXAMPLES)
    assert flow_context(v) == ["v", "B", "s", "A"]


def test_defaults_fill_in():
    s = base()
    for k in ("spline", "conditioner", "training", "control_limits", "mass_tolerance", "data"):
        s.pop(k, None)
    v = validate_spec(s, base=EXAMPLES)
    assert v["training"]["epochs"] == 300 and v["training"]["keep"] == 30
    assert v["conditioner"]["widths"] == [256, 128] and v["spline"]["bins"] == 64
    assert v["control_limits"] is None and v["mass_tolerance"] == 1e-3
    assert v["data"]["train"] == str((EXAMPLES / "data/train").resolve())

"""`nleflow golden` reproduces the committed RDM golden files from their cards
(M1 acceptance (c)): rdm_ss_base_s801_e271 (the A = 0 flow) and
rdm_rdm_small_e195 (the small RDM flow, whose golden file is the EMC2 test
tree's), both to 1e-12."""
import json

import numpy as np
import pytest

from nleflow.golden import main as golden_main

CARDS = ["rdm_ss_base_s801_e271", "rdm_rdm_small_e195", "ddm_sc_s1106_e245"]


def read(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def edge_tolerance(card, z, k=16):
    """Per-row bound for a scaled card: 1e-12 plus k rounding errors of the knot
    positions, amplified by 1/(width of the bin holding z). The top and bottom
    bins sit at the minimum width, where one ulp of a knot moves the log
    Jacobian by ~ulp/width, so the same file differs across BLAS builds there."""
    sp = card["spline"]
    z = np.where(np.isfinite(z), z, 0.0)
    w = np.maximum(sp["min_bin_size"], np.minimum(z - sp["range_min"], sp["range_max"] - z))
    return 1e-12 + k * np.finfo(float).eps * np.maximum(1.0, np.abs(z)) / w


@pytest.mark.parametrize("tag", CARDS)
def test_golden_reproduces_committed_file(export_dir, tmp_path, tag):
    out = tmp_path / f"{tag}_golden.csv"
    golden_main(str(export_dir / f"{tag}_weights.json"), str(out))
    new, ref = read(out), read(export_dir / f"{tag}_golden.csv")
    assert new.dtype.names == ref.dtype.names
    assert len(new) == len(ref) == (1150 if tag.startswith("ddm") else 575)
    card = json.load(open(export_dir / f"{tag}_weights.json"))
    scaled = card.get("scaling") is not None
    tol = edge_tolerance(card, ref["z"]) if scaled else np.full(len(ref), 1e-12)
    worst = 0.0
    for c in new.dtype.names:
        fa, fb = np.isfinite(new[c]), np.isfinite(ref[c])
        assert np.array_equal(fa, fb) and np.array_equal(new[c][~fa], ref[c][~fb])   # -Inf where -Inf
        with np.errstate(invalid="ignore"):
            diff = np.abs(new[c] - ref[c])
        assert np.all(diff[fa] < tol[fa]), f"{tag} {c}: max |diff| {diff[fa].max():.2e} exceeds the bound at row {int(np.argmax(np.where(fa, diff / tol, 0)))}"
        worst = max(worst, float(np.max(diff[fa])))


def test_zero_box_rows_are_exact_zeros(export_dir, tmp_path):
    out = tmp_path / "g.csv"
    golden_main(str(export_dir / "rdm_ss_base_s801_e271_weights.json"), str(out))
    g = read(out)
    assert np.sum(g["A"] == 0.0) == 5 * 23          # the first five parameter sets sit at A = 0


def test_default_output_name(export_dir, tmp_path):
    import shutil
    card = tmp_path / "x_weights.json"
    shutil.copy(export_dir / "rdm_rdm_small_e195_weights.json", card)
    golden_main(str(card))
    assert (tmp_path / "x_golden.csv").exists()

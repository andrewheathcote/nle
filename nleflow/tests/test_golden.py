"""`nleflow golden` reproduces the committed RDM golden files from their cards
(M1 acceptance (c)): rdm_ss_base_s801_e271 (the A = 0 flow) and
rdm_rdm_small_e195 (the small RDM flow, whose golden file is the EMC2 test
tree's), both to 1e-12."""
import numpy as np
import pytest

from nleflow.golden import main as golden_main

CARDS = ["rdm_ss_base_s801_e271", "rdm_rdm_small_e195", "ddm_sc_s1106_e245"]


def read(path):
    return np.genfromtxt(path, delimiter=",", names=True)


@pytest.mark.parametrize("tag", CARDS)
def test_golden_reproduces_committed_file(export_dir, tmp_path, tag):
    out = tmp_path / f"{tag}_golden.csv"
    golden_main(str(export_dir / f"{tag}_weights.json"), str(out))
    new, ref = read(out), read(export_dir / f"{tag}_golden.csv")
    assert new.dtype.names == ref.dtype.names
    assert len(new) == len(ref) == (1150 if tag.startswith("ddm") else 575)
    worst = 0.0
    for c in new.dtype.names:
        fa, fb = np.isfinite(new[c]), np.isfinite(ref[c])
        assert np.array_equal(fa, fb) and np.array_equal(new[c][~fa], ref[c][~fb])   # -Inf where -Inf
        worst = max(worst, float(np.max(np.abs(new[c][fa] - ref[c][fb]))))
    assert worst < 1e-12, f"{tag}: max |diff| {worst:.2e}"


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

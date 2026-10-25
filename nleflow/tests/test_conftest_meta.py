from pathlib import Path


def test_emc2_lib_skip_names_dmc2():
    """emc2_lib fixture's skip message must name DMC2 main, not an old branch."""
    conftest = (Path(__file__).parent / "conftest.py").read_text()
    assert "DMC2 main" in conftest
    assert "dev-flow" not in conftest and "dev-nle" not in conftest

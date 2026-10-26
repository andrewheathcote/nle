from pathlib import Path


def test_emc2_lib_skip_names_dmc2_main():
    """emc2_lib fixture's skip message names the EMC2 the R checks need: DMC2 main."""
    conftest = (Path(__file__).parent / "conftest.py").read_text()
    assert "DMC2 main" in conftest, (
        "emc2_lib skip reason does not name DMC2 main, the EMC2 the R checks need"
    )

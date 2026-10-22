from pathlib import Path


def test_emc2_lib_skip_names_dev_flow():
    """emc2_lib fixture's skip message must reference dev-flow, not dev-nle-m0."""
    conftest = (Path(__file__).parent / "conftest.py").read_text()
    assert "dev-flow" in conftest, (
        "emc2_lib skip reason still names dev-nle-m0; update to dev-flow"
    )

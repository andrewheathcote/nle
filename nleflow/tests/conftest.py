import os
from pathlib import Path

import pytest

NLEFLOW = Path(__file__).resolve().parents[1]
REPO = NLEFLOW.parent
EXPORT = REPO / "nleCards"
# EMC2's source tree, for tests/testthat/golden: NLEFLOW_EMC2_SRC, else EMC2 beside the nle repo
EMC2_SRC = Path(os.environ.get("NLEFLOW_EMC2_SRC") or REPO.parent / "EMC2")
EMC2_GOLDEN = EMC2_SRC / "tests" / "testthat" / "golden"


@pytest.fixture(scope="session")
def export_dir():
    if not EXPORT.exists():
        pytest.skip(f"the reference exports are not at {EXPORT}")
    return EXPORT


@pytest.fixture(scope="session")
def emc2_lib():
    """A library holding the EMC2 of DMC2 main (NLEFLOW_EMC2_LIB), or skip."""
    lib = os.environ.get("NLEFLOW_EMC2_LIB", "")
    if not lib or not (Path(lib) / "EMC2").exists():
        pytest.skip("set NLEFLOW_EMC2_LIB to a library holding the EMC2 of DMC2 main for the R checks")
    return lib

import os
from pathlib import Path

import pytest

NLEFLOW = Path(__file__).resolve().parents[1]
REPO = NLEFLOW.parent
EXPORT = REPO / "handover" / "nle-methods" / "port" / "export"


@pytest.fixture(scope="session")
def export_dir():
    if not EXPORT.exists():
        pytest.skip(f"the reference exports are not at {EXPORT}")
    return EXPORT


@pytest.fixture(scope="session")
def emc2_lib():
    """A library holding EMC2 dev-nle-m0 (NLEFLOW_EMC2_LIB), or skip."""
    lib = os.environ.get("NLEFLOW_EMC2_LIB", "")
    if not lib or not (Path(lib) / "EMC2").exists():
        pytest.skip("set NLEFLOW_EMC2_LIB to a library holding EMC2 dev-nle-m0 for the R checks")
    return lib

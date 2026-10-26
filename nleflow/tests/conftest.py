import os
from pathlib import Path

import pytest

NLEFLOW = Path(__file__).resolve().parents[1]
REPO = NLEFLOW.parent
# the reference exports: cards, their golden files, and the published LAN with the card and
# onnxruntime values EMC2's own tests hold
EXPORT = REPO / "nleCards"


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

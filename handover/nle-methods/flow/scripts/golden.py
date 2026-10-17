#!/usr/bin/env python3
"""Regenerate a card's golden vectors from its weights JSON alone.

The implementation moved into the nleflow package (nleflow/src/nleflow/golden.py)
at milestone M1; this script is the same command for callers that have the
repository but not the package installed:

    python golden.py <card>_weights.json [<out>_golden.csv]

numpy and scipy only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "nleflow" / "src"))
from nleflow.golden import main  # noqa: E402

if __name__ == "__main__":
    main(*sys.argv[1:])

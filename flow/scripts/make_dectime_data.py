"""Add a decision-time column d = rt - t0 to an existing DDM dataset.

The (rt - t0) parameterisation trains the flow on DECISION time rather than
response time, dropping t0 from the conditioning set entirely. t0 then enters
the likelihood only as an exact shift, and because d(rt - t0)/drt = 1 the
Jacobian is one -- so the stored `ll` is ALREADY the correct target for d and
does not need recomputing.

t0 is stored on the sampled (log) scale, so the natural value is exp(t0).

  python flow/scripts/make_dectime_data.py \
      --in data/train/st0zero400k/DDM/trials \
      --out data/train/dectime400k/DDM/trials
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    a = ap.parse_args()
    src, dst = REPO / a.src, REPO / a.dst
    dst.mkdir(parents=True, exist_ok=True)

    shards = sorted(src.glob("*.parquet"))
    assert shards, f"no parquet under {src}"
    n_tot = n_bad = 0
    dmin = np.inf
    for s in shards:
        t = pq.read_table(s)
        rt = t.column("rt").to_numpy(zero_copy_only=False)
        t0 = np.exp(t.column("t0").to_numpy(zero_copy_only=False))
        d = rt - t0
        n_tot += len(d)
        n_bad += int((d <= 0).sum())
        dmin = min(dmin, float(d.min()))
        t = t.append_column("d", pa.array(d.astype(np.float64)))
        pq.write_table(t, dst / s.name)
    print(f"wrote {len(shards)} shards -> {dst}")
    print(f"  rows {n_tot:,}   min d = {dmin:.6e}   rows with d <= 0: {n_bad}")
    assert n_bad == 0, "decision time must be strictly positive"


if __name__ == "__main__":
    main()

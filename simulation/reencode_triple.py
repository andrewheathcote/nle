"""Re-encode an existing DDM training set into the (SZ0, SZ, SZ1) parameterisation.

Andrew's proposal: replace (a, Z, SZ) with three positive quantities

    SZ0 = lower gap, SZ = start-point width, SZ1 = upper gap
    a = SZ0 + SZ + SZ1,   w = (SZ0 + SZ/2)/a,   sw = SZ/a

The point is to remove EMC2's Ttransform,  sw = 2 * SZprop * min(Z, 1-Z),
from what the conditioner has to represent: it is non-differentiable at
Z = 0.5 and couples SZ to both Z and a.

This is an EXACT re-encoding of an existing dataset rather than a fresh
simulation, and that is deliberate. Drawing SZ0/SZ/SZ1 from an independent
log-uniform box changes the induced prior (it shrinks the spread of w by ~19%
and piles 2.3x more mass at w = 0.5), which would confound the comparison with
the baseline. Re-encoding holds rt, ll, and the parameter distribution exactly
fixed, so the ONLY thing that changes is the representation the flow is
conditioned on. That is the controlled version of the experiment.

Stored columns are log(SZ0), log(SZ), log(SZ1), matching the log scale the
other positive parameters already use.

    python simulation/reencode_triple.py <in_dir> <out_dir>
"""
import sys, pathlib
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from scipy.stats import norm

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/train/tpt40p400k/DDM")
dst = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "data/train/reparam40/DDM")
(dst / "trials").mkdir(parents=True, exist_ok=True)

files = sorted((src / "trials").glob("*.parquet"))
assert files, f"no parquet files under {src/'trials'}"
tot = 0
lo_seen = np.inf
for f in files:
    t = pq.read_table(f)
    d = t.to_pandas()
    a = np.exp(d["a"].to_numpy())          # log scale -> natural
    w = norm.cdf(d["Z"].to_numpy())        # probit -> relative start point
    szp = norm.cdf(d["SZ"].to_numpy())     # probit -> SZ proportion
    sw = 2.0 * szp * np.minimum(w, 1.0 - w)   # EMC2 Ttransform

    SZ = sw * a
    SZ0 = a * w - SZ / 2.0
    SZ1 = a - SZ0 - SZ
    # SZ0, SZ1 >= 0 by construction; guard only against float cancellation
    eps = 1e-12
    lo_seen = min(lo_seen, SZ0.min(), SZ1.min(), SZ.min())
    SZ0 = np.maximum(SZ0, eps); SZ1 = np.maximum(SZ1, eps); SZ = np.maximum(SZ, eps)

    # round-trip check on this shard
    a_r = SZ0 + SZ + SZ1
    w_r = (SZ0 + SZ / 2.0) / a_r
    assert np.allclose(a_r, a, rtol=1e-9, atol=1e-9), "a round-trip failed"
    assert np.allclose(w_r, w, rtol=1e-7, atol=1e-9), "w round-trip failed"

    d["a"] = np.log(SZ0); d["Z"] = np.log(SZ); d["SZ"] = np.log(SZ1)
    d = d.rename(columns={"a": "SZ0", "Z": "SZw", "SZ": "SZ1"})
    pq.write_table(pa.Table.from_pandas(d, preserve_index=False),
                   dst / "trials" / f.name)
    tot += len(d)
    print(f"  {f.name}: {len(d):,}", flush=True)

print(f"\n{tot:,} rows -> {dst}")
print(f"smallest raw SZ0/SZ/SZ1 before the epsilon guard: {lo_seen:.3e}")

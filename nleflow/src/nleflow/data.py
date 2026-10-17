"""The data contract (NLEFLOW_PLAN.md section 7.2) and the in-memory batches.

One row per trial. Columns: `param_id`; each context name on the stored
(EMC2 sampled) scale, `-Inf` at an exact zero of a zero-box input; `rt`; and
optionally `ll` (the analytic log density, kept for diagnostics; not used by
the loss) and `R`. A dataset is a directory of parquet files (`.parquet`) or,
when R has no `arrow`, of CSV files (`.csv`, as `write.csv` writes them, with
`-Inf` spelled so); the two load identically.

The context encoding is the training-time counterpart of the card's
`context_transforms`: a zero-box input stored as log x (or probit x) becomes
asinh(x / c), computed in float64 from the stored value, exactly 0 at an exact
zero; every other input is used as stored. The standardiser is fitted on the
encoded training context and written to the run directory as scaler.json,
from where the exporter copies it into the card.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from scipy.special import ndtr

from .spec import flow_context

DATA = "rt"
DECISION_TIME = "d"


def open_dataset(path):
    """A pyarrow dataset over the parquet or CSV files under `path`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no data at {path}")
    if path.is_file():
        files = [path]
    else:
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in (".parquet", ".csv"))
    if not files:
        raise FileNotFoundError(f"no .parquet or .csv files under {path}")
    exts = {p.suffix.lower() for p in files}
    if len(exts) > 1:
        raise ValueError(f"{path} mixes parquet and csv files")
    fmt = "parquet" if exts == {".parquet"} else "csv"
    return ds.dataset([str(p) for p in files], format=fmt)


def load_table(dset, spec):
    """The columns a race spec needs, as a pyarrow table in file order."""
    cols = [DATA, *spec["context"]]
    names = set(dset.schema.names)
    missing = [c for c in cols if c not in names]
    if missing:
        raise ValueError(f"the data lack columns {missing}; present: {sorted(names)}")
    tbl = dset.to_table(columns=cols)
    return tbl


def encode_context(tbl, spec):
    """(time, context) as float64 arrays: the flow's time column (rt, or
    rt - t0 for a decision-time spec) and the encoded context in flow order."""
    rt = tbl[DATA].to_numpy(zero_copy_only=False).astype(np.float64)
    if not np.all(np.isfinite(rt) & (rt > 0)):
        bad = int(np.sum(~(np.isfinite(rt) & (rt > 0))))
        raise ValueError(f"{bad} rows have a non-finite or non-positive rt")
    time = rt
    if spec["pre"] is not None:
        t0 = np.exp(tbl[spec["pre"]].to_numpy(zero_copy_only=False).astype(np.float64))
        time = rt - t0
        n_bad = int(np.sum(time <= 0))
        if n_bad:
            raise ValueError(f"{n_bad} rows have rt <= {spec['pre']}: not decision-time data")
    names = flow_context(spec)
    cols = []
    for c in names:
        x = tbl[c].to_numpy(zero_copy_only=False).astype(np.float64)
        if spec["transforms"][c] == "zerobox":
            nat = ndtr(x) if spec["stored"][c] == "probit" else np.exp(x)   # -Inf -> 0 exactly
            x = np.arcsinh(nat / spec["zerobox_c"][c])
        if not np.all(np.isfinite(x)):
            raise ValueError(f"non-finite values in context column '{c}' after encoding")
        cols.append(x)
    ctx = np.stack(cols, axis=1)
    return time, ctx


class Scaler:
    """Standardisation of the encoded context, (x - mean) / scale per input,
    with the moments of the training set (population standard deviation, as
    sklearn's StandardScaler computes it)."""

    def __init__(self, mean, scale, names):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.names = list(names)

    @classmethod
    def fit(cls, ctx, names):
        mean = ctx.mean(axis=0)
        sd = ctx.std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)        # a constant column stays as it is
        return cls(mean, sd, names)

    def transform(self, ctx):
        return (ctx - self.mean) / self.scale

    def to_dict(self):
        return {"names": self.names, "mean": self.mean.tolist(), "scale": self.scale.tolist()}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["mean"], d["scale"], d["names"])


class Batches:
    """A dataset held in device memory, iterated in shuffled batches of
    (time, standardised context) each epoch; the last partial batch is dropped
    (train_race.py's PreloadedBatches)."""

    def __init__(self, time, ctx_std, batch_size, shuffle, rng=None):
        import jax
        import jax.numpy as jnp
        self.n = int(time.shape[0])
        if self.n < batch_size:
            raise ValueError(f"{self.n} rows is fewer than one batch of {batch_size}")
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.rng = rng if rng is not None else np.random.default_rng()
        self._time = jax.device_put(jnp.asarray(time.astype(np.float32)))
        self._ctx = jax.device_put(jnp.asarray(ctx_std.astype(np.float32)))
        self._jnp = jnp

    def __len__(self):
        return self.n // self.batch_size

    def __iter__(self):
        idx = self.rng.permutation(self.n) if self.shuffle else np.arange(self.n)
        end = len(self) * self.batch_size
        for start in range(0, end, self.batch_size):
            sl = self._jnp.asarray(idx[start:start + self.batch_size])
            yield self._time[sl], self._ctx[sl]


def write_parquet(path, columns):
    """Write one parquet file from a dict of equal-length arrays (the tests'
    and the Python-side simulators' way of meeting the data contract)."""
    import pyarrow.parquet as pq
    tbl = pa.table({k: pa.array(np.asarray(v)) for k, v in columns.items()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, str(path))

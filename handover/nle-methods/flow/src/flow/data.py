"""Batch generator and per-trial RT signing for flow / classifier training.

Extracted from training.py; semantics unchanged.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from sklearn import preprocessing
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path


def sign_rts(rts, response):
    """Sign RTs based on response."""
    return rts * jnp.where(response == 1, -1, 1)


def batches(dset, data_names, context_names, target_names, response_name,
            batch_size=8192, shuffle_files=True, sign=False, context_encoding=None):
    frags = list(dset.get_fragments())
    if shuffle_files:
        np.random.shuffle(frags)

    cols = [data_names] + context_names + [target_names]
    if sign and response_name not in cols:
        cols = cols + [response_name]

    for frag in frags:
        for rb in frag.to_batches(batch_size=batch_size, columns=cols):
            data = rb.column(data_names).to_numpy(zero_copy_only=False).astype(np.float32)
            context = np.stack(
                [rb.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in context_names],
                axis=1,
            )
            context = apply_context_encoding(context, context_names, context_encoding)
            y = rb.column(target_names).to_numpy(zero_copy_only=False).astype(np.float32)

            data_j = jnp.asarray(data)
            if sign:
                response = jnp.asarray(
                    rb.column(response_name).to_numpy(zero_copy_only=False).astype(np.int32),
                    dtype=jnp.int32,
                )
                data_j = sign_rts(data_j, response)

            yield data_j, jnp.asarray(context), jnp.asarray(y)


# a and Z jointly set where the accumulator starts RELATIVE to each boundary:
# distance to the upper bound is a(1-Z), to the lower a*Z. The conditioner gets
# a and Z as separate standardised inputs and must learn to form those
# products - a multiplicative interaction MLPs represent inefficiently. That
# would explain why the residual bias is shared by exactly these two
# parameters, is systematic across all 15 seeds, and survived capacity, data,
# ensembling and the classifier fix. These append the products directly.
def add_geometry_features(context, context_names, response_name=None):
    ia, iz = context_names.index("a"), context_names.index("Z")
    a = np.exp(context[:, ia])          # a is stored on the log scale
    Z = _norm_cdf(context[:, iz])       # Z on the probit scale
    new = np.concatenate([(a * Z)[:, None],            # distance to lower
                          (a * (1.0 - Z))[:, None]], axis=1)
    # R must stay the LAST column: downstream code slices context[:, -1] for the
    # response and context[:, :n_theta] for parameters, so insert before it.
    if response_name is not None and response_name in context_names:
        return np.concatenate([context[:, :-1], new, context[:, -1:]], axis=1)
    return np.concatenate([context, new], axis=1)


def _norm_cdf(x):
    from math import sqrt
    import scipy.special as sp
    return 0.5 * (1.0 + sp.erf(x / sqrt(2.0)))



# --- context encodings -------------------------------------------------------
# The parquet stores parameters on EMC2's sampled scale: log for a, t0, s, sv,
# st0 and probit for Z, SZ, with -Inf at an exact zero. The "zerobox" encoding
# re-expresses sv, SZ and st0 as asinh(x / c), which is exactly 0 at 0, linear
# below c and log-like above it, so one network can be conditioned on exact
# zeros. Constants match simulation/zerobox.R and are written into the export.
# "eta" (= sv a / s^2) is the scaled DDM's input (handover/SCALED_DDM_PLAN.md); it is stored on
# the log scale like sv, and only the columns present in a context are encoded.
ZEROBOX_C = {"sv": 0.01, "SZ": 0.05, "st0": 0.002, "eta": 0.01}


def apply_context_encoding(context, context_names, encoding):
    """Return context re-encoded in place of the stored EMC2-scale columns."""
    if encoding in (None, "emc2"):
        return context
    if encoding != "zerobox":
        raise ValueError(f"unknown context encoding {encoding!r}")
    out = context.copy()
    for name, c in ZEROBOX_C.items():
        if name not in context_names:
            continue
        j = context_names.index(name)
        col = context[:, j].astype(np.float64)
        nat = _norm_cdf(col) if name == "SZ" else np.exp(col)   # -Inf -> 0 exactly
        out[:, j] = np.arcsinh(nat / c).astype(np.float32)
    assert np.isfinite(out).all(), "zerobox encoding left a non-finite context value"
    return out

def _load_dataset(dset, data_names, context_names, target_names,
                  response_name, sign, standardize=False, scaler=None,
                  geometry=False, context_encoding=None):
    # target_names may be a LIST, in which case y comes back as (n, k). Needed
    # by the score-matching loss (ll plus 8 true-score columns) and by the
    # off-sample grid loss (ll plus a flag marking rows that are not model
    # samples, which must feed the MSE term but not the NLL).
    tnames = list(target_names) if isinstance(target_names, (list, tuple)) \
        else [target_names]
    cols = [data_names] + context_names + tnames
    if sign and response_name not in cols:
        cols = cols + [response_name]

    table = dset.to_table(columns=cols)

    data = table.column(data_names).to_numpy(zero_copy_only=False).astype(np.float32)
    context = np.stack(
        [table.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
         for c in context_names],
        axis=1,
    )

    context = apply_context_encoding(context, context_names, context_encoding)

    if geometry:
        context = add_geometry_features(context, context_names, response_name)

    if standardize:
        if response_name in context_names:
            feat, resp = context[:, :-1], context[:, -1:]
            if scaler is None:
                scaler = preprocessing.StandardScaler().fit(feat)
            context = np.concatenate([scaler.transform(feat), resp], axis=1)
        else:
            if scaler is None:
                scaler = preprocessing.StandardScaler().fit(context)
            context = scaler.transform(context)

    if len(tnames) == 1:
        y = table.column(tnames[0]).to_numpy(zero_copy_only=False).astype(np.float32)
    else:
        y = np.stack([table.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
                      for c in tnames], axis=1)

    if sign:
        response = table.column(response_name).to_numpy(zero_copy_only=False).astype(np.int32)
        data = data * np.where(response == 1, -1, 1).astype(np.float32)

    return (
        jax.device_put(jnp.asarray(data)),
        jax.device_put(jnp.asarray(context)),
        jax.device_put(jnp.asarray(y)),
        scaler if standardize else None,
    )


class PreloadedBatches:
    """Load a parquet dataset once into device memory; iterate batches each epoch.

    Iterating yields ``(data, context, y)`` as JAX arrays, matching ``batches``.
    The rng is stored on the instance so each ``__iter__`` advances its state
    and produces a different permutation per epoch (reproducible if seeded).
    """

    def __init__(self, dset, data_names, context_names, target_names,
                 response_name, batch_size=8192, shuffle=True, sign=False,
                 drop_last=True, rng=None, standardize=False, scaler=None,
                 pair_rows=False, geometry=False, context_encoding=None):
        self._data, self._context, self._y, self._scaler = _load_dataset(
            dset, data_names, context_names, target_names, response_name, sign,
            standardize, scaler, geometry
        , context_encoding=context_encoding)
        self._n = int(self._data.shape[0])
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._rng = rng if rng is not None else np.random.default_rng()
        # pair_rows: the simulator writes exactly 2 trials per parameter draw,
        # adjacently and in file order. The score penalty needs both members of
        # a pair in the same batch at positions 2k / 2k+1, so shuffle PAIRS
        # rather than rows. Verified on data/train/large/DDM/trials: every
        # fragment holds an even row count and t[0::2] == t[1::2] on param_id.
        self._pair_rows = pair_rows
        if pair_rows:
            if self._n % 2:
                raise ValueError(
                    "pair_rows=True needs an even row count; got %d" % self._n)
            if batch_size % 2:
                raise ValueError(
                    "pair_rows=True needs an even batch_size; got %d" % batch_size)

    def __iter__(self):
        if self._pair_rows:
            n_pairs = self._n // 2
            order = (self._rng.permutation(n_pairs) if self._shuffle
                     else np.arange(n_pairs))
            idx = np.empty(self._n, dtype=np.int64)
            idx[0::2] = 2 * order
            idx[1::2] = 2 * order + 1
        elif self._shuffle:
            idx = self._rng.permutation(self._n)
        else:
            idx = np.arange(self._n)

        n_full = (self._n // self._batch_size) * self._batch_size
        end = n_full if self._drop_last else self._n

        for start in range(0, end, self._batch_size):
            sl = jnp.asarray(idx[start:start + self._batch_size])
            yield self._data[sl], self._context[sl], self._y[sl]

def get_inp_data(parquet_path: Path, params_path: Path, context: list[str]):
    """Load the DDM data and extract the context and target arrays."""
    df = pq.read_table(parquet_path).to_pandas()
    params_df = pd.read_csv(params_path)
    params_df = params_df[context]
    
    return df, params_df
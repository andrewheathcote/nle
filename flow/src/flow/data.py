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
            batch_size=8192, shuffle_files=True, sign=False):
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
            y = rb.column(target_names).to_numpy(zero_copy_only=False).astype(np.float32)

            data_j = jnp.asarray(data)
            if sign:
                response = jnp.asarray(
                    rb.column(response_name).to_numpy(zero_copy_only=False).astype(np.int32),
                    dtype=jnp.int32,
                )
                data_j = sign_rts(data_j, response)

            yield data_j, jnp.asarray(context), jnp.asarray(y)


def _load_dataset(dset, data_names, context_names, target_names,
                  response_name, sign, standardize=False, scaler=None):
    cols = [data_names] + context_names + [target_names]
    if sign and response_name not in cols:
        cols = cols + [response_name]

    table = dset.to_table(columns=cols)

    data = table.column(data_names).to_numpy(zero_copy_only=False).astype(np.float32)
    context = np.stack(
        [table.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
         for c in context_names],
        axis=1,
    )

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

    y = table.column(target_names).to_numpy(zero_copy_only=False).astype(np.float32)

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
                 drop_last=True, rng=None, standardize=False, scaler=None):
        self._data, self._context, self._y, self._scaler = _load_dataset(
            dset, data_names, context_names, target_names, response_name, sign, standardize, scaler
        )
        self._n = int(self._data.shape[0])
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._rng = rng if rng is not None else np.random.default_rng()

    def __iter__(self):
        if self._shuffle:
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
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: likelihood-kde-normflow
#     language: python
#     name: python3
# ---

# %% [markdown]
# # RDM large-budget flow (standalone replication)
#
# Self-contained replication of the large-budget RDM section of
# `flow_training.py` — same seed (`jax.random.key(100)`), architecture
# (dims [5, 32], 32 spline bins), optimizer/schedule, and data.
#
# **The parameters are already on the log scale.** The training data was
# sampled uniformly on the natural scale within the designs.R bounds and then
# log-transformed to the sampled scale for *all five* RDM parameters
# (`param_matrix` in simulation/sim_functions.R defaults `transforms = NULL`
# to `rep("log", dim)`). So the flow conditions on
# log(v), log(B), log(t0), log(s), log(A) — no transformation is needed here,
# and applying one would double-log the data.
#
# **Run cell-by-cell from `flow/notebooks/`** (paths are `../../data/...`).
# Outputs go to the canonical locations:
#
#     checkpoints:  results/model_states/large_budget/RDM/
#     predictions:  results/RDM/large/flow_ll_pred.csv

# %%
import pyarrow.dataset as ds
import numpy as np, jax.numpy as jnp
from pathlib import Path
import jax
from flax import nnx
import optax
import json

from flow.config import make_mlp_conditioner, spline_flow
from flow.training import (make_checkpoint_manager, restore_checkpoint,
                           train_model, save_obj, plot_loss,
                           plot_scatter_normalisation, evaluate_pdf_sf)
from flow.data import PreloadedBatches

# %% [markdown]
# ## Data — large budget (context columns are log-scale on disk)

# %%
DATA = "rt"  # The normflow is only trained on one accumulator
CONTEXT = ["v", "B", "t0", "s", "A"]
TARGET = "ll"
RESPONSE = "R"
TRAIN_PATH = Path('../../data/train/large/RDM_single/trials')
VAL_PATH = Path('../../data/val/RDM_single/trials')

dset_train = ds.dataset(TRAIN_PATH, format="parquet")
dset_val = ds.dataset(VAL_PATH, format="parquet")

# %% [markdown]
# ## Model, optimizer, loaders

# %%
NUM_BINS = 32
NUM_SPLINES = 1
NUM_EPOCHS = 200
BATCH_SIZE_TRAIN = 128
USE_NORM = False
STANDARDIZE = True

train_key = jax.random.key(100)
conditioner_key, sampling_key = jax.random.split(train_key, 2)
rngs = nnx.Rngs(default=conditioner_key, sampling=sampling_key)

dims = [len(CONTEXT), 32]

conditioner_RDM = make_mlp_conditioner(
    dims=dims,
    num_bins=NUM_BINS,
    num_splines=NUM_SPLINES,
    rngs=rngs,
    use_norm=USE_NORM,
)

total_steps = (dset_train.count_rows() // BATCH_SIZE_TRAIN) * NUM_EPOCHS
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.,
    peak_value=1e-4,
    warmup_steps=0.1 * total_steps,
    decay_steps=total_steps
)

optimizer = nnx.Optimizer(
    conditioner_RDM, optax.adamw(learning_rate=schedule), wrt=nnx.Param,
)
train_metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
val_metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

train_loader = PreloadedBatches(
    dset_train,
    data_names=DATA,
    context_names=CONTEXT,
    target_names=TARGET,
    response_name=RESPONSE,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    standardize=STANDARDIZE
)

val_loader = PreloadedBatches(
    dset_val,
    data_names=DATA,
    context_names=CONTEXT,
    target_names=TARGET,
    response_name=RESPONSE,
    batch_size=256,
    shuffle=False,
    standardize=STANDARDIZE,
    scaler=train_loader._scaler
)

# %% [markdown]
# ## Train (long cell — 200 epochs over 1M trials; hours on CPU)

# %%
train_loss, val_loss, _, _ = train_model(
    conditioner_RDM,
    optimizer,
    train_metrics,
    val_metrics,
    train_loader,
    val_loader,
    batch_size_train=BATCH_SIZE_TRAIN,
    batch_size_val=256,
    dset_train=dset_train,
    dset_val=dset_val,
    num_epochs=NUM_EPOCHS,
    data_names=DATA,
    context_names=CONTEXT,
    target_names=TARGET,
    response_name=RESPONSE,
    checkpoint_path="../../results/model_states/large_budget/RDM",
    early_stopping=False,
)

# %%
plot_loss(train_loss, val_loss)

# %%
save_obj(train_loader, "../../results/model_states/large_budget/RDM/train_loader.pkl")
with open("../../results/model_states/large_budget/RDM/schedule_config.json", "w") as f:
    json.dump({
        "schedule_type": "warmup_cosine_decay",
        "init_value": 0,
        "peak_value": 1e-4,
        "warmup_steps": 0.1 * total_steps,
        "decay_steps": total_steps
    }, f)

# %% [markdown]
# ## Restore the best-validation-loss checkpoint

# %%
def cond_factory():
    return make_mlp_conditioner(nnx.Rngs(0), dims=dims, num_bins=NUM_BINS,
                                num_splines=NUM_SPLINES, use_norm=USE_NORM)

def opt_factory():
    # must match the optimizer used at save time (same optax tx, same hparams)
    return nnx.Optimizer(cond_factory(), optax.adamw(schedule), wrt=nnx.Param)

mgr = make_checkpoint_manager("../../results/model_states/large_budget/RDM")
conditioner_best, _ = restore_checkpoint(
    mgr, mgr.best_step(), cond_factory, opt_factory
)

# %% [markdown]
# ## Test evaluation — two-accumulator race
#
# The test context columns are on the same (log) sampled scale as training,
# so they only pass through the fitted standardization scaler.

# %%
TEST_PATH = Path('../../data/test/RDM/trials')

ds_test = ds.dataset(TEST_PATH, format="parquet").to_table()
np.random.seed(0)
n_test = ds_test.num_rows
idx = np.random.permutation(np.arange(n_test))

rt     = jnp.asarray(ds_test.column("rt").to_numpy(zero_copy_only=False))
choice = jnp.asarray(np.int32(ds_test.column("R").to_numpy(zero_copy_only=False)))  # 1 or 2
y      = jnp.asarray(ds_test.column("ll").to_numpy(zero_copy_only=False))

# Per-accumulator context, SAME column order as training: [v, B, t0, s, A]
def stack_ctx(suffix):
    cols = [f"v_lR{suffix}", f"B_lR{suffix}", "t0", f"s_lR{suffix}", f"A_lR{suffix}"]
    return jnp.asarray(np.stack(
        [ds_test.column(c).to_numpy(zero_copy_only=False) for c in cols], axis=1))

scaler = train_loader._scaler
ctx_1 = scaler.transform(stack_ctx(1))
ctx_2 = scaler.transform(stack_ctx(2))

log_pdf_1, log_sf_1 = evaluate_pdf_sf(conditioner_best, rt, ctx_1)
log_pdf_2, log_sf_2 = evaluate_pdf_sf(conditioner_best, rt, ctx_2)

pred_ll = jnp.where(
    choice == 1,
    log_pdf_1.squeeze() + log_sf_2.squeeze(),
    log_pdf_2.squeeze() + log_sf_1.squeeze(),
)

# %%
mask = np.asarray(choice == 1)
_, flow = spline_flow(rt[mask], ctx_1[mask], conditioner_best)
z_test = np.asarray(flow.bijector.inverse(rt[mask]))

pred_ll = jnp.maximum(pred_ll, jnp.log(1e-10))
plot_scatter_normalisation(y[idx[:50000]], pred_ll[idx[:50000]], z_test, num_bins=200)
print(f"Correlation: {jnp.corrcoef(y, pred_ll)[0, 1]:.4f}")
print(f"MSE(y, pred_ll) = {jnp.mean((y - pred_ll)**2):.4f}")

# %% [markdown]
# Compare against Julian's deposit run of the same section
# (`julian/results/RDM/large/flow_ll_pred.csv`): correlation and MSE should
# match closely (not bit-exactly on different hardware).

# %%
test_pd = ds_test.to_pandas()
test_pd["flow_ll_pred"] = pred_ll

SAVE_PATH = Path('../../results/RDM/large')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

test_pd.to_csv(SAVE_PATH / "flow_ll_pred.csv", index=False)

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

# %%
import pyarrow.dataset as ds
import numpy as np, jax.numpy as jnp
from pathlib import Path
import jax
from flax import nnx
import optax
from flow.config import make_mlp_conditioner
from flow.training import make_checkpoint_manager, restore_checkpoint, train_model, save_obj
import orbax.checkpoint as ocp
from flow.data import PreloadedBatches
import json

# %%
from jax.sharding import SingleDeviceSharding
cpu_sharding = SingleDeviceSharding(jax.devices("cpu")[0])

def _with_sharding(x):
    if isinstance(x, jax.ShapeDtypeStruct):
        return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=cpu_sharding)
    return x

def restore_on_cpu(mgr, step, cond_factory, opt_factory):
    abs_cond = nnx.eval_shape(cond_factory)
    cond_gd, abs_cond_state = nnx.split(abs_cond)
    abs_cond_state = jax.tree.map(_with_sharding, abs_cond_state)

    abs_opt = nnx.eval_shape(opt_factory)
    opt_gd, abs_opt_state = nnx.split(abs_opt)
    abs_opt_state = jax.tree.map(_with_sharding, abs_opt_state)

    restored = mgr.restore(
        step,
        args=ocp.args.Composite(
            conditioner=ocp.args.StandardRestore(abs_cond_state),
            optimizer=ocp.args.StandardRestore(abs_opt_state),
        ),
    )
    return nnx.merge(cond_gd, restored["conditioner"]), nnx.merge(opt_gd, restored["optimizer"])



# %%
DATA = "rt" # The normflow is only trained on one accumulator
CONTEXT = ["m", "s", "t0"]
TARGET   = "ll"
RESPONSE = "R"
TRAIN_PATH = Path('../../data/train/medium/LNR_single/trials')

dset_train = ds.dataset(TRAIN_PATH, format="parquet")

# %%
batch_loader = PreloadedBatches(dset_train, DATA, CONTEXT, TARGET, RESPONSE, standardize=True, drop_last=False)

# %%
batches = np.empty((0, len(CONTEXT)))
for batch in iter(batch_loader):
    batches = np.concatenate([batches, batch[1]])
    print(batch[0].shape, batch[1].shape, batch[2].shape)
    print(batch[0][:5], batch[1][:5], batch[2][:5])
    np.mean(batch[0]), np.mean(batch[1]), np.mean(batch[2])

# %%
print(batches.shape)
# Check if standardization worked
print("Mean:", np.mean(batches, axis=0))
print("Std:", np.std(batches, axis=0))

# %% [markdown]
# # Normalizing flow training for RDM, LNR and full DDM 
# ## RDM
# ### Small budget

# %%
DATA = "rt" # The normflow is only trained on one accumulator
CONTEXT = ["v", "B", "t0", "s", "A"]
TARGET   = "ll"
RESPONSE = "R"
TRAIN_PATH = Path('../../data/train/small/RDM_single/trials')
VAL_PATH = Path('../../data/val/RDM_single/trials')

dset_train = ds.dataset(TRAIN_PATH, format="parquet")
dset_val = ds.dataset(VAL_PATH, format="parquet")

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

# set up lr schedule
total_steps = (dset_train.count_rows() // BATCH_SIZE_TRAIN) * NUM_EPOCHS
# schedule = optax.cosine_decay_schedule(1e-5, decay_steps=total_steps)
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0., 
    peak_value=1e-4, 
    warmup_steps=0.1 * total_steps, 
    decay_steps=total_steps
)

optimizer = nnx.Optimizer(
    conditioner_RDM, optax.adamw(learning_rate=schedule), wrt=nnx.Param,
)
train_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)
val_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

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
    checkpoint_path="../../results/model_states/small_budget/RDM",
    early_stopping=False,
)

# %%
save_obj(train_loader, "../../results/model_states/small_budget/RDM/train_loader.pkl")
with open("../../results/model_states/small_budget/RDM/schedule_config.json", "w") as f:
    json.dump({
        "schedule_type": "warmup_cosine_decay",
        "init_value": 0,
        "peak_value": 1e-4,
        "warmup_steps": 0.1 * total_steps,
        "decay_steps": total_steps
    }, f)

# %%
from flow.training import plot_loss
plot_loss(train_loss, val_loss)


# %%
def cond_factory():
    return make_mlp_conditioner(nnx.Rngs(0), dims=dims, num_bins=NUM_BINS, num_splines=NUM_SPLINES, use_norm=USE_NORM)

def opt_factory():
    # must match the optimizer used at save time (same optax tx, same hparams)
    return nnx.Optimizer(cond_factory(), optax.adamw(schedule), wrt=nnx.Param)

mgr = make_checkpoint_manager("../../results/model_states/small_budget/RDM")
conditioner_best, optimizer = restore_checkpoint(
    mgr, mgr.best_step(), cond_factory, opt_factory
)


# %%
state = nnx.state(conditioner_best, nnx.Param)
total = 0
for path, leaf in jax.tree_util.tree_leaves_with_path(state):
    name = jax.tree_util.keystr(path)
    n = leaf.size
    total += n
    print(f"{name:55s} {str(leaf.shape):18s} {n:>10,}")
print(f"{'TOTAL':55s} {'':18s} {total:>10,}")

# %%
from flow.training import evaluate_pdf_sf
from flow.config import spline_flow

TEST_PATH = Path('../../data/test/RDM/trials')

ds_test = ds.dataset(TEST_PATH, format="parquet").to_table()
np.random.seed(0)
idx = np.random.permutation(np.arange(ds_test.num_rows))

rt     = jnp.asarray(ds_test.column("rt").to_numpy(zero_copy_only=False))
choice = jnp.asarray(np.int32(ds_test.column("R").to_numpy(zero_copy_only=False)))  # 1 or 2
y      = jnp.asarray(ds_test.column("ll").to_numpy(zero_copy_only=False))

# Per-accumulator context, SAME column order as training: [v, B, t0, s, A]
def stack_ctx(suffix):
    cols = [f"v_lR{suffix}", f"B_lR{suffix}", "t0", f"s_lR{suffix}", f"A_lR{suffix}"]
    return jnp.asarray(np.stack(
        [ds_test.column(c).to_numpy(zero_copy_only=False) for c in cols], axis=1))

if train_loader._scaler is not None:
    scaler = train_loader._scaler
    ctx_1 = scaler.transform(stack_ctx(1))
    ctx_2 = scaler.transform(stack_ctx(2))
else:
    ctx_1 = stack_ctx(1)
    ctx_2 = stack_ctx(2)

log_pdf_1, log_sf_1 = evaluate_pdf_sf(conditioner_best, rt, ctx_1)
log_pdf_2, log_sf_2 = evaluate_pdf_sf(conditioner_best, rt, ctx_2)

pred_ll = jnp.where(
    choice == 1,
    log_pdf_1.squeeze() + log_sf_2.squeeze(),  
    log_pdf_2.squeeze() + log_sf_1.squeeze(),
    
)

mask = np.asarray(choice == 1)
_, flow = spline_flow(rt[mask], ctx_1[mask], conditioner_best)
z_test = np.asarray(flow.bijector.inverse(rt[mask]))

from flow.training import plot_scatter_normalisation

pred_ll = jnp.maximum(pred_ll, jnp.log(1e-10))
plot_scatter_normalisation(y[idx[:50000]], pred_ll[idx[:50000]], z_test, num_bins=200)

print(f"corr(y, pred_ll) = {jnp.corrcoef(y, pred_ll)[0,1]:.4f}")
print(f"MSE(y, pred_ll) = {jnp.mean((y - pred_ll)**2):.4f}")

# %%
test_pd = ds_test.to_pandas()
test_pd["flow_ll_pred"] = pred_ll

SAVE_PATH = Path('../../results/RDM/small')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

test_pd.to_csv(SAVE_PATH / "flow_ll_pred.csv", index=False)

# %% [markdown]
# ### Medium budget

# %%
TRAIN_PATH = Path('../../data/train/medium/RDM_single/trials')
dset_train = ds.dataset(TRAIN_PATH, format="parquet")

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

# set up lr schedule
total_steps = (dset_train.count_rows() // BATCH_SIZE_TRAIN) * NUM_EPOCHS
# schedule = optax.cosine_decay_schedule(1e-5, decay_steps=total_steps)
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0., 
    peak_value=1e-4, 
    warmup_steps=0.1 * total_steps, 
    decay_steps=total_steps
)

optimizer = nnx.Optimizer(
    conditioner_RDM, optax.adamw(learning_rate=schedule), wrt=nnx.Param,
)
train_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)
val_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

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
    checkpoint_path="../../results/model_states/medium_budget/RDM",
    early_stopping=False,
)

# %%
save_obj(train_loader, "../../results/model_states/medium_budget/RDM/train_loader.pkl")
with open("../../results/model_states/medium_budget/RDM/schedule_config.json", "w") as f:
    json.dump({
        "schedule_type": "warmup_cosine_decay",
        "init_value": 0,
        "peak_value": 1e-4,
        "warmup_steps": 0.1 * total_steps,
        "decay_steps": total_steps
    }, f)

# %%
from flow.training import plot_loss
plot_loss(train_loss, val_loss)


# %%
def cond_factory():
    return make_mlp_conditioner(nnx.Rngs(0), dims=dims, num_bins=NUM_BINS, num_splines=NUM_SPLINES, use_norm=USE_NORM)

def opt_factory():
    # must match the optimizer used at save time (same optax tx, same hparams)
    return nnx.Optimizer(cond_factory(), optax.adamw(schedule), wrt=nnx.Param)

mgr = make_checkpoint_manager("../../results/model_states/medium_budget/RDM")
conditioner_best, optimizer = restore_checkpoint(
    mgr, mgr.best_step(), cond_factory, opt_factory
)

# %%
state = nnx.state(conditioner_best, nnx.Param)
for path, leaf in jax.tree_util.tree_leaves_with_path(state):
    name = jax.tree_util.keystr(path)
    n = leaf.size
    print(f"{name:55s} {str(leaf.shape):18s} {n:>10,}")

# %%
from flow.training import evaluate_pdf_sf
from flow.config import spline_flow

TEST_PATH = Path('../../data/test/RDM/trials')


ds_test = ds.dataset(TEST_PATH, format="parquet").to_table()
np.random.seed(0)
idx = np.random.permutation(np.arange(ds_test.num_rows))

rt     = jnp.asarray(ds_test.column("rt").to_numpy(zero_copy_only=False))
choice = jnp.asarray(np.int32(ds_test.column("R").to_numpy(zero_copy_only=False)))  # 1 or 2
y      = jnp.asarray(ds_test.column("ll").to_numpy(zero_copy_only=False))

# Per-accumulator context, SAME column order as training: [v, B, t0, s, A]
def stack_ctx(suffix):
    cols = [f"v_lR{suffix}", f"B_lR{suffix}", "t0", f"s_lR{suffix}", f"A_lR{suffix}"]
    return jnp.asarray(np.stack(
        [ds_test.column(c).to_numpy(zero_copy_only=False) for c in cols], axis=1))

if train_loader._scaler is not None:
    scaler = train_loader._scaler
    ctx_1 = scaler.transform(stack_ctx(1))
    ctx_2 = scaler.transform(stack_ctx(2))
else:
    ctx_1 = stack_ctx(1)
    ctx_2 = stack_ctx(2)

log_pdf_1, log_sf_1 = evaluate_pdf_sf(conditioner_best, rt, ctx_1)
log_pdf_2, log_sf_2 = evaluate_pdf_sf(conditioner_best, rt, ctx_2)

pred_ll = jnp.where(
    choice == 1,
    log_pdf_1.squeeze() + log_sf_2.squeeze(),  
    log_pdf_2.squeeze() + log_sf_1.squeeze(),
    
)

mask = np.asarray(choice == 1)
_, flow = spline_flow(rt[mask], ctx_1[mask], conditioner_best)
z_test = np.asarray(flow.bijector.inverse(rt[mask]))

from flow.training import plot_scatter_normalisation

pred_ll = jnp.maximum(pred_ll, jnp.log(1e-10))
plot_scatter_normalisation(y[idx[:50000]], pred_ll[idx[:50000]], z_test, num_bins=200)

print(f"corr(y, pred_ll) = {jnp.corrcoef(y, pred_ll)[0,1]:.4f}")
print(f"MSE(y, pred_ll) = {jnp.mean((y - pred_ll)**2):.4f}")

# %%
test_pd = ds_test.to_pandas()
test_pd["flow_ll_pred"] = pred_ll

SAVE_PATH = Path('../../results/RDM/medium')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

test_pd.to_csv(SAVE_PATH / "flow_ll_pred.csv", index=False)

# %% [markdown]
# ### Large budget

# %%
TRAIN_PATH = Path('../../data/train/large/RDM_single/trials')
dset_train = ds.dataset(TRAIN_PATH, format="parquet")

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

# set up lr schedule
total_steps = (dset_train.count_rows() // BATCH_SIZE_TRAIN) * NUM_EPOCHS
# schedule = optax.cosine_decay_schedule(1e-5, decay_steps=total_steps)
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0., 
    peak_value=1e-4, 
    warmup_steps=0.1 * total_steps, 
    decay_steps=total_steps
)

optimizer = nnx.Optimizer(
    conditioner_RDM, optax.adamw(learning_rate=schedule), wrt=nnx.Param,
)
train_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)
val_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

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


# %%
def cond_factory():
    return make_mlp_conditioner(nnx.Rngs(0), dims=dims, num_bins=NUM_BINS, num_splines=NUM_SPLINES, use_norm=USE_NORM)

def opt_factory():
    # must match the optimizer used at save time (same optax tx, same hparams)
    return nnx.Optimizer(cond_factory(), optax.adamw(schedule), wrt=nnx.Param)

mgr = make_checkpoint_manager("../../results/model_states/large_budget/RDM")
conditioner_best, optimizer = restore_checkpoint(
    mgr, mgr.best_step(), cond_factory, opt_factory
)

# %%
from flow.training import evaluate_pdf_sf
from flow.config import spline_flow

TEST_PATH = Path('../../data/test/RDM/trials')

ds_test = ds.dataset(TEST_PATH, format="parquet").to_table()
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

if train_loader._scaler is not None:
    scaler = train_loader._scaler
    ctx_1 = scaler.transform(stack_ctx(1))
    ctx_2 = scaler.transform(stack_ctx(2))
else:
    ctx_1 = stack_ctx(1)
    ctx_2 = stack_ctx(2)

log_pdf_1, log_sf_1 = evaluate_pdf_sf(conditioner_best, rt, ctx_1)
log_pdf_2, log_sf_2 = evaluate_pdf_sf(conditioner_best, rt, ctx_2)

pred_ll = jnp.where(
    choice == 1,
    log_pdf_1.squeeze() + log_sf_2.squeeze(),  
    log_pdf_2.squeeze() + log_sf_1.squeeze(),
    
)

mask = np.asarray(choice == 1)
_, flow = spline_flow(rt[mask], ctx_1[mask], conditioner_best)
z_test = np.asarray(flow.bijector.inverse(rt[mask]))

from flow.training import plot_scatter_normalisation

pred_ll = jnp.maximum(pred_ll, jnp.log(1e-10))
plot_scatter_normalisation(y[idx[:50000]], pred_ll[idx[:50000]], z_test, num_bins=200)
print(f"Correlation: {jnp.corrcoef(y, pred_ll)[0, 1]:.4f}")
print(f"MSE(y, pred_ll) = {jnp.mean((y - pred_ll)**2):.4f}")

# %%
test_pd = ds_test.to_pandas()
test_pd["flow_ll_pred"] = pred_ll

SAVE_PATH = Path('../../results/RDM/large')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

test_pd.to_csv(SAVE_PATH / "flow_ll_pred.csv", index=False)

# %% [markdown]
# ## LNR
# ### Small budget

# %%
DATA = "rt" # The normflow is only trained on one accumulator
CONTEXT_LNR = ["m", "s", "t0"]
TARGET   = "ll"
TRAIN_PATH = Path('../../data/train/small/LNR_single/trials')
VAL_PATH = Path('../../data/val/LNR_single/trials')

dset_train_LNR = ds.dataset(TRAIN_PATH, format="parquet")
dset_val_LNR = ds.dataset(VAL_PATH, format="parquet")

# %%
NUM_BINS = 16
NUM_SPLINES = 1
NUM_EPOCHS = 200
BATCH_SIZE_TRAIN = 128
USE_NORM = False
STANDARDIZE = True

train_key = jax.random.key(100)
conditioner_key, sampling_key = jax.random.split(train_key, 2)
rngs = nnx.Rngs(default=conditioner_key, sampling=sampling_key)

dims = [len(CONTEXT_LNR), 16]

conditioner_LNR = make_mlp_conditioner(
    dims=dims,
    num_bins=NUM_BINS,
    num_splines=NUM_SPLINES,
    rngs=rngs,
    use_norm=USE_NORM,
)

# set up lr schedule
total_steps = (dset_train_LNR.count_rows() // BATCH_SIZE_TRAIN) * NUM_EPOCHS
# schedule = optax.cosine_decay_schedule(1e-5, decay_steps=total_steps)
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0., 
    peak_value=1e-4, 
    warmup_steps=0.1 * total_steps, 
    decay_steps=total_steps
)


optimizer = nnx.Optimizer(
    conditioner_LNR, optax.adamw(learning_rate=schedule), wrt=nnx.Param,
)
train_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)
val_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

train_loader = PreloadedBatches(
    dset_train_LNR, 
    data_names=DATA, 
    context_names=CONTEXT_LNR, 
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True, 
    standardize=STANDARDIZE
)


val_loader = PreloadedBatches(
    dset_val_LNR, 
    data_names=DATA, 
    context_names=CONTEXT_LNR, 
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=256, 
    shuffle=False, 
    standardize=STANDARDIZE, 
    scaler=train_loader._scaler
)

# %%
train_loss, val_loss, _, _ = train_model(
    conditioner_LNR,
    optimizer,
    train_metrics,
    val_metrics,
    train_loader,
    val_loader,
    batch_size_train=BATCH_SIZE_TRAIN,
    batch_size_val=256, 
    dset_train=dset_train_LNR, 
    dset_val=dset_val_LNR, 
    num_epochs=NUM_EPOCHS, 
    data_names=DATA, 
    context_names=CONTEXT_LNR, 
    target_names=TARGET, 
    response_name=RESPONSE,
    checkpoint_path="../../results/model_states/small_budget/LNR",
    early_stopping=False,
)

# %%
with open("../../results/model_states/small_budget/LNR/schedule_config.json", "w") as f:
    json.dump({
        "schedule_type": "warmup_cosine_decay",
        "init_value": 0,
        "peak_value": 1e-4,
        "warmup_steps": 0.1 * total_steps,
        "decay_steps": total_steps
    }, f)

save_obj(train_loader, "../../results/model_states/small_budget/LNR/train_loader.pkl")

# %%
from flow.training import plot_loss
plot_loss(train_loss, val_loss)


# %%
def cond_factory():
    return make_mlp_conditioner(nnx.Rngs(0), dims=dims, num_bins=NUM_BINS, num_splines=NUM_SPLINES, use_norm=USE_NORM)

def opt_factory():
    # must match the optimizer used at save time (same optax tx, same hparams)
    return nnx.Optimizer(cond_factory(), optax.adamw(schedule), wrt=nnx.Param)

mgr = make_checkpoint_manager("../../results/model_states/small_budget/LNR/")
conditioner_best, _ = restore_checkpoint(
    mgr, mgr.best_step(), cond_factory, opt_factory
)

# %%
state = nnx.state(conditioner_best, nnx.Param)
total = 0
for path, leaf in jax.tree_util.tree_leaves_with_path(state):
    name = jax.tree_util.keystr(path)
    n = leaf.size
    total += n
    print(f"{name:55s} {str(leaf.shape):18s} {n:>10,}")
print(f"{'TOTAL':55s} {'':18s} {total:>10,}")

# %%
from flow.training import evaluate_pdf_sf
from flow.config import spline_flow

TEST_PATH = Path('../../data/test/LNR/trials')

ds_test = ds.dataset(TEST_PATH, format="parquet").to_table()
np.random.seed(0)
n_test = ds_test.num_rows
idx = np.random.permutation(np.arange(n_test))

rt     = jnp.asarray(ds_test.column("rt").to_numpy(zero_copy_only=False))
choice = jnp.asarray(np.int32(ds_test.column("R").to_numpy(zero_copy_only=False)))  # 1 or 2
y      = jnp.asarray(ds_test.column("ll").to_numpy(zero_copy_only=False))

# Per-accumulator context, SAME column order as training: [v, B, t0, s, A]
def stack_ctx(suffix):
    cols = [f"m_lR{suffix}", f"s_lR{suffix}", "t0"]
    return jnp.asarray(np.stack(
        [ds_test.column(c).to_numpy(zero_copy_only=False) for c in cols], axis=1))

if train_loader._scaler is not None:
    scaler = train_loader._scaler
    ctx_1 = scaler.transform(stack_ctx(1))
    ctx_2 = scaler.transform(stack_ctx(2))
else:
    ctx_1 = stack_ctx(1)
    ctx_2 = stack_ctx(2)

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

from flow.training import plot_scatter_normalisation

pred_ll = jnp.maximum(pred_ll, jnp.log(1e-10))
plot_scatter_normalisation(y[idx[:50000]], pred_ll[idx[:50000]], z_test, num_bins=200)
print(f"Correlation: {jnp.corrcoef(y, pred_ll)[0, 1]:.4f}")
print(f"MSE(y, pred_ll) = {jnp.mean((y - pred_ll)**2):.4f}")

# %%
test_pd = ds_test.to_pandas()
test_pd["flow_ll_pred"] = pred_ll

SAVE_PATH = Path('../../results/LNR/small')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

test_pd.to_csv(SAVE_PATH / "flow_ll_pred.csv", index=False)

# %% [markdown]
# ### Medium budget

# %%
DATA = "rt" # The normflow is only trained on one accumulator
CONTEXT_LNR = ["m", "s", "t0"]
TARGET   = "ll"
TRAIN_PATH = Path('../../data/train/medium/LNR_single/trials')
VAL_PATH = Path('../../data/val/LNR_single/trials')

dset_train_LNR = ds.dataset(TRAIN_PATH, format="parquet")
dset_val_LNR = ds.dataset(VAL_PATH, format="parquet")

# %%
NUM_BINS = 16
NUM_SPLINES = 1
NUM_EPOCHS = 200
BATCH_SIZE_TRAIN = 128
USE_NORM = False
STANDARDIZE = True

train_key = jax.random.key(100)
conditioner_key, sampling_key = jax.random.split(train_key, 2)
rngs = nnx.Rngs(default=conditioner_key, sampling=sampling_key)

dims = [len(CONTEXT_LNR), 16]

conditioner_LNR = make_mlp_conditioner(
    dims=dims,
    num_bins=NUM_BINS,
    num_splines=NUM_SPLINES,
    rngs=rngs,
    use_norm=USE_NORM,
)

# set up lr schedule
total_steps = (dset_train_LNR.count_rows() // BATCH_SIZE_TRAIN) * NUM_EPOCHS
# schedule = optax.cosine_decay_schedule(1e-5, decay_steps=total_steps)
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0., 
    peak_value=1e-4, 
    warmup_steps=0.1 * total_steps, 
    decay_steps=total_steps
)


optimizer = nnx.Optimizer(
    conditioner_LNR, optax.adamw(learning_rate=schedule), wrt=nnx.Param,
)
train_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)
val_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

train_loader = PreloadedBatches(
    dset_train_LNR, 
    data_names=DATA, 
    context_names=CONTEXT_LNR, 
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True, 
    standardize=STANDARDIZE
)


val_loader = PreloadedBatches(
    dset_val_LNR, 
    data_names=DATA, 
    context_names=CONTEXT_LNR, 
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=256, 
    shuffle=False, 
    standardize=STANDARDIZE, 
    scaler=train_loader._scaler
)

# %%
train_loss, val_loss, _, _ = train_model(
    conditioner_LNR,
    optimizer,
    train_metrics,
    val_metrics,
    train_loader,
    val_loader,
    batch_size_train=BATCH_SIZE_TRAIN,
    batch_size_val=256, 
    dset_train=dset_train_LNR, 
    dset_val=dset_val_LNR, 
    num_epochs=NUM_EPOCHS, 
    data_names=DATA, 
    context_names=CONTEXT_LNR, 
    target_names=TARGET, 
    response_name=RESPONSE,
    checkpoint_path="../../results/model_states/medium_budget/LNR",
    early_stopping=False,
)

# %%
with open("../../results/model_states/medium_budget/LNR/schedule_config.json", "w") as f:
    json.dump({
        "schedule_type": "warmup_cosine_decay",
        "init_value": 0,
        "peak_value": 1e-4,
        "warmup_steps": 0.1 * total_steps,
        "decay_steps": total_steps
    }, f)

save_obj(train_loader, "../../results/model_states/medium_budget/LNR/train_loader.pkl")

# %%
from flow.training import plot_loss
plot_loss(train_loss, val_loss)


# %%
def cond_factory():
    return make_mlp_conditioner(nnx.Rngs(0), dims=dims, num_bins=NUM_BINS, num_splines=NUM_SPLINES, use_norm=USE_NORM)

def opt_factory():
    # must match the optimizer used at save time (same optax tx, same hparams)
    return nnx.Optimizer(cond_factory(), optax.adamw(schedule), wrt=nnx.Param)

mgr = make_checkpoint_manager("../../results/model_states/medium_budget/LNR/")
conditioner_best, _ = restore_checkpoint(
    mgr, mgr.best_step(), cond_factory, opt_factory
)

# %%
from flow.training import evaluate_pdf_sf
from flow.config import spline_flow

TEST_PATH = Path('../../data/test/LNR/trials')

ds_test = ds.dataset(TEST_PATH, format="parquet").to_table()
np.random.seed(0)
n_test = ds_test.num_rows
idx = np.random.permutation(np.arange(n_test))

rt     = jnp.asarray(ds_test.column("rt").to_numpy(zero_copy_only=False))
choice = jnp.asarray(np.int32(ds_test.column("R").to_numpy(zero_copy_only=False)))  # 1 or 2
y      = jnp.asarray(ds_test.column("ll").to_numpy(zero_copy_only=False))

# Per-accumulator context, SAME column order as training: [v, B, t0, s, A]
def stack_ctx(suffix):
    cols = [f"m_lR{suffix}", f"s_lR{suffix}", "t0"]
    return jnp.asarray(np.stack(
        [ds_test.column(c).to_numpy(zero_copy_only=False) for c in cols], axis=1))

if train_loader._scaler is not None:
    scaler = train_loader._scaler
    ctx_1 = scaler.transform(stack_ctx(1))
    ctx_2 = scaler.transform(stack_ctx(2))
else:
    ctx_1 = stack_ctx(1)
    ctx_2 = stack_ctx(2)

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

from flow.training import plot_scatter_normalisation

pred_ll = jnp.maximum(pred_ll, jnp.log(1e-10))
plot_scatter_normalisation(y[idx[:50000]], pred_ll[idx[:50000]], z_test, num_bins=200)
print(f"Correlation: {jnp.corrcoef(y, pred_ll)[0, 1]:.4f}")
print(f"MSE(y, pred_ll) = {jnp.mean((y - pred_ll)**2):.4f}")

# %%
test_pd = ds_test.to_pandas()
test_pd["flow_ll_pred"] = pred_ll

SAVE_PATH = Path('../../results/LNR/medium')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

test_pd.to_csv(SAVE_PATH / "flow_ll_pred.csv", index=False)

# %% [markdown]
# ### Large budget

# %%
DATA = "rt" # The normflow is only trained on one accumulator
CONTEXT_LNR = ["m", "s", "t0"]
TARGET   = "ll"
TRAIN_PATH = Path('../../data/train/large/LNR_single/trials')
VAL_PATH = Path('../../data/val/LNR_single/trials')

dset_train_LNR = ds.dataset(TRAIN_PATH, format="parquet")
dset_val_LNR = ds.dataset(VAL_PATH, format="parquet")

# %%
NUM_BINS = 16
NUM_SPLINES = 1
NUM_EPOCHS = 200
BATCH_SIZE_TRAIN = 128
USE_NORM = False
STANDARDIZE = True

train_key = jax.random.key(100)
conditioner_key, sampling_key = jax.random.split(train_key, 2)
rngs = nnx.Rngs(default=conditioner_key, sampling=sampling_key)

dims = [len(CONTEXT_LNR), 16]

conditioner_LNR = make_mlp_conditioner(
    dims=dims,
    num_bins=NUM_BINS,
    num_splines=NUM_SPLINES,
    rngs=rngs,
    use_norm=USE_NORM,
)

# set up lr schedule
total_steps = (dset_train_LNR.count_rows() // BATCH_SIZE_TRAIN) * NUM_EPOCHS
# schedule = optax.cosine_decay_schedule(1e-5, decay_steps=total_steps)
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0., 
    peak_value=1e-4, 
    warmup_steps=0.1 * total_steps, 
    decay_steps=total_steps
)

optimizer = nnx.Optimizer(
    conditioner_LNR, optax.adamw(learning_rate=schedule), wrt=nnx.Param,
)
train_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)
val_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

train_loader = PreloadedBatches(
    dset_train_LNR, 
    data_names=DATA, 
    context_names=CONTEXT_LNR, 
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True, 
    standardize=STANDARDIZE
)

val_loader = PreloadedBatches(
    dset_val_LNR, 
    data_names=DATA, 
    context_names=CONTEXT_LNR, 
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=256, 
    shuffle=False, 
    standardize=STANDARDIZE, 
    scaler=train_loader._scaler
)

# %%
train_loss, val_loss, _, _ = train_model(
    conditioner_LNR,
    optimizer,
    train_metrics,
    val_metrics,
    train_loader,
    val_loader,
    batch_size_train=BATCH_SIZE_TRAIN,
    batch_size_val=256, 
    dset_train=dset_train_LNR, 
    dset_val=dset_val_LNR, 
    num_epochs=NUM_EPOCHS, 
    data_names=DATA, 
    context_names=CONTEXT_LNR, 
    target_names=TARGET, 
    response_name=RESPONSE,
    checkpoint_path="../../results/model_states/large_budget/LNR",
    early_stopping=False,
)

# %%
save_obj(train_loader, "../../results/model_states/large_budget/LNR/train_loader.pkl")
with open("../../results/model_states/large_budget/LNR/schedule_config.json", "w") as f:
    json.dump({
        "schedule_type": "warmup_cosine_decay",
        "init_value": 0,
        "peak_value": 1e-4,
        "warmup_steps": 0.1 * total_steps,
        "decay_steps": total_steps
    }, f)

# %%
from flow.training import plot_loss
plot_loss(train_loss, val_loss)


# %%
def cond_factory():
    return make_mlp_conditioner(nnx.Rngs(0), dims=dims, num_bins=NUM_BINS, num_splines=NUM_SPLINES, use_norm=USE_NORM)

def opt_factory():
    # must match the optimizer used at save time (same optax tx, same hparams)
    return nnx.Optimizer(cond_factory(), optax.adamw(schedule), wrt=nnx.Param)

mgr = make_checkpoint_manager("../../results/model_states/large_budget/LNR/")
conditioner_best, _ = restore_checkpoint(
    mgr, mgr.best_step(), cond_factory, opt_factory
)

# %%
from flow.training import evaluate_pdf_sf
from flow.config import spline_flow

TEST_PATH = Path('../../data/test/LNR/trials')

ds_test = ds.dataset(TEST_PATH, format="parquet").to_table()
n_test = ds_test.num_rows
idx = np.random.permutation(np.arange(n_test))

rt     = jnp.asarray(ds_test.column("rt").to_numpy(zero_copy_only=False))
choice = jnp.asarray(np.int32(ds_test.column("R").to_numpy(zero_copy_only=False)))  # 1 or 2
y      = jnp.asarray(ds_test.column("ll").to_numpy(zero_copy_only=False))

# Per-accumulator context, SAME column order as training: [v, B, t0, s, A]
def stack_ctx(suffix):
    cols = [f"m_lR{suffix}", f"s_lR{suffix}", "t0"]
    return jnp.asarray(np.stack(
        [ds_test.column(c).to_numpy(zero_copy_only=False) for c in cols], axis=1))

if train_loader._scaler is not None:
    scaler = train_loader._scaler
    ctx_1 = scaler.transform(stack_ctx(1))
    ctx_2 = scaler.transform(stack_ctx(2))
else:
    ctx_1 = stack_ctx(1)
    ctx_2 = stack_ctx(2)

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

from flow.training import plot_scatter_normalisation

pred_ll = jnp.maximum(pred_ll, jnp.log(1e-10))
plot_scatter_normalisation(y[idx[:50000]], pred_ll[idx[:50000]], z_test, num_bins=200)
print(f"Correlation: {jnp.corrcoef(y, pred_ll)[0, 1]:.4f}")
print(f"MSE(y, pred_ll) = {jnp.mean((y - pred_ll)**2):.4f}")

# %%
test_pd = ds_test.to_pandas()
test_pd["flow_ll_pred"] = pred_ll

SAVE_PATH = Path('../../results/LNR/large')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

test_pd.to_csv(SAVE_PATH / "flow_ll_pred.csv", index=False)

# %% [markdown]
# ## DDM
# ### Small budget

# %%
from flow.config import make_choice_classifier
from flow.training import train_model_classifier

# %%
DATA = "rt" # The normflow is only trained on one accumulator
CONTEXT_DDM = ["v", "a", "t0", "s", "Z", "SZ", "sv", "st0"]
TARGET   = "ll"
RESPONSE = "R"
TRAIN_PATH = Path('../../data/train/small/DDM/trials')
VAL_PATH = Path('../../data/val/DDM/trials')

dset_train_DDM = ds.dataset(TRAIN_PATH, format="parquet")
dset_val_DDM = ds.dataset(VAL_PATH, format="parquet")

# %%
dims_clf = [len(CONTEXT_DDM), 32, 16]
BATCH_SIZE_TRAIN = 128
STANDARDIZE = True
NUM_EPOCHS = 300

clf = make_choice_classifier(
    dims=dims_clf,
    rngs=nnx.Rngs(0)
)

optimizer_clf = nnx.Optimizer(
    clf, optax.adamw(learning_rate=1e-4), wrt=nnx.Param,
)

train_metrics_clf = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

val_metrics_clf = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

train_loader = PreloadedBatches(
    dset_train_DDM, 
    data_names=DATA, 
    context_names=CONTEXT_DDM + [RESPONSE],
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True, 
    standardize=STANDARDIZE
)

val_loader = PreloadedBatches(
    dset_val_DDM, 
    data_names=DATA, 
    context_names=CONTEXT_DDM + [RESPONSE],
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=256, 
    shuffle=False, 
    standardize=STANDARDIZE, 
    scaler=train_loader._scaler
)


# %%
train_loss, val_loss, _, _ = train_model_classifier(
    clf,
    optimizer_clf,
    train_metrics_clf,
    val_metrics_clf,
    train_loader,
    val_loader,
    batch_size_train=BATCH_SIZE_TRAIN,
    batch_size_val=256, 
    dset_train=dset_train_DDM, 
    dset_val=dset_val_DDM, 
    num_epochs=NUM_EPOCHS, 
    checkpoint_path="../../results/model_states/small_budget/DDM_classifier",
    early_stopping=False,
)

# %%
from flow.training import plot_loss
plot_loss(train_loss, val_loss)

# %%
save_obj(train_loader, "../../results/model_states/small_budget/DDM_classifier/train_loader.pkl")
with open("../../results/model_states/small_budget/DDM_classifier/schedule_config.json", "w") as f:
    json.dump({
        "schedule_type": "constant",
        "learning_rate": 1e-4
    }, f)

# %%
cond_factory_clf = lambda: make_choice_classifier(dims=dims_clf, rngs=nnx.Rngs(0))
opt_factory_clf = lambda: nnx.Optimizer(cond_factory_clf(), optax.adamw(1e-4), wrt=nnx.Param)
mgr_clf = make_checkpoint_manager("../../results/model_states/small_budget/DDM_classifier")
classifier_best, _ = restore_checkpoint(
    mgr_clf, mgr_clf.best_step(), cond_factory_clf, opt_factory_clf
)
print(mgr_clf.best_step())

# %%
state = nnx.state(classifier_best, nnx.Param)
total = 0
for path, leaf in jax.tree_util.tree_leaves_with_path(state):
    name = jax.tree_util.keystr(path)
    n = leaf.size
    total += n
    print(f"{name:55s} {str(leaf.shape):18s} {n:>10,}")
print(f"{'TOTAL':55s} {'':18s} {total:>10,}")

# %%
train_key = jax.random.key(100)
conditioner_key, sampling_key = jax.random.split(train_key, 2)
rngs = nnx.Rngs(default=conditioner_key, sampling=sampling_key)

dims = [len(CONTEXT_DDM + [RESPONSE]), 32, 16] # flow conditioner sees full context including response
NUM_BINS = 32
NUM_SPLINES = 1
BATCH_SIZE_TRAIN = 128
STANDARDIZE = True
USE_NORM = False
NUM_EPOCHS = 300


conditioner_DDM = make_mlp_conditioner(
    dims=dims,
    num_bins=NUM_BINS,
    num_splines=NUM_SPLINES,
    rngs=rngs,
    use_norm=USE_NORM,
)

# set up lr schedule
total_steps = (dset_train_DDM.count_rows() // BATCH_SIZE_TRAIN) * NUM_EPOCHS
# schedule = optax.cosine_decay_schedule(1e-5, decay_steps=total_steps)
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0., 
    peak_value=1e-4, 
    warmup_steps=0.1 * total_steps, 
    decay_steps=total_steps
)

optimizer_flow = nnx.Optimizer(
    conditioner_DDM, optax.adamw(learning_rate=schedule), wrt=nnx.Param,
)
train_metrics_flow = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)
val_metrics_flow = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

train_loader_flow = PreloadedBatches(
    dset_train_DDM, 
    data_names=DATA, 
    context_names=CONTEXT_DDM + [RESPONSE],
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True, 
    standardize=STANDARDIZE
)

val_loader_flow = PreloadedBatches(
    dset_val_DDM, 
    data_names=DATA, 
    context_names=CONTEXT_DDM + [RESPONSE],
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=256, 
    shuffle=False, 
    standardize=STANDARDIZE, 
    scaler=train_loader_flow._scaler
)

# %%
train_loss, val_loss, train_time, val_time = train_model(
    conditioner_DDM,
    optimizer_flow,
    train_metrics_flow,
    val_metrics_flow, 
    train_loader_flow,
    val_loader_flow,
    batch_size_train=BATCH_SIZE_TRAIN,
    batch_size_val=256, 
    dset_train=dset_train_DDM, 
    dset_val=dset_val_DDM, 
    num_epochs=NUM_EPOCHS, 
    data_names=DATA, 
    checkpoint_path="../../results/model_states/small_budget/DDM_flow", 
    early_stopping=False,
)

# %%
from flow.training import plot_loss
plot_loss(train_loss, val_loss)

# %%
save_obj(train_loader, "../../results/model_states/small_budget/DDM_flow/train_loader.pkl")
with open("../../results/model_states/small_budget/DDM_flow/schedule_config.json", "w") as f:
    json.dump({
        "schedule_type": "warmup_cosine_decay",
        "init_value": 0,
        "peak_value": 1e-4,
        "warmup_steps": 0.1 * total_steps,
        "decay_steps": total_steps
    }, f)

# %%
cond_factory_flow = lambda: make_mlp_conditioner(dims=dims, rngs=nnx.Rngs(0), num_bins=NUM_BINS, num_splines=NUM_SPLINES, use_norm=USE_NORM)
opt_factory_flow = lambda: nnx.Optimizer(cond_factory_flow(), optax.adamw(schedule), wrt=nnx.Param)
mgr_flow = make_checkpoint_manager("../../results/model_states/small_budget/DDM_flow")
flow_best, _ = restore_checkpoint(
    mgr_flow, mgr_flow.best_step(), cond_factory_flow, opt_factory_flow
)
print(mgr_flow.best_step())

# %%
state = nnx.state(flow_best, nnx.Param)
total = 0
for path, leaf in jax.tree_util.tree_leaves_with_path(state):
    name = jax.tree_util.keystr(path)
    n = leaf.size
    total += n
    print(f"{name:55s} {str(leaf.shape):18s} {n:>10,}")
print(f"{'TOTAL':55s} {'':18s} {total:>10,}")

# %%
from flow.config import joint_log_prob, spline_flow
from flow.training import plot_scatter_normalisation

TEST_PATH = Path('../../data/test/DDM/trials')
ds_test = ds.dataset(TEST_PATH, format="parquet").to_table()
n_test = ds_test.num_rows
idx_DDM = np.random.permutation(np.arange(n_test))

rt     = jnp.asarray(ds_test.column("rt").to_numpy(zero_copy_only=False))
response = jnp.asarray(np.int32(ds_test.column("R").to_numpy(zero_copy_only=False)))  # 1 or 2
y      = jnp.asarray(ds_test.column("ll").to_numpy(zero_copy_only=False))
ctx    = jnp.asarray(np.stack([ds_test.column(c).to_numpy(zero_copy_only=False) for c in CONTEXT_DDM], axis=1))

if train_loader_flow._scaler is not None:
    scaler = train_loader_flow._scaler
    ctx = scaler.transform(ctx)

jitted = nnx.jit(joint_log_prob)
pred_ll = jitted(flow_best, classifier_best, rt, ctx, response)
pred_ll = jnp.maximum(pred_ll, jnp.log(1e-10))


full_context = jnp.concatenate([ctx, jnp.expand_dims(response, -1)], axis=-1)
_, flow = spline_flow(rt, full_context, flow_best)
z_test = flow.bijector.inverse(rt)

# %%
plot_scatter_normalisation(y[idx_DDM[:50000]], pred_ll[idx_DDM[:50000]], z_test, num_bins=200)
print(f"Correlation: {jnp.corrcoef(y, pred_ll)[0, 1]:.4f}")
print(f"MSE(y, pred_ll) = {jnp.mean((y - pred_ll)**2):.4f}")

# %%
test_pd = ds_test.to_pandas()
test_pd["flow_ll_pred"] = pred_ll

SAVE_PATH = Path('../../results/DDM/small')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

test_pd.to_csv(SAVE_PATH / "flow_ll_pred.csv", index=False)

# %% [markdown]
# ### Medium budget

# %%
DATA = "rt" # The normflow is only trained on one accumulator
CONTEXT_DDM = ["v", "a", "t0", "s", "Z", "SZ", "sv", "st0"]
TARGET   = "ll"
RESPONSE = "R"
TRAIN_PATH = Path('../../data/train/medium/DDM/trials')
VAL_PATH = Path('../../data/val/DDM/trials')

dset_train_DDM = ds.dataset(TRAIN_PATH, format="parquet")
dset_val_DDM = ds.dataset(VAL_PATH, format="parquet")

# %%
dims_clf = [len(CONTEXT_DDM), 32, 16]
BATCH_SIZE_TRAIN = 128
STANDARDIZE = True
NUM_EPOCHS = 300

clf = make_choice_classifier(
    dims=dims_clf,
    rngs=nnx.Rngs(0)
)

optimizer_clf = nnx.Optimizer(
    clf, optax.adamw(learning_rate=1e-4), wrt=nnx.Param,
)

train_metrics_clf = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

val_metrics_clf = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

train_loader = PreloadedBatches(
    dset_train_DDM, 
    data_names=DATA, 
    context_names=CONTEXT_DDM + [RESPONSE],
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True, 
    standardize=STANDARDIZE
)

val_loader = PreloadedBatches(
    dset_val_DDM, 
    data_names=DATA, 
    context_names=CONTEXT_DDM + [RESPONSE],
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=256, 
    shuffle=False, 
    standardize=STANDARDIZE, 
    scaler=train_loader._scaler
)


# %%
train_loss, val_loss, _, _ = train_model_classifier(
    clf,
    optimizer_clf,
    train_metrics_clf,
    val_metrics_clf,
    train_loader,
    val_loader,
    batch_size_train=BATCH_SIZE_TRAIN,
    batch_size_val=256, 
    dset_train=dset_train_DDM, 
    dset_val=dset_val_DDM, 
    num_epochs=NUM_EPOCHS, 
    checkpoint_path="../../results/model_states/medium_budget/DDM_classifier",
    early_stopping=False,
)

# %%
plot_loss(train_loss, val_loss)

# %%
save_obj(train_loader, "../../results/model_states/medium_budget/DDM_classifier/train_loader.pkl")
with open("../../results/model_states/medium_budget/DDM_classifier/schedule_config.json", "w") as f:
    json.dump({
        "schedule_type": "constant",
        "learning_rate": 1e-4
    }, f)

# %%
cond_factory_clf = lambda: make_choice_classifier(dims=dims_clf, rngs=nnx.Rngs(0))
opt_factory_clf = lambda: nnx.Optimizer(cond_factory_clf(), optax.adamw(1e-4), wrt=nnx.Param)
mgr_clf = make_checkpoint_manager("../../results/model_states/medium_budget/DDM_classifier")
classifier_best, _ = restore_checkpoint(
    mgr_clf, mgr_clf.best_step(), cond_factory_clf, opt_factory_clf
)
print(mgr_clf.best_step())

# %%
train_key = jax.random.key(100)
conditioner_key, sampling_key = jax.random.split(train_key, 2)
rngs = nnx.Rngs(default=conditioner_key, sampling=sampling_key)

dims = [len(CONTEXT_DDM + [RESPONSE]), 32, 16] # flow conditioner sees full context including response
NUM_BINS = 32
NUM_SPLINES = 1
BATCH_SIZE_TRAIN = 128
STANDARDIZE = True
USE_NORM = False
NUM_EPOCHS = 300


conditioner_DDM = make_mlp_conditioner(
    dims=dims,
    num_bins=NUM_BINS,
    num_splines=NUM_SPLINES,
    rngs=rngs,
    use_norm=USE_NORM,
)

# set up lr schedule
total_steps = (dset_train_DDM.count_rows() // BATCH_SIZE_TRAIN) * NUM_EPOCHS
# schedule = optax.cosine_decay_schedule(1e-5, decay_steps=total_steps)
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0., 
    peak_value=1e-4, 
    warmup_steps=0.1 * total_steps, 
    decay_steps=total_steps
)

optimizer_flow = nnx.Optimizer(
    conditioner_DDM, optax.adamw(learning_rate=schedule), wrt=nnx.Param,
)
train_metrics_flow = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)
val_metrics_flow = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

train_loader_flow = PreloadedBatches(
    dset_train_DDM, 
    data_names=DATA, 
    context_names=CONTEXT_DDM + [RESPONSE],
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True, 
    standardize=STANDARDIZE
)

val_loader_flow = PreloadedBatches(
    dset_val_DDM, 
    data_names=DATA, 
    context_names=CONTEXT_DDM + [RESPONSE],
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=256, 
    shuffle=False, 
    standardize=STANDARDIZE, 
    scaler=train_loader_flow._scaler
)

# %%
train_loss, val_loss, train_time, val_time = train_model(
    conditioner_DDM,
    optimizer_flow,
    train_metrics_flow,
    val_metrics_flow, 
    train_loader_flow,
    val_loader_flow,
    batch_size_train=BATCH_SIZE_TRAIN,
    batch_size_val=256, 
    dset_train=dset_train_DDM, 
    dset_val=dset_val_DDM, 
    num_epochs=NUM_EPOCHS, 
    data_names=DATA, 
    checkpoint_path="../../results/model_states/medium_budget/DDM_flow", 
    early_stopping=False,
)

# %%
plot_loss(train_loss, val_loss)

# %%
save_obj(train_loader, "../../results/model_states/medium_budget/DDM_flow/train_loader.pkl")
with open("../../results/model_states/medium_budget/DDM_flow/schedule_config.json", "w") as f:
    json.dump({
        "schedule_type": "warmup_cosine_decay",
        "init_value": 0,
        "peak_value": 1e-4,
        "warmup_steps": 0.1 * total_steps,
        "decay_steps": total_steps
    }, f)

# %%
cond_factory_flow = lambda: make_mlp_conditioner(dims=dims, rngs=nnx.Rngs(0), num_bins=NUM_BINS, num_splines=NUM_SPLINES, use_norm=USE_NORM)
opt_factory_flow = lambda: nnx.Optimizer(cond_factory_flow(), optax.adamw(schedule), wrt=nnx.Param)
mgr_flow = make_checkpoint_manager("../../results/model_states/medium_budget/DDM_flow")
flow_best, _ = restore_checkpoint(
    mgr_flow, mgr_flow.best_step(), cond_factory_flow, opt_factory_flow
)
print(mgr_flow.best_step())

# %%
from flow.config import joint_log_prob, spline_flow
from flow.training import plot_scatter_normalisation

TEST_PATH = Path('../../data/test/DDM/trials')
ds_test = ds.dataset(TEST_PATH, format="parquet").to_table()
n_test = ds_test.num_rows
idx_DDM = np.random.permutation(np.arange(n_test))

rt     = jnp.asarray(ds_test.column("rt").to_numpy(zero_copy_only=False))
response = jnp.asarray(np.int32(ds_test.column("R").to_numpy(zero_copy_only=False)))  # 1 or 2
y      = jnp.asarray(ds_test.column("ll").to_numpy(zero_copy_only=False))
ctx    = jnp.asarray(np.stack([ds_test.column(c).to_numpy(zero_copy_only=False) for c in CONTEXT_DDM], axis=1))

if train_loader_flow._scaler is not None:
    scaler = train_loader_flow._scaler
    ctx = scaler.transform(ctx)

jitted = nnx.jit(joint_log_prob)
pred_ll = jitted(flow_best, classifier_best, rt, ctx, response)
pred_ll = jnp.maximum(pred_ll, jnp.log(1e-10))


full_context = jnp.concatenate([ctx, jnp.expand_dims(response, -1)], axis=-1)
_, flow = spline_flow(rt, full_context, flow_best)
z_test = flow.bijector.inverse(rt)

# %%
plot_scatter_normalisation(y[idx_DDM[:50000]], pred_ll[idx_DDM[:50000]], z_test, num_bins=200)
print(f"Correlation: {jnp.corrcoef(y, pred_ll)[0, 1]:.4f}")
print(f"MSE(y, pred_ll) = {jnp.mean((y - pred_ll)**2):.4f}")

# %%
test_pd = ds_test.to_pandas()
test_pd["flow_ll_pred"] = pred_ll

SAVE_PATH = Path('../../results/DDM/medium')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

test_pd.to_csv(SAVE_PATH / "flow_ll_pred.csv", index=False)

# %% [markdown]
# ### Large budget

# %%
DATA = "rt" # The normflow is only trained on one accumulator
CONTEXT_DDM = ["v", "a", "t0", "s", "Z", "SZ", "sv", "st0"]
TARGET   = "ll"
RESPONSE = "R"
TRAIN_PATH = Path('../../data/train/large/DDM/trials')
VAL_PATH = Path('../../data/val/DDM/trials')

dset_train_DDM = ds.dataset(TRAIN_PATH, format="parquet")
dset_val_DDM = ds.dataset(VAL_PATH, format="parquet")

# %%
dims_clf = [len(CONTEXT_DDM), 32, 16]
BATCH_SIZE_TRAIN = 128
STANDARDIZE = True
NUM_EPOCHS = 300

clf = make_choice_classifier(
    dims=dims_clf,
    rngs=nnx.Rngs(0)
)

optimizer_clf = nnx.Optimizer(
    clf, optax.adamw(learning_rate=1e-4), wrt=nnx.Param,
)

train_metrics_clf = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

val_metrics_clf = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

train_loader = PreloadedBatches(
    dset_train_DDM, 
    data_names=DATA, 
    context_names=CONTEXT_DDM + [RESPONSE],
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True, 
    standardize=STANDARDIZE
)

val_loader = PreloadedBatches(
    dset_val_DDM, 
    data_names=DATA, 
    context_names=CONTEXT_DDM + [RESPONSE],
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=256, 
    shuffle=False, 
    standardize=STANDARDIZE, 
    scaler=train_loader._scaler
)


# %%
train_loss, val_loss, _, _ = train_model_classifier(
    clf,
    optimizer_clf,
    train_metrics_clf,
    val_metrics_clf,
    train_loader,
    val_loader,
    batch_size_train=BATCH_SIZE_TRAIN,
    batch_size_val=256, 
    dset_train=dset_train_DDM, 
    dset_val=dset_val_DDM, 
    num_epochs=NUM_EPOCHS, 
    checkpoint_path="../../results/model_states/large_budget/DDM_classifier",
    early_stopping=False,
)

# %%
with open("../../results/model_states/large_budget/DDM_classifier/training_log.json", "r") as f:
    training_log = json.load(f)
plot_loss(training_log['train_losses'], training_log['val_losses'])

# %%
save_obj(train_loader, "../../results/model_states/large_budget/DDM_classifier/train_loader.pkl")
with open("../../results/model_states/large_budget/DDM_classifier/schedule_config.json", "w") as f:
    json.dump({
        "schedule_type": "constant",
        "learning_rate": 1e-4
    }, f)

# %%
cond_factory_clf = lambda: make_choice_classifier(dims=dims_clf, rngs=nnx.Rngs(0))
opt_factory_clf = lambda: nnx.Optimizer(cond_factory_clf(), optax.adamw(1e-4), wrt=nnx.Param)
mgr_clf = make_checkpoint_manager("../../results/model_states/large_budget/DDM_classifier")
classifier_best, _ = restore_checkpoint(
    mgr_clf, mgr_clf.best_step(), cond_factory_clf, opt_factory_clf
)
print(mgr_clf.best_step())

# %%
train_key = jax.random.key(100)
conditioner_key, sampling_key = jax.random.split(train_key, 2)
rngs = nnx.Rngs(default=conditioner_key, sampling=sampling_key)

dims = [len(CONTEXT_DDM + [RESPONSE]), 32, 16] # flow conditioner sees full context including response
NUM_BINS = 32
NUM_SPLINES = 1
BATCH_SIZE_TRAIN = 128
STANDARDIZE = True
USE_NORM = False
NUM_EPOCHS = 300


conditioner_DDM = make_mlp_conditioner(
    dims=dims,
    num_bins=NUM_BINS,
    num_splines=NUM_SPLINES,
    rngs=rngs,
    use_norm=USE_NORM,
)

# set up lr schedule
total_steps = (dset_train_DDM.count_rows() // BATCH_SIZE_TRAIN) * NUM_EPOCHS
# schedule = optax.cosine_decay_schedule(1e-5, decay_steps=total_steps)
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0., 
    peak_value=1e-4, 
    warmup_steps=0.1 * total_steps, 
    decay_steps=total_steps
)

optimizer_flow = nnx.Optimizer(
    conditioner_DDM, optax.adamw(learning_rate=schedule), wrt=nnx.Param,
)
train_metrics_flow = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)
val_metrics_flow = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

train_loader_flow = PreloadedBatches(
    dset_train_DDM, 
    data_names=DATA, 
    context_names=CONTEXT_DDM + [RESPONSE],
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True, 
    standardize=STANDARDIZE
)

val_loader_flow = PreloadedBatches(
    dset_val_DDM, 
    data_names=DATA, 
    context_names=CONTEXT_DDM + [RESPONSE],
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=256, 
    shuffle=False, 
    standardize=STANDARDIZE, 
    scaler=train_loader_flow._scaler
)

# %%
train_loss, val_loss, train_time, val_time = train_model(
    conditioner_DDM,
    optimizer_flow,
    train_metrics_flow,
    val_metrics_flow, 
    train_loader_flow,
    val_loader_flow,
    batch_size_train=BATCH_SIZE_TRAIN,
    batch_size_val=256, 
    dset_train=dset_train_DDM, 
    dset_val=dset_val_DDM, 
    num_epochs=NUM_EPOCHS, 
    data_names=DATA, 
    checkpoint_path="../../results/model_states/large_budget/DDM_flow", 
    early_stopping=True,
    early_stopping_patience=25,
    early_stopping_min_delta=1e-3,
)

# %%
from flow.training import plot_loss
plot_loss(train_loss, val_loss)

# %%
save_obj(train_loader, "../../results/model_states/large_budget/DDM_flow/train_loader.pkl")
with open("../../results/model_states/large_budget/DDM_flow/schedule_config.json", "w") as f:
    json.dump({
        "schedule_type": "warmup_cosine_decay",
        "init_value": 0,
        "peak_value": 1e-4,
        "warmup_steps": 0.1 * total_steps,
        "decay_steps": total_steps
    }, f)

# %%
cond_factory_flow = lambda: make_mlp_conditioner(dims=dims, rngs=nnx.Rngs(0), num_bins=NUM_BINS, num_splines=NUM_SPLINES, use_norm=USE_NORM)
opt_factory_flow = lambda: nnx.Optimizer(cond_factory_flow(), optax.adamw(schedule), wrt=nnx.Param)
mgr_flow = make_checkpoint_manager("../../results/model_states/large_budget/DDM_flow")
flow_best, _ = restore_checkpoint(
    mgr_flow, mgr_flow.best_step(), cond_factory_flow, opt_factory_flow
)
print(mgr_flow.best_step())

# %%
from flow.config import joint_log_prob, spline_flow
from flow.training import plot_scatter_normalisation

TEST_PATH = Path('../../data/test/DDM/trials')
ds_test = ds.dataset(TEST_PATH, format="parquet").to_table()
n_test = ds_test.num_rows
idx_DDM = np.random.permutation(np.arange(n_test))

rt     = jnp.asarray(ds_test.column("rt").to_numpy(zero_copy_only=False))
response = jnp.asarray(np.int32(ds_test.column("R").to_numpy(zero_copy_only=False)))  # 1 or 2
y      = jnp.asarray(ds_test.column("ll").to_numpy(zero_copy_only=False))
ctx    = jnp.asarray(np.stack([ds_test.column(c).to_numpy(zero_copy_only=False) for c in CONTEXT_DDM], axis=1))

if train_loader_flow._scaler is not None:
    scaler = train_loader_flow._scaler
    ctx = scaler.transform(ctx)

jitted = nnx.jit(joint_log_prob)
pred_ll = jitted(flow_best, classifier_best, rt, ctx, response)
pred_ll = jnp.maximum(pred_ll, jnp.log(1e-10))


full_context = jnp.concatenate([ctx, jnp.expand_dims(response, -1)], axis=-1)
_, flow = spline_flow(rt, full_context, flow_best)
z_test = flow.bijector.inverse(rt)

# %%
plot_scatter_normalisation(y[idx_DDM[:50000]], pred_ll[idx_DDM[:50000]], z_test, num_bins=200)
print(f"Correlation: {jnp.corrcoef(y, pred_ll)[0, 1]:.4f}")
print(f"MSE(y, pred_ll) = {jnp.mean((y - pred_ll)**2):.4f}")

# %%
test_pd = ds_test.to_pandas()
test_pd["flow_ll_pred"] = pred_ll

SAVE_PATH = Path('../../results/DDM/large')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

test_pd.to_csv(SAVE_PATH / "flow_ll_pred.csv", index=False)

# %% [markdown]
# # OLD CODE !!!

# %% [markdown]
# ### Trying out a simpler model

# %%
DATA = "rt" # The normflow is only trained on one accumulator
CONTEXT_DDM = ["v", "a", "t0", "s", "Z", "SZ", "sv", "st0"]
TARGET   = "ll"
RESPONSE = "R"
TRAIN_PATH = Path('../../data/train/large/DDM/trials')
VAL_PATH = Path('../../data/val/DDM/trials')

dset_train_DDM = ds.dataset(TRAIN_PATH, format="parquet")
dset_val_DDM = ds.dataset(VAL_PATH, format="parquet")

# %% [markdown]
# Classifier stays as is for now

# %%
from flow.config import make_choice_classifier

dims_clf = [len(CONTEXT_DDM), 32, 16]
BATCH_SIZE_TRAIN = 128
STANDARDIZE = True
NUM_EPOCHS = 500

clf = make_choice_classifier(
    dims=dims_clf,
    rngs=nnx.Rngs(0)
)

optimizer_clf = nnx.Optimizer(
    clf, optax.adamw(learning_rate=1e-4), wrt=nnx.Param,
)

train_metrics_clf = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

val_metrics_clf = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

train_loader = PreloadedBatches(
    dset_train_DDM, 
    data_names=DATA, 
    context_names=CONTEXT_DDM + [RESPONSE],
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True, 
    standardize=STANDARDIZE
)

val_loader = PreloadedBatches(
    dset_val_DDM, 
    data_names=DATA, 
    context_names=CONTEXT_DDM + [RESPONSE],
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=256, 
    shuffle=False, 
    standardize=STANDARDIZE, 
    scaler=train_loader._scaler
)



cond_factory_clf = lambda: make_choice_classifier(dims=dims_clf, rngs=nnx.Rngs(0))
opt_factory_clf = lambda: nnx.Optimizer(cond_factory_clf(), optax.adamw(1e-4), wrt=nnx.Param)
mgr_clf = make_checkpoint_manager("../../results/model_states/large_budget/DDM_classifier")
classifier_best, _ = restore_checkpoint(
    mgr_clf, mgr_clf.best_step(), cond_factory_clf, opt_factory_clf
)
print(mgr_clf.best_step())

# %%
train_key = jax.random.key(100)
conditioner_key, sampling_key = jax.random.split(train_key, 2)
rngs = nnx.Rngs(default=conditioner_key, sampling=sampling_key)

dims = [len(CONTEXT_DDM + [RESPONSE]), 32, 16] # flow conditioner sees full context including response
NUM_BINS = 32
NUM_SPLINES = 1
BATCH_SIZE_TRAIN = 128
STANDARDIZE = True
USE_NORM = False
NUM_EPOCHS = 150


conditioner_DDM = make_mlp_conditioner(
    dims=dims,
    num_bins=NUM_BINS,
    num_splines=NUM_SPLINES,
    rngs=rngs,
    use_norm=USE_NORM,
)

# set up lr schedule
total_steps = (dset_train_DDM.count_rows() // BATCH_SIZE_TRAIN) * NUM_EPOCHS
# schedule = optax.cosine_decay_schedule(1e-5, decay_steps=total_steps)
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0., 
    peak_value=1e-4, 
    warmup_steps=0.1 * total_steps, 
    decay_steps=total_steps
)

optimizer_flow = nnx.Optimizer(
    conditioner_DDM, optax.adamw(learning_rate=schedule), wrt=nnx.Param,
)
train_metrics_flow = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)
val_metrics_flow = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)

train_loader_flow = PreloadedBatches(
    dset_train_DDM, 
    data_names=DATA, 
    context_names=CONTEXT_DDM + [RESPONSE],
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True, 
    standardize=STANDARDIZE
)

val_loader_flow = PreloadedBatches(
    dset_val_DDM, 
    data_names=DATA, 
    context_names=CONTEXT_DDM + [RESPONSE],
    target_names=TARGET, 
    response_name=RESPONSE,
    batch_size=256, 
    shuffle=False, 
    standardize=STANDARDIZE, 
    scaler=train_loader_flow._scaler
)
    

# %%
train_loss, val_loss, train_time, val_time = train_model(
    conditioner_DDM,
    optimizer_flow,
    train_metrics_flow,
    val_metrics_flow, 
    train_loader_flow,
    val_loader_flow,
    batch_size_train=BATCH_SIZE_TRAIN,
    batch_size_val=256, 
    dset_train=dset_train_DDM, 
    dset_val=dset_val_DDM, 
    num_epochs=NUM_EPOCHS, 
    data_names=DATA, 
    checkpoint_path="../../results/ddm_trials/model_states/large_budget/DDM_flow", 
    early_stopping=False,
    early_stopping_patience=25,
    early_stopping_min_delta=1e-3,
)

# %%
from flow.training import plot_loss
plot_loss(train_loss, val_loss)

# %%
save_obj(train_loader, "../../results/ddm_trials/model_states/large_budget/DDM_flow/train_loader.pkl")
with open("../../results/ddm_trials/model_states/large_budget/DDM_flow/schedule_config.json", "w") as f:
    json.dump({
        "schedule_type": "warmup_cosine_decay",
        "init_value": 0,
        "peak_value": 1e-4,
        "warmup_steps": 0.1 * total_steps,
        "decay_steps": total_steps
    }, f)

# %%
cond_factory_flow = lambda: make_mlp_conditioner(dims=dims, rngs=nnx.Rngs(0), num_bins=NUM_BINS, num_splines=NUM_SPLINES, use_norm=USE_NORM)
opt_factory_flow = lambda: nnx.Optimizer(cond_factory_flow(), optax.adamw(schedule), wrt=nnx.Param)
mgr_flow = make_checkpoint_manager("../../results/ddm_trials/model_states/large_budget/DDM_flow")
flow_best, _ = restore_checkpoint(
    mgr_flow, mgr_flow.best_step(), cond_factory_flow, opt_factory_flow
)
print(mgr_flow.best_step())

# %%
from flow.config import joint_log_prob, spline_flow
from flow.training import plot_scatter_normalisation

TEST_PATH = Path('../../data/test/DDM/trials')
ds_test = ds.dataset(TEST_PATH, format="parquet").to_table()
n_test = ds_test.num_rows
idx_DDM = np.random.permutation(np.arange(n_test))

rt     = jnp.asarray(ds_test.column("rt").to_numpy(zero_copy_only=False))
response = jnp.asarray(np.int32(ds_test.column("R").to_numpy(zero_copy_only=False)))  # 1 or 2
y      = jnp.asarray(ds_test.column("ll").to_numpy(zero_copy_only=False))
ctx    = jnp.asarray(np.stack([ds_test.column(c).to_numpy(zero_copy_only=False) for c in CONTEXT_DDM], axis=1))

if train_loader_flow._scaler is not None:
    scaler = train_loader_flow._scaler
    ctx = scaler.transform(ctx)

jitted = nnx.jit(joint_log_prob)
pred_ll = jitted(flow_best, classifier_best, rt, ctx, response)
pred_ll = jnp.maximum(pred_ll, jnp.log(1e-10))


full_context = jnp.concatenate([ctx, jnp.expand_dims(response, -1)], axis=-1)
_, flow = spline_flow(rt, full_context, flow_best)
z_test = flow.bijector.inverse(rt)

# %%
plot_scatter_normalisation(y[idx_DDM[:50000]], pred_ll[idx_DDM[:50000]], z_test, num_bins=200)
print(f"Correlation: {jnp.corrcoef(y, pred_ll)[0, 1]:.4f}")
print(f"MSE(y, pred_ll) = {jnp.mean((y - pred_ll)**2):.4f}")

# %% [markdown]
# #### RT hist density overlay

# %%
import pyarrow.compute as pc
from flow.training import load_obj

train_loader = load_obj("../../results/ddm_trials/model_states/large_budget/DDM_flow/train_loader.pkl")

expr = pc.field("param_id") == 482

pd_dat = ds_test.filter(expr).to_pandas()
pd_dat["R"] = pd_dat["R"].astype(int)
r_1 = pd_dat[pd_dat["R"] == 1]["rt"] #- np.exp(pd_dat[pd_dat["R"] == 1]["t0"])
r_2 = pd_dat[pd_dat["R"] == 2]["rt"] #- np.exp(pd_dat[pd_dat["R"] == 2]["t0"])
ll_1 = pd_dat[pd_dat["R"] == 1]["ll"]
ll_2 = pd_dat[pd_dat["R"] == 2]["ll"]
p_r1 = len(r_1) / len(pd_dat)
p_r2 = len(r_2) / len(pd_dat)

xs = jnp.linspace(1e-10, 15, 500)
xs = jnp.concatenate([xs, xs])
rs = jnp.repeat(jnp.array([1, 2]), repeats=500)
rs_2 = jnp.repeat(jnp.array([2]), repeats=len(r_2))
rs_1 = jnp.repeat(jnp.array([1]), repeats=len(r_1))
params = pd_dat.loc[0, CONTEXT_DDM].to_numpy().reshape(1, len(CONTEXT_DDM))

if train_loader._scaler is not None:
    scaler = train_loader._scaler
    params = scaler.transform(params)
ctx = jnp.repeat(params, repeats=1000, axis=0)
ctx_2 = jnp.repeat(params, repeats=len(r_2), axis=0)
ctx_1 = jnp.repeat(params, repeats=len(r_1), axis=0)

# %%
jitted = nnx.jit(joint_log_prob)
pred_ll = jitted(flow_best, classifier_best, xs, ctx, rs)
pred_ll_rt = jitted(flow_best, classifier_best, r_2.to_numpy(), ctx_2, rs_2)
pred_ll_rt_1 = jitted(flow_best, classifier_best, r_1.to_numpy(), ctx_1, rs_1)

# %%
import matplotlib.pyplot as plt
plt.plot(xs[500:1000], jnp.exp(pred_ll[500:1000]) / p_r2, label="R=2")
plt.plot(r_2, jnp.exp(pred_ll_rt) / p_r2, ".", alpha=0.5)
#plt.plot(r_2, np.exp(ll_2) / p_r2, ".", alpha=0.5)
plt.hist(r_2, bins=50, density=True, alpha=0.5)
plt.xlim(min(r_2) - 0.1, max(r_2) + 0.1)
plt.show()
plt.close()

plt.plot(xs[:500], jnp.exp(pred_ll[:500]) / p_r1, label="R=1")
plt.hist(r_1, bins=50, density=True, alpha=0.5)
#plt.plot(r_1, jnp.exp(pred_ll_rt_1) / p_r1, ".", alpha=0.5)
plt.plot(r_1, np.exp(ll_1) / p_r1, ".", alpha=0.5)
plt.xlim(min(r_1) - 0.1, max(r_1) + 0.1)
plt.show()

# %%
import importlib
import flow.training
importlib.reload(flow.training)
from flow.training import plot_rt_overlay

# %%
max_idx = np.int32(ds_test.column("param_id").to_numpy().max()) + 1
param_id = np.random.randint(0, max_idx)
plot_rt_overlay(ds_test, param_id, CONTEXT_DDM, flow_best, classifier_best, train_loader._scaler, precision=5000, eval_at_data=False)

# %% [markdown]
# # DDM training with separate classifier

# %%
DATA = "rt" # The normflow is only trained on one accumulator
CONTEXT_DDM = ["v", "a", "t0", "s", "Z", "SZ", "sv", "st0"]
RESPONSE = "R"
STIMULUS = "S"
TARGET   = "ll"
TRAIN_PATH = Path('../data/train/small/DDM/trials')

VAL_PATH = Path('../data/train/small/DDM/trials')

dset_train_DDM = ds.dataset(TRAIN_PATH, format="parquet")
dset_val_DDM = ds.dataset(VAL_PATH, format="parquet")

# %%
import importlib
import flow.config as flow_config
import flow.training as training
importlib.reload(training)
importlib.reload(flow_config)
from flow.config import make_mlp_conditioner, MLP, make_choice_classifier, joint_log_prob, spline_flow
from flow.training import train_model_classifier, train_model

# %%
# build models
NUM_PARAMS = len(CONTEXT_DDM)         # without R

# 1. context_names now includes both S and R, in this order:
context_names = CONTEXT_DDM + [STIMULUS, RESPONSE]   # e.g. CONTEXT_DDM + ["S", "R"]

# 2. Bump conditioner input dim by 1 (was NUM_PARAMS + 1, now + 2):
conditioner = make_mlp_conditioner(
    nnx.Rngs(0),
    dims=[NUM_PARAMS + 2, 256, 256],   # + S + R
    num_bins=32, num_splines=5,
)

# 3. Classifier input dim also up by 1 (was NUM_PARAMS, now + 1 for S):
classifier = make_choice_classifier(
    nnx.Rngs(1),
    dims=[NUM_PARAMS + 1, 64, 64],     # + S
)


# %%
# build optimizers (separate)
opt_flow = nnx.Optimizer(conditioner,
    optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3)))
opt_clf  = nnx.Optimizer(classifier,
    optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3)))

# metrics
train_metrics_flow = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
val_metrics_flow  = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
train_metrics_clf = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
val_metrics_clf = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

# %%
# 1. classifier first
clf_train, clf_val, *_ = train_model_classifier(
    classifier, opt_clf, train_metrics_clf, val_metrics_clf,
    batches, dset_train_DDM, dset_val_DDM,
    num_epochs=20,
    data_names=DATA, context_names=context_names, target_names=TARGET, response_name=RESPONSE,
    checkpoint_path="model_states/DDM/clf", sign=False,
)

# %%
train_loss, val_loss, *_ = train_model(
    conditioner, opt_flow, train_metrics_flow, val_metrics_flow,
    batches, dset_train_DDM, dset_val_DDM,
    num_epochs=20,
    data_names=DATA, context_names=context_names, target_names=TARGET, response_name=RESPONSE,
    checkpoint_path="model_states/DDM/flow", sign=False,
)

# %%
mgr_flow = make_checkpoint_manager("../server_training/model_states/DDM/flow")

def cond_factory():
    return make_mlp_conditioner(
        nnx.Rngs(0),
        dims=[NUM_PARAMS + 2, 256, 256],   # + S + R
        num_bins=32, num_splines=5,
    )

def opt_factory():
    return nnx.Optimizer(
        cond_factory(),
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3)),
    )


mgr_clf = make_checkpoint_manager("../server_training/model_states/DDM/clf")

def clf_factory():
    return make_choice_classifier(
        nnx.Rngs(1),
        dims=[NUM_PARAMS + 1, 64, 64],     # + S
    )

def clf_opt_factory():
    return nnx.Optimizer(
        clf_factory(),
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3)),
    )





# %%
import jax
from jax.sharding import SingleDeviceSharding
from flax import nnx
import orbax.checkpoint as ocp

cpu_sharding = SingleDeviceSharding(jax.devices("cpu")[0])

def _with_sharding(x):
    if isinstance(x, jax.ShapeDtypeStruct):
        return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=cpu_sharding)
    return x

def restore_on_cpu(mgr, step, cond_factory, opt_factory):
    abs_cond = nnx.eval_shape(cond_factory)
    cond_gd, abs_cond_state = nnx.split(abs_cond)
    abs_cond_state = jax.tree.map(_with_sharding, abs_cond_state)

    abs_opt = nnx.eval_shape(opt_factory)
    opt_gd, abs_opt_state = nnx.split(abs_opt)
    abs_opt_state = jax.tree.map(_with_sharding, abs_opt_state)

    restored = mgr.restore(
        step,
        args=ocp.args.Composite(
            conditioner=ocp.args.StandardRestore(abs_cond_state),
            optimizer=ocp.args.StandardRestore(abs_opt_state),
        ),
    )
    return nnx.merge(cond_gd, restored["conditioner"]), nnx.merge(opt_gd, restored["optimizer"])

conditioner, opt_flow = restore_on_cpu(
    mgr_flow, mgr_flow.best_step(), cond_factory, opt_factory
)


classifier, opt_clf = restore_on_cpu(
    mgr_clf, mgr_clf.best_step(),
    clf_factory, clf_opt_factory,
)

# %%
from flow.training import plot_scatter_normalisation

TEST_PATH = Path('../data/test/DDM/trials')

ds_test = ds.dataset(TEST_PATH, format="parquet").to_table()
row_idx = np.random.permutation(np.arange(ds_test.num_rows))[:200000]
ds_test = ds_test.take(row_idx)

rt       = jnp.asarray(ds_test.column("rt").to_numpy(zero_copy_only=False))
response = jnp.asarray(np.float32(ds_test.column("R").to_numpy(zero_copy_only=False)))  # 1 or 2
stimulus = jnp.asarray(np.float32(ds_test.column("S").to_numpy(zero_copy_only=False)))  # 1 or 2
y        = jnp.asarray(ds_test.column("ll").to_numpy(zero_copy_only=False))
params   = jnp.asarray(np.stack([ds_test.column(c).to_numpy(zero_copy_only=False).astype(np.float32) 
                                 for c in CONTEXT_DDM], axis=1))


pred_ll = joint_log_prob(conditioner, classifier, rt, params, stimulus, response)


full_context = jnp.concatenate([params, jnp.expand_dims(stimulus, -1), jnp.expand_dims(response, -1)], axis=-1)

pred_ll = jnp.maximum(pred_ll, jnp.log(1e-10))

_, flow = spline_flow(rt, full_context, conditioner)
z_test = flow.bijector.inverse(rt)
plot_scatter_normalisation(y, pred_ll, z_test, num_bins=200)

# %% [markdown]
# Finding trials within the stripe

# %%
strip_idx = jnp.where((y < -5) & ( (pred_ll < 2) & (pred_ll > -3) ))[0]

strip_dat = jnp.concat([
    jnp.expand_dims(y[strip_idx], axis=1),
    jnp.expand_dims(pred_ll[strip_idx], axis=1),
    jnp.expand_dims(rt[strip_idx], axis=1),
    params[strip_idx],
    jnp.expand_dims(stimulus[strip_idx], axis=1),
    jnp.expand_dims(response[strip_idx], axis=1),
], axis=1)

import pandas as pd

strip_df = pd.DataFrame(strip_dat, columns=["EMC2_ll", "pred_ll", "rt"] + CONTEXT_DDM + ["S", "R"])

strip_df.to_csv("strip_df.csv", index=False)

# %% [markdown]
# # Trying out some code

# %%
batches_DDM = batches(dset_train_DDM, data_names=DATA, context_names=CONTEXT_DDM_RESPONSE, target_names=TARGET, response_name=RESPONSE)

_, params_test, _ = next(iter(batches_DDM))

# %%
(params_test[:, -1] == 2.0).astype(jnp.float32)

# %%
test_out = classifier(params_test[:, :-1])  # Exclude the response column for the classifier input
print(test_out)

optax.sigmoid_binary_cross_entropy(test_out, (params_test[:, -1] == 2.0).astype(jnp.float32))

# %%
print(jnp.all(jax.nn.log_sigmoid(test_out) == -jax.nn.softplus(-test_out)))

print(jnp.allclose(jnp.log(1-jax.nn.sigmoid(test_out)), -jax.nn.softplus(test_out)))

# %%
1-jax.nn.sigmoid(test_out)

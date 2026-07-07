import os
import json
import time
import pickle
import jax
import jax.scipy.stats as jstats
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from flax.training.early_stopping import EarlyStopping
import pyarrow.compute as pc

from .config import spline_flow, joint_log_prob
from .data import batches, sign_rts, PreloadedBatches
from .checkpoints import make_checkpoint_manager, save_checkpoint, restore_checkpoint


def loss_fn(conditioner, data, context, min_log_prob: float = 1e-12):
  data_flat = data.squeeze()

  is_valid = jnp.isfinite(data_flat) & (data_flat > 0.0)

  safe_data = jnp.where(is_valid, data_flat, 1.0)

  safe_log_probs, _ = conditioner.spline_flow(safe_data, context, conditioner)

  final_log_probs = jnp.where(is_valid, safe_log_probs, jnp.log(min_log_prob))
  
  return -jnp.mean(final_log_probs)

@nnx.jit
def train_step(conditioner, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, data, context):
  grad_fn = nnx.value_and_grad(loss_fn)
  loss, grads = grad_fn(conditioner, data, context)
  metrics.update(loss=loss)  # In-place updates.
  optimizer.update(grads)  # In-place updates.

def eval_step(conditioner, metrics: nnx.MultiMetric, data, context):
  loss = loss_fn(conditioner, data, context)
  metrics.update(loss=loss)


def train_one_epoch(conditioner, optimizer, metrics, batches, dset_train, batch_size=8192):
    pbar = tqdm(batches, total = dset_train.count_rows() // batch_size, desc="Training", unit="batches") 
    conditioner.train()
    metrics.reset()

    for batch, (data, context, y) in enumerate(pbar):
        train_step(conditioner, optimizer, metrics, data, context)
        
        if (batch + 1) % 10 == 0:
            avg_loss = float(metrics.compute()["loss"])
            pbar.set_postfix({"Avg. Loss": f"{avg_loss:.4f}"})
    
    return avg_loss

def validate_model(conditioner, metrics, batches, dset_val, batch_size=8192*2):
    pbar = tqdm(batches, total = dset_val.count_rows() // batch_size, desc="Validating", unit="batches") 
    conditioner.eval()
    metrics.reset()

    for batch, (data, context, y) in enumerate(pbar):
        eval_step(conditioner, metrics, data, context)
        
        if (batch + 1) % 10 == 0:
            avg_val_loss = float(metrics.compute()["loss"])
            pbar.set_postfix({"Avg. Val Loss": f"{avg_val_loss:.4f}"})
    
    return avg_val_loss


def train_model(
        conditioner,
        optimizer,
        train_metrics,
        val_metrics,
        train_loader,
        val_loader,
        dset_train,
        dset_val,
        batch_size_train=8192,
        batch_size_val=8192*2,
        num_epochs=10,
        start_at_epoch=0,
        checkpoint_path=None,
        early_stopping = True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        *args,
        **kwargs
):
    train_losses = []
    val_losses = []
    train_times = []
    val_times = []

    mgr = make_checkpoint_manager(checkpoint_path, clear_dir=True) if checkpoint_path else None

    if early_stopping:
        early_stop = EarlyStopping(min_delta=early_stopping_min_delta, patience=early_stopping_patience)  # Example: stop if val_loss doesn't improve by at least 0.001 for 3 epochs

    try:
        for epoch in range(start_at_epoch, num_epochs + start_at_epoch):
            print(f"Epoch {epoch + 1}/{num_epochs + start_at_epoch}")

            t0 = time.perf_counter()
            train_loss = train_one_epoch(
                conditioner,
                optimizer,
                train_metrics,
                train_loader,
                dset_train,
                batch_size=batch_size_train
            )
            t1 = time.perf_counter()
            val_loss = validate_model(
                conditioner,
                val_metrics,
                val_loader,
                dset_val,
                batch_size=batch_size_val
            )
            t2 = time.perf_counter()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_times.append(t1 - t0)
            val_times.append(t2 - t1)

            print(f"  train {t1 - t0:.2f}s | val {t2 - t1:.2f}s")

            if not np.logical_and(np.isfinite(train_loss), np.isfinite(val_loss)):
                print("Loss is not finite. Terminating training...")
                break

            if mgr is not None:
                save_checkpoint(mgr, epoch, conditioner, optimizer, val_loss)

            if checkpoint_path is not None:
                log_path = os.path.join(os.path.abspath(checkpoint_path), "training_log.json")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "w") as f:
                    json.dump({
                        "train_losses": [float(x) for x in train_losses],
                        "val_losses": [float(x) for x in val_losses],
                        "train_times_s": train_times,
                        "val_times_s": val_times,
                        "total_time_s": sum(train_times) + sum(val_times),
                        "compile_overhead_est_s": train_times[0] - (
                            sum(train_times[1:]) / max(len(train_times) - 1, 1)
                        ) if len(train_times) > 1 else None,
                    }, f, indent=2)

            if early_stopping:
                # Update early stopping and check if we should stop
                early_stop = early_stop.update(val_loss)
                if early_stop.should_stop:
                    print(f"Early stopping triggered at epoch {epoch + 1}. No improvement in validation loss for {early_stop.patience} consecutive epochs.")
                    break
    finally:
        # Always let in-flight async saves finalize, even if the loop is
        # interrupted (KeyboardInterrupt / kernel stop / exception). Otherwise
        # Orbax leaves unfinalized step dirs that max_to_keep/best_fn can never
        # garbage-collect.
        if mgr is not None:
            mgr.wait_until_finished()

    return train_losses, val_losses, train_times, val_times


def evaluate_pdf_sf(conditioner, data, context):
    conditioner.eval()

    _, flow = spline_flow(data.squeeze(), context, conditioner)
    z = flow.bijector.inverse(data)

    return flow.log_prob(data), jstats.norm.logsf(z)

def ll_racing_models_timed(conditioner, data, ctx_1, ctx_2, response, n_repeats=10, warmup=True):
    jitted = nnx.jit(evaluate_pdf_sf)

    if warmup:
        lp = jitted(conditioner, data, ctx_1)
        jax.block_until_ready(lp)

    elapsed = []
    for _ in range(n_repeats):
        t0 = time.perf_counter_ns()
        log_pdf_1, log_sf_1 = jitted(conditioner, data, ctx_1)
        log_pdf_2, log_sf_2 = jitted(conditioner, data, ctx_2)

        pred_ll = jnp.where(
            response == 1,
            log_pdf_1.squeeze() + log_sf_2.squeeze(),  
            log_pdf_2.squeeze() + log_sf_1.squeeze(),   
        )
        t1 = time.perf_counter_ns()
        elapsed.append(t1 - t0)

    return pred_ll, elapsed


def plot_scatter_normalisation(true_ll, pred_ll, z, num_bins=50):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(true_ll.ravel(), pred_ll.ravel(), s=0.5, alpha=0.3)
    lims = [true_ll.min(), true_ll.max()]
    ax[0].plot(lims, lims, "r--", linewidth=1, label="identity")
    ax[0].set_xlabel("Analytical log-likelihood")
    ax[0].set_ylabel("Predicted log-likelihood")
    ax[0].set_title("Predicted vs Analytical log-likelihood")
    ax[0].legend()

    x = np.linspace(-4, 4, 100)
    y_norm = stats.norm.pdf(x)
    
    ax[1].hist(z, bins=num_bins, density=True, alpha=0.6, color='g', label='Transformed Data')
    ax[1].plot(x, y_norm, 'r--', label='Standard Normal PDF')
    ax[1].set_xlabel("Transformed Data")
    ax[1].set_ylabel("Density")
    ax[1].set_title("Histogram of Transformed Data")
    ax[1].legend()
    plt.tight_layout()
    plt.show()




# Losses for classifier and flow (DDM)

import optax

# --- conditioner / flow loss is unchanged; data and context come from your batches() ---
# conditioner now receives a context that already includes R.

def classifier_loss_fn(classifier, clf_input, response):
    """
    clf_input: (N, num_params) — just params, i.e. context without R.
    response:  int array with values in {1, 2}.
    """
    logits  = classifier(clf_input)
    targets = (response == 2).astype(jnp.float32)   # 0 if R=1, 1 if R=2
    return optax.sigmoid_binary_cross_entropy(logits, targets).mean()



@nnx.jit
def classifier_train_step(classifier, optimizer, metrics, clf_input, response):
    grad_fn = nnx.value_and_grad(classifier_loss_fn)
    loss, grads = grad_fn(classifier, clf_input, response)
    metrics.update(loss=loss)
    optimizer.update(grads)


def classifier_eval_step(classifier, metrics, clf_input, response):
    loss = classifier_loss_fn(classifier, clf_input, response)
    metrics.update(loss=loss)


# ---------------------------------------------------------------------------
# Classifier training loop (parallel to train_one_epoch / validate_model /
# train_model). The classifier consumes the SAME `batches(...)` generator.
#
# Context layout: [params, R]
#   classifier input = context[..., :-1]    # params
#   classifier target = context[..., -1]    # R
# ---------------------------------------------------------------------------

def train_one_epoch_classifier(classifier, optimizer, metrics, batches, dset_train, batch_size=8192):
    pbar = tqdm(batches, total=dset_train.count_rows() // batch_size,
                desc="Training", unit="batches")
    classifier.train()
    metrics.reset()

    avg_loss = float("nan")
    for batch, (data, context, y) in enumerate(pbar):
        clf_input = context[..., :-1]                       # params
        response  = context[..., -1].astype(jnp.float32)    # R
        classifier_train_step(classifier, optimizer, metrics, clf_input, response)

        if (batch + 1) % 10 == 0:
            avg_loss = float(metrics.compute()["loss"])
            pbar.set_postfix({"Avg. Loss": f"{avg_loss:.4f}"})

    return avg_loss


def validate_model_classifier(classifier, metrics, batches, dset_val, batch_size=8192*2):
    pbar = tqdm(batches, total=dset_val.count_rows() // batch_size,
                desc="Validating", unit="batches")
    classifier.eval()
    metrics.reset()

    avg_val_loss = float("nan")
    for batch, (data, context, y) in enumerate(pbar):
        clf_input = context[..., :-1]                       # params
        response  = context[..., -1].astype(jnp.float32)    # R
        classifier_eval_step(classifier, metrics, clf_input, response)

        if (batch + 1) % 10 == 0:
            avg_val_loss = float(metrics.compute()["loss"])
            pbar.set_postfix({"Avg. Val Loss": f"{avg_val_loss:.4f}"})

    return avg_val_loss

# Status: the DDM choice-classifier training head. Used by the notebook's
# "DDM training with separate classifier" section; mirrors train_model's
# (train_loader, val_loader, dset_train, dset_val) argument order.
def train_model_classifier(
        classifier,
        optimizer,
        train_metrics,
        val_metrics,
        train_loader,
        val_loader,
        dset_train,
        dset_val,
        batch_size_train=8192,
        batch_size_val=8192*2,
        num_epochs=10,
        start_at_epoch=0,
        checkpoint_path=None,
        early_stopping = True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        *args,
        **kwargs
):
    train_losses = []
    val_losses = []
    train_times = []
    val_times = []

    mgr = make_checkpoint_manager(checkpoint_path, clear_dir=True) if checkpoint_path else None
    
    if early_stopping:
        early_stop = EarlyStopping(min_delta=early_stopping_min_delta, patience=early_stopping_patience)
    
    try:
        for epoch in range(start_at_epoch, num_epochs + start_at_epoch):
            print(f"Epoch {epoch + 1}/{num_epochs + start_at_epoch}")

            t0 = time.perf_counter()
            train_loss = train_one_epoch_classifier(
                classifier,
                optimizer,
                train_metrics,
                train_loader,
                dset_train,
                batch_size=batch_size_train,
            )
            t1 = time.perf_counter()
            val_loss = validate_model_classifier(
                classifier,
                val_metrics,
                val_loader,
                dset_val,
                batch_size=batch_size_val,
            )
            t2 = time.perf_counter()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_times.append(t1 - t0)
            val_times.append(t2 - t1)

            print(f"  train {t1 - t0:.2f}s | val {t2 - t1:.2f}s")

            if not np.logical_and(np.isfinite(train_loss), np.isfinite(val_loss)):
                print("Loss is not finite. Terminating training...")
                break

            if mgr is not None:
                save_checkpoint(mgr, epoch, classifier, optimizer, val_loss)

            if checkpoint_path is not None:
                log_path = os.path.join(os.path.abspath(checkpoint_path), "training_log.json")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "w") as f:
                    json.dump({
                        "train_losses": [float(x) for x in train_losses],
                        "val_losses": [float(x) for x in val_losses],
                        "train_times_s": train_times,
                        "val_times_s": val_times,
                        "total_time_s": sum(train_times) + sum(val_times),
                        "compile_overhead_est_s": train_times[0] - (
                            sum(train_times[1:]) / max(len(train_times) - 1, 1)
                        ) if len(train_times) > 1 else None,
                    }, f, indent=2)

            if early_stopping:
                early_stop = early_stop.update(val_loss)
                if early_stop.should_stop:
                    print(f"Early stopping triggered at epoch {epoch + 1}. No improvement in validation loss for {early_stop.patience} consecutive epochs.")
                    break
    finally:
        # Always let in-flight async saves finalize, even if the loop is
        # interrupted (KeyboardInterrupt / kernel stop / exception). Otherwise
        # Orbax leaves unfinalized step dirs that max_to_keep/best_fn can never
        # garbage-collect.
        if mgr is not None:
            mgr.wait_until_finished()

    return train_losses, val_losses, train_times, val_times

def plot_loss(train_loss: list, val_loss: list):
    x = np.arange(1, len(train_loss) + 1)
    
    plt.plot(x, train_loss, label = "train")
    plt.plot(x, val_loss, label = "val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()
    plt.close()

def save_obj(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_obj(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

def plot_rt_overlay(ds_test, param_id, param_names, conditioner, classifier, scaler=None, precision=500, eval_at_data=False):
    expr = pc.field("param_id") == param_id

    pd_dat = ds_test.filter(expr).to_pandas()
    pd_dat["R"] = pd_dat["R"].astype(int)
    r_1 = pd_dat[pd_dat["R"] == 1]["rt"]
    r_2 = pd_dat[pd_dat["R"] == 2]["rt"]
    ll_1 = pd_dat[pd_dat["R"] == 1]["ll"]
    ll_2 = pd_dat[pd_dat["R"] == 2]["ll"]
    p_r1 = len(r_1) / len(pd_dat)
    p_r2 = len(r_2) / len(pd_dat)

    xs = jnp.linspace(1e-10, 15, precision)
    xs = jnp.concatenate([xs, xs])
    rs = jnp.repeat(jnp.array([1, 2]), repeats=precision)
    params = pd_dat.loc[0, param_names].to_numpy().reshape(1, len(param_names))

    if scaler is not None:
        params = scaler.transform(params)
    ctx = jnp.repeat(params, repeats=precision*2, axis=0)
        

    pred_ll = joint_log_prob(conditioner, classifier, xs, ctx, rs)

    fig, ax = plt.subplots(2, 1)
    plt.suptitle(f"RT distribution overlay for param_id={param_id}")

    if eval_at_data:
        rs_2 = jnp.repeat(jnp.array([2]), repeats=len(r_2))
        rs_1 = jnp.repeat(jnp.array([1]), repeats=len(r_1))
        ctx_2 = jnp.repeat(params, repeats=len(r_2), axis=0)
        ctx_1 = jnp.repeat(params, repeats=len(r_1), axis=0)
        pred_ll_rt_2 = joint_log_prob(conditioner, classifier, r_2.to_numpy(), ctx_2, rs_2)
        pred_ll_rt_1 = joint_log_prob(conditioner, classifier, r_1.to_numpy(), ctx_1, rs_1)

    if len(r_1) > 0:

        if eval_at_data:
            ax[0].plot(r_1, jnp.exp(pred_ll_rt_1) / p_r1, ".", alpha=0.5, label="R=1 flow")
        else:
            ax[0].plot(xs[:precision], jnp.exp(pred_ll[:precision]) / p_r1, label="R=1 flow")

        ax[0].hist(r_1, bins=50, density=True, alpha=0.5)
        ax[0].plot(r_1, np.exp(ll_1) / p_r1, ".", alpha=0.5, label="R=1 true")
        ax[0].set_xlim(min(r_1) - 0.1, max(r_1) + 0.1)
        ax[0].legend()
    if len(r_2) > 0:
        if eval_at_data:
            ax[1].plot(r_2, jnp.exp(pred_ll_rt_2) / p_r2, ".", alpha=0.5, label="R=2 flow")
        else:
            ax[1].plot(xs[precision:], jnp.exp(pred_ll[precision:]) / p_r2, label="R=2 flow")

        ax[1].plot(r_2, np.exp(ll_2) / p_r2, ".", alpha=0.5, label="R=2 true")
        ax[1].hist(r_2, bins=50, density=True, alpha=0.5)
        ax[1].set_xlim(min(r_2) - 0.1, max(r_2) + 0.1)
        ax[1].legend()
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
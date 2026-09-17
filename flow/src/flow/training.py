import os
import json
import time
import pickle
from functools import partial

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


def loss_fn_edge(conditioner, data, context, edge_amp: float, edge_scale: float,
                 t0_mu: float, t0_sd: float, st0_mu: float, st0_sd: float,
                 min_log_prob: float = 1e-12):
  """NLL with the LEADING EDGE upweighted.

  Phase 13 measured where the flow's density error actually sits: it
  over-predicts by up to +0.12 in log density inside the st0 smear
  (0 < rt - t0 < st0) and is near-unbiased beyond it. That region holds ~6% of
  the probability mass and contributes ~0.0014 to the mean log-likelihood —
  the same size as the seed-to-seed spread in validation loss (0.001). So plain
  MLE cannot prioritise the region that determines t0; the signal is buried in
  seed noise.

  This reweights each row by 1 + edge_amp * exp(-u / edge_scale) with
  u = (rt - t0) / st0, amplifying the onset region by up to (1 + edge_amp)
  while leaving the bulk essentially untouched. edge_amp = 0 reproduces
  loss_fn exactly.

  t0 and st0 arrive standardised in the context (columns 2 and 7 of the DDM
  layout v, a, t0, s, Z, SZ, sv, st0, R), so their scaler statistics are passed
  in to recover the natural scale.
  """
  data_flat = data.squeeze()
  is_valid = jnp.isfinite(data_flat) & (data_flat > 0.0)
  safe_data = jnp.where(is_valid, data_flat, 1.0)
  safe_log_probs, _ = conditioner.spline_flow(safe_data, context, conditioner)
  final_log_probs = jnp.where(is_valid, safe_log_probs, jnp.log(min_log_prob))

  t0 = jnp.exp(context[:, 2] * t0_sd + t0_mu)
  st0 = jnp.exp(context[:, 7] * st0_sd + st0_mu)
  u = jnp.clip((data_flat - t0) / jnp.maximum(st0, 1e-6), 0.0, 50.0)
  w = 1.0 + edge_amp * jnp.exp(-u / edge_scale)
  w = jnp.where(is_valid, w, 0.0)
  return -jnp.sum(w * final_log_probs) / jnp.maximum(jnp.sum(w), 1e-6)


def loss_fn_censored(conditioner, data, context, censor,
                     min_log_prob: float = 1e-12):
  """Censored MLE: rts below `censor` contribute the usual log-density;
  rts at/above it contribute the log survivor mass at the censor,
  log(1 - CDF(censor | context)). The flow's shape above the censor is
  thereby unconstrained (see sbc/censor_design.md in the analysis repo)."""
  data_flat = data.squeeze()

  is_valid = jnp.isfinite(data_flat) & (data_flat > 0.0)
  is_cens = is_valid & (data_flat >= censor)
  is_obs = is_valid & ~is_cens

  safe_data = jnp.where(is_obs, data_flat, 1.0)
  safe_log_probs, flow = conditioner.spline_flow(safe_data, context, conditioner)

  # survivor at the censor: same spline inversion the evaluators use
  z_c = flow.bijector.inverse(jnp.full_like(data_flat, censor))
  log_sf_c = jstats.norm.logsf(z_c)

  final = jnp.where(is_obs, safe_log_probs,
                    jnp.where(is_cens, log_sf_c, jnp.log(min_log_prob)))
  return -jnp.mean(final)


def score_penalty(conditioner, data, context, n_theta: int, weights=None):
  """Unbiased estimator of the flow's squared SCORE BIAS, ||E_rt|theta[s]||^2,
  where s = d log f(rt | theta, R) / d theta.

  Why this and not the density error. MLE training minimises cross-entropy,
  which constrains the LEVEL of the density error and says nothing about its
  gradient in theta. A flow can fit the density well and still carry a
  systematic score bias, and it is the score bias — not the density error —
  that displaces a posterior mode: for the true model E_rt|theta[s] = 0, so any
  departure is a standing force on every fit. Measured on the shipped DDM
  ensemble this term is +0.029 per trial in st0, which reproduces the observed
  SBC miscalibration across N=100/200/400 (sbc/SBC_PLAN.md, Phase 3).

  Estimator. ||E[s]||^2 cannot be estimated from one draw without bias, but
  with TWO conditionally independent rts sharing a theta it can:
      E[<s(rt1,theta), s(rt2,theta)>] = ||E_rt|theta[s]||^2.
  The training data carries exactly 2 trials per parameter draw, so rows are
  consumed in pairs: rows 2k and 2k+1 must share theta. Use
  PreloadedBatches(..., pair_rows=True), which permutes PAIRS rather than rows.
  Individual pair terms may be negative; the mean is what is unbiased.

  Gradients are taken w.r.t. the STANDARDIZED context, so the penalty weights
  each parameter by its training-prior scale. The response coordinate is the
  last context column and is excluded via `n_theta`.
  """
  data_flat = data.squeeze()
  is_valid = jnp.isfinite(data_flat) & (data_flat > 0.0)
  safe_data = jnp.where(is_valid, data_flat, 1.0)

  def total_lp(ctx):
    lp, _ = conditioner.spline_flow(safe_data, ctx, conditioner)
    return jnp.sum(jnp.where(is_valid, lp, 0.0))

  # One reverse pass gives every row's own gradient, because a row's log
  # density depends only on its own context row.
  s = jax.grad(total_lp)(context)
  # Masked rather than sliced so this works whether n_theta arrives concrete
  # (jitted steps mark it static) or traced (a bare value_and_grad call, where
  # a dynamic stop index cannot be used with static slicing). The strided
  # slices below are static either way.
  keep = (jnp.arange(s.shape[-1]) < n_theta).astype(s.dtype)
  s = s * keep
  # Optional per-parameter weights. The unweighted penalty treats every
  # standardized context dimension alike, but what actually matters is the
  # posterior DISPLACEMENT each score bias causes,
  #     z_p = N * g_p / sqrt(1/psd_p^2 + N * I_p),
  # so minimising sum_p z_p^2 means weighting g_std,p^2 by
  #     w_p = 1 / (sigma_p^2 * (1/psd_p^2 + N * I_p)).
  # This matters because the unit information I varies enormously across the
  # DDM's parameters: measured against the analytic likelihood, a has I=5.83
  # while sv has I~0, i.e. sv is essentially unidentified and its posterior is
  # prior-dominated. An unweighted penalty therefore spends capacity on a
  # (which tolerates large score bias) and under-protects sv (which cannot
  # tolerate any) — exactly the failure seen at lambda=1, where st0 reached
  # its target while sv coverage collapsed to .36.
  prod = s[0::2] * s[1::2]          # per-pair, per-parameter estimate of g_p^2
  if weights is not None:
    # Applied to the PRODUCT, so w_p weights g_p^2 linearly. (Scaling each
    # score vector instead would make the weights enter squared.) Weights
    # cover the n_theta parameter columns; the trailing response column is
    # already zeroed by `keep`, so pad with ones to match width.
    wv = jnp.asarray(weights, dtype=prod.dtype)
    wv = jnp.concatenate([wv, jnp.ones(prod.shape[-1] - wv.shape[0], prod.dtype)])
    prod = prod * wv[None, :]
  pen = jnp.mean(jnp.sum(prod, axis=-1))
  # CLIPPED AT ZERO. <s1,s2> is unbiased for ||E[s]||^2 but only in
  # expectation: empirically it is unbounded below, and the optimiser will
  # drive it negative by anti-correlating the two draws' scores on the
  # specific pairs in the data, which does nothing to the true bias. An
  # unclipped 300-epoch run at lambda=0.1 did exactly this — 122/300 epochs
  # had NEGATIVE validation loss, minimum -6236 — and since the checkpoint
  # manager selects on minimum val loss, the saved checkpoint was the most
  # exploited state (port/export/REJECTED/). Clipping the BATCH MEAN (not
  # each pair, which would bias it badly given the per-pair noise) removes the
  # gradient reward for going negative while leaving the estimator untouched
  # wherever it is positive, i.e. wherever there is real bias to remove.
  return jnp.maximum(pen, 0.0)


def loss_fn_score(conditioner, data, context, lam: float, n_theta: int,
                  weights=None, min_log_prob: float = 1e-12):
  """NLL plus lam * (squared score bias). lam = 0 reproduces loss_fn exactly."""
  nll = loss_fn(conditioner, data, context, min_log_prob)
  return nll + lam * score_penalty(conditioner, data, context, n_theta, weights)


# n_theta is static: it selects which context columns are parameters, so it is
# structural, not data. Without @nnx.jit this step re-traces the (nested-grad)
# graph on every batch — measured at 537 s/epoch versus 6.4 s/epoch for the
# plain MLE step, i.e. 84x, which makes a 300-epoch run infeasible.
def loss_fn_distill(conditioner, classifier, data, context, y, beta: float,
                    n_theta: int, nll_weight: float = 1.0,
                    min_log_prob: float = 1e-12):
  """NLL plus beta * MSE against the ANALYTIC log-density.

  Why. The score penalty drives a moment (E[score]) to zero, which is
  satisfiable by flattening log f in theta - and since I = -E[grad^2 log f],
  flattening costs curvature (measured: kappa 1.027 -> 0.70 at lambda=0.01,
  -> 0.46 at lambda=1). A POINTWISE TARGET does not have that degeneracy:
  matching a function everywhere pins its derivatives too. This is the
  Brehmer et al. (2020) distinction between SCANDAL (score regressed onto a
  target) and a targetless constraint.

  This is a BENCHMARK, not a method: it uses the analytic likelihood we are
  trying to avoid needing. Its purpose is to separate "this architecture
  cannot represent the likelihood well enough" from "MLE does not weight the
  right regions" - i.e. to establish whether the N-ceiling is architectural.

  Target. The parquet `ll` column is the JOINT defective density
  log p(rt, R | theta), while the flow models the CONDITIONAL log p(rt | theta, R).
  They differ by log P(R | theta), supplied by the (frozen) choice classifier:
      log P(R=2) = -softplus(-logit),  log P(R=1) = -softplus(logit)
  so the flow's target is  y - log P(R | theta).  Omitting this correction
  would ask a normalised density to absorb an rt-independent offset it
  structurally cannot, biasing the fit.
  """
  data_flat = data.squeeze()
  # y may be (n,) = ll, or (n, 2) = [ll, is_grid]. Off-sample GRID rows are not
  # draws from the model, so they carry no valid maximum-likelihood
  # contribution: they feed the MSE term only. That is the whole point of them
  # - an analytic likelihood can be evaluated where the simulator never puts
  # samples, which is exactly where the flow's errors live (leading edge, the
  # 1-3 s shoulder) and where MLE therefore has no gradient signal.
  if y.ndim == 2:
    ll_t = y[:, 0]
    # column 1 is param_id; off-sample grid rows are written with an offset of
    # 1e7 by simulation/sim_grid_data.R, so no flag column is needed.
    is_grid = y[:, 1] >= 1e6
  else:
    ll_t = y
    is_grid = jnp.zeros_like(data_flat, dtype=bool)
  is_valid = jnp.isfinite(data_flat) & (data_flat > 0.0) & jnp.isfinite(ll_t)
  is_sample = is_valid & ~is_grid
  safe_data = jnp.where(is_valid, data_flat, 1.0)
  log_f, _ = conditioner.spline_flow(safe_data, context, conditioner)

  nll = -(jnp.sum(jnp.where(is_sample, log_f, 0.0))
          / jnp.maximum(jnp.sum(is_sample), 1))

  logit = classifier(context[:, :n_theta])
  R2 = context[:, -1] > 1.5                      # R coded 1 / 2
  log_p_R = jnp.where(R2, -jax.nn.softplus(-logit), -jax.nn.softplus(logit))
  target = ll_t - log_p_R

  se = jnp.where(is_valid, (log_f - target) ** 2, 0.0)
  mse = jnp.sum(se) / jnp.maximum(jnp.sum(is_valid), 1)
  # nll_weight = 0 gives PURE regression onto the analytic log-density (no
  # maximum-likelihood term at all). Note beta=10 was already no better than
  # beta=1 on density error (RMSE .1375 vs .1327) and pushed Z and SZ outside
  # the seed band, so the NLL term appears to be doing useful work; expect
  # pure regression to be worse, not better.
  return nll_weight * nll + beta * mse


def loss_fn_scorematch(conditioner, classifier, data, context, y, alpha: float,
                       n_theta: int, sigma, delta: float = 1.0,
                       min_log_prob: float = 1e-12):
  """NLL plus alpha * MSE between the flow's score and the TRUE score.

  This is the targeted counterpart of score_penalty(). That one drove a MOMENT
  (E[score]) to zero, which is satisfiable by flattening log f in theta - and
  because I = -E[grad^2 log f], flattening destroys curvature (kappa 1.027 ->
  0.46 at lambda=1). Matching the score POINTWISE pins the theta-dependence of
  log f, so the flattening solution is no longer available. This is SCANDAL's
  form (Brehmer et al. 2020, Eq 14) with an exact target rather than a
  simulator-derived or classifier-estimated one, which isolates "does targeted
  score matching work" from "how good is the estimated target".

  y: (n, 1 + n_theta) - column 0 is `ll`, columns 1.. are d ll / d theta on the
  SAMPLED scale, precomputed by central differences (simulate_trials_score).

  Two conversions are essential and easy to get wrong:
    * the flow's autodiff is w.r.t. the STANDARDISED context, so the target
      must be multiplied by sigma (chain rule, z = (theta - mu)/sigma);
    * the stored score is of the JOINT defective density, as `ll` is, while
      the flow models the conditional - so d log P(R|theta)/d theta is
      subtracted, obtained by autodiff through the frozen classifier.
  """
  data_flat = data.squeeze()
  ll_t = y[:, 0]
  score_t = y[:, 1:1 + n_theta]
  is_valid = jnp.isfinite(data_flat) & (data_flat > 0.0) & jnp.isfinite(ll_t)
  safe_data = jnp.where(is_valid, data_flat, 1.0)

  def total_lp(ctx):
    lp, _ = conditioner.spline_flow(safe_data, ctx, conditioner)
    return jnp.sum(jnp.where(is_valid, lp, 0.0))
  log_f, _ = conditioner.spline_flow(safe_data, context, conditioner)
  nll = -jnp.mean(jnp.where(is_valid, log_f, jnp.log(min_log_prob)))
  s_flow = jax.grad(total_lp)(context)[:, :n_theta]        # d log f / d z

  # d log P(R|theta) / d z, through the frozen classifier
  R2 = context[:, -1] > 1.5
  def total_lpR(ctx):
    logit = classifier(ctx[:, :n_theta])
    lpR = jnp.where(R2, -jax.nn.softplus(-logit), -jax.nn.softplus(logit))
    return jnp.sum(lpR)
  s_R = jax.grad(total_lpR)(context)[:, :n_theta]

  sig = jnp.asarray(sigma, dtype=score_t.dtype)[None, :n_theta]
  target = score_t * sig - s_R                              # conditional, in z units

  # HUBER, not squared error. t0 and st0 shift the support boundary, so their
  # TRUE scores explode near the leading edge: measured p99 of |score_t0| is
  # 243 against a median of 1.2, and |score_st0| 233 against 0.79. Under MSE a
  # single such row contributes ~60,000 and the other seven parameters are
  # never seen - the plain-MSE run reached val 101.9 with |val|>10 in all 300
  # epochs. These outliers are REAL information, not corruption, so they are
  # down-weighted rather than dropped: quadratic within delta, linear beyond.
  r = jnp.where(is_valid[:, None], s_flow - target, 0.0)
  ar = jnp.abs(r)
  huber = jnp.where(ar <= delta, 0.5 * r ** 2, delta * (ar - 0.5 * delta))
  loss_sm = jnp.sum(huber) / jnp.maximum(jnp.sum(is_valid) * n_theta, 1)
  return nll + alpha * loss_sm


@partial(nnx.jit, static_argnums=(8,))
def train_step_scorematch(conditioner, classifier, optimizer: nnx.Optimizer,
                          metrics: nnx.MultiMetric, data, context, y, alpha,
                          n_theta, sigma, delta=1.0):
  grad_fn = nnx.value_and_grad(loss_fn_scorematch)
  loss, grads = grad_fn(conditioner, classifier, data, context, y, alpha,
                        n_theta, sigma, delta)
  metrics.update(loss=loss)
  optimizer.update(grads)


@partial(nnx.jit, static_argnums=(7,))
def eval_step_scorematch(conditioner, classifier, metrics: nnx.MultiMetric,
                         data, context, y, alpha, n_theta, sigma, delta=1.0):
  metrics.update(loss=loss_fn_scorematch(conditioner, classifier, data,
                                         context, y, alpha, n_theta, sigma,
                                         delta))


@partial(nnx.jit, static_argnums=(8,))
def train_step_distill(conditioner, classifier, optimizer: nnx.Optimizer,
                       metrics: nnx.MultiMetric, data, context, y, beta,
                       n_theta, nll_weight=1.0):
  grad_fn = nnx.value_and_grad(loss_fn_distill)          # wrt conditioner only
  loss, grads = grad_fn(conditioner, classifier, data, context, y, beta,
                        n_theta, nll_weight)
  metrics.update(loss=loss)
  optimizer.update(grads)


@partial(nnx.jit, static_argnums=(7,))
def eval_step_distill(conditioner, classifier, metrics: nnx.MultiMetric,
                      data, context, y, beta, n_theta, nll_weight=1.0):
  metrics.update(loss=loss_fn_distill(conditioner, classifier, data, context,
                                      y, beta, n_theta, nll_weight))


@partial(nnx.jit, static_argnums=(6,))
def train_step_score(conditioner, optimizer: nnx.Optimizer,
                     metrics: nnx.MultiMetric, data, context, lam, n_theta,
                     weights=None):
  grad_fn = nnx.value_and_grad(loss_fn_score)
  loss, grads = grad_fn(conditioner, data, context, lam, n_theta, weights)
  metrics.update(loss=loss)
  optimizer.update(grads)


@partial(nnx.jit, static_argnums=(5,))
def eval_step_score(conditioner, metrics: nnx.MultiMetric, data, context,
                    lam, n_theta, weights=None):
  metrics.update(loss=loss_fn_score(conditioner, data, context, lam, n_theta,
                                    weights))


@nnx.jit
def train_step_edge(conditioner, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric,
                    data, context, edge_amp, edge_scale, t0_mu, t0_sd, st0_mu, st0_sd):
  grad_fn = nnx.value_and_grad(loss_fn_edge)
  loss, grads = grad_fn(conditioner, data, context, edge_amp, edge_scale,
                        t0_mu, t0_sd, st0_mu, st0_sd)
  optimizer.update(grads)
  metrics.update(loss=loss)
  return loss


@nnx.jit
def train_step(conditioner, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, data, context):
  grad_fn = nnx.value_and_grad(loss_fn)
  loss, grads = grad_fn(conditioner, data, context)
  metrics.update(loss=loss)  # In-place updates.
  optimizer.update(grads)  # In-place updates.

def eval_step(conditioner, metrics: nnx.MultiMetric, data, context):
  loss = loss_fn(conditioner, data, context)
  metrics.update(loss=loss)

@nnx.jit
def train_step_censored(conditioner, optimizer: nnx.Optimizer,
                        metrics: nnx.MultiMetric, data, context, censor):
  grad_fn = nnx.value_and_grad(loss_fn_censored)
  loss, grads = grad_fn(conditioner, data, context, censor)
  metrics.update(loss=loss)
  optimizer.update(grads)

def eval_step_censored(conditioner, metrics: nnx.MultiMetric, data, context, censor):
  loss = loss_fn_censored(conditioner, data, context, censor)
  metrics.update(loss=loss)


def train_one_epoch(conditioner, optimizer, metrics, batches, dset_train, batch_size=8192, censor=None,
                    score_lambda=None, n_theta=None, score_weights=None,
                    distill_beta=None, classifier=None, nll_weight=1.0,
                    score_match=None, sigma=None, huber_delta=1.0,
                    edge_amp=None, edge_scale=1.0, edge_stats=None):
    pbar = tqdm(batches, total = dset_train.count_rows() // batch_size, desc="Training", unit="batches") 
    conditioner.train()
    metrics.reset()

    censor_j = None if censor is None else jnp.asarray(censor, dtype=jnp.float32)
    for batch, (data, context, y) in enumerate(pbar):
        if score_match is not None:
            train_step_scorematch(conditioner, classifier, optimizer, metrics,
                                  data, context, y, score_match, n_theta, sigma,
                                  huber_delta)
        elif distill_beta is not None:
            train_step_distill(conditioner, classifier, optimizer, metrics,
                               data, context, y, distill_beta, n_theta,
                               nll_weight)
        elif score_lambda is not None:
            train_step_score(conditioner, optimizer, metrics, data, context,
                             score_lambda, n_theta, score_weights)
        elif edge_amp is not None:
            train_step_edge(conditioner, optimizer, metrics, data, context,
                            edge_amp, edge_scale, *edge_stats)
        elif censor_j is None:
            train_step(conditioner, optimizer, metrics, data, context)
        else:
            train_step_censored(conditioner, optimizer, metrics, data, context, censor_j)
        
        if (batch + 1) % 10 == 0:
            avg_loss = float(metrics.compute()["loss"])
            pbar.set_postfix({"Avg. Loss": f"{avg_loss:.4f}"})
    
    return avg_loss

def validate_model(conditioner, metrics, batches, dset_val, batch_size=8192*2, censor=None,
                   score_lambda=None, n_theta=None, score_weights=None,
                   distill_beta=None, classifier=None, nll_weight=1.0,
                   score_match=None, sigma=None, huber_delta=1.0):
    pbar = tqdm(batches, total = dset_val.count_rows() // batch_size, desc="Validating", unit="batches") 
    conditioner.eval()
    metrics.reset()

    censor_j = None if censor is None else jnp.asarray(censor, dtype=jnp.float32)
    for batch, (data, context, y) in enumerate(pbar):
        if score_match is not None:
            eval_step_scorematch(conditioner, classifier, metrics, data,
                                 context, y, score_match, n_theta, sigma,
                                 huber_delta)
        elif distill_beta is not None:
            eval_step_distill(conditioner, classifier, metrics, data, context,
                              y, distill_beta, n_theta, nll_weight)
        elif score_lambda is not None:
            eval_step_score(conditioner, metrics, data, context, score_lambda,
                            n_theta, score_weights)
        elif censor_j is None:
            eval_step(conditioner, metrics, data, context)
        else:
            eval_step_censored(conditioner, metrics, data, context, censor_j)
        
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
        censor=None,
        score_lambda=None,
        n_theta=None,
        score_weights=None,
        distill_beta=None,
        classifier=None,
        nll_weight=1.0,
        score_match=None,
        sigma=None,
        huber_delta=1.0,
        edge_amp=None,
        edge_scale=1.0,
        edge_stats=None,
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
                batch_size=batch_size_train,
                censor=censor,
                score_lambda=score_lambda,
                n_theta=n_theta,
                score_weights=score_weights,
                distill_beta=distill_beta, classifier=classifier,
                nll_weight=nll_weight,
                score_match=score_match, sigma=sigma,
                huber_delta=huber_delta,
                edge_amp=edge_amp, edge_scale=edge_scale, edge_stats=edge_stats
            )
            t1 = time.perf_counter()
            val_loss = validate_model(
                conditioner,
                val_metrics,
                val_loader,
                dset_val,
                batch_size=batch_size_val,
                censor=censor,
                score_lambda=score_lambda,
                n_theta=n_theta,
                score_weights=score_weights,
                distill_beta=distill_beta, classifier=classifier,
                nll_weight=nll_weight,
                score_match=score_match, sigma=sigma,
                huber_delta=huber_delta
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
    
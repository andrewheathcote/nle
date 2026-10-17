import math
import time

import distrax
import optax
import jax
import jax.numpy as jnp
from flax import nnx


# Affine stage on log time (option `affine`, off by default):
#   scale = softplus(raw_scale + AFFINE_OFFSET) + AFFINE_MIN_SCALE,
# with the offset chosen so that raw_scale = 0 gives scale = 1 exactly. The
# exporter writes both numbers into the card's `affine_scale`.
AFFINE_MIN_SCALE = 1e-3
AFFINE_OFFSET = math.log(math.expm1(1.0 - AFFINE_MIN_SCALE))


# @nnx.jit
def spline_flow(data, context, conditioner):
    num_splines = conditioner.num_splines
    num_bins = conditioner.num_bins
    # spline support in log-rt; conditioners built before this attribute
    # existed default to the original [-5, 5]
    range_min = getattr(conditioner, "range_min", -5.0)
    range_max = getattr(conditioner, "range_max", 5.0)
    # per-context location and scale on log time (see make_mlp_conditioner);
    # conditioners built without the attribute are plain
    affine = getattr(conditioner, "affine", False)
    params = conditioner(context)                 # (N, dout)
    spline_size = 3 * num_bins + 1
    idx = 0
    layers = []

    for i in range(num_splines):
        sp = params[..., idx:idx + spline_size]
        idx += spline_size
        layers.append(distrax.RationalQuadraticSpline(
            sp, range_min=range_min, range_max=range_max,
            boundary_slopes="identity", min_bin_size=1e-4,
        ))
        if i < num_splines - 1:                    # affine between splines, not after the last
            shift     = params[..., idx:idx + 1].squeeze(-1)
            log_scale = params[..., idx + 1:idx + 2].squeeze(-1)
            idx += 2
            layers.append(distrax.ScalarAffine(shift=shift, log_scale=log_scale))

    if affine:
        # log t = loc + scale * spline(z): the two outputs AFTER the 3K+1
        # spline outputs are [loc, raw_scale]. Exactly the card semantics of
        # port/R/flow_race.R (`affine`, `affine_scale`) and of Malte Lueken's
        # confrdm_jax.flows_affine, so zeroed outputs give loc 0, scale 1.
        loc = params[..., idx]
        scale = jax.nn.softplus(params[..., idx + 1] + AFFINE_OFFSET) + AFFINE_MIN_SCALE
        layers.append(distrax.ScalarAffine(shift=loc, scale=scale))

    exp_layer = distrax.Lambda(
        forward=jnp.exp,
        inverse=jnp.log,
        forward_log_det_jacobian=lambda z: z,
        inverse_log_det_jacobian=lambda x: -jnp.log(x),
        event_ndims_in=0,
        event_ndims_out=0,
    )

    # exp at the end, same as before
    layers.append(exp_layer)

    bijector = distrax.Chain(list(reversed(layers)))   # Chain composes in reverse order
    flow = distrax.Transformed(distrax.Normal(0.0, 1.0), bijector)
    return flow.log_prob(data), flow

def spline_flow_ddm(data, context, conditioner):
    """Status: NOT used in the documented training path (kept for reference).

    The DDM flow instead uses `spline_flow` with a separate
    choice classifier (see `joint_log_prob`).
    """
    num_splines = conditioner.num_splines
    num_bins = conditioner.num_bins
    params = conditioner(context)                 # (N, dout)
    spline_size = 3 * num_bins + 1
    idx = 0
    layers = []

    arcsinh_layer = distrax.Lambda(
        forward=jnp.arcsinh,
        inverse=jnp.sinh,
        forward_log_det_jacobian=lambda z: -0.5 * jnp.log(z**2 + 1), 
        inverse_log_det_jacobian=lambda x: jnp.abs(x) + jnp.log1p(jnp.exp(-2 * jnp.abs(x))) - jnp.log(2.0),
        event_ndims_in=0,
        event_ndims_out=0,
    )

    layers.append(arcsinh_layer)

    for i in range(num_splines):
        sp = params[..., idx:idx + spline_size]
        idx += spline_size
        layers.append(distrax.RationalQuadraticSpline(
            sp, range_min=-5.0, range_max=15.0,
            boundary_slopes="identity", min_bin_size=1e-4,
        ))
        if i < num_splines - 1:                    # affine between splines, not after the last
            shift     = params[..., idx:idx + 1].squeeze(-1)
            log_scale = params[..., idx + 1:idx + 2].squeeze(-1)
            idx += 2
            layers.append(distrax.ScalarAffine(shift=shift, log_scale=log_scale))

    # exp_layer = distrax.Lambda(
    #     forward=jnp.exp,
    #     inverse=jnp.log,
    #     forward_log_det_jacobian=lambda z: z,
    #     inverse_log_det_jacobian=lambda x: -jnp.log(x),
    #     event_ndims_in=0,
    #     event_ndims_out=0,
    # )

    # # exp at the end, same as before
    # layers.append(exp_layer)

    bijector = distrax.Chain(list(reversed(layers)))   # Chain composes in reverse order
    flow = distrax.Transformed(distrax.Normal(0.0, 1.0), bijector)
    return flow.log_prob(data), flow


class MLP(nnx.Module):
    def __init__(self, dims: list[int], dout: int, *, rngs: nnx.Rngs):
        widths = list(dims) + [dout]
        self.linears = [
            nnx.Linear(widths[i], widths[i + 1], rngs=rngs)
            for i in range(len(widths) - 2)
        ]
        self.norms = [nnx.LayerNorm(widths[i + 1], rngs=rngs)
                      for i in range(len(widths) - 2)]
        self.out = nnx.Linear(
            widths[-2], widths[-1],
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x):
        for linear, norm in zip(self.linears, self.norms):
            x = nnx.gelu(norm(linear(x)))
        return self.out(x)

class MLP_no_norm(nnx.Module):
    def __init__(self, dims: list[int], dout: int, *, rngs: nnx.Rngs):
        widths = list(dims) + [dout]
        self.linears = [
            nnx.Linear(widths[i], widths[i + 1], rngs=rngs)
            for i in range(len(widths) - 2)
        ]
        self.out = nnx.Linear(
            widths[-2], widths[-1],
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x):
        for linear in self.linears:
            x = nnx.gelu(linear(x))
        return self.out(x)


def make_mlp_conditioner(rngs, dims, num_bins=16, num_splines=3, spline_flow = spline_flow, use_norm=True,
                         range_min=-5.0, range_max=5.0, affine=False):
    # each spline needs 3*num_bins + 1 params
    # each affine between splines needs 2 params (shift, log_scale)
    num_affine = num_splines - 1
    dout = num_splines * (3 * num_bins + 1) + num_affine * 2
    if affine:
        # two more outputs, [loc, raw_scale], after the spline's 3K+1. One
        # spline only: the layout 3K+3 is what the card evaluators read.
        if num_splines != 1:
            raise ValueError("affine=True needs num_splines == 1")
        dout += 2
    if use_norm:
        m = MLP(dims, dout, rngs=rngs)
    else:
        m = MLP_no_norm(dims, dout, rngs=rngs)
    m.num_bins = num_bins
    m.num_splines = num_splines
    m.spline_flow = spline_flow
    m.range_min = range_min
    m.range_max = range_max
    if affine:            # attribute absent on a plain conditioner, as before
        m.affine = True
    
    return m



# Classifier for DDM
#
# Context column layout assumed throughout (set this when building the
# parquet-reader `context_names` list in the notebook):
#
#     context = [ p_1, p_2, ..., p_K, R ]
#                |---- params -----|  ^
#                                     +-- response  (target for classifier)
#
#   - Flow conditioner       : sees full context  → input dim = K + 1
#   - Choice classifier      : sees params        → input dim = K
#                              target = R
#
# Slicing helpers:
#   classifier_input(ctx) = ctx[..., :-1]
#   response(ctx)         = ctx[..., -1]

class ChoiceClassifier(nnx.Module):
    """MLP that maps params -> logit for P(R=2 | params)."""
    def __init__(self, dims: list[int], *, rngs: nnx.Rngs):
        # dims = [num_params + 1, h1, h2, ...]; output head is single logit
        widths = list(dims) + [1]
        self.layers = [
            nnx.Linear(widths[i], widths[i + 1], rngs=rngs)
            for i in range(len(widths) - 1)
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = nnx.gelu(layer(x))
        return self.layers[-1](x).squeeze(-1)   # (N,)


def make_choice_classifier(rngs, dims):
    return ChoiceClassifier(dims, rngs=rngs)


# Joint (classifier & flow) training step
@nnx.jit
def joint_train_step(conditioner, classifier,
                     opt_flow, opt_clf,
                     metrics_flow, metrics_clf,
                     data, context):
    # context layout: [params, R]   (see top-of-section docstring)
    clf_input = context[..., :-1]                       # params
    response  = context[..., -1].astype(jnp.int32)      # R

    # flow uses full context (params, R)
    grad_fn_flow = nnx.value_and_grad(loss_fn)
    loss_flow, grads_flow = grad_fn_flow(conditioner, data, context)
    metrics_flow.update(loss=loss_flow)
    opt_flow.update(grads_flow)

    # classifier uses params; target is R
    grad_fn_clf = nnx.value_and_grad(classifier_loss_fn)
    loss_clf, grads_clf = grad_fn_clf(classifier, clf_input, response)
    metrics_clf.update(loss=loss_clf)
    opt_clf.update(grads_clf)


def joint_log_prob(conditioner, classifier, rt, params, response,
                   clf_params=None):
    """Joint log p(rt, R | params).

    params:   (N, K)    — DDM parameters in sampled scale, no R
    response: (N,)      — R, 1 or 2
    clf_params: (N, K)  — optional separate encoding for the classifier. Needed
                only when the flow and the classifier are conditioned on
                different parameterisations of the same draw, as under
                train_ddm.py --triple, where the flow takes (SZ0, SZ, SZ1) and
                the frozen classifier still takes EMC2's (a, Z, SZ). Defaults
                to `params`, i.e. the ordinary single-encoding case.
    """
    R_col = response[..., None].astype(jnp.float32)
    full_context = jnp.concatenate([params, R_col], axis=-1)
    clf_input    = params if clf_params is None else clf_params

    log_p_rt_given, _ = spline_flow(rt, full_context, conditioner)

    logits   = classifier(clf_input)
    log_p_R  = -jax.nn.softplus(jnp.where(response == 2, -logits, logits))

    return log_p_rt_given + log_p_R


def joint_log_prob_timed(conditioner, classifier, rt, params, response,
                         warmup=True, n_repeats=1):
    """Evaluate `joint_log_prob` and return (log_prob, mean_elapsed_seconds).

    Uses `nnx.jit` so the timed call reflects compiled execution, calls
    `jax.block_until_ready` to defeat JAX's async dispatch, and (by default)
    runs one untimed warmup so JIT compile time is not included.

    n_repeats: time `n_repeats` calls and return the mean per-call wall time.
    """
    jitted = nnx.jit(joint_log_prob)

    if warmup:
        lp = jitted(conditioner, classifier, rt, params, response)
        jax.block_until_ready(lp)

    elapsed = []
    for _ in range(n_repeats):
        t0 = time.perf_counter_ns()
        lp = jitted(conditioner, classifier, rt, params, response)
        jax.block_until_ready(lp)
        elapsed.append(time.perf_counter_ns() - t0)

    return lp, elapsed

import json
from tqdm import tqdm
import pandas as pd
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow.parquet as pq
from flax import nnx
from pathlib import Path
from flow import (
    joint_log_prob,
    joint_log_prob_timed,
    make_choice_classifier,
    make_checkpoint_manager,
    make_mlp_conditioner,
    restore_checkpoint,
)
from flow.data import get_inp_data
from flow.training import load_obj

_NUM_BINS = 32
_NUM_SPLINES = 1
_CONTEXT_DDM = ["v", "a", "t0", "s", "Z", "SZ", "sv", "st0"]
dims_clf = [len(_CONTEXT_DDM), 32, 16]
dims = [len(_CONTEXT_DDM) + 1, 32, 16]
_USE_NORM = False

def load_model(clf_dir, flow_dir, dims_clf, dims, num_bins, num_splines, use_norm, schedule_clf, schedule_flow):
    # Load classifier
    cond_factory_clf = lambda: make_choice_classifier(dims=dims_clf, rngs=nnx.Rngs(0))
    opt_factory_clf = lambda: nnx.Optimizer(cond_factory_clf(), optax.adamw(schedule_clf), wrt=nnx.Param)
    mgr_clf = make_checkpoint_manager(clf_dir)
    classifier, _ = restore_checkpoint(mgr_clf, mgr_clf.best_step(), cond_factory_clf, opt_factory_clf)

    # Load flow conditioner
    cond_factory_flow = lambda: make_mlp_conditioner(dims=dims, rngs=nnx.Rngs(0), num_bins=num_bins, num_splines=num_splines, use_norm=use_norm)
    opt_factory_flow = lambda: nnx.Optimizer(cond_factory_flow(), optax.adamw(schedule_flow), wrt=nnx.Param)
    mgr_flow = make_checkpoint_manager(flow_dir)
    conditioner, _ = restore_checkpoint(mgr_flow, mgr_flow.best_step(), cond_factory_flow, opt_factory_flow)

    return classifier, conditioner


# train_loader & schedule
train_loader = load_obj("./results/model_states/large_budget/DDM_flow/train_loader.pkl") # scaler is the same for flow and classifier since they were trained on the same data
with open("./results/model_states/large_budget/DDM_flow/schedule_config.json", "r") as f:
    schedule_config_flow = json.load(f)
schedule_flow = optax.warmup_cosine_decay_schedule(
    init_value=schedule_config_flow["init_value"], 
    peak_value=schedule_config_flow["peak_value"], 
    warmup_steps=schedule_config_flow["warmup_steps"], 
    decay_steps=schedule_config_flow["decay_steps"]
)
with open("./results/model_states/large_budget/DDM_classifier/schedule_config.json", "r") as f:
    schedule_config_clf = json.load(f)
schedule_clf = schedule_config_clf["learning_rate"]

# Load model
flow_path = Path("./results/model_states/large_budget/DDM_flow")
clf_path = Path("./results/model_states/large_budget/DDM_classifier")

classifier, conditioner = load_model(clf_path, flow_path, dims_clf, dims, _NUM_BINS, _NUM_SPLINES, _USE_NORM, schedule_clf, schedule_flow)


# Load data
dat_path = Path("./benchmark/scripts_julian/DDM/N=1000/trials")
params_path = Path("./benchmark/scripts_julian/DDM/eval_params.csv")

dat, params = get_inp_data(dat_path, params_path, _CONTEXT_DDM)


# Feed through classifier and flow
rt = jnp.array(dat.loc[:, "rt"])
pars_true = jnp.array(dat.loc[:, _CONTEXT_DDM])
response = jnp.array(dat.loc[:, "R"], dtype=jnp.float32)

if train_loader._scaler is not None:
    scaler = train_loader._scaler
    pars_true = scaler.transform(pars_true)

jitted = nnx.jit(joint_log_prob)
pred_ll = jitted(
    conditioner,
    classifier,
    rt,
    pars_true,
    response
)
print(np.corrcoef(pred_ll, dat.ll.to_numpy())[0, 1])



# now the full loop
for bname in ["N=100", "N=1000", "N=10000"]:
    dat_path = Path(f"benchmark/scripts_julian/DDM/{bname}/trials")
    params_path = Path(f"benchmark/scripts_julian/DDM/eval_params.csv")
    dat, params = get_inp_data(dat_path, params_path, _CONTEXT_DDM)

    rt = jnp.array(dat.loc[:, "rt"])
    response = jnp.array(dat.loc[:, "R"], dtype=jnp.float32)

    ll_times_dict = {}
    include_ll = True
    for i in tqdm(range(len(params)), desc="Timing param sets"):
        params_temp = jnp.repeat(params.iloc[i].to_numpy().reshape(1, len(_CONTEXT_DDM)), len(dat), axis=0)

        if train_loader._scaler is not None:
            scaler = train_loader._scaler
            params_temp = scaler.transform(params_temp)

        if include_ll:
            ll, times = joint_log_prob_timed(
                conditioner,
                classifier,
                rt,
                params_temp,
                response,
                n_repeats=10,
            )
            ll_times_dict[i + 1] = {"times": times, "ll": ll.tolist()}
        else:
            _, times = joint_log_prob_timed(
                conditioner,
                classifier,
                rt,
                params_temp,
                response,
                n_repeats=10,
            )
            ll_times_dict[i + 1] = {"times": times}


    with open(f"results/DDM/timing/{bname}/flow_timing.json", "w") as f:
        json.dump(ll_times_dict, f)

# # Change to numpy for numba processing
# dat = dat.to_numpy()
# params = params.to_numpy()

# @njit(parallel=True)
# def compute_times(params):
#     times = np.empty((len(params), 10))
    
#     for i in prange(len(params)):
#         params_temp = jnp.repeat(params[i].reshape(1, len(_CONTEXT_DDM)), len(dat), axis=0)
#         _, times = joint_log_prob_timed(
#                 conditioner,
#                 classifier,
#                 rt,
#                 params_temp,
#                 response,
#                 n_repeats=10,
#             )
        
#         times[i, :] = times
#     return times

# comp_times = compute_times(params)
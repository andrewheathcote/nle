from tqdm import tqdm

import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from flax import nnx
from pathlib import Path
from flow import (
    make_checkpoint_manager,
    make_mlp_conditioner,
    restore_checkpoint,
)
from flow.training import ll_racing_models_timed, load_obj
from flow.data import get_inp_data
import json
import pandas as pd

def load_model(dir, dims, num_bins, num_splines, use_norm, schedule):
    cond_factory = lambda: make_mlp_conditioner(nnx.Rngs(0), dims=dims, num_bins=num_bins, num_splines=num_splines, use_norm=use_norm)
    opt_factory = lambda: nnx.Optimizer(cond_factory(), optax.adamw(schedule), wrt=nnx.Param)
    mgr = make_checkpoint_manager(dir)
    conditioner, _ = restore_checkpoint(
        mgr, mgr.best_step(), cond_factory, opt_factory
    )
    return conditioner

# Per-accumulator context, SAME column order as training: [v, B, t0, s, A]
def stack_ctx(suffix, dat):
    cols = [f"v_lR{suffix}", f"B_lR{suffix}", "t0", f"s_lR{suffix}", f"A_lR{suffix}"]
    return jnp.asarray(np.stack(
        [dat.loc[:, c].to_numpy() for c in cols], axis=1))



DATA = "rt" # The normflow is only trained on one accumulator
CONTEXT_SINGLE = ["v", "B", "t0", "s", "A"]
CONTEXT = ["v_lR1", "B_lR1", "t0", "s_lR1", "A_lR1", "v_lR2", "B_lR2", "s_lR2", "A_lR2"]
TARGET   = "ll"
RESPONSE = "R"

train_loader = load_obj("./results/model_states/large_budget/RDM/train_loader.pkl")

with open("./results/model_states/large_budget/RDM/schedule_config.json", "r") as f:
    schedule_config = json.load(f)

schedule = optax.warmup_cosine_decay_schedule(
    init_value=schedule_config["init_value"], 
    peak_value=schedule_config["peak_value"], 
    warmup_steps=schedule_config["warmup_steps"], 
    decay_steps=schedule_config["decay_steps"]
)


conditioner = load_model(
    "./results/model_states/large_budget/RDM", 
    dims=[len(CONTEXT), 32],
    num_bins=32,
    num_splines=1,
    use_norm=False,
    schedule=schedule
)

# try this out with a single budget
dat, params = get_inp_data(
    Path("benchmark/scripts_julian/RDM/N=1000/trials"),
    Path("benchmark/scripts_julian/RDM/eval_params.csv"),
    CONTEXT
)

rt = jnp.array(dat.loc[:, "rt"])
pars_true = jnp.array(dat.loc[:, CONTEXT])
response = jnp.array(dat.loc[:, "R"], dtype=jnp.float32)

if train_loader._scaler is not None:
    scaler = train_loader._scaler
    ctx_1 = scaler.transform(stack_ctx(1, dat))
    ctx_2 = scaler.transform(stack_ctx(2, dat))
else:
    ctx_1 = stack_ctx(1, dat)
    ctx_2 = stack_ctx(2, dat)


pred_ll, times = ll_racing_models_timed(conditioner, rt, ctx_1, ctx_2, response, n_repeats=1, warmup=True)

jnp.corrcoef(pred_ll, dat.ll.to_numpy())[0, 1]

# now the full loop
for bname in ["N=100", "N=1000", "N=10000"]:
    dat_path = Path(f"benchmark/scripts_julian/RDM/{bname}/trials")
    params_path = Path(f"benchmark/scripts_julian/RDM/eval_params.csv")
    dat, params = get_inp_data(dat_path, params_path, CONTEXT)

    rt = jnp.array(dat.loc[:, "rt"])
    response = jnp.array(dat.loc[:, "R"], dtype=jnp.float32)

    ll_times_dict = {}
    include_ll = True

    for i in tqdm(range(len(params)), desc="Timing param sets"):
        
        params_temp = pd.concat([params.iloc[[i]]] * len(dat), ignore_index=True)
        if train_loader._scaler is not None:
            scaler = train_loader._scaler
            ctx_1 = scaler.transform(stack_ctx(1, params_temp))
            ctx_2 = scaler.transform(stack_ctx(2, params_temp))
        else:
            ctx_1 = stack_ctx(1, params_temp)
            ctx_2 = stack_ctx(2, params_temp)

        if include_ll:
            pred_ll, times = ll_racing_models_timed(conditioner, rt, ctx_1, ctx_2, response, n_repeats=10, warmup=True)
            ll_times_dict[i + 1] = {"times": times, "ll": pred_ll.tolist()}
        else:
            _, times = ll_racing_models_timed(conditioner, rt, ctx_1, ctx_2, response, n_repeats=10, warmup=True)
            ll_times_dict[i + 1] = {"times": times}

    with open(f"results/RDM/timing/{bname}/flow_timing.json", "w") as f:
        json.dump(ll_times_dict, f)


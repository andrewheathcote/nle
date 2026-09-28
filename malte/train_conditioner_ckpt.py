"""Malte Lueken's RDM flow "as published", trained with his code, with checkpoints retained.

Stage M7 (seed study), arm 5. His `scripts/train_conditioner.py` saves one conditioner, at
the end. The seed study screens 30 checkpoints per run, so this is his training loop,
statement for statement, with a save every `--save-every` steps over the last
`--keep` * `--save-every` steps. A save is a side effect: the weights at step N are
the ones his script has at step N, and the last retained checkpoint is the one his script
writes. Objective, optimiser step, sampler, prior and checkpoint format are imported from
his packages; the configuration is composed from his `conf_jax` with the overrides of
`slurm/train_conditioner_rdm_affine_log_deep_clip_box.sh` (DEFAULT_OVERRIDES below, minus
`device=gpu`). Nothing in his clone is edited.

    malte/.venv/bin/python malte/train_conditioner_ckpt.py --out results/malte_runs/malte_s2026 \
        --seed 2026 [--steps 100000] [--save-every 1000] [--keep 30]

Writes <out>/ckpt_s<step>/ (Orbax + eamax_conditioner.json sidecar, what
flow/scripts/malte_to_card.py reads), <out>/training_log.json and <out>/run_config.json.
The box for malte_to_card.py --bounds is printed and recorded in run_config.json.
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONF = ROOT / "neural-race-model-estimation" / "conf_jax"

DEFAULT_OVERRIDES = [
    "model=rdm",
    "model.flow_affine=true",
    "model.flow_log_inputs=true",
    "model.flow_num_hidden=2",
    "model.num_bins=12",
    "model.num_mid=128",
    "model.training_prior.s_min=0.25",
    "model.training_prior.s_max=3.5",
    "model.training_prior.b_min=0.25",
    "model.training_prior.b_max=3.5",
    "optimizer=adam_cosine_decay_clip",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026, help="his train_seed (published: 2026)")
    ap.add_argument("--steps", type=int, default=100000, help="his train_steps (published: 100000)")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--keep", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("overrides", nargs="*", help="further Hydra overrides, after the published ones")
    a = ap.parse_args()
    if a.keep * a.save_every > a.steps or a.steps % a.save_every:
        raise SystemExit("need keep * save_every <= steps and steps a multiple of save_every")

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    overrides = DEFAULT_OVERRIDES + [f"train_steps={a.steps}", f"train_seed={a.seed}",
                                     f"device={a.device}"] + list(a.overrides)
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    import jax
    from flax import nnx
    from confrdm_jax import configure_jax
    from confrdm_jax.flows_affine import (Maximum, flow_options, make_mlp_conditioner,
                                          save_conditioner, train_step)

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # ---- from here to the loop: scripts/train_conditioner.py, unchanged ----
    configure_jax(cfg["device"], require_device=cfg["require_device"])
    prior = instantiate(cfg["model"]["training_prior"])
    sampler = instantiate(cfg["model"]["sampler"], _partial_=True)
    train_steps = cfg["train_steps"]
    eval_every = max(1, int(train_steps // 10))
    batch_shape = (cfg["train_batch_size"], cfg["train_num_obs"])
    t_max = cfg["model"]["sampler"].get("t_max", None)
    context_names = list(cfg["model"]["context_names"])

    train_key = jax.random.key(cfg["train_seed"])
    conditioner_key, sampling_key = jax.random.split(train_key, 2)
    rngs = nnx.Rngs(default=conditioner_key, sampling=sampling_key)
    conditioner = make_mlp_conditioner(
        num_in=len(context_names),
        num_bins=cfg["model"]["num_bins"],
        num_mid=cfg["model"]["num_mid"],
        **flow_options(cfg["model"]),
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(conditioner, instantiate(cfg["optimizer"]), wrt=nnx.Param)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        grad_norm=nnx.metrics.Average("grad_norm"),
        grad_norm_max=Maximum("grad_norm"),
    )

    bounds = OmegaConf.to_container(cfg["model"]["flow_context_bounds"], resolve=True)
    box = ",".join(f"{n}={float(bounds[n][0]):g}:{float(bounds[n][1]):g}" for n in context_names)
    (out / "run_config.json").write_text(json.dumps({
        "overrides": overrides, "context_names": context_names, "bounds": box,
        "train_steps": int(train_steps), "train_seed": int(cfg["train_seed"]),
        "batch_shape": list(batch_shape), "save_every": a.save_every, "keep": a.keep}, indent=2))
    print(f"[{time.strftime('%H:%M:%S')}] malte rdm  seed={cfg['train_seed']}  steps={train_steps}  "
          f"batch={batch_shape}  context={context_names}  box {box}  -> {out}", flush=True)

    first_kept = train_steps - a.keep * a.save_every
    log = {"step": [], "loss": [], "grad_norm": [], "grad_norm_max": [], "saved": []}
    t_start = time.time()
    for step in range(train_steps):
        data, context = sampler(rngs.sampling(), batch_shape, prior)
        conditioner.train()
        train_step(conditioner, optimizer, metrics, data, context, t_max)

        if step == train_steps - 1 or (step > 0 and step % eval_every == 0):
            computed = metrics.compute()
            metrics.reset()
            log["step"].append(step)
            for k in ("loss", "grad_norm", "grad_norm_max"):
                log[k].append(float(computed[k]))
            print(f"[{time.strftime('%H:%M:%S')}] step {step}  Loss: {float(computed['loss']):.5f} | "
                  f"grad norm mean {float(computed['grad_norm']):.4g} max "
                  f"{float(computed['grad_norm_max']):.4g}", flush=True)

        done = step + 1                      # weights after `done` optimiser steps
        if done > first_kept and done % a.save_every == 0:
            save_conditioner(conditioner, out / f"ckpt_s{done:06d}", context_names=context_names,
                             num_mid=cfg["model"]["num_mid"], num_bins=cfg["model"]["num_bins"])
            log["saved"].append(done)
            (out / "training_log.json").write_text(json.dumps(log))

    log["total_time_s"] = time.time() - t_start
    (out / "training_log.json").write_text(json.dumps(log))
    print(f"[{time.strftime('%H:%M:%S')}] training done in {log['total_time_s'] / 3600:.2f} h, "
          f"{len(log['saved'])} checkpoints -> {out}", flush=True)


if __name__ == "__main__":
    main()

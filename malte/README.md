# malte/ — environment and fixtures for Malte Lüken's flows

Separate from `flow/.venv` (his `eamax` needs jax >= 0.9; ours has 0.5.3). CPU only.

Rebuild:

    cd malte
    git clone https://github.com/maltelueken/eamax.git                      # v0.1.1, 6d7f2ff
    git clone -b development https://github.com/maltelueken/neural-race-model-estimation.git   # fae8add
    uv venv --python 3.11 .venv && source .venv/bin/activate
    uv pip install -e "./eamax[flows,inference]" scipy pyyaml hydra-core   # hydra-core: his trainer's config (stage M7a)
    uv pip install --no-deps -e ./neural-race-model-estimation   # its git pin of eamax conflicts with the local clone
    python make_fixtures.py       # -> fixtures/<name>/{ckpt,card.json,golden.csv}

Then `Rscript port/tests/validate_malte.R` from the repo root. `requirements.lock` is the
resolved environment. The clones, `.venv` and `fixtures/*/ckpt` are not tracked.
Converter: `flow/scripts/malte_to_card.py`.

Training his RDM flow as published, with checkpoints retained (seed study, stage M7):
`malte/.venv/bin/python malte/train_conditioner_ckpt.py --out results/malte_runs/<tag> --seed 2026`.
It is his loop with his objective, sampler, optimiser and configuration
(`slurm/train_conditioner_rdm_affine_log_deep_clip_box.sh`), plus a save every 1000 steps over
the last 30 000. About 40 steps per second on 8 laptop cores, so 100k steps take about 45 minutes.

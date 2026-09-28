# malte/ — environment and fixtures for Malte Lüken's flows

Separate from `flow/.venv` (his `eamax` needs jax >= 0.9; ours has 0.5.3). CPU only.

Rebuild:

    cd malte
    git clone https://github.com/maltelueken/eamax.git                      # v0.1.1, 6d7f2ff
    git clone -b development https://github.com/maltelueken/neural-race-model-estimation.git   # fae8add
    uv venv --python 3.11 .venv && source .venv/bin/activate
    uv pip install -e "./eamax[flows,inference]" scipy pyyaml
    uv pip install --no-deps -e ./neural-race-model-estimation   # its git pin of eamax conflicts with the local clone
    python make_fixtures.py       # -> fixtures/<name>/{ckpt,card.json,golden.csv}

Then `Rscript port/tests/validate_malte.R` from the repo root. `requirements.lock` is the
resolved environment. The clones, `.venv` and `fixtures/*/ckpt` are not tracked.
Converter: `flow/scripts/malte_to_card.py`.

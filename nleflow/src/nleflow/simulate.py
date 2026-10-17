"""`nleflow simulate <spec>`: run the spec's simulator command.

The command runs in the spec file's directory with these variables set, and
must write the data contract (data.py) to NLEFLOW_TRAIN and NLEFLOW_VAL:

    NLEFLOW_SPEC       the resolved spec as JSON (box, transforms, strata, ...)
    NLEFLOW_TRAIN      directory for the training files
    NLEFLOW_VAL        directory for the validation files
    NLEFLOW_ROWS       training rows (training.rows; empty: the simulator's default)
    NLEFLOW_VAL_ROWS   validation rows (training.val_rows; empty likewise)
    NLEFLOW_SEED       training.seed; the simulator derives the validation seed
    NLEFLOW_CORES      cores to use (--cores, at most 8 by default)
    NLEFLOW_TRIALS_PER_DRAW   training.trials_per_draw
  and for a choice model (exact P(R = 2) targets for the classifier):
    NLEFLOW_CHOICE_TRAIN, NLEFLOW_CHOICE_VAL   the two files to write
    NLEFLOW_CHOICE_ROWS, NLEFLOW_CHOICE_VAL_ROWS   their row counts (may be empty)
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .spec import dump_json


def default_cores():
    return max(1, min(8, os.cpu_count() or 1))


def simulate(spec, cores=None, say=print):
    if not spec.get("simulator"):
        raise ValueError("the spec has no 'simulator' command")
    cwd = Path(spec["spec_file"]).parent if spec.get("spec_file") else Path.cwd()
    resolved = cwd / f".{spec['name']}.resolved.json"
    dump_json(spec, resolved)
    tr = spec["training"]
    env = {**os.environ,
           "NLEFLOW_SPEC": str(resolved),
           "NLEFLOW_TRAIN": spec["data"]["train"],
           "NLEFLOW_VAL": spec["data"]["val"],
           "NLEFLOW_ROWS": "" if tr["rows"] is None else str(tr["rows"]),
           "NLEFLOW_VAL_ROWS": "" if tr["val_rows"] is None else str(tr["val_rows"]),
           "NLEFLOW_SEED": str(tr["seed"]),
           "NLEFLOW_TRIALS_PER_DRAW": str(tr["trials_per_draw"]),
           "NLEFLOW_CORES": str(cores or default_cores())}
    if spec["response"] == "choice":
        env.update({"NLEFLOW_CHOICE_TRAIN": spec["data"]["choice_train"],
                    "NLEFLOW_CHOICE_VAL": spec["data"]["choice_val"],
                    "NLEFLOW_CHOICE_ROWS": "" if tr["choice_rows"] is None else str(tr["choice_rows"]),
                    "NLEFLOW_CHOICE_VAL_ROWS": "" if tr["choice_val_rows"] is None else str(tr["choice_val_rows"])})
    say(f"running in {cwd}: {spec['simulator']}")
    r = subprocess.run(spec["simulator"], shell=True, cwd=str(cwd), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"the simulator exited with status {r.returncode}")
    return spec["data"]

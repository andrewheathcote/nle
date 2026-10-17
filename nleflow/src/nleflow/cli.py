"""The command line: nleflow spec | simulate | train | export | golden."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__


def _limit_threads(cores):
    cores = str(cores)
    for v in ("OMP_NUM_THREADS", "XLA_FLAGS"):
        if v == "XLA_FLAGS":
            os.environ.setdefault(v, f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={cores}")
        else:
            os.environ.setdefault(v, cores)


def cmd_spec(a):
    from .spec import load_spec
    spec = load_spec(a.spec)
    print(json.dumps(spec, indent=1))


def cmd_simulate(a):
    from .simulate import simulate
    from .spec import load_spec
    simulate(load_spec(a.spec), cores=a.cores)


def cmd_train(a):
    from .simulate import default_cores
    _limit_threads(a.cores or default_cores())
    from .spec import load_spec
    from .train import train
    spec = load_spec(a.spec)
    runs = a.runs or str(Path(spec["spec_file"]).parent / "runs")
    train(spec, runs=runs, epochs=a.epochs, classifier_from=a.classifier_from)


def cmd_export(a):
    from .simulate import default_cores
    _limit_threads(a.cores or default_cores())
    from .export import export
    export(a.run, step=a.step, out=a.out, tag=a.tag)


def cmd_golden(a):
    from .golden import main
    main(a.card, a.out)


def cmd_onnx(a):
    from .onnx_export import export_onnx
    for sig in (["trialwise", "shared"] if a.signature == "both" else [a.signature]):
        export_onnx(a.card, out=None if a.out is None or a.signature == "both" else a.out, signature=sig)


def cmd_onnx_check(a):
    import json
    from .onnx_export import check_onnx
    ok = check_onnx(a.model, a.card, golden=a.golden, cases=a.cases, tol_density=a.tol, tol_window=a.tol_window)
    sys.exit(0 if ok else 1)


def cmd_steps(a):
    from .export import checkpoint_steps, step_metrics
    mgr, steps, best, latest = checkpoint_steps(a.run)
    for s in steps:
        v = step_metrics(a.run, s).get("val_loss")
        print(f"step {s}: val_loss {'?' if v is None else f'{v:.6f}'}" + ("  (best)" if s == best else ""))


def build_parser():
    p = argparse.ArgumentParser(prog="nleflow",
                                description="Train a normalizing-flow neural likelihood from a model "
                                            "specification and export it as an EMC2 card.")
    p.add_argument("--version", action="version", version=f"nleflow {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("spec", help="validate a spec and print it resolved")
    s.add_argument("spec"); s.set_defaults(fn=cmd_spec)

    s = sub.add_parser("simulate", help="run the spec's simulator command (see nleflow.simulate)")
    s.add_argument("spec"); s.add_argument("--cores", type=int, default=None)
    s.set_defaults(fn=cmd_simulate)

    s = sub.add_parser("train", help="train the spec; checkpoints under <runs>/<name>")
    s.add_argument("spec")
    s.add_argument("--runs", default=None, help="runs directory (default: runs/ beside the spec)")
    s.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    s.add_argument("--cores", type=int, default=None, help="CPU threads (default: min(8, cores))")
    s.add_argument("--classifier-from", default=None,
                   help="choice models: reuse this run's classifier instead of training one")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("export", help="export one checkpoint of a run as a card_version 1 card")
    s.add_argument("run")
    s.add_argument("--step", type=int, default=None, help="checkpoint step (default: best validation loss)")
    s.add_argument("--out", default=None, help="output directory (default: <run>/export)")
    s.add_argument("--tag", default=None, help="file tag (default: <name>_e<step>)")
    s.add_argument("--cores", type=int, default=None)
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("steps", help="list a run's kept checkpoints with their validation loss")
    s.add_argument("run"); s.set_defaults(fn=cmd_steps)

    s = sub.add_parser("golden", help="rewrite a card's golden vectors from its weights JSON")
    s.add_argument("card"); s.add_argument("out", nargs="?", default=None)
    s.set_defaults(fn=cmd_golden)

    s = sub.add_parser("onnx", help="export a DDM flow card (flow_joint, flow_scaled) as an ONNX graph")
    s.add_argument("card")
    s.add_argument("--signature", default="both", choices=["trialwise", "shared", "both"],
                   help="trialwise [N, k+2] (the LAN shape), shared theta[k] + rt[N] + choice[N], or both (default)")
    s.add_argument("--out", default=None, help="output path (one signature only; default <card>_<signature>.onnx)")
    s.set_defaults(fn=cmd_onnx)

    s = sub.add_parser("onnx-check", help="check an ONNX graph against the card's golden rows and EMC2 cases")
    s.add_argument("model"); s.add_argument("card")
    s.add_argument("--golden", default=None, help="the card's golden CSV (default: beside the card)")
    s.add_argument("--cases", default=None, help="EMC2 reference cases from scripts/onnx_cases_scaled.R")
    s.add_argument("--tol", type=float, default=1e-10, help="max |dlog p| where the likelihood is a density")
    s.add_argument("--tol-window", type=float, default=1e-8,
                   help="max |dlog p| on the st0 > 0 branch (a CDF difference; see the README)")
    s.set_defaults(fn=cmd_onnx_check)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        a.fn(a)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"nleflow {a.cmd}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

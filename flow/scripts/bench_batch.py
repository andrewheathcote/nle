"""How much of the machine does a training step actually use, by batch size?

Measured 2026-09-12: a production run (--batch 128, [256,128] conditioner,
64 bins) sustains ~2.8 of 12 cores. 74 threads exist and 5 are running. The
matmuls at batch 128 are far too small for XLA to parallelise, so the run is
bound by per-step overhead, not compute — every 12 h training run in this
project has left roughly three quarters of the machine idle.

This measures throughput AND core utilisation across batch sizes, on synthetic
data of the right shape (no disk I/O), so the answer is about the computation
rather than the loader.

Batch size changes the optimisation trajectory, so it is NOT a free speedup for
a run being compared against an existing one. It is a real speedup for new work.

  python scripts/bench_batch.py
"""
import os, sys, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import jax, jax.numpy as jnp, optax
from flax import nnx
from flow.config import make_mlp_conditioner
from flow.training import loss_fn

BATCHES = [128, 256, 512, 1024, 4096, 16384]
NCTX, WIDTHS, BINS = 9, [256, 128], 64
STEPS = 40


def cpu_percent(stop, out):
    """Sample this process's CPU% while the timed loop runs."""
    import subprocess
    pid = os.getpid()
    vals = []
    while not stop.is_set():
        try:
            r = subprocess.run(["ps", "-o", "%cpu=", "-p", str(pid)],
                               capture_output=True, text=True, timeout=3)
            vals.append(float(r.stdout.strip()))
        except Exception:
            pass
        time.sleep(0.4)
    out.extend(vals)


def main():
    ncpu = os.cpu_count()
    print(f"machine: {ncpu} logical cores; jax devices: {jax.devices()}")
    print(f"conditioner dims [{NCTX}]+{WIDTHS}, {BINS} bins, {STEPS} timed steps\n")
    print(f"{'batch':>7} {'ms/step':>9} {'rows/s':>12} {'rel':>7} {'cpu%':>7} {'cores':>6}")
    base = None
    for bs in BATCHES:
        cond = make_mlp_conditioner(nnx.Rngs(0), dims=[NCTX] + WIDTHS, num_bins=BINS,
                                    num_splines=1, use_norm=False,
                                    range_min=-5.0, range_max=5.0)
        opt = nnx.Optimizer(cond, optax.adamw(1e-4), wrt=nnx.Param)
        metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

        @nnx.jit
        def step(c, o, m, d, x):
            gf = nnx.value_and_grad(loss_fn)
            l, g = gf(c, d, x)
            o.update(g); m.update(loss=l); return l

        rng = np.random.default_rng(0)
        ctx = jnp.asarray(rng.normal(size=(bs, NCTX)).astype("float32"))
        dat = jnp.asarray(np.abs(rng.normal(size=bs)).astype("float32") + 0.2)

        for _ in range(5):                      # warm up / compile
            step(cond, opt, metrics, dat, ctx)
        jax.block_until_ready(metrics.compute()["loss"])

        stop = threading.Event(); samples = []
        t = threading.Thread(target=cpu_percent, args=(stop, samples)); t.start()
        t0 = time.perf_counter()
        for _ in range(STEPS):
            step(cond, opt, metrics, dat, ctx)
        jax.block_until_ready(metrics.compute()["loss"])
        el = time.perf_counter() - t0
        stop.set(); t.join()

        ms = el / STEPS * 1000
        rows = bs * STEPS / el
        if base is None:
            base = rows
        cpu = float(np.median(samples)) if samples else float("nan")
        print(f"{bs:>7} {ms:>9.2f} {rows:>12,.0f} {rows/base:>6.1f}x {cpu:>6.0f}% {cpu/100:>6.1f}")

    print("\n'rel' is throughput relative to batch 128 (the current default).")
    print("An epoch is 16M rows; divide by rows/s for epoch seconds.")


if __name__ == "__main__":
    main()

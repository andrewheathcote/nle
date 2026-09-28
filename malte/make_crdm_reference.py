"""Golden values of eamax's Volterra solver and simulator for the CRDM box (stage M2).

    cd /Users/andrew/Downloads/julian/nle/malte && source .venv/bin/activate && python make_crdm_reference.py

Writes crdm_ref/{sets.csv, grid_dt<dt>.csv.gz, draws.csv.gz}. float64, CPU.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import gzip
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc
from eamax.accumulators.volterra import solve_volterra_fpt
from eamax.accumulators.diffusion import SimulatedPulsedWald

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crdm_ref")
os.makedirs(OUT, exist_ok=True)
T_MAX = 3.0
DTS = (0.002, 0.0005)
SIM_DT = 0.001
N_SETS, N_DRAWS, SIM_SETS = 20, 100_000, (0, 5, 10, 15, 19)
lower = np.array([0.0, 0.0, 0.01, 0.25, 0.25])   # v amp tau s b  (crdm_final box)
upper = np.array([8.0, 1.0, 0.5, 3.0, 3.0])

u = qmc.Sobol(5, scramble=True, seed=20260928).random(N_SETS)
theta = lower + u * (upper - lower)
theta[[3, 8, 13], 1] = 0.0                       # three amp = 0 sets (Wald limit)
theta[:, 0] = np.maximum(theta[:, 0], 0.5)       # keep some mass inside T_MAX
theta[:, 4] = np.minimum(theta[:, 4], 2.0)
names = ["v", "amp", "tau", "s", "b"]
with open(f"{OUT}/sets.csv", "w") as f:
    f.write("set," + ",".join(names) + "\n")
    for i, th in enumerate(theta):
        f.write(f"{i}," + ",".join(repr(float(x)) for x in th) + "\n")

for dt in DTS:
    n = int(round(T_MAX / dt))
    t0 = time.time()
    rows = []
    for i, (v, amp, tau, s, b) in enumerate(theta):
        g, G = solve_volterra_fpt(v, amp, tau, s, b, dt, n)
        g, G = np.asarray(g), np.asarray(G)
        rows.append(np.column_stack([np.full(n, i), np.arange(1, n + 1) * dt, g, G]))
    a = np.vstack(rows)
    with gzip.open(f"{OUT}/grid_dt{dt}.csv.gz", "wt") as f:
        f.write("set,t,pdf,cdf\n")
        np.savetxt(f, a, delimiter=",", fmt=["%d", "%.17g", "%.17g", "%.17g"])
    print(f"dt={dt}: {N_SETS} sets in {time.time() - t0:.1f}s")

sim = SimulatedPulsedWald(dt=SIM_DT, t_max=T_MAX)
draws = []
for k, i in enumerate(SIM_SETS):
    p = {nm: jnp.full(N_DRAWS, theta[i, j]) for j, nm in enumerate(names)}
    draws.append(np.asarray(sim.sample(jax.random.PRNGKey(100 + k), p)))
with gzip.open(f"{OUT}/draws.csv.gz", "wt") as f:
    f.write("set,draw\n")
    for i, d in zip(SIM_SETS, draws):
        np.savetxt(f, np.column_stack([np.full(len(d), i), d]), delimiter=",", fmt=["%d", "%.17g"])
print("censored fraction per simulated set:", [float(np.isinf(d).mean()) for d in draws])
print("sim dt", SIM_DT, "t_max", T_MAX)

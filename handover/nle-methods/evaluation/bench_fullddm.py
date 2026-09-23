# HSSM's blackbox full_ddm likelihood (hddm_wfpt.wiener_logp_array, err=1e-8),
# called exactly as hssm.likelihoods.blackbox.logp_full_ddm calls it.
#   python bench_fullddm.py v a w t0 st0 sv sz n reps
import sys, time, json, numpy as np
from hddm_wfpt import wfpt
v, a_w, w, t0, st0, sv, sz = map(float, sys.argv[1:8])
n, reps = int(sys.argv[8]), int(sys.argv[9])
rng = np.random.default_rng(7)
data = (rng.uniform(t0 + 0.05, t0 + 1.2, n) * rng.choice([1.0, -1.0], n)).astype(np.float64)
B = lambda x: np.broadcast_to(np.array(x, dtype=np.float64), n)
call = lambda: wfpt.wiener_logp_array(x=data, v=B(v), sv=B(sv), a=B(a_w), z=B(w),
                                      sz=B(sz), t=B(t0 + st0 / 2), st=B(st0), err=1e-8)
out = call()
t = time.perf_counter()
for _ in range(reps): call()
print(json.dumps({"ms_per_1000": (time.perf_counter() - t) / reps * 1000 * 1000 / n,
                  "finite": int(np.isfinite(out).sum())}))

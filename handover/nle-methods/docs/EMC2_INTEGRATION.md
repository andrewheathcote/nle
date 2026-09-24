# Bringing a flow into EMC2

Everything here was learned by getting it wrong first. The failure modes are
silent, which is why each has a verification step attached.

---

## 1. Install the dev-nn build into a DEDICATED library

The `dev-nn` branch of `ampl-psych/EMC2` carries the compiled port and the
`DDMnn` / `RDMnn` / `LNRnn` model functions.

**It reports the same version (3.4.1) as the release build**, so the two
overwrite each other silently. That happened here: a plain 3.4.1 landed in the
default user library and every likelihood floored at `-23.03` until it was
found.

```r
.libPaths(c(normalizePath(".Rlibs"), .libPaths()))   # before library(EMC2)
library(EMC2)
find.package("EMC2")   # CHECK THIS, not packageVersion()
```

Every script in `evaluation/` opens with that `.libPaths` line. Keep it.

## 2. Weights live in extdata, under HARDCODED names

`DDMnn` → `dDDMnn` → `ddmnn_get("ddm_ens5")` → reads
`<EMC2>/extdata/flownn/ddm_ens5.rds`.

The name is hardcoded, so **swapping a model means replacing the file**, not
adding one. Consequences:

- back the original up before overwriting, and restore it afterwards *whatever
  the outcome* — a killed job once left a test bundle installed and every later
  local run silently used the wrong model;
- two SBC cells using different bundles cannot run concurrently against the same
  install;
- **print which artefact is installed at the start of every run.** All the
  drivers here do:

```r
b <- readRDS(system.file("extdata", "flownn", "ddm_ens5.rds", package = "EMC2"))
m <- if (is.null(b$members)) b else b$members[[1]]
cat(sprintf("installed: budget=%s step=%s bins=%d context=%s\n", m$budget,
            m$checkpoint_step_flow, m$spline$num_bins,
            paste(m$context_names, collapse = ",")))
```

## 3. The theta builders hardcode the context order

`EMC2:::.theta_DDMnn` is:

```r
cbind(pars[,"v"], log(pars[,"a"]), log(pars[,"t0"]), log(pars[,"s"]),
      qnorm(pars[,"Z"]), qnorm(pars[,"SZ"]), log(pars[,"sv"]), log(pars[,"st0"]))
```

It **ignores the export's `context_names`**. `dRDMnn` has the same property.

- A flow trained on a different **order** is silently mis-wired. The signature
  is a score bias ~500× too large that is *uniform across checkpoints* — a badly
  trained model varies, a mis-wired one does not. Use
  `evaluation/permute_context.R` to re-express a flow in another order.
- A flow with a different **width** needs a parallel model function. See
  `evaluation/DDMnn7.R` for the 7-parameter (st0 = 0) case: it copies
  `EMC2::DDMnn` and changes only the theta builder and the bundle name, and
  asserts `context_names` on load so a mismatch fails loudly.

**The compiled port itself is dimension-generic.** Both
`ddm_eval_trials_cpp` and `ddm_ens_eval_trials_cpp` accept a 7-column bundle and
return pdf and cdf. No C++ change is needed for a different parameter count —
only the R-level theta builder.

## 4. EMC2 gives st0 (and sv) a default of -Inf

```r
DDM()$p_types      # st0 = -Inf,  transform "exp"
```

So **omitting st0 from a design formula sets it to exactly 0** — no `-Inf` need
be passed by hand. Verified through `mapped_pars`.

This is convenient for simulation and a trap for evaluation: with st0 omitted,
`pars` still has an `st0` column (of zeros), and the 8-column theta builder then
feeds `log(0) = -Inf` to the flow.

## 5. `prior()` takes pmean POSITIONALLY

`sampled_pars()` returns parameters in **formula order**, not the canonical
model order. Writing `formula = list(..., Z ~ 1, SZ ~ 1, sv ~ 1)` rather than
`(Z, sv, SZ)` silently transposes two priors.

Look the values up by name in whatever order `sampled_pars` returns:

```r
PM <- c(v = 1, a = log(.8), t0 = log(.3), Z = qnorm(.5), sv = log(.1), SZ = qnorm(.2))
sp <- names(sampled_pars(des))
stopifnot(all(sp %in% names(PM)), length(sp) == length(PM))
pmean <- PM[sp]
```

An assertion caught this one before it ran; a hardcoded vector would not have.

## 6. Building and verifying a bundle

`evaluation/make_bundle_st0zero.R` is the pattern. It asserts the exact context
and encoding, checks that every member shares a scaler and spline geometry, and
round-trips the saved bundle against the loaded exports (`max |diff| < 1e-12`)
before declaring success.

Verify a freshly installed artefact three ways:

1. **Golden vectors** — `port/tests/validate_ddm.R`, the float64 R port against
   the exported JAX values.
2. **Total mass** — `∫ Σ_R exp(f̂) drt` should be 1. A flow gets this by
   construction; a regression net does not.
3. **The installed-bundle line** above, on every run.

## 7. Known gap

`evaluation/DDMnn7.R` is not in EMC2 — it is sourced alongside an unmodified
install. Folding it in (as `DDMnn7` / `.theta_DDMnn7` / `pDDMnn7` plus a
`ddm_st0zero.rds` bundle name) is a small PR onto `dev-nn` and is the natural
home for it. The same applies to any decision-time (`rt - t0`) variant, which
additionally needs the shift applied in the evaluator before the flow is called.


---

## 8. An ONNX build, for HSSM

`artefacts/ddm_cap256w_c4.onnx` is the same flow as a single ONNX graph, built
by `flow/scripts/export_onnx_ddm.py` (with the `onnx` helper API; no torch
needed); `artefacts/ddm_cap256w_c4_shared.onnx` is the same model with a
shared-theta signature, 25x faster, built by passing `shared` as the third
argument. It exists so the artefact can be offered to HSSM, whose likelihood
interface is exactly this: a graph from (parameters, rt, choice) to a joint
log-likelihood.

**Interface.** One input, float64 `[N, 10]`:

| columns | contents |
|---|---|
| 0-7 | the context on the **sampled** scale, in the export's order: `v`, `log a`, `log t0`, `log s`, `probit Z`, `probit SZ`, `log sv`, `log st0` |
| 8 | `rt` in seconds |
| 9 | `choice`, −1 = lower boundary, +1 = upper (our R = 1, 2) |

Output `[N]`: `log p(rt, choice | theta)`, the joint defective log-density — the
same quantity a LAN returns, so it drops into the same slot.

**Why the sampled scale.** The probit transform has no exact ONNX op. Taking
natural-scale `Z` and `SZ` would mean baking in a polynomial approximation;
instead the caller applies `log()` and `qnorm()`, where they are exact. Given
how much a silently mis-wired context has cost this project, an explicit
transform outside the graph is the safer contract.

**Verified**, `flow/scripts/onnx_check.py` against cases from the R port
(`flow/scripts/onnx_cases.R`): **max |diff| 2.4e-12, median 1.7e-14** over 400
random draws. The only deliberate difference is out of the training box, where
the port returns `-Inf` and the graph returns `ll_floor_log` (−23.03) — which is
also what EMC2's `log_likelihood_ddm` floors to.

**Two builds, because the interface decides the speed.**

| build | signature | ms / 1000 trials, one core |
|---|---|---|
| `ddm_cap256w_c4.onnx` | `[N, 10]`, LAN-compatible | 14.5 |
| `ddm_cap256w_c4_shared.onnx` | `theta [8]`, `rt [N]`, `choice [N]` | **0.58** |
| compiled C++ port, via EMC2 | — | 1.70 |

The 25x between the two ONNX builds is **not** an ONNX inefficiency: it is what
the LAN-style interface costs. A trial-wise graph re-runs the whole network for
every trial — 59,776 MACs of conditioner plus 133,376 of classifier, so 193
MMACs per 1000 trials. When the trials share one theta, as they do inside any
likelihood call, the conditioner needs to run twice (once per response) and the
classifier once: 0.25 MMACs, then a 64-bin spline inversion per trial. That is
**764x less network work**, and it is why the compiled port was always faster
than the trial-wise graph.

So: use `_shared` for fitting (it is faster than our own compiled path, since it
also avoids re-deriving theta per row and the R-to-C++ marshalling), and ship
the trial-wise build only where a LAN-shaped interface is required.

float32 was tested and rejected: 0.51 ms against 0.57, an 11% gain for a max
error of 1.9e-04 against the R port instead of 2.4e-12.

**A note for anyone benchmarking against a published LAN.** Those files are
exported with a **rank-1 signature** (shape `[7]`, one trial per call) and HSSM
batches by converting the graph, not by calling onnxruntime per trial. So a
like-for-like batched timing between the two ONNX files is not available: our
graph takes `[N, 10]`, theirs takes `[7]`. Timing theirs per call gives 8.0 ms
per 1000, which is call-overhead dominated and should not be quoted as a
comparison. The honest speed comparison remains the one in `METHODS.md` §3.8,
where both networks were re-implemented identically in R.

**What a contribution would still need**: a `ddm_full` model config in HSSM with
`list_params` `["v", "a", "z", "t", "sz", "sv", "st"]`, the transform wrapper
above, bounds taken from `bounds_natural`, and — the part this project would
insist on — an SBC cell run through HSSM's own sampler rather than EMC2's.


---

## 9. Where the time goes in `dDDMnn` (profiled 2026-09-24)

Milliseconds per call, one core, `cap256w_c4`:

| stage | n = 1 | n = 1000 | n = 10000 |
|---|---|---|---|
| `ddmnn_get()` (pointer cache) | 0.000 | 0.000 | 0.000 |
| `.theta_DDMnn()` (R: log/qnorm/cbind) | 0.005 | 0.065 | 0.600 |
| **`ddm_ens_eval_trials_cpp`** | **0.865** | 1.220 | 4.500 |
| `dDDMnn()` total | 0.845 | 1.300 | 5.250 |

So a call costs **~0.87 ms fixed plus ~0.36 ms per 1000 trials**, and the fixed
part is entirely inside the C++ evaluator. The R side is negligible.

**The fixed cost is the per-theta conditioning, and it is scalar code.**
Conditioning one theta is 0.25 MMACs (the conditioner twice, once per response,
plus the classifier once). Taking 0.87 ms for that is ~0.6 GFLOP/s, which is
what a hand-written triple loop gives: `mlp_forward` in `port/src/flow_ddm.cpp`
multiplies with nested `for` loops and no BLAS call. The same arithmetic runs in
0.13 ms under onnxruntime.

**The fix is local, and it is not "switch to ONNX".** Route `mlp_forward`'s
matrix products through BLAS (`R_ext/BLAS.h`'s `dgemv_`, already linked in every
R package) instead of scalar loops. On these sizes that should take the fixed
cost from ~0.87 ms toward ~0.2 ms, i.e. **roughly a third off a typical N = 400
likelihood call**, with no new dependency and no second evaluation path to keep
in sync. An ensemble bundle gains proportionally more, since it conditions once
per member.

For the record, the ONNX alternative was measured rather than assumed: its
marginal cost is the same as the C++ port's (0.45 vs 0.43 ms per 1000 trials),
so the only thing it would buy is that fixed overhead — and calling it from R
needs either reticulate or an Rcpp binding to the ORT C API, either of which
gives some of it back. Fix the loop instead.

**A trap this profiling exposed.** With a DISTINCT theta per row — any design
with a trial-varying continuous covariate — the same call takes **855 ms per
1000 trials** against 1.22 ms with a shared theta, because the conditioning runs
once per unique row. That is a 700x cliff, and it is invisible in single-subject
fitting where every trial shares one theta. Before using a flow likelihood with
a continuous trial-level regressor, budget for it: the cost becomes
(unique theta count) x 0.87 ms.

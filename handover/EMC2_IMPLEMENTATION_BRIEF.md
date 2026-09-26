# EMC2: two pieces of work on the neural-likelihood path

A brief for whoever redoes the flow integration against current `dev` EMC2.
Everything here was measured on 2026-09-24; the numbers are milliseconds per
1000 trial evaluations, one core, macOS, `cap256w_c4`.

Source material lives in the NLE project (`~/Downloads/julian/nle`):
`handover/nle-methods/` (the shipped package), `port/` (the float64 R and C++
evaluators), `handover/nle-methods/docs/EMC2_INTEGRATION.md` §§8-10 (the
measurements this brief is built on).

---

## 0. Read these first — the traps that have already cost this project weeks

1. **EMC2 ships its own compiled copy of the port.** `EMC2:::ddm_eval_trials_cpp`
   is EMC2's build, not `port/src/flow_ddm.cpp`. Editing one and testing the
   other proves nothing. Any change below must be made in both, or the source of
   truth must be made explicit.
2. **The dev-nn build reports the same version (3.4.1) as the release**, so the
   two silently overwrite each other. Install into a dedicated library and check
   `find.package("EMC2")`, never `packageVersion()`.
3. **Bundle filenames are hardcoded** in `ddmnn_get()`. Swapping a model means
   replacing the file; print which artefact is loaded at the start of every run.
4. **A model's `p_types` defaults differ between analytic and nn variants.**
   `DDM`'s `st0` default is `-Inf` (= 0); `DDMnn`'s is `-2.996` (= 0.05 s).
   Omitting a parameter from a formula therefore simulates a *different model*
   under the two. This invalidated four phases of work before it was found. Any
   new nn model function must either inherit the analytic defaults or assert the
   mapped values (`mapped_pars(des, pmean)`).
5. **`prior()` takes `pmean` positionally** while `sampled_pars()` returns
   formula order. Index a named vector.

---

## 1. Make the compiled evaluator fast (do this during the rebase)

### What is slow, and why

Profiled cost of `EMC2:::dDDMnn`:

| stage | n = 1 | n = 1000 | n = 10000 |
|---|---|---|---|
| `ddmnn_get()` pointer cache | 0.000 | 0.000 | 0.000 |
| `.theta_DDMnn()` (R: log/qnorm/cbind) | 0.005 | 0.065 | 0.600 |
| **`ddm_ens_eval_trials_cpp`** | **0.865** | 1.220 | 4.500 |
| total | 0.845 | 1.300 | 5.250 |

So **~0.87 ms fixed per call plus ~0.36 ms per 1000 trials**, with the fixed
part entirely inside the C++ evaluator. The R side is negligible.

The fixed cost is the per-theta conditioning: the conditioner MLP runs twice
(once per response) and the classifier once, 0.25 MMACs in total. Taking 0.87 ms
for that is ~0.6 GFLOP/s — the signature of scalar code. `mlp_forward` in
`port/src/flow_ddm.cpp` multiplies with nested `for` loops and calls no BLAS.
The same arithmetic runs in 0.13 ms under onnxruntime.

### Two changes, same place

**(a) Route `mlp_forward` through BLAS.** `dgemv_` from `R_ext/BLAS.h` is linked
into every R package already — no new dependency. Expected: fixed cost from
~0.87 ms toward ~0.2 ms, i.e. **roughly a third off a typical N = 400 likelihood
call**, and proportionally more for ensemble bundles, which condition once per
member.

**(b) Batch the conditioner over unique theta rows.** Today the evaluator
conditions once per unique row, one row at a time. With a shared theta that is
one conditioning and costs nothing; with a **distinct theta per row** — any
design with a continuous trial-level covariate — it is 1000 scalar conditionings
and the call takes **839 ms per 1000 trials** against 1.28 ms shared. A batched
GEMM over the unique rows is the same arithmetic: our ONNX graph does exactly
that and takes 14.6 ms. So this is worth ~57x on trial-varying designs.

Concretely, the evaluator already finds unique theta rows in order to cache
them; the change is to stop conditioning them one at a time. Collect the U
unique rows into a U x n_ctx matrix, run each MLP layer as ONE `dgemm_` call
(U x n_in by n_in x n_out) instead of U `dgemv_` calls, apply the activation
over the whole block, build the U x 2 sets of spline knots from the result, then
scatter to trials as now. The per-trial spline inversion is untouched, and with
U = 1 the code path is identical to today's. Do (a) and (b) together: the same
`mlp_forward` rewrite serves both, with U = 1 for the shared-theta case and
U = number of unique rows otherwise.

### Acceptance tests

- `port/tests/validate_ddm.R` — the golden vectors, exported from JAX. Must pass
  unchanged, and note that they compare float64 R against float32 JAX, so the
  tolerance is about the export gate (5e-3 on log density), not about this change.
- Tighter, and the one that actually protects you: the R port and the compiled
  path must agree to ~1e-14 on matched theta and rt. `sbc/test_emc2_vs_port.R`
  in the NLE project does this over many theta, not just at one point.
- One SBC cell before and after is NOT needed — this is an arithmetic change with
  an exact reference — but re-running `evaluation/score_bias_draws.R` on
  `cap256w_c4` and getting the same g to 1e-6 is cheap insurance.

---

## 2. An ONNX reader in EMC2 — for models EMC2 cannot otherwise fit

### Why, in order of importance

1. **Capability.** `lnccbrown/HSSM` ships likelihood approximation networks
   (LANs) for `angle`, `levy`, `ornstein`, `weibull`, `gamma_drift`, `ddm_sdv`,
   `ddm_uniform_st`, `ddm_seq2_no_bias` and several race and LBA variants. EMC2
   has analytic likelihoods for almost none of these. One reader makes the whole
   library available to EMC2 designs.
2. **Trial-varying designs, where no analytic exists.** This is the argument
   that generalises beyond the DDM. A neural likelihood whose network must run
   per trial is what a regression design forces, and a batched ONNX runtime
   handles that at 14.6 ms per 1000 trials where a scalar per-row loop takes
   839. For the DDM this does not matter, because WienR is available and costs
   5.2 ms — but for `angle`, `weibull` or any future flow for an intractable
   model **there is no analytic to fall back on**, and the batched path is the
   difference between a usable regression design and an unusable one. Note that
   §1(b) buys the same win for our own compiled models; the point is that ONNX
   gets it for free, for models we did not build.
3. Not speed on our flows. Measured: the ONNX marginal cost equals the compiled
   port's (0.45 vs 0.43 ms per 1000). ONNX would buy only the fixed overhead
   that §1(a) addresses without a new dependency.

### What the files look like

- Input: a **rank-1** float32 tensor, e.g. shape `[7]` for `ddm_uniform_st`
  (`v, a, z, t, st, rt, choice`), one trial per call. Output: a scalar log-
  likelihood. HSSM batches by converting the graph, not by calling a runtime per
  trial.
- Called per trial through onnxruntime this measures **8.0 ms per 1000 trials**,
  which is call overhead, not arithmetic. Either accept that, or edit the graph
  to take a leading batch dimension — for a plain MLP this is a small,
  mechanical change to the input/output value_info and any `Reshape` nodes, and
  it is worth doing.
- `choice` is ±1 (lower/upper). Parameter conventions are NOT WienR's; for
  `ddm_uniform_st` the mapping, determined empirically and verified to 4.3e-07,
  is `a_wienr = 2 * a_lan`, `w = z`, `t0 = t - st`, `st0 = 2 * st`, `sv = 0`,
  `sw = 0`. **Verify the mapping of any new network against WienR before
  trusting it**; do not infer it from parameter names.

### Suggested shape in EMC2

- A loader that takes an `.onnx` path plus a declared context order and returns
  a pointer, mirroring `ddmnn_get()`; an evaluator that takes the `pars` matrix,
  builds the network's own input layout, and returns the density.
- A model function per network, following the `DDMnn7` pattern: copy the nearest
  analytic model, replace `dfun`/`pfun`, and **assert the context order on load**
  rather than trusting metadata.
- Bounds: take them from the network's own training box, not from the analytic
  model's. `ddm_uniform_st`'s is `v [-3,3], a [0.3,2.5], z [0.3,0.7],
  t [0.25,2.25], st [0.001,0.25]` (LAN convention). Outside it, return the
  model's floor rather than extrapolating; these networks carry an explicit
  `label_floor` (−16.12 for that one).

### How to test the reader, using a model whose answers are already known

Do not debug an ONNX reader against a network you cannot check. The NLE project
ships **our own flow as ONNX**, in two signatures, both verified against the R
port to **max |diff| 2.4e-12**:

- `handover/nle-methods/artefacts/ddm_cap256w_c4.onnx` — `[N, 10]`, LAN-shaped
  (8 context values on the sampled scale, rt, choice).
- `handover/nle-methods/artefacts/ddm_cap256w_c4_shared.onnx` — `theta [8]`,
  `rt [N]`, `choice [N]`; 25x faster, and the shape a batched reader should aim
  for.

Point the new reader at those, compare against `EMC2:::dDDMnn` on the same
inputs, and you have an exact reference before any LAN is involved. The build
script is `flow/scripts/export_onnx_ddm.py` and the checker is
`flow/scripts/onnx_check.py` (float64 throughout; float32 was tested and
rejected — 11% faster for a max error of 1.9e-04).

### A note on scope

(a) and (b) are internal to the evaluator and change no interface. The reader in
§2 adds one. Section 3 is a larger proposal and should be costed separately.

### The warning that must travel with any LAN exposed this way

Their calibration is unknown, and the one we measured is not good. Scored with
our estimator on matched scales over a shared region, `ddm_uniform_st` carries
a score bias on `log a` of **−0.122** against our flow's +0.071 over the same
region — with both analytic controls at exactly 0. That is a statement about
score bias, not about SBC, and this project has learned the hard way that only
SBC settles a calibration question. So: expose them, and document that any
network reached this way needs its own SBC cell with a matched analytic control
before it is used for inference.


---

## 3. "Bring your own network" — a contract, not a special case

Worth considering for the plan, because everything below already exists in some
form and is currently hardcoded per model.

### Why it is viable

Our export JSON already carries everything a generic loader needs:
`context_names`, `context_transforms` (identity/log/probit per parameter),
`parameter_scale`, the standardiser, `bounds_natural` and `bounds_sampled`,
`response_coding`, the spline geometry, and `ll_floor_log`. That is a model
card. What makes EMC2 model-specific today is not the maths but three hardcoded
things:

1. `.theta_DDMnn()` builds the context with a literal `cbind(...)` in a fixed
   order, ignoring the export's own `context_names` (the same is true of
   `dRDMnn`). A generic builder driven by `context_names` + `context_transforms`
   replaces every one of these, and removes the mis-wiring trap in §0 entirely.
2. Bundle filenames are hardcoded in the `*_get()` functions, so a model is
   swapped by overwriting a file. A registry keyed by a user-supplied path
   removes that.
3. The evaluator assumes the DDM's 8-column layout. The compiled port is already
   dimension-generic (verified: it accepts 7- and 6-column bundles unchanged);
   only the R-level builders assume a width.

So the work is mostly deletion of assumptions, not new machinery.

### Suggested shape

```r
m <- register_nn_model(
  path      = "my_flow.json",        # or .onnx + a sidecar model card
  pars      = c("v", "B", "A", "t0"),# EMC2 parameter names, in the net's order
  transforms= c("identity", "log", "log", "log"),
  bounds    = list(lower = ..., upper = ...),
  response  = "joint",               # joint density, or "conditional" + a classifier
  cdf       = FALSE                  # no survivor -> race assembly unavailable
)
design(..., model = m)
```

with three tiers of what a network can be:

| tier | interchange | what it buys |
|---|---|---|
| A | our flow JSON | any flow trained with this project's exporter, any parameter count |
| B | ONNX + model card | any architecture, including the HSSM LANs (§2) and plain regression MLPs |
| C | an R closure | an escape hatch for anything else, at R speed |

Tier A is the smallest step and covers "a user retrained our recipe on their own
model". Tier B is the general one and shares all its machinery with §2. Tier C
costs nothing and makes the contract testable.

### What must come with it, or it does harm

A likelihood that loads silently and fits silently is exactly the failure mode
this project spent months documenting. Any such interface should ship a
validation kit that runs from the registration object:

- **total mass** — `integral over rt of sum over R exp(f_hat)` should be 1. A flow
  gets this by construction; a regression net does not, and what matters is not
  the offset but its variation with theta.
- **in-box check** — refuse, loudly, outside the declared training box rather
  than extrapolating.
- **score bias with its analytic control**, where an analytic exists
  (`evaluation/score_bias_draws.R` generalises: it needs only the model's own
  density and a reference).
- **the posterior-shift screen** (`evaluation/test_posterior_shift.R`), ~1 minute
  per model, which predicted SBC within ~0.1 on every DDM flow tested.
- **an SBC cell with a matched analytic control**, which is the only verdict.
  The harness (`evaluation/sbc_specs.R`, `run_sbc_cell`) is already generic over
  models; it needs the cell definition to come from the registration object.

### Cost and risk, honestly

- Tier A plus the generic theta builder: small, and it deletes more code than it
  adds.
- Tier B: the ONNX reader from §2, plus a model-card schema. Moderate.
- The validation kit: this is the real work, and it is also the part with
  research value — it is the difference between "EMC2 can load your network" and
  "EMC2 can tell you whether your network is fit to infer with".
- The risk to manage is support burden: publish the contract as a short document
  with a worked example, and make the validation kit the price of admission
  rather than an optional extra.


---

## 4. After it lands: this project migrates onto it

Not optional, and worth stating in the plan now, because it is also the only
honest test of §3. The NLE project currently carries its own versions of
everything the new interface is meant to provide, and each one becomes a test
case:

| what this project has now | what it becomes |
|---|---|
| `evaluation/DDMnn7.R`, `DDMnn6.R`, `DDMreg.R` — hand-written model functions, one per parameterisation | one `register_nn_model()` call each; if the registry cannot express them, the contract is wrong |
| `evaluation/permute_context.R` — re-expresses a flow to match a hardcoded context order | deleted: a generic theta builder reads `context_names` |
| the `.theta_DDMnn` family's literal `cbind()` | deleted |
| `evaluation/score_bias_draws.R`, `score_bias_t0.R`, `score_bias_race.R`, `score_bias_regress.R`, `test_posterior_shift.R` | the validation kit, generalised to any registered model |
| `evaluation/sbc_specs.R` + `run_sbc_cell` | the kit's SBC harness, with the cell taken from the registration object |
| `evaluation/make_bundle_st0zero.R` and the swap-a-file dance | a path in a registry |

**The acceptance test for the whole exercise** is that the four artefacts in
`handover/nle-methods/artefacts/` — `cap256w_c4`, `st0zero`, `rdm_small`, and
the regression net — load through the new interface and reproduce their recorded
numbers: the same log-densities as the R port (~1e-14), the same score bias to
1e-6, and the same SBC standardised biases within Monte Carlo error on a
re-run cell. Those numbers are in `results/` and in `METHODS.md`, so the test is
already written down; nothing new has to be invented to check it.

### The usability gate — what "dev-nle is usable" means, concretely

Five checks, all exact or near-exact, ~25 minutes at 8 cores, none an SBC
cell. Runnable as `Rscript sbc/gate_dev_nle.R ~/R/nle-sbc-lib` from the NLE
repo root (it uses dev-nle's own kit -- `nn_score_bias()`, `nn_posterior_shift()`,
`nn_sbc_cell()` -- plus one exact comparison of each model function's `dfun`
against this project's R port). The same procedure is in the EMC2 session's
notes: `SBC-staging/2026-09-24_dev-nle_phase6/USABILITY_GATE.md`. First run,
2026-09-26 against dev-nle 3.4.1.9001: checks 1-2 pass, with the compiled
likelihood matching the R port to ~1e-13 for cap256w_c4, st0zero and reg_s402.

| # | check | how | pass when |
|---|---|---|---|
| 1 | the artefacts load | `cap256w_c4`, `st0zero`, `rdm_small`, `reg_s402` through the new interface, contexts asserted on load | no error, asserted `context_names` match the exports |
| 2 | same likelihood, across theta | `sbc/test_emc2_vs_port.R` (already written: EMC2's full likelihood path vs the R port over 120 theta x 3 datasets, not just at one point) | max abs diff ~1e-12 or better; 1e-9 if the BLAS rewrite reorders sums |
| 3 | same score bias | `evaluation/score_bias_draws.R 200 cap256w_c4` and `score_bias_regress.R` | every parameter within 1e-6 of `results/score_bias/` |
| 4 | same screen | `evaluation/test_posterior_shift.R` on `cap256w_c4` and `z0_s503_e282` | predictions within Monte Carlo noise of `results/score_bias/postshift_*.rds` (same seeds: identical) |
| 5 | the SBC pipeline runs | ONE cell, N = 100, 12 replicates, any DDM artefact (~10 min) | completes; `mapped_pars()` assertion passes; no bound rejections |

Check 5 is a smoke test of the plumbing (design, priors, defaults, bound
exceptions, the sampler loop), not a measurement — 12 replicates cannot measure
anything, which is exactly why it is cheap enough to run.

Why this replaces the "server batch" (a `cap256w_c4` control cell plus a
`reg_s402` cell): an SBC cell has Monte Carlo error 0.045 per parameter at 500
replicates, so it would pass an evaluator that was wrong by 0.02 in standardised
bias and cost three hours (flow) to four days (regression net) to say so. Check
2 catches a difference of 1e-10 in seconds. The cells measure the *model*; the
gate measures the *code*, and only the code changed.

**Compute, and what does NOT need a server (decided 2026-09-26, compute-bound
on tux26 until 1 Oct).** Acceptance does not require re-running SBC cells on
the new branch. An SBC cell is an expensive and weak test of an evaluator
change — its Monte Carlo error is 0.045 per parameter at 500 replicates — while
the exact tests are cheap and strict:

| check | cost | what it proves |
|---|---|---|
| log-densities on matched (theta, rt, R) vs the R port, all four artefacts | seconds | the evaluator is the same function, to ~1e-14 |
| score bias on `cap256w_c4` and `reg_s402` reproduced to 1e-6 (`evaluation/score_bias_draws.R`, `score_bias_regress.R`) | minutes | the estimator path is intact |
| posterior-shift screen reproduces its recorded predictions (`results/score_bias/postshift_*.rds`) | ~1 min per 7-parameter flow | the sampler-free path is intact |

So the "server batch" of a `cap256w_c4` control cell and a `reg_s402` cell is
**not** needed: the `reg_s402` cell is already running on the present install
(its result is the number the document needs, whichever branch computed it),
and `cap256w_c4` may be superseded by the zero-box network under evaluation.
The only SBC compute that has to happen is the zero-box network's own cells
(`ZERO_BOX_RETRAIN_PLAN.md` §4). If the new branch is usable when those run,
run them there and they double as the acceptance test; if not, run them on the
present install and re-run ONE of them on the new branch later, which is the
one SBC-level acceptance check worth its cost.

Two consequences worth planning for:

- **The migration will find contract gaps.** `DDMreg` has no CDF, the
  decision-time flow needs `rt - t0` applied before the network sees it, and the
  st0 = 0 build needs a bound exception at zero. Any interface that cannot carry
  those three will not carry a stranger's model either.
- **The documentation for the validation kit should be this project's methods
  document, condensed.** `docs/TRAPS.md` is already the checklist a user needs,
  and `METHODS.md` §6 is the argument for why loading a network is not the same
  as being able to infer with it. Point at them rather than writing something
  new and thinner.

**Result (2026-09-26):** `Rscript sbc/gate_dev_nle.R ~/R/nle-sbc-lib` passed
5/5 in 1 minute (`sbc/gate_dev_nle.out`): exact check 2 at 8.9e-14 /
7.4e-14 / 1.4e-14 for `cap256w_c4` / `st0zero` / `reg_s402`; score bias `a`
+0.0126 (0.0013) vs recorded +0.0136; posterior shift t0 +0.22 vs +0.28; the
12-replicate `nn_sbc_cell()` ran and summarised. dev-nle is usable now. The zero-box registration (asinh transform, `-Inf`
`p_types`, bound exceptions at 0, st0 = 0 support rule) landed the same day
(EMC2-nle d49d190b) and passed the NLE acceptance script
`sbc/gate_zerobox_dev_nle.R` 3/3 (dfun vs R port ≤ 1.3e-12 in every zero
regime; `-Inf` matched at 450 support-boundary points; exact zeros survive
`design()`/`make_data()`/`nn_sbc_cell()`).

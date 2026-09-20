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

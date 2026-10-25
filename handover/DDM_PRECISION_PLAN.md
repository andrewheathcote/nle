# Plan: user control of the DDM likelihood precision in DMC2

Status: plan only, nothing implemented (2026-10-25). Written against DMC2 `main` ee8f2375.

## 1. Why

EMC2's DDM evaluates the Wiener first-passage density to a fixed accuracy, WienR's `precision = 5e-3`.
The user cannot change it:

- **R path** (`R/model_DDM.R`): `dDDM(rt, R, pars, precision = 5e-3)`, `pDDM`, `rDDM` take a `precision`
  argument, but the model list calls them as `dfun = function(rt, R, pars) dDDM(rt, R, pars)`, so the default
  is always used.
- **Compiled path** (`src/model_DDM.cpp`, the only path the sampler accepts; `native_likelihood_inputs()` stops
  with "Sampling requires a compiled likelihood" for a model without `c_name`): the accuracy is three
  compile-time constants, `DDM_EPS = 5e-3`, `DDM_LOG_EPS = log(5e-3)`, `DDM_LOG_INNER_EPS = log(5e-4)`, plus the
  cubature's `DDM_EPS * 0.9` (line 442) and `DDM_NEVAL = 6000` evaluations.

Consequences: (a) no fit at, say, 1e-8 can be run, so the vignette comparison "DDMnle against the DDM at
standard and at 1e-8 accuracy" can only be done by reweighting posterior draws on the R path, not by fitting;
(b) the exact-likelihood oracle in our tests (`dDDM(..., precision = 1e-8)`) is reachable only by calling
internals; (c) a user who wants to know whether a posterior depends on the likelihood accuracy cannot ask.

## 2. Goal

A user states the precision when choosing the model, and both paths honour it:

```r
design(..., model = DDM)                          # unchanged: precision 5e-3, results bit-identical
design(..., model = function() DDM(precision = 1e-8))   # new
```

Requirements: (1) the default is bit-identical to today's output (no change to any existing result or snapshot);
(2) the same value reaches the R path (`dfun`, `pfun`, `rfun`) and the compiled path; (3) the value is validated
(finite, in a documented range); (4) `register_nn_model(..., twin = ...)` and `nn_cell(..., control = ...)` accept
the precise DDM as the twin or control; (5) nothing else in the sampler changes.

## 3. Design options

| option | how | for | against |
|---|---|---|---|
| **A. a model argument (recommended)** | `DDM <- function(precision = 5e-3)` stores `precision` in the model list; R `dfun/pfun/rfun` close over it; the spec handed to C++ carries it | explicit in the design, no hidden state, the value is part of the model that is cached and printed, same pattern as `nn` (which the sampler already threads) | a model is passed as a function, so the call is `function() DDM(precision = 1e-8)`, a little clumsy; we can add a helper `DDM_precise(1e-8)` |
| B. a global option `options(emc.ddm_precision = ...)` | read in R and C++ | trivial to type | hidden global state; a fit's result depends on a session setting; cache and reproducibility hazards |
| C. a design constant `constants = c(precision = ...)` | a pseudo-parameter | no new API | it is not a parameter; it would pollute `p_types`, priors and the parameter table |

Recommendation: A, with a small exported helper so the call reads `model = DDM_precision(1e-8)`
(returns a model function). Option B only as a testing aid, if at all.

## 4. Touchpoints (all small)

R:
- `R/model_DDM.R`: `DDM(precision = 5e-3)`; `dfun/pfun/rfun` pass it; the model list gets `precision`;
  validation (e.g. `1e-12 <= precision <= 1e-1`; the range to be settled by the accuracy study in section 6);
  roxygen for `DDM` and for the new helper; the `.emc2_ll_cache_version` is bumped only if the cache key does not
  already include the model list (check; the default must still hit old caches or the bump must be justified).
- `R/sampling_cpp.R` (`native_likelihood_inputs()`) and `R/sampling.R` (the `calc_ll_manager` call, lines ~268-280):
  pass `precision` the way `nn` is passed (`if (identical(model$c_name, "DDM")) model$precision`).
- `R/nn_register.R` / `nn_cell()`: when the twin or control is DDM, keep its precision; the control's printout names it.

C++:
- `src/model_DDM.cpp`: replace the three constants by values derived from a runtime `eps`:
  `log_eps = log(eps)`, `log_inner = log(0.1 * eps)`, cubature tolerance `0.9 * eps`. They enter through
  `dwiener(..., log_eps)` (already a parameter), `Ks/Kl/ks` call sites (lines 375-405), `int_ddiff`'s inner
  call (line 456) and `ddiff`'s `hcubature` call (line 442). Thread `eps` through `my_params` (the cubature
  struct) and `fill_ddm`.
- `src/RaceSpec.h` / `src/particle_ll.cpp`: carry `ddm_eps` in the DDM setup (built where `make_race_setup`
  is called, line ~778, from a new constructor argument next to `nn`); `c_log_likelihood_DDM` passes it to
  `fill_ddm`. `src/model_DDM.h`: new signature. `d_DDM_Wien` (line 601) keeps a default.
- `DDM_NEVAL = 6000`: at 1e-8 the cubature (sv, SZ or st0 > 0) may reach this cap before it reaches the
  tolerance. Decide in the accuracy study whether the cap becomes a function of `eps` (and make the cap-hit
  observable: a counter, or a warning, rather than a silent inaccurate density).
- `Rcpp::compileAttributes()` and `devtools::document()` regenerate `RcppExports` and `NAMESPACE`; never edit
  them by hand (DMC2 rule).

## 5. Tests (TDD: each fails first)

1. Default unchanged: compiled log-likelihood at the default equals the stored values of the current build, to
   0 absolute difference, on the cases of `tests/testthat/test-nn-reference.R` and on a design with sv, SZ and
   st0 free; the existing DDM test files pass unmodified.
2. Argument reaches the compiled path: at `precision = 1e-8` the compiled log-likelihood agrees with the R path
   at `precision = 1e-8` (which is WienR) to a stated tolerance, for the zero-box regimes (all combinations of
   sv, SZ, st0 equal to or above zero) and for rts from near t0 to 5 s; at the default it still agrees with the R
   path at 5e-3.
3. Monotone accuracy: the error against a 1e-12 reference falls from 5e-3 to 1e-5 to 1e-8 on a fixed grid.
4. Validation: values outside the range and non-numeric values stop with a message naming the argument.
5. Twin and control: `register_nn_model(card, twin = DDM_precision(1e-8))` registers and `nn_golden_check()` passes;
   `nn_cell(DDMnle, control = ...)` uses the precise control.
6. `rfun` (simulation) at the precision: no change in the distribution (KS against the default), since WienR's
   sampler also takes `precision`.
7. Full suite from source, `NOT_CRAN=true`, as the DMC2 notes give (5282 / 0 / 0 on `main`).

## 6. Accuracy and speed study (before fixing the API range)

On the laptop, at most 8 cores, no sampling: for `precision` in 5e-3, 1e-4, 1e-6, 1e-8, 1e-10, on a grid of
parameters (the six variability cases of the DDMnle validation) and rts: (a) achieved error against a 1e-12 R-path
reference; (b) milliseconds per 400 trials, single thread, against the default; (c) whether `DDM_NEVAL` is hit.
Report the range at which the compiled code is trustworthy; set the validation range and the NEVAL rule from it.
This also gives the exact-likelihood speed table that the DDMnle speed-up claim (2.9 to 9.8 times against WienR at
N = 400) should be quoted against: the default is not the exact likelihood.

## 7. Repository and branch rules (DMC2 notes)

The change touches only `model_DDM.*`, the DDM wiring in `particle_ll.cpp`, `sampling*.R` and tests; it does not
depend on hardfit, rating or the neural-likelihood code, except for item 5 of section 5. So:
- implement on a branch off `upstream/dev` (EMC2-bound, a PR to ampl-psych/EMC2 stays possible), run the suite
  there, then merge the branch into DMC2 `main` as well; the nn-dependent test (item 5) is added in a second
  commit on a branch off `main`;
- do not edit `NEWS.md` or `DESCRIPTION` (EMC2's); notes go in `NEWS_DMC2.md`;
- never touch AH's clone in `~/Documents/Projects/EMCpackage/EMCrelease/DMC2`; work in a clone under `$HOME`;
- sync order after merging: `git branch -f upstream-dev upstream/dev` only when `upstream/dev` itself moved.

## 8. Risks

- The ported series and cubature code was written for 5e-3. Tolerances far below may not be reached (series term
  counts, `hcubature` evaluation cap, `min_ll = log(1e-10)` flooring, which is itself a precision floor): this is why
  section 6 comes first.
- Run time rises steeply with precision for sv/SZ/st0 > 0 (the cubature); a fit at 1e-8 may be slow. That is
  acceptable (it is a reference), but the vignette must say so and use N = 100 to 400, one participant.
- Cached likelihood results (`.emc2_ll_cache_version`) must not mix precisions.
- Hardfit, rating or other DMC2 code that calls `dDDM` directly keeps its default.

## 9. Order of work

1. Accuracy and speed study (section 6), laptop, about 1 h of compute. 2. Failing tests (section 5), then the R
side, then the C++ side. 3. Suite on the `upstream/dev` branch; merge into `main`; suite there. 4. Helper and docs
(`?DDM`, `NEWS_DMC2.md`). 5. Then the vignette work: update `using-a-trained-flow` to follow AH's example, and the
new single-participant DDMnle vignette with the 1e-8 comparison as a real fit, not a reweighting.

## 10. Next stage prompt

See the end of the session that wrote this file; the prompt is for stage P1 (sections 6 and 5.1-5.2) and is
reproduced there.

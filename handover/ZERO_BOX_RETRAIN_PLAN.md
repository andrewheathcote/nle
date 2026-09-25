# One DDM flow for sv, SZ, st0 at zero and above

Plan, 2026-09-26. Goal: a single network that fits the full 8-parameter DDM and
any model in which one or more of `sv`, `SZ`, `st0` is fixed at **exactly
zero**, assessed against the exact-zero analytic solution — so one artefact
serves every case without swapping bundles.

## 0. Verdict: feasible, with one design change and one hard corner

**The context.** EMC2 floors these parameters above zero (`st0 >= 0.01`,
`sv >= 0.01`, `SZ >= 0.05`) because the numerical DDM likelihood is unstable
near zero, and handles exactly zero as a special case. A flow has no such
instability of its own — but it inherits one from its training labels, and it
cannot represent zero at all in the current encoding. Both were checked before
writing this.

**What was measured (2026-09-26):**

1. **The encoding cannot express zero.** The context is `log sv`, `log st0`,
   `probit SZ`; all three map 0 to −Inf. `simulation/designs.R` says as much in
   the comment on the `st0` bound. So this is a change of parameterisation for
   three inputs, not an extension of a box.
2. **The labels are sound where the data are.** `WienR::dWDM` at precision
   1e-8, 40 random θ across the training box, each of the three parameters
   stepped 1e-5 → 1e-2 against its exact-zero solution: `sv` converges as x²
   everywhere (worst |Δ| 1.4e-09 at 1e-5); `sw` and `st0` show the instability
   you described — up to 5 log units of disagreement at x = 1e-5 — **but every
   one of the 36 such cases is at rt = t0 + 5 ms with exact-zero log density
   between −27 and −326.** That region carries no probability mass and no
   simulated training row lands there. In-mass, the gap between zero and the
   floor is smooth and continuous in all three parameters.
3. **The exact-zero regime has a passing precedent.** The dedicated st0 = 0
   flow passes SBC at N = 400 once the cell is specified correctly (Phase 37:
   worst +0.282, coverage nominal, rank non-uniformity on a/t0/Z).
4. **The simulator is fine.** `EMC2::make_data` with constants at −Inf under the
   analytic `DDM` design reproduces `WienR::rWDM` in all four zero regimes
   (mean rt and P(upper) agree to three decimals).
5. **Fixing a parameter at the floor is not a substitute.** Phase 29 measured
   it: at `st0 = 0.05` the t0 bias is essentially intact and `Z` degrades; at
   the floor `st0 = 0.01` everything degrades (a +0.448, Z −0.405).

The hard corner is `st0 = 0`, where the density has an exact support boundary
at `t0`; §3 handles it.

---

## 1. Specification

### 1.1 Parameterisation

Replace the three transforms with one that is exactly 0 at 0, smooth, and
log-like above the old floor so nothing is lost where the network is already
good:

    u = asinh(x / c)        c_sv = 0.01,  c_SZ = 0.05,  c_st0 = 0.002

`c` is the scale below which the coordinate is linear. For `sv` and `SZ` the
old floor is the natural choice. For `st0` it is set smaller because that is
where the density moves fastest: at 10 ms past onset, log f changes by 2.4 units
between st0 = 0 and 0.01, so the network needs resolution *below* the old
floor, not just at it. `v`, `a`, `t0`, `s`, `Z` keep their transforms.

Alternatives considered: natural scale (loses resolution near 0), `sqrt`
(acceptable for SZ, poor for st0), log plus a zero-indicator input (the density
is continuous as x → 0, so a discrete flag adds an input the network does not
need).

### 1.2 Box

| parameter | lower | upper | change |
|---|---|---|---|
| sv | **0** | 4.0 | was 0.01 |
| SZ | **0** | 0.99 | was 0.05 |
| st0 | **0** | 0.49 | was 0.01 |
| others | unchanged | | |

### 1.3 Sampling: stratified zero patterns

Independent draws would leave the all-zero model — the plain Wiener DDM most
people fit — at about 1% of the data. Stratify on the eight regimes:

| regime | share |
|---|---|
| all three > 0 | 40% |
| st0 = 0 | 14% |
| sv = 0 | 8% |
| SZ = 0 | 8% |
| st0 = sv = 0 | 7% |
| st0 = SZ = 0 | 7% |
| sv = SZ = 0 | 6% |
| all three = 0 | 10% |

Within a stratum, free variability parameters are uniform in the `asinh`
coordinate, so the gap between 0 and the old floor is densely covered. Stored
context is on the new sampled scale (the current pipeline would store −Inf).

### 1.4 Budget

**24e6 rows** (40 trials per draw, 600k draws) rather than 16e6, so the
full-model stratum keeps 240k draws against today's 400k rather than dropping
to 160k. Simulation is ~18 min; training ~1.5x today's.

### 1.5 Architecture and training

Unchanged: [256, 128] conditioner, 64-bin spline, batch 1024, 300 epochs,
30 retained checkpoints, corrected-target classifier retrained on the new box.
Three seeds. Nothing about the recipe depends on the encoding.

---

## 2. Implementation, in order

| step | where | change |
|---|---|---|
| 1 | `simulation/designs.R` | bounds to 0; `transformsDDM` gains `asinh` with per-parameter `c` |
| 2 | `simulation/sim_functions.R` | `param_matrix()` learns `asinh`; stratified zero-pattern sampler |
| 3 | `simulation/sim_ddm.R`, `sim_choice_prob.R` | regenerate at 24e6 with the corrected P(R) target |
| 4 | `flow/scripts/train_ddm.py` | `--context-encoding zerobox`; scaler and data loading follow |
| 5 | `flow/scripts/export_flow_ddm.py` | write the new `context_transforms` and `c` constants; golden vectors per regime |
| 6 | `port/R/flow_ddm.R`, `port/src/flow_ddm.cpp` | none in the spline code — the port takes the sampled-scale context; **add the st0 = 0 support rule (§3)** |
| 7 | EMC2 model function | `DDMnn0` in the `DDMnn7` pattern now; `register_nn_model()` later (§5) |
| 8 | `sbc/nci/` drivers | priors on the new scale; zero regimes as named constants at −Inf with bound exceptions at 0; `mapped_pars()` asserted |

Steps 1–3 and 4–5 can proceed in parallel; 6–8 need the export.

---

## 3. The st0 = 0 corner

With `st0 = 0` the density is exactly zero below `t0`. A spline over `log rt` is
positive everywhere, so it cannot produce −Inf; what it can do — and what the
Phase 37 result shows it does well enough to pass the gate — is put negligible
mass there (measured: ~1e-5 below the true t0).

**Rule to build in:** when `st0 = 0`, the evaluator returns `log f = −Inf` for
`rt <= t0`, in both the R port and the compiled path. It is a true property of
the model, it costs nothing, and Phase 32b measured that it does not bind at
the likelihood maximum — so it is insurance against a class of failure, not a
patch that changes results.

Also give this regime a golden vector at rt within 2 ms of `t0`, where the
exact-zero and near-zero solutions differ most, so a mis-encoded `c_st0` fails
loudly.

---

## 4. Testing — three questions, in this order

Each regime is assessed against its own exact analytic. Three questions have to
come out right, and they are not the same question:

**Q1. Is it accurate at exactly zero?** (the point of the exercise)
For each zero regime: score bias with its analytic control; the posterior-shift
screen (~1 min); SBC at N = 400, 500 replicates, **with a matched analytic
control** — same design, the zero parameters as named constants, `mapped_pars()`
asserted. Cells: st0 = 0, sv = 0, SZ = 0, all-zero. Analytic controls exist for
st0 = 0; three are new.

**Q2. Is it still as good above the floor?** (the cost of the exercise)
The full-model regime's score bias over the SBC prior must not be worse than
`cap256w_c4`'s, and one SBC cell at N = 400 must land within Monte Carlo error
of the shipped flow's. If this fails, ship both networks rather than replace
the shipped one.

**Q3. Is it right in the gap?** (the region EMC2 cannot reach)
Between zero and the old floor there is no EMC2 evaluator at all; the flow would
be the only one. Score bias against WienR 1e-8 at, say, st0 ∈ {0.001, 0.003},
sv ∈ {0.001, 0.003}, SZ ∈ {0.01, 0.03}, with the analytic control at the same
points. The control is meaningful there because in-mass WienR is smooth (§0.2);
report the density-weighted error, not the max, which lives in the massless
edge.

Screening and selection as in the recipe: 90 checkpoints on score bias, best
confirmed by SBC, selection declared.

---

## 5. Sequencing against the EMC2 work

The new encoding needs a theta builder that applies `asinh`, which no built-in
EMC2 model has. That is precisely what `register_nn_model()` in
`EMC2_IMPLEMENTATION_BRIEF.md` §3 provides, and this flow is its natural first
client — a network whose transforms differ from every built-in. But training
and testing do not wait on it: write `DDMnn0` in the `DDMnn7` pattern (a copy
with the new builder, `p_types` defaults of −Inf for the three, bound
exceptions at 0), run the cells with it, and migrate to the registry when it
lands.

---

## 6. Cost

| stage | wall time |
|---|---|
| simulation, 24e6 rows + choice targets | ~25 min |
| training, 3 seeds concurrently on tux26 | ~6–8 h |
| export, golden vectors, checkpoint screening | ~2 h |
| posterior-shift screens, 5 regimes | ~3 h |
| SBC: 5 flow cells + 3 new analytic controls, 48 cores | ~24 h |
| **total** | **~2–3 days on tux26, dominated by SBC** |

All of it fits on tux26 after the regression cell finishes; none of it needs
the laptop.

---

## 7. Speed expectations — measured, so the paper does not overclaim

The architecture is unchanged, so the flow's cost per call is the same in every
regime: the transform is applied once per theta and is negligible. What changes
is the **analytic's** cost, which collapses when a variability parameter is
exactly zero because WienR takes its special-case branch and skips an
integration. ms per 1000 trials, one core (flow timed with the current artefact
as a stand-in):

| regime | WienR 1e-8 | EMC2 default 5e-3 | flow | flow advantage |
|---|---|---|---|---|
| full (sv .1, SZ .2, st0 .05) | 6.20 | 2.80 | 1.10 | 5.6x / 2.5x |
| full, larger variability (sv .5, .2, .2) | 96.70 | 3.75 | 1.15 | 84x / 3.3x |
| st0 = 0 | 1.85 | 1.50 | 1.15 | 1.6x / 1.3x |
| sv = 0 | 6.20 | 2.75 | 1.15 | 5.4x / 2.4x |
| SZ = 0 | 1.80 | 1.50 | 1.15 | 1.6x / 1.3x |
| **all zero (plain Wiener)** | **0.30** | **0.25** | 1.20 | **0.3x / 0.2x — the analytic wins** |

So:

- The margin holds for the full model and for `sv = 0` (it is the `st0` and
  `sw` integrations that cost, not `sv`).
- Fixing `st0` or `SZ` to zero shrinks it to 1.3–1.6x: still a win, but not
  one to headline.
- For the plain Wiener model the flow is **4x slower** than the analytic. The
  case for the single network there is convenience and one validated artefact,
  not speed, and §1 of the methods document should say so.

Two-thirds of the flow's 1.15 ms at n = 1000 is the fixed per-call cost that
`EMC2_IMPLEMENTATION_BRIEF.md` §1 attributes to scalar matrix loops. With that
fix (~0.55 ms per call) the st0 = 0 and SZ = 0 margins become ~3x and the plain
Wiener deficit ~2x — worth having, but it does not change which column wins.

## 8. Risks

- **Interior degradation** — eight regimes now share the capacity. Q2 is the
  gate; the fallback is two artefacts.
- **The st0 = 0 edge** — known survivable at the gate level, not known clean.
  The combined flow may inherit the existing st0 = 0 flow's rank
  non-uniformity on a/t0/Z. That is a reportable outcome, not a failure of the
  plan.
- **Seed variance** — half the seeds of this recipe fail SBC (Phase 11). Three
  seeds and screening are the minimum; selection must be declared.
- **Mis-wiring** — a new parameterisation is a new chance for the project's
  most expensive recurring bug. Assert `context_names` and `context_transforms`
  on load; one golden vector per regime.

## 9. Decisions needed

All five taken 2026-09-26 (AH: yes to every recommendation), subject to the
speed expectations in §7 being stated rather than hidden.

1. Transform: `asinh(x/c)` with `c_st0 = 0.002` (recommended) or a common `c`.
2. Budget: 24e6 (recommended) or keep 16e6.
3. Strata shares as in §1.3, or a different emphasis.
4. Q2's fallback: replace the shipped artefact only if not worse, else ship both.
5. Wrapper: bespoke `DDMnn0` now (recommended) or wait for the registry.

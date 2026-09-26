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
5. **The pipeline's own labels are exact at zero and in the gap.** The stored
   `ll` comes from `EMC2:::dDDM` at precision 5e-10 on the EMC2-scale
   parameter vector; with zeros passed as −Inf it matches WienR 1e-8 to
   ≤ 2e-8 in every regime — full, st0 = 0, sv = 0, SZ = 0, all-zero, and
   st0 ∈ {1e-3, 3e-3}. No new likelihood code is needed for the labels.
6. **EMC2's admissible set has a gap only for `st0`.** `DDM()$bound` allows
   `sv ∈ {0} ∪ [0.01, 10]` and `SZ ∈ {0} ∪ [0.01, 0.99]` — `make_data` refuses
   `sv = 1e-3` outright — but `st0 ∈ [0, 0.5]` continuously. The box below
   mirrors that: for `sv` and `SZ` the network learns exactly zero plus the
   region above the floor, which is all a user can fit, and interpolates across
   a gap it never sees (harmless: both effects are O(x²)). For `st0` the gap is
   real, allowed, and where the density moves fastest, so it is sampled densely.
7. **Fixing a parameter at the floor is not a substitute.** Phase 29 measured
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

| parameter | admissible set | change |
|---|---|---|
| sv | **{0} ∪ [0.01, 4.0]** | zero added; mirrors `DDM()$bound` |
| SZ | **{0} ∪ [0.05, 0.99]** | zero added; 0.05 keeps `sw ≥ 0.01` at the worst `Z` |
| st0 | **[0, 0.49]** | continuous down to zero, as EMC2 allows |
| others | unchanged | |

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
coordinate over their admissible range: for `sv` and `SZ` that is log-like
above the floor (as today); for `st0` it runs to zero, so the gap is densely
covered. The simulation stores parameters on **EMC2's** sampled scale, with
−Inf at zeros — the pipeline's likelihood accepts that exactly (§0.5) — and the
`asinh` context is computed from it in one place, the training loader and
exporter (`zerobox_context()`), never stored twice.

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

**Q3. Is it right in the `st0` gap?** (the region below EMC2's old nn floor)
Between zero and 0.01 the analytic runs but the current flow was never trained.
Score bias against WienR 1e-8 at st0 ∈ {0.001, 0.003}, with the analytic
control at the same points. (No gap exists for `sv` and `SZ`: EMC2 admits zero
or the floor and above, and so does the box.) The control is meaningful there because in-mass WienR is smooth (§0.2);
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

## 8. Capacity, staging, and which model does what

### Capacity on tux26

The only job there is the regression-baseline SBC cell (last seen 169/250,
~29 h to go as of 2026-09-26 00:44, ~57 cores). Nothing needs to be paused:

- **Training does not need the cores the cell is using.** Three seeds
  concurrently take ~12 cores (each run uses under 4). With the other user at
  ~23 and the cell at ~57, that is 92 of 96 — tight but it fits, and training
  is ~6-8 h.
- **The new flow's SBC cells come after training, export and screening**, by
  which time the regression cell has finished on its own schedule.

If it ever is necessary to reclaim cores without losing work, **suspend rather
than kill**: `pkill -STOP -f "SBCddmreg[.]R"` freezes every worker in place
(memory stays allocated, progress is kept) and `pkill -CONT -f ...` resumes.
`run_sbc` writes nothing until the end, so killing it forfeits the whole cell.
The progress watcher counts processes, and stopped processes still count, so it
keeps working through a suspension.

### Staging, and which model for which stage

The work splits cleanly into mechanical stages that produce compact, checkable
output and judgement stages where a wrong call is expensive. Match the model to
the stage; the token cost of this project so far has been dominated by reading
long outputs, not by reasoning.

| stage | what it is | model | why |
|---|---|---|---|
| S1 simulation: box, transforms, stratified sampler, 24e6 rows, choice targets | code edits with an exact acceptance test (stored `ll` vs WienR at zeros = 0.000000; strata counts as specified) | **Sonnet** | routine implementation against a written spec; the test decides, not the model |
| S2 training: launch 3 seeds, export 90 checkpoints, golden vectors | scripted; long-running; output is pass/fail per checkpoint | **Haiku** for launch/monitor/collect; **Sonnet** if an export gate trips | nothing to decide unless something fails |
| S3 screening: score bias per checkpoint, posterior-shift screen per regime | scripted; produces one table | **Haiku** to run, **Sonnet** to read the table and pick the checkpoint by the stated rule | the selection rule is written down in advance |
| S4 EMC2 wrapper `DDMnn0`, SBC drivers, `mapped_pars()` assertions, bound exceptions | small but trap-dense (§0 of the EMC2 brief) | **Opus / Fable** | this is where four phases were lost last time; one careful pass beats three cheap ones |
| S5 SBC cells: launch, watch, collect | scripted, days of wall time | **Haiku** | pure orchestration |
| S6 reading the results against Q1-Q3, deciding replace-or-ship-both, writing the document sections | judgement, and the part the paper depends on | **Opus / Fable** | disagreements between screen and SBC have so far always been the cell's fault, and finding that takes reasoning |

Rules that save more than the model choice does:

- **Every long job runs in the background with a watcher that prints a table,
  never a log.** Reading a 700-line SBC log costs more than the analysis of it.
- **Scripts print the acceptance test result as one line.** `PASS max|diff|
  2.4e-12` is a token; a dump of the diffs is a page.
- **One launcher script per batch of cells**, checked once, rather than five
  hand-launched cells each needing a verification exchange.
- **Run the posterior-shift screen before any SBC cell.** A regime the screen
  puts far inside the gate still needs its cell for the record, but one that
  fails the screen should be understood before three hours of cluster time are
  spent confirming it.
- **Hand the plan file, not the conversation, to the next stage.** Each stage
  starts from `ZERO_BOX_RETRAIN_PLAN.md` and the spec in it; the context that
  needs carrying is written here, not remembered.

## 9. Risks

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

## 10. Decisions needed

All five taken 2026-09-26 (AH: yes to every recommendation), subject to the
speed expectations in §7 being stated rather than hidden.

1. Transform: `asinh(x/c)` with `c_st0 = 0.002` (recommended) or a common `c`.
2. Budget: 24e6 (recommended) or keep 16e6.
3. Strata shares as in §1.3, or a different emphasis.
4. Q2's fallback: replace the shipped artefact only if not worse, else ship both.
5. Wrapper: bespoke `DDMnn0` now (recommended) or wait for the registry.


---

## Progress log

**2026-09-26 — S1 and S2 complete; training launched.**

- S1 (`simulation/zerobox.R`, `sim_ddm_zerobox.R`, `sim_val_zerobox.R`,
  `sim_choice_prob_zerobox.R`, `test_zerobox.R`): eight acceptance checks pass;
  stored `ll` matches WienR 1e-8 to 3.9e-10 across all regimes (with the
  s-scaling in the reference — omit it and every s ≠ 1 draw fails, the Phase 7
  trap). Training set 600k draws / 24e6 rows, 552 MB, 113 min at 8 cores, 0% at
  the floor; validation 25k draws; classifier targets 200k/20k. One packaging
  bug found and fixed on the way: run from the package directory, no `.Rlibs`
  is found and the default-library EMC2 floors every label in the parallel
  workers — the script now resolves `EMC2NN_LIB` or a repo-root `.Rlibs`,
  exports it as `R_LIBS` for the workers, and asserts on `find.package`.
- S2: `data.py` `apply_context_encoding()` (both loaders), `train_ddm.py
  --zerobox`, `export_flow_ddm.py --zerobox` (constants and bounds in the
  export, golden vectors with forced exact zeros), R and C++ ports with the
  encoding, the box check on the network scale, and the st0 = 0 support rule.
  `port/tests/validate_zerobox_plumbing.R`: R and C++ agree to 8.7e-12 over
  five regimes with exact zeros; the rule fires identically in both. Note the
  ports exist in two copies (repo root and package) whose committed versions
  differ; both carry the change now, and a working-directory reset put one
  patch in the wrong copy once — check `git status` on both after editing.
- `DDMnn0` wrapper (`handover/zerobox/DDMnn0.R`): builds the asinh context in
  R so EMC2's own compiled evaluator needs no change; `p_types` −Inf and bound
  exceptions at 0 verified; omitting SZ/st0 from a formula maps them to exactly
  0.
- Training: `flow/scripts/train_zerobox_tux.sh` on tux26, three seeds
  (zb_s901–903), 4 threads each, placeholder classifier `large`; imports the
  package's patched `flow` via `PYTHONPATH` (the venv's editable install points
  at the unpatched repo tree). A one-epoch local smoke run passed. The
  placeholder classifier's scaler differs from the flow's, so export requires
  the classifier retrained with `retrain_classifier.py --zerobox --scaler-from
  <seed> --data-suffix _zerobox --out-json`, converted with
  `json_to_classifier_ckpt.py`, and installed into each seed with its
  `train_loader.pkl` — exactly `sbc/fix_z0_classifier.sh`'s sequence.

**Next (S3):** export all 90 checkpoints with `--zerobox`; regime-specific
screening (the existing `score_bias_draws.R` differentiates on EMC2's log scale
and cannot be used at an exact zero — Q1 needs draws per regime with the zero
parameters held at zero); posterior-shift screen per regime; select; then S4.

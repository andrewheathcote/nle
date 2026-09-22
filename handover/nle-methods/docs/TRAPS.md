# Traps — ways to get a wrong number and not know it

A checklist. Every entry was hit in this project, most of them more than once.
They are properties of the method, not of any particular team, which is why this
is the most useful document here.

Ordered by how expensive they were.

---

## Measurement

**A score-bias estimator with no analytic control.**
Run the estimator on the *analytic* likelihood first. It must return ≈ 0. Ours
was silently 25–40% wrong until that check existed. Without it there is nothing
to distinguish "the model is biased" from "the estimator is biased", and the
numbers mean nothing.

**A finite-difference step that is too large.**
Halve `H`; the analytic control should fall by 4×. At `H = 0.05` the control was
14% of the DDM's signal; at `H = 0.01` it is under 1%. Rankings survive a common
offset, absolute values do not.

**Omitting the integration measure.**
`E[·]` over a grid needs the trapezoid widths. Without them the analytic control
returns −0.25 instead of 0.

**A support-boundary parameter (t0).**
The usual estimator returns −Inf or is silently wrong, because moving t0 puts
zero analytic density where the truth has mass. Restrict to a region past the
smear, where the target is `d/dθ log P(A|θ)` rather than zero, and validate
against the closed form.

**Score bias is LOCAL and cannot see surface distortion.**
`g = d/dθ E_true[log f̂]` is a derivative *at the truth*. A model can have small
`g` at every θ and still have a badly distorted likelihood *surface* between θs
— which is what the posterior explores. This is why `g` ranks checkpoints within
an architecture (ρ = −0.989) but not across architectures, and why one of our
artefacts had total |g| = 0.064 and SBC coverage of 0.000. **Score bias is
necessary, not sufficient. Only SBC probes the surface.**

---

## Comparison

**Comparing models that carry different auxiliary components.**
Splice a common classifier into every export *before* comparing flows, or you
are measuring the classifier as much as the flow. Two of our classifiers both
labelled "c4" turned out to be independently trained networks.

**Aggregate bias hiding component cancellation.**
Report flow and classifier shares separately. Measured: `a` = +0.0010 total was
flow +0.0121 cancelling classifier −0.0111 — and the cancellation did not
survive a change of seed.

**Comparing arms trained on different machines or in different eras.**
Half of an apparent 8.7× variance effect was provenance. Against a matched
same-machine arm it was 2.84× (p = 0.34). If two arms differ in anything besides
the variable under test, they are not a comparison.

**Believing a single run.**
Run-to-run variance at fixed hyperparameters is comparable to any effect size
worth reporting. **Every claim in this project that rested on one run per
condition was later retracted; none replicated across ≥ 3 seeds was.** Two
separate interventions — batch size and distillation — were announced and then
withdrawn on this. Measure the between-seed spread *first*, then decide whether
your effect is visible against it.

**Believing an explanation because it is consistent.**
A mechanism that explains the data is a hypothesis. Write down what it predicts,
then test the prediction, *before* writing it up as a finding. We published a
boundary-crossing mechanism and refuted it with a ten-minute test one commit
later.

**Diagnosing from n = 1.**
A single simulated dataset suggested an exact likelihood was itself badly biased.
Over 500 replicates its mean bias was +0.0001. Single datasets set hypotheses;
they do not settle them.

---

## Targets and data

**An analytic target computed wrongly.**
Then a correct model looks wrong and a wrong model looks right. Our `P(R)`
target omitted the s-scaling — the DDM depends on `v, a, sv` only through
`v/s, a/s, sv/s` — giving a mean error of 0.055 and making the *shipped*
classifier look 6× worse than it was.

**A validation set from the wrong parameterisation.**
Checkpoints are retained by best validation loss, so a mismatched val set
silently selects the wrong checkpoints. Training an st0 = 0 flow against a val
set simulated with st0 > 0 would have actively preferred checkpoints that
*over-smooth the leading edge* — the exact failure being studied.

**Reading a log-scale quantity as natural units.**
Label the scale on every table of parameter-space numbers. A t0 offset was
reported as "~3 ms" for months; it is 0.28%, which is 0.83 ms at t0 = 0.3 s.
Most parameters here are log- or probit-transformed.

---

## Plumbing

**Context column order.**
Produces ~500× worse score bias — and the tell is that it is *uniform across
checkpoints*. A badly trained model varies from checkpoint to checkpoint; a
mis-wired one does not. Assert `context_names` rather than trusting it.

**Two installs of the same package version.**
Check `find.package()`, not the version string. A second build reporting the
same version silently returned floored likelihoods (`-23.03`) for every
evaluation until it was found.

**A stale artefact left installed.**
Bundle filenames are hardcoded, so testing means overwriting. Always restore,
and always print which artefact is installed at the start of a run. A killed job
left a test bundle in place and every subsequent local evaluation used the wrong
model.

**A neural model's spec is not its analytic twin's — check defaults AND bounds.**
EMC2's `DDMnn` carries the training box, not the DDM's parameter space, and two
entries differ from `DDM` in ways that are silent and fatal:

| | `DDM` | `DDMnn` |
|---|---|---|
| `p_types["st0"]` | `-Inf` (st0 = 0) | `-2.996` (**st0 = 0.05**) |
| `bound$exception` | `c(sv = 0, SZ = 0, st0 = 0)` | none; `st0` bounded to [0.01, 0.49] |

So omitting `st0` from a formula SIMULATES at st0 = 0.05 under a flow design and
at st0 = 0 under the analytic design. Two SBC cells here were run that way
against flows trained at st0 = 0, and both failed catastrophically (t0 +3.1 sd,
coverage 0.000) — a model mismatch, not a flow defect. The analytic "control"
used the `DDM` model, so it simulated at st0 = 0, passed, and appeared to
exonerate the pipeline. Correcting the default and the bound turns the same
flow, same data-generating design, into a PASS (t0 +0.35, a +0.04).

**How to detect:** never rely on a default for a parameter you are not sampling
— name it in `constants` — and assert the simulated value:
`stopifnot(mapped_pars(des, pmean, remove_subjects = FALSE)[1, "st0"] == 0)`.
Confirm empirically as well: 100k simulated trials from the design matched
st0 = 0.05 at max |ECDF - CDF| = 0.002 against 0.145 for st0 = 0. And the tell
that generalises — **when the analytic and the approximation are displaced
TOGETHER, suspect the data, not the approximation.** On these datasets the
analytic MLE was displaced +13.4 ms where matched data give +1.0 ms.

**Priors passed by position.**
EMC2's `prior()` takes `pmean` and `psd` positionally, and `sampled_pars()`
returns parameters in *formula* order, not the canonical model order. Writing
the formula as `(Z, SZ, sv)` against a prior vector written `(Z, sv, SZ)`
silently swaps two priors, and SBC then draws truths from one prior and fits
under another. It shows up as a calibration failure on exactly those two
parameters. Write the prior as a named vector and index it by
`names(sampled_pars(design))`, as the `SBCddm*.R` drivers do.

**A frozen classifier whose input width no longer matches the flow.**
Copying a classifier in at training time (`--classifier-from`) is convenient and
silently wrong the moment the flow's context changes. It bit twice here, two
days apart, on the 7- and 6-parameter builds. The exporter's width assertion
caught it both times — **keep that assertion**; without it a mismatched head
reaches SBC and shows up as a calibration failure with no obvious cause.

Retraining the head is usually cheap and sometimes unnecessary: `P(R | θ)` is
*exactly* independent of both `st0` and `t0` (measured spreads 1.5e-11 and 0.0),
because they shift *when* a response occurs, not *which* boundary is hit. So the
same choice-probability targets serve any of these builds with the irrelevant
columns dropped — no new simulation.

**Port-vs-reference guards calibrated on a small model.**
A bigger network trips them on float32 accumulation. Report the error
*distribution*, not the max — and when a checkpoint trips the gate, check
whether its percentiles match a known-good model before dropping it. Dropping
the checkpoints that trip biases whichever arm trips more.

**`pgrep -f` matching its own command string.**
A watcher that greps for a pattern contained in its own command line matches
itself and waits forever, or reports a finished job as running. Anchor on the
interpreter or use a bracket class.

**Wait-predicates that match nothing.**
The opposite failure, and worse: a predicate anchored on a path that never
appears (a venv symlink resolves to the real interpreter) makes the wait a
no-op, and the queued job starts immediately on top of the running one. Always
verify a predicate matches while the thing it waits for is alive.

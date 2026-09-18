# Spec: the st0 = 0 DDM flow (7-parameter build)

**Status:** ready to run. Every claim below was verified rather than assumed;
where something was checked, the number is given.

## Why this exists

The shipped 8-parameter DDM flow (`cap256w_c4`) is calibrated on every parameter
except t0, which carries a systematic +0.28% (0.83 ms at t0 = 0.3 s) upward
offset that grows in standardised terms with N (+0.118 -> +0.371 from N = 100 to
800) and leaks into `a` (r = -0.28), `SZ` (+0.33) and `Z` (+0.19).

The mechanism is a **t0/st0 leading-edge trade-off**: t0 and st0 jointly
determine where the density turns on, so a flow that smooths a sharp onset trades
one against the other. Two independent lines support it — st0's own SBC bias is
opposite in sign to t0's, and the RDM, which has **no st0**, shows no t0 bias at
either N = 100 or N = 400 (standardised -0.013 and -0.001).

If the mechanism is right, a DDM flow trained with **st0 = 0** should not have
the problem. That is what this build tests, and if it passes it is the artefact
most users should actually be given: st0 is rarely of substantive interest and is
very often fixed in practice.

## Why the existing artefact cannot just be used at st0 = 0

`st0 = 0` is **outside the trained box**. The box is st0 in [0.01, 0.49], i.e.
log st0 in [-4.61, -0.71], and log(0) = -Inf. The lowest st0 the shipped flow can
represent is **10 ms**. The two running SBC cells (st0 fixed at 0.05 and at 0.01)
probe how close the existing artefact gets; they cannot reach zero. A genuinely
st0-free likelihood has to be trained as a 7-parameter model.

## What does NOT need redoing: the classifier

**The choice probability is exactly independent of st0.** Verified analytically:
P(upper) varies by **1.5e-11** across st0 in [0.001, 0.4] at fixed other
parameters, and `pWDM` at the CDF limit is identical to 7 decimals. This is
structural, not numerical — st0 is non-decision-time variability, so it shifts rt
but cannot change which boundary is hit.

The **trained** c4 classifier inherited the invariance: varying only st0 across
[0.01, 0.45] moves its P(R=2) by a mean of 0.000429 (max 0.000791) over 100 SBC
draws, which is **below its own accuracy floor** of 0.000845 against the analytic.

**Consequence: reuse the existing c4 classifier unchanged.** Feed it any fixed
st0 inside its box (use the prior mean, 0.05). Do not retrain it, and do not
train a 7-input classifier — you would be spending compute to relearn a function
that is constant in the dropped argument.

So this is a **flow-only** retrain.

## Step 1 — training data (~1-2 h)

EMC2 makes st0 = 0 easy: `DDM()$p_types` gives st0 a default of **-Inf** on the
sampled scale, and the transform is `exp`, so **omitting st0 from the formula
sets it to exactly 0**. No -Inf has to be passed by hand.

Use `handover/st0_zero/designs_st0zero.R` and `handover/st0_zero/sim_ddm_st0zero.R`
(both written and included). They are the existing `designs.R` / `sim_ladder16m.R`
with st0 removed from the formula, the box and the transform vector, and nothing
else changed — same seed, same 400k draws x 40 rows = 16e6 rows, same target
column `ll`.

Output: `data/train/st0zero400k/DDM/trials`.

## Step 2 — one edit to `flow/scripts/train_ddm.py`

`CONTEXT` is a module-level constant:

```python
CONTEXT = ["v", "a", "t0", "s", "Z", "SZ", "sv", "st0"]
```

Add a 7-parameter variant and a flag to select it:

```python
CONTEXT_ST0ZERO = ["v", "a", "t0", "s", "Z", "SZ", "sv"]
# ... in main(), alongside the existing --triple branch:
#   CTX = CONTEXT_ST0ZERO if args.st0_zero else (CONTEXT_TRIPLE if args.triple else CONTEXT)
```

Everything downstream is already dimension-generic: the conditioner's input width
is `len(CTX) + 1`, and the exporter asserts
`f_widths[0] == len(CONTEXT) + 1`, so `export_flow_ddm.py` needs the **same**
constant and the same flag. Those two asserts are the safety net — keep them.

> **Do not make this edit while a `train_ddm.py` run is in flight.** The
> distillation sweep launches fresh training processes at its stage 4, and a
> syntax error would kill it silently hours in.

## Step 3 — train (~10 h locally, 3 concurrent at 3 cores each)

Settings carried over from the 8-parameter work, all of which were established
empirically (see `sbc/SBC_PLAN.md`):

```
--batch 1024 --epochs 300 --widths 256,128 --num-bins 64
--keep-checkpoints 30 --classifier-from c4clf
--train-data data/train/st0zero400k/DDM/trials
```

`--train-data` is **not** optional: it defaults to a 1M-row set, and every
comparison model used the 16M one.

Keep 30 checkpoints, not 10: roughly half the apparent seed-to-seed variance is
within-run epoch-to-epoch variance, so screening a window from one run is ~5x
cheaper than training multiple seeds, and the screening spread is wider at lower
capacity.

## Step 4 — export, screen, select

Export every retained checkpoint, then rank on score bias and pick the best.
Retry any checkpoint that trips the float32/float64 port gate at
`--port-tol 2.5e-2` rather than dropping it — dropping biases the screen toward
smooth checkpoints.

`sbc/score_bias_draws.R` needs `PARS` and `FI` reduced to the 7 parameters
(drop `st0`), and `sbc/score_bias_t0.R` needs its analytic reference called with
`st0 = 0`. **With st0 = 0 the support boundary is a hard jump**, exactly as in the
RDM, so use the restricted-region estimator with the cut clear of t0 — there is
no smear to clear any more, so the RDM's simpler `cut <- t0n * exp(H) * 1.05`
applies rather than the DDM's st0-aware version.

Always run each estimator's analytic control first; it must return ~0.

## Step 5 — EMC2 integration: the one real obstacle

`EMC2:::.theta_DDMnn` **hardcodes the 8-column order** and ignores the export's
`context_names`:

```r
function (pars) cbind(pars[, "v"], log(pars[, "a"]), log(pars[, "t0"]),
                      log(pars[, "s"]), qnorm(pars[, "Z"]), qnorm(pars[, "SZ"]),
                      log(pars[, "sv"]), log(pars[, "st0"]))
```

This is the same trap already documented for `dRDMnn`. A 7-parameter flow cannot
be dropped in: the design omits st0, so `pars` has no `st0` column and this
errors — which is the good failure. The silent failure would be a *reordered*
context, so keep the `context_names` assertion in the port.

Add a parallel model function rather than editing the existing one:

- `.theta_DDMnn7` — the same `cbind` minus the `st0` term
- `dDDMnn7` / `pDDMnn7` / `DDMnn7`, reading `ddmnn_get("ddm_st0zero")`
- bundle installed at `<EMC2>/extdata/flownn/ddm_st0zero.rds`

Bundle filenames are hardcoded in `ddmnn_get()`, so a new model means a new
name, not a new argument.

## Step 6 — SBC, the verdict

One cell at N = 400, 500 replicates, design as in step 1 (st0 simply absent from
the formula). ~2.5 h at 48 cores on tux26; cost is flat in N, so N = 400 is no
more expensive than N = 100.

**The comparison that matters** is against the 8-parameter cell at the same N:
t0 standardised bias +0.319 with rank KS p 1.8e-07. If the st0 = 0 build lands
near zero with a uniform rank distribution, the mechanism is established from the
DDM's own side rather than only by analogy to the RDM, and the artefact is
recommendable without a correction.

## Total

| step | cost |
|---|---|
| 1 simulate | 1-2 h |
| 2 code edit | minutes |
| 3 train | ~10 h |
| 4 export + screen | ~1 h |
| 5 EMC2 integration | a few hours of coding |
| 6 SBC | ~2.5 h on 48 cores |

About **two days** including the integration, most of it unattended.

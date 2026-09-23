# Open items at handover

Status 2026-09-23. One result is outstanding; the rest are judgement calls that
do not block the document.

---

## 1. RUNNING: the direct-regression baseline's SBC cell

Everything else in this package is finished and self-consistent. This is the one
number the document promises and does not have.

**What is running.** `reg_s402`, N = 400, **250 replicates**, launched on tux26
2026-09-23 05:36 (CEST) at 48 cores (`SBC_CORES=16` x 3 chains) by
`sbc/run_reg_sbc.sh`.

| | |
|---|---|
| driver | `evaluation/SBCddmreg.R` (on tux26: `sbc/nci/SBCddmreg.R`) |
| model wrapper | `evaluation/DDMreg.R` (on tux26: `handover/regress/DDMreg.R`) |
| net | `results/model_states/reg_s402_budget/regress.json`, md5 `979c83158bd0ba0edf94d49dd895db77` |
| log | tux26 `~/emc2nn/nle/sbc/reg_sbc.log` |
| result | tux26 `~/emc2nn/nle/sbc/nci/results_reg_s402/DDMreg_N400_R250.rds` |
| marker | the line `reg sbc done` appended to `~/emc2nn/nle/sbc/phase37.log` |

**Expected duration ~5.7 days** from launch: the regression net costs 43 ms per
400-trial call against the flow's 0.80 ms on that machine (reference BLAS, 53x),
and the observed rate was 0.55 h per replicate on a contended machine. Nothing
is recoverable if it is killed — `run_sbc` writes the drawn truths at the start
and everything else at the end, so the `_tmp.RData` holds no partial results.

**Check progress:**

```bash
ssh andrew@tux26psy.fmg.uva.nl 'cd ~/emc2nn/nle
  grep -c "Sampling stopped" sbc/reg_sbc.log          # replicates finished, of 250
  sed -n "/=====.*done/,\$p" sbc/reg_sbc.log'          # the table, once finished
```

**Collect it:**

```bash
scp andrew@tux26psy.fmg.uva.nl:~/emc2nn/nle/sbc/nci/results_reg_s402/DDMreg_N400_R250.rds \
    results/sbc/reg_s402/
shasum -a 256 results/sbc/reg_s402/DDMreg_N400_R250.rds >> ARTEFACTS.sha256
```

**What the screen predicts**, so the result can be read against a registered
expectation rather than after the fact (`results/score_bias/postshift_reg_s402*.rds`):

| par | predicted std bias |
|---|---|
| t0 | +0.08 |
| Z | +0.35 |
| v | −0.18 |
| sv | +0.11 |
| SZ | −0.10 |
| st0 | −0.08 |
| a | −0.05 |

`Z` is the parameter to watch: it is the largest displacement screened for any
model in this project, and the only one near the 0.5 gate.

**Where to write the result — four places:**

1. `docs/METHODS.md` §4 — replace "the SBC cell is running and its result is NOT
   in this document" with the table, and settle the division-of-labour wording.
2. `docs/RECIPE.md` — the artefact status table.
3. `README.md` — the artefact status table.
4. `../../sbc/SBC_PLAN.md` — a Phase 38 entry, with the screen's prediction
   scored against the outcome (it has now been right on four cells; a miss is
   as informative as a hit).

**If it disagrees with the screen**, do not assume the screen is wrong: on the
st0 = 0 and decision-time cells, the disagreement was the cell's fault (§3.7 of
`METHODS.md`, Phase 37). Check the design first — `mapped_pars` for every
parameter not being sampled, and that the analytic control differs from the cell
in exactly one respect.

---

## 2. Judgement calls, not blockers

- **Which DDM artefact ships.** `cap256w_c4` remains the recommendation. The
  st0 = 0 build (`z0_s503_e282`) is now a real alternative: it passes the gate
  at 500 replicates with nominal coverage, and fails rank uniformity on `a`, t0
  and `Z` where the shipped flow fails on t0 alone. Neither is clean.
- **An independent cell for the t0 correction.** Phase 27's multiplicative
  correction is still calibrated and tested on the same four N values,
  leave-one-out only. A cell at an N outside {100, 200, 400, 800} closes it.
- **A compiled evaluator for the regression net.** Would turn a 5-day cell into
  a 3-hour one and make the baseline a practical artefact rather than a
  comparison. Note that a naive C++ loop will not beat reference BLAS by much;
  this needs an optimised BLAS or a vectorised library, and RcppEigen is not
  installed on tux26.
- **Phases 33 and 36 could be re-measured** on correctly specified cells. Their
  conclusions are already marked as resting on mis-specified ones, and §6 no
  longer depends on them, so this is for completeness only.

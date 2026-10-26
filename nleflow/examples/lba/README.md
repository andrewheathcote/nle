# The LBA example

One accumulator's finishing-time density of EMC2's LBA (rates `N(v, sv)`
truncated to be positive, `b = B + A`), the training box reaching `A = 0`
exactly. Context `v, sv, B, A, t0` (EMC2's LBA parameters), `v` as it is
(identity, the box reaches `v = -1`), `sv`, `B`, `t0` as logs, `A` as
`asinh(A / 0.05)`; the RDM example's three-part `A` mixture; [256, 128],
64 bins. The card registers with `twin = LBA`.

| input | box (natural) |
|---|---|
| v | [-1, 5] |
| sv | [0.5, 2] (v / sv >= -2: P(rate > 0) >= 0.023) |
| B | [0.1, 3] |
| A | [0, 2], 15 % of draws exactly 0 |
| t0 | [0.05, 1] |

The finishing time of an LBA accumulator has a 1/t survivor tail (the rate
density is positive at 0), heavier than the RDM's. The spline's range is
[-4, 10] in log seconds, so the identity tails carry what lies beyond
e^10 s, about 2e-5 of the training rows; the simulator prints that share.

```sh
cd nleflow/examples/lba
export NLEFLOW_RLIB=<library holding EMC2 built from DMC2 main> R_PROFILE_USER=/dev/null
uv run --project ../.. nleflow spec     spec_smoke.yaml   # the spec, resolved
uv run --project ../.. nleflow simulate spec_smoke.yaml   # 1 s: data_smoke/{train,val}
R_LIBS=$NLEFLOW_RLIB Rscript check_simulator.R             # the simulator vs pLBA
uv run --project ../.. nleflow train    spec_smoke.yaml   # 20 s: runs/lba_smoke/
uv run --project ../.. nleflow export   runs/lba_smoke    # 6 s
CARD=runs/lba_smoke/export/lba_smoke_e1_weights.json
R_LIBS=$NLEFLOW_RLIB Rscript -e 'source(system.file("scripts", "nle_verify.R", package = "EMC2"))' $CARD
R_LIBS=$NLEFLOW_RLIB Rscript check_in_emc2.R $CARD data_smoke/val
```

Checks:

| command | check |
|---|---|
| `simulate` | `=== simulated 240,000 rows ... A = 0 exactly in 0.15 ...` and a `val` line |
| `check_simulator.R` | three lines (all rows, A = 0, v < 0), KS p-values not small, deciles near 0.100 |
| `export` | `float64 gate: ... (tolerance 1e-08)`, then three `wrote` lines |
| `nle_verify.R` | `PASS golden check of 'lba_smoke_e1_weights.json' (flow_race, 575 rows)` and `all checks passed` |
| `check_in_emc2.R` | `compiled vs R path ... max \|diff\|` below 1e-11; the network vs the analytic LBA, per design and per accumulator |

`check_simulator.R` tests the simulator, not the network: `pLBA(rt | θ)` of
every simulated row is uniform when the data follow EMC2's LBA.
`check_in_emc2.R` registers the card, builds a two-accumulator design
(`v ~ lM`) through `nn_cell()` and compares the compiled likelihood with the
R path and with the analytic LBA over 50 prior draws; then, per accumulator,
the card's density with the analytic `ll` of 1e5 simulated rows (the mean of
`log p - log q` estimates the KL divergence). After 2 epochs on 1 % of the data the
smoke card is far from the LBA: it tests the pipeline, not the fit.

The full recipe is `spec.yaml`: 24 000 000 rows, 1 000 000 validation rows,
300 epochs, 30 checkpoints kept, the RDM example's recipe. On an Apple M2 Max at
8 cores an epoch took 71 to 86 s + 3 s (419 min in all) and the simulation under a minute;
`run_full.sh` runs both. The analytic floor of the validation loss (the mean of `-ll`
over `data/val`) is 0.908.

## The full card

`lba_e299` (the best validation loss, 0.909400 at step 299; the analytic floor
0.9078), in `handover/nle-methods/port/export/` as `lba_e299_weights.json`
with its golden file and intermediates; sha256 `65fe6178cc83e51b1157e73dee9e689b5ac93c48b1907453ec96faa99789429c`.

| check | result |
|---|---|
| float64 gate (port vs JAX float64, 575 points) | 4.6e-13 (log pdf), 3.8e-14 (z), against 1e-8 |
| float32 gap (information) | 1.9e-3 |
| `nleflow golden` from the card alone | byte-identical golden file |
| `nn_golden_check()` in EMC2 | PASS, 4.5e-12 against 1e-9 |
| `nle_verify.R <card>` | every line PASS, exit 0 |
| compiled vs R path, 50 draws × 200 trials | 1.7e-13 |
| per accumulator vs `dLBA`, 1e5 validation rows | KL 0.0019; \|Δ log p\| median 0.006, 90 % 0.022, 99 % 0.068 |
| the same at A = 0 / v < 0 / rt > 10 s | medians 0.006 / 0.008 / 0.018 |
| race likelihood vs the analytic LBA, 50 draws × 200 trials | mean \|Δ\| 1.5 (0.008 per trial), worst draw 23 |

The worst draw is a parameter vector far from the data: its fastest trials lie
just above the leading edge, where the analytic density is e^-15 to e^-22 and the
flow's e^-10 to e^-13 (the training data hold almost no rows there); in the bulk the
two agree to 0.01. The card records no validation: score bias over the 30 kept
checkpoints and an SBC cell are EMC2's validation kit, not run here.

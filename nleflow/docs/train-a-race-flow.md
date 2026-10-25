# Train a race flow

A race flow (`response: race`) models rt for one accumulator of an evidence-accumulation
race given the parameters; the example is the RDM with `A = 0` allowed (`examples/rdm_a0/`).
Needs the Python environment and, for the simulator, R with EMC2 (neural-likelihood build)
([install and check](install-and-check.md)). Run from `nleflow/`; the smoke spec is 1 % of
the data and 2 epochs, a pipeline check and not a fit.

**Training resolution.** Train on response times at full precision: do not floor or round them
(the example simulator writes them unrounded). The shipped cards were trained and validated at
1 ms or finer, and we recommend `rt_resolution = 1/1000` when sampling (START_HERE, "rt resolution").

```sh
export NLEFLOW_RLIB=$HOME/R/nle-lib R_PROFILE_USER=/dev/null
cd examples/rdm_a0     # from nleflow/; spec_smoke.yaml is nleflow/examples/rdm_a0/spec_smoke.yaml
uv run --project ../.. nleflow spec     spec_smoke.yaml   # the spec, resolved
uv run --project ../.. nleflow simulate spec_smoke.yaml   # data_smoke/{train,val}
uv run --project ../.. nleflow train    spec_smoke.yaml   # runs/rdm_a0_smoke/
uv run --project ../.. nleflow steps    runs/rdm_a0_smoke # the kept checkpoints
```

Checks, in order:

| command | check |
|---|---|
| `spec` | prints the resolved JSON, exit 0; a bad key is refused by name |
| `simulate` | `=== simulated 240,000 rows ... A = 0 exactly in 0.15 ===` and a `val` line |
| `train` | one `epoch n/2  train ...  val ...` line per epoch, then `flow done` |
| `steps` | `step 1: val_loss ... (best)` |

The recipe of the full run is `examples/rdm_a0/spec.yaml` (24 000 000 rows, 300 epochs); the spec's keys
are documented in `src/nleflow/spec.py` and in the README ("The spec"). Training is the NLE
project's `train_race.py`: the zero-box encoding computed in float64 from the stored value,
the context standardised by the training moments (`scaler.json` in the run), the spline
flow's mean negative log density as the loss, AdamW under a warmup-cosine schedule, one
Orbax checkpoint per epoch with the `keep` latest kept and the best validation loss
marked, no early stopping.

A second race example is the LBA (`examples/lba/`, `twin = LBA`): an identity input (`v`,
whose box reaches below 0), the twin's own parameter names, a wider spline range for the
LBA's 1/t survivor tail, and two scripts that check the simulator against `pLBA` and the
card inside EMC2's race likelihood against the analytic LBA; its README has the commands.

Your own model: write a `simulator` that follows the data contract (README, "The data
contract") and a spec; any program that writes one row per trial with the columns
`param_id`, each context name on the stored scale, `rt` will do.

Next: [export and check a card](export-and-check-a-card.md).

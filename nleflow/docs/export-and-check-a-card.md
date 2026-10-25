# Export and check a card

A card is the JSON file EMC2 loads with `register_nn_model()`, written beside golden
vectors that every evaluator of the card is tested against. The card format is EMC2's:
`vignette("neural-likelihood-cards")` in the EMC2 repository. Run after training
([race](train-a-race-flow.md) or [choice](train-a-choice-flow.md)); the commands below use
the choice smoke run, the race run is the same with `rdm_a0` and `twin = RDM`.

```sh
cd examples/ddmnle
CARD=runs/ddmnle_smoke/export/ddmnle_smoke_e1_weights.json
uv run --project ../.. nleflow export runs/ddmnle_smoke       # best validation loss; --step N for another
uv run --project ../.. nleflow golden $CARD /tmp/golden.csv   # the golden vectors from the card alone
cmp /tmp/golden.csv runs/ddmnle_smoke/export/ddmnle_smoke_e1_golden.csv
```

Checks:

| command | check |
|---|---|
| `export` | `float64 gate: ... (tolerance 1e-08)` printed with the float32 gap for information, then three `wrote` lines (`_weights.json`, `_golden.csv`, `_intermediates.json`) |
| `golden` | `wrote ... flow_scaled, 1150 rows` (575 rows for a race card) |
| `cmp` | no output: the file rewritten from the card is the exported one |

Before anything is written, the float64 NumPy port that produces the golden vectors is gated
against the flow itself evaluated by JAX in float64: log density and spline inversion must
agree to 1e-8 over the golden points, or the export refuses. On the smoke card of the cloud
instance: float64 gate 3.8e-13 (log pdf) and 1.1e-14 (z), float32 gap 7.5e-5. The card
carries no `validation` block until the spec records something in it, so EMC2 says "the
card records no validation" when it registers: loading a network is not the same as being
able to infer with it.

## In EMC2

```sh
export R_LIBS=$HOME/R/nle-lib R_PROFILE_USER=/dev/null
Rscript -e 'library(EMC2); m <- register_nn_model("'$CARD'", twin = DDM); nn_golden_check(m)'
Rscript -e 'source(system.file("scripts", "nle_verify.R", package = "EMC2"))' $CARD
```

Checks: `PASS golden check of 'ddmnle_smoke_e1_weights.json' (flow_scaled, 1150 rows): max
|diff| ... against 1e-09` (3.6e-13 on the cloud instance), and from `nle_verify.R` also
`PASS compiled vs R path ... against 1e-11` and `all checks passed`. A race card registers
with `twin = RDM` and `kind flow_race`. The tolerance is the card's own
`validation.golden.tolerance` when it has one, else 1e-9.

Next: [ONNX out](onnx-out.md) for use outside EMC2, or fit with the card in EMC2.

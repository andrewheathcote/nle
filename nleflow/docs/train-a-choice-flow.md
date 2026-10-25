# Train a choice flow

A choice model (`response: choice`) has two responses: a flow over rt given the
parameters and the response, and a classifier for P(R = 2). The example is the scaled DDM
(DDMnle, `examples/ddmnle/`): the flow models the unit-scale decision time
u = (rt - t0)(s/a)^2 given kappa, eta, Z and SZ, and t0, st0, a and s are applied exactly by
the card's evaluator. Needs the Python environment, R with EMC2 (neural-likelihood build) and `WienR` (the
exact choice probabilities); `arrow` for parquet, else CSV
([install and check](install-and-check.md)). The smoke spec is 1 % of the data and 2 epochs.

**Training resolution.** Train on response times at full precision: do not floor or round them
(the example simulator writes them unrounded). The shipped cards were trained and validated at
1 ms or finer, and we recommend `rt_resolution = 1/1000` when sampling (START_HERE, "rt resolution").

```sh
export NLEFLOW_RLIB=$HOME/R/nle-lib R_PROFILE_USER=/dev/null
cd examples/ddmnle     # from nleflow/; spec_smoke.yaml is nleflow/examples/ddmnle/spec_smoke.yaml
uv run --project ../.. nleflow simulate spec_smoke.yaml   # data_smoke/{train,val,choice}
uv run --project ../.. nleflow train    spec_smoke.yaml   # the flow, then the classifier
```

Checks:

| command | check |
|---|---|
| `simulate` | the flow rows are written, then `choice set .../choice/train.parquet: 2000 draws, P(R = 2) in [0.0000, 1.0000]` and a `val` line |
| `train` | the flow's epoch lines, then `classifier epoch 194 ... val mean \|P_hat - P\| ...` and `classifier done` |

The classifier maps the standardised context to the logit of P(R = 2), is trained after the
flow on the exact probabilities of the two choice sets (sigmoid output, MSE on the
probability, AdamW under warmup-cosine), with the flow's standardiser, and is written to
the run as `classifier.json`. `--classifier-from <run>` reuses another run's classifier when
the two standardisers agree to 1e-3. The choice sets are written in the format the
simulator is set to (`NLEFLOW_FORMAT`, parquet when `arrow` is installed) whatever the
spec's file names say, and the reader finds either.

Measured on the cloud instance (4 cores): data in 10 s, flow 2 epochs in 11 s, classifier
194 epochs in 6 s; the flow's validation loss after 2 epochs on 1 % is -1.107 against the
project's -1.436 after 300 epochs on all of it: a smoke run, not a fit. DDMnle itself is
not retrained here: the committed card `ddm_sc_s1106_e245` is the reference the tests use.
The DDM's own eight-parameter flows (`flow_joint` cards, no `scaling`) are the same path
without the scaling block: `examples/ddm_svst0/` trains one with sv and st0 free and s = 1,
SZ = 0 held fixed (the spec's `fixed`, recorded in the card for `register_nn_model(...,
exceptions = ...)`); its `examples/ddm_svst0/run_smoke.sh` runs the whole chain with one check per step.

Next: [export and check a card](export-and-check-a-card.md).

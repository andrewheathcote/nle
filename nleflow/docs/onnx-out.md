# ONNX out

`nleflow onnx` writes a `flow_joint` or `flow_scaled` card as an ONNX graph with the shape of a
published LAN, float64, opset 18, IR version 10 (onnxruntime <= 1.30 loads it), in two
signatures:

| signature | inputs | output |
|---|---|---|
| `trialwise` | one `[N, k + 2]` tensor: the k parameters, rt, choice (-1 lower, +1 upper) | `[N]` log p(rt, choice \| theta) |
| `shared` | `theta [k]`, `rt [N]`, `choice [N]`; the conditioner runs twice (once per response) and the classifier once | `[N]` |

**Trialwise or shared, and why it matters.** The two graphs compute the same number. Both pass the same checks to about 1e-12 on the golden rows, and the check lines further down are run on each. They differ in how often the network's parameter-dependent part, the conditioner that turns the parameters into the spline's shape, has to be run.

- **`trialwise`** takes a row per trial, each with its own parameter vector. It must run the conditioner for every row. Use it when the parameters really differ from trial to trial (a trial-level covariate, a trend, a hierarchical design evaluated at many subjects at once). Its input has the same layout as a published LAN's (`[N, k + 2]`: parameters, rt, choice), so code written for a LAN can call it unchanged.
- **`shared`** takes one parameter vector `theta [k]` and vectors of `rt` and `choice`. It runs the conditioner twice, once per response, and the classifier once, however many trials there are, and then only evaluates the cheap spline per trial. Use it when a block of trials shares its parameters, which is the usual case when fitting: all the trials of one participant in one design cell, or all the data scored at one posterior draw.

The cost difference is large: on the shipped `ddm_svst0` card, 100 000 trials at one parameter vector take 16 µs per trial with `trialwise` and 1.1 µs with `shared` (one thread; [examples/onnx_use](../examples/onnx_use/README.md)), about 15 times faster, and the gap grows with the block size. The rule: if the parameters repeat across trials, group the trials by parameter vector and call `shared` once per group. `shared` cannot take a different vector per trial, so `trialwise` is the only correct choice there. Calling `trialwise` on a block with one repeated vector is correct, only slower.

For a `flow_scaled` card the k inputs are the eight natural DDM parameters v, a, t0, s, Z, SZ
(the start-point proportion), sv, st0; the graph computes kappa, eta, the encodings and the
st0 branch exactly as EMC2's `nn_eval_scaled()` does, gives -Inf for rt <= t0 and the card's
floor outside the box. Needs the Python environment with `onnx` and `onnxruntime`
([install and check](install-and-check.md)); `--cases` needs R with EMC2.

```sh
cd examples/ddmnle
E=runs/ddmnle_smoke/export
uv run --project ../.. nleflow onnx $E/ddmnle_smoke_e1_weights.json            # both signatures
uv run --project ../.. nleflow onnx-check $E/ddmnle_smoke_e1_trialwise.onnx $E/ddmnle_smoke_e1_weights.json
uv run --project ../.. nleflow onnx-check $E/ddmnle_smoke_e1_shared.onnx    $E/ddmnle_smoke_e1_weights.json
```

Checks: `onnx` prints two `wrote ... .onnx` lines; `onnx-check` prints `golden rows of ...:
1150 rows, ... above the floor: max |dlog p| ...; -Inf rows agree: True; floor rows agree:
True` and `PASS against 1e-10`.

Against EMC2's own evaluation over prior draws (200 draws x 100 trials in each of the six
regimes of `sbc/scaled_cell.R`, half with s != 1):

```sh
export NLEFLOW_RLIB=$HOME/R/nle-lib R_PROFILE_USER=/dev/null
Rscript ../../scripts/onnx_cases_scaled.R $E/ddmnle_smoke_e1_weights.json cases.csv 200 100
uv run --project ../.. nleflow onnx-check $E/ddmnle_smoke_e1_shared.onnx $E/ddmnle_smoke_e1_weights.json --cases cases.csv
```

For a `flow_joint` card (`examples/ddm_svst0/`) the cases are the card's own inputs on the
stored scale with `rt`, `choice` and `logp`, written by that example's `check_in_emc2.R` (`nleflow/examples/ddm_svst0/check_in_emc2.R`); the
check is one `EMC2 cases (the card's inputs)` line against 1e-10.

Check: two more `PASS` lines, `EMC2 cases, st0 = 0 (a density)` against 1e-10 and `EMC2 cases,
st0 > 0 (a CDF difference)` against 1e-08. Measured on the committed DDMnle card, both
signatures:

| comparison | rows above the floor | max \|dlog p\| | median |
|---|---|---|---|
| golden rows | 1010 | 5.3e-11 | 7e-15 |
| EMC2, st0 = 0 (a density) | 38 400 | 4.4e-11 | 1e-14 |
| EMC2, st0 > 0 (a CDF difference) | 78 400 | 5.6e-9 | 4e-15 |

The 5e-11 rows sit at z = 2.0, the spline's range boundary, where the identity tail takes
over; on the st0 > 0 branch the difference of two CDFs amplifies round-off, and the NumPy
float64 port differs from EMC2 by the same 3e-9 on those rows. Hence the two defaults, 1e-10
and 1e-8 (`--tol`, `--tol-window`). onnxruntime has no double-precision `Erf` or `Asinh`, so
the graph carries R's own pnorm (Cody) and qnorm (AS 241) as graph operations.

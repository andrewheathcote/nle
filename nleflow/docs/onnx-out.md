# ONNX out

`nleflow onnx` writes a `flow_joint` or `flow_scaled` card as an ONNX graph with the shape of a
published LAN, float64, opset 18, IR version 10 (onnxruntime <= 1.30 loads it), in two
signatures:

| signature | inputs | output |
|---|---|---|
| `trialwise` | one `[N, k + 2]` tensor: the k parameters, rt, choice (-1 lower, +1 upper) | `[N]` log p(rt, choice \| theta) |
| `shared` | `theta [k]`, `rt [N]`, `choice [N]`; the conditioner runs twice (once per response) and the classifier once | `[N]` |

For a `flow_scaled` card the k inputs are the eight natural DDM parameters v, a, t0, s, Z, SZ
(the start-point proportion), sv, st0; the graph computes kappa, eta, the encodings and the
st0 branch exactly as EMC2's `nn_eval_scaled()` does, gives -Inf for rt <= t0 and the card's
floor outside the box. Needs the Python environment with `onnx` and `onnxruntime`
([install and check](install-and-check.md)); `--cases` needs R with EMC2.

```sh
cd examples/ddm_scaled
E=runs/ddm_scaled_smoke/export
uv run --project ../.. nleflow onnx $E/ddm_scaled_smoke_e1_weights.json            # both signatures
uv run --project ../.. nleflow onnx-check $E/ddm_scaled_smoke_e1_trialwise.onnx $E/ddm_scaled_smoke_e1_weights.json
uv run --project ../.. nleflow onnx-check $E/ddm_scaled_smoke_e1_shared.onnx    $E/ddm_scaled_smoke_e1_weights.json
```

Checks: `onnx` prints two `wrote ... .onnx` lines; `onnx-check` prints `golden rows of ...:
1150 rows, ... above the floor: max |dlog p| ...; -Inf rows agree: True; floor rows agree:
True` and `PASS against 1e-10`.

Against EMC2's own evaluation over prior draws (200 draws x 100 trials in each of the six
regimes of `sbc/scaled_cell.R`, half with s != 1):

```sh
export NLEFLOW_RLIB=$HOME/R/nle-lib R_PROFILE_USER=/dev/null
Rscript ../../scripts/onnx_cases_scaled.R $E/ddm_scaled_smoke_e1_weights.json cases.csv 200 100
uv run --project ../.. nleflow onnx-check $E/ddm_scaled_smoke_e1_shared.onnx $E/ddm_scaled_smoke_e1_weights.json --cases cases.csv
```

For a `flow_joint` card (`examples/ddm_svst0/`) the cases are the card's own inputs on the
stored scale with `rt`, `choice` and `logp`, written by that example's `check_in_emc2.R`; the
check is one `EMC2 cases (the card's inputs)` line against 1e-10.

Check: two more `PASS` lines, `EMC2 cases, st0 = 0 (a density)` against 1e-10 and `EMC2 cases,
st0 > 0 (a CDF difference)` against 1e-08. Measured on the committed scaled DDM card, both
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

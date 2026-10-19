# A card's ONNX graphs from Python, onnxruntime only

An nleflow card exported to ONNX and used with nothing but `numpy` and
`onnxruntime`: no nleflow, no JAX, no EMC2. This is how HSSM-style code (or any
other ONNX consumer) would take a flow trained here. [docs/onnx-out.md](../../docs/onnx-out.md)
describes the export and `nleflow onnx-check`.

```sh
cd nleflow/examples/onnx_use
./run.sh                 # the ddm_svst0 card in handover/nle-methods/port/export; about 5 s
./run.sh <card_weights.json>    # another flow_joint card, its golden file beside it
```

`use_onnx.py` is written for `flow_joint` cards; it reads their JSON layout
(`kind`, `context_names`, `context_transforms`). A `flow_scaled` card (DDMnle) has
another layout: its graph exports and passes `nleflow onnx-check` the same way, and
takes the eight natural DDM parameters (docs/onnx-out.md), but this script does not
read that card.

`run.sh` copies the card and its golden file to `out/`, writes both graphs with
`nleflow onnx` (the only nleflow step), then runs `use_onnx.py`, which reads the card's
JSON only for the input names and transforms.

## The two graphs

| graph | inputs | use |
|---|---|---|
| `<tag>_trialwise.onnx` | `input [N, k + 2]`: the k parameters on the stored scale, then rt (s), then choice ±1 | a parameter vector per trial (hierarchical models, a design with trial-level covariates) |
| `<tag>_shared.onnx` | `theta [k]`, `rt [N]`, `choice [N]` | one parameter vector for a block of trials; computes the conditioner once, so much faster |

Both return `log_likelihood [N]`, the log joint defective density
log p(rt, choice | θ), in float64: −Inf below the onset when `st0 = 0`, and the card's
floor (log 1e-10) outside the training box. The graph's description (printed first)
states all of this.

**The stored scale.** Parameters go in as the card stores them, not on the natural
scale: `identity` as is; `log` as log x; a zero-box input (`sv`, `st0`) as log x with
−Inf for an exact 0; `probit` (`Z`) as Φ⁻¹(x). `stored()` in `use_onnx.py` is the
whole conversion; the names and transforms come from the card's `context_names` and
`context_transforms`. Choice −1 is EMC2's first response (lower boundary), +1 the
second (upper). Parameters the card holds fixed (here `s = 1`, `SZ = 0`, the card's
`fixed`) are not inputs.

## What it prints, on `ddm_svst0_e299` (Apple M2 Max)

```
ddm_svst0_e299: flow_joint card for EMC2's DDM; inputs ['v', 'a', 't0', 'Z', 'sv', 'st0'], fixed {'s': 1.0, 'SZ': 0.0}
  trialwise in  input ['N', 8] tensor(double)
            out log_likelihood ['N']
  shared    in  theta [6] tensor(double); rt ['N'] tensor(double); choice ['N'] tensor(double)
            out log_likelihood ['N']
natural v=1, a=1.2, t0=0.3, Z=0.5, sv=0.5, st0=0.1 -> inputs [1.0, 0.1823, -1.204, 0.0, -0.6931, -2.3026]
  P(lower) 0.2469 + P(upper) 0.7528 = 0.9998 (rt to 30 s; 1 for a trained card); log p at rt = t0 - 0.01: -10.256, at t0 + 0.001: -10.557
natural v=1, a=1.2, t0=0.3, Z=0.5, sv=0.5, st0=0 -> inputs [1.0, 0.1823, -1.204, 0.0, -0.6931, -inf]
  P(lower) 0.2469 + P(upper) 0.7527 = 0.9995 (rt to 30 s; 1 for a trained card); log p at rt = t0 - 0.01: -inf, at t0 + 0.001: -11.515
golden rows, trialwise: 1150 rows, 1106 above the floor: max |dlog p| 3.31e-12; -Inf rows agree True; floor rows agree True: PASS against 1e-10
golden rows, shared: 1150 rows, 1106 above the floor: max |dlog p| 3.31e-12; -Inf rows agree True; floor rows agree True: PASS against 1e-10
100000 trials, one parameter vector, trialwise: 1640 ms (16.40 us per trial, one thread)
100000 trials, one parameter vector, shared: 108 ms (1.08 us per trial, one thread)
```

How to read it:

- **The density integrates to 1** (0.9998 and 0.9995 by the trapezoid rule to 30 s;
  the small deficit at `st0 = 0` is the rule at the jump at t0). The two responses
  split as the DDM does.
- **`st0 = 0` gives an exact onset** (−Inf just below t0); with `st0 > 0` the flow
  models the smear and is small but not zero below t0 (log p −10.3 at 10 ms below).
- **The golden rows match** EMC2's own evaluation of the card (`log_pdf_joint` in the
  golden file) to 3e-12 with both graphs, including which rows are −Inf and which are
  at the floor. This is the acceptance test of a graph; `nleflow onnx-check` runs the
  same comparison and also against EMC2 cases (see the ddm_svst0 README).
- **Use the shared graph for blocks of trials**: 1.1 µs a trial against 16 µs, one
  thread. Set `intra_op_num_threads` to use more.

On the 2-epoch smoke card from `examples/ddm_svst0/run_smoke.sh` the same script
passes the golden rows (2.3e-13) but the mass is 0.95 at `st0 = 0`: an untrained
card is a pipeline check, not a density.

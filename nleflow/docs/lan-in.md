# LAN in

A published likelihood approximation network (Fengler et al.; HSSM's LANs) is a plain
multilayer perceptron whose ONNX file holds weights and an input width. `nleflow convert`
reads the graph and writes an EMC2 card of kind `mlp_joint` (card_version 1) that EMC2's
compiled MLP evaluator runs; EMC2 never runs ONNX and has no converter of its own. Needs
the Python environment with `onnx` and `onnxruntime` ([install and check](install-and-check.md)).

What a graph does not record comes from a preset (`src/nleflow/presets/*.yaml`):

```sh
uv run nleflow convert --list-presets
uv run nleflow convert path/to/ddm_uniform_st.onnx --preset ddm_uniform_st \
    --out lan_ddm_uniform_st_weights.json --verify 300
```

(`path/to/` is where the published `ddm_uniform_st.onnx` is. The current file is
`https://huggingface.co/Eitanm/ddm-st-lans/resolve/40e0327c22a4/ddm_uniform_st.onnx`; EMC2's
`tests/testthat/golden/` holds the superseded 3 × 100 file. Both convert; see Presets below.
[examples/lan_in](../examples/lan_in/README.md) runs the whole path.) Checks:

| command | check |
|---|---|
| `--list-presets` | `ddm_uniform_st: huggingface.co/Eitanm/ddm-st-lans :: ddm_uniform_st.onnx`, its parameters, box and mapping onto EMC2's DDM |
| `convert ... --verify 300` | ends with `PASS` and writes `lan_ddm_uniform_st_golden.csv` beside the card |

On the cloud instance `--verify` printed: card (float64) vs a float64 copy of the graph
1.07e-14 (gate 1e-10); vs the graph as published (float32) 1.08e-05 (gate 1e-04, float32
round-off); the published graph, one-trial calls vs batched, 0.00e+00. 1e-6 is out of reach
for any float64 evaluator of a float32 network, and EMC2's own test of this card uses 1e-4.

## In EMC2

A LAN has no analytic twin in its own parameterisation, so it registers with defaults
for its parameters (`p_types`). The converter writes the preset's into the card, where
`nle_verify.R` reads them; `register_nn_model()` takes them as an argument:

The card is the file `lan_ddm_uniform_st_weights.json` that `nleflow/examples/lan_in/run.sh`
writes to `nleflow/examples/lan_in/out/`; run these from that folder:

```sh
export R_LIBS=$HOME/R/nle-lib R_PROFILE_USER=/dev/null
Rscript -e 'library(EMC2); f <- "lan_ddm_uniform_st_weights.json"
    m <- register_nn_model(f, p_types = unlist(jsonlite::fromJSON(f)$p_types)); nn_golden_check(m)'
Rscript -e 'source(system.file("scripts", "nle_verify.R", package = "EMC2"))' lan_ddm_uniform_st_weights.json
```

(A card converted without a preset has no `p_types`: give them with
`--p-types=v=0,a=1,...` to `nle_verify.R`.)

Checks: `PASS golden check of 'lan_ddm_uniform_st_weights.json' (mlp_joint, 300 rows): max
|diff| 1.08e-05 (log_pdf) against 1e-04`, and `all checks passed`. The tolerance is not
typed: the converter records it in the card (`validation.golden.tolerance`) and
`nn_golden_check()` reads it.

## What it takes and refuses

One input, rank 1 `[k]` (one trial per call, as the published LANs) or rank 2 `[N, k]`;
MatMul + Add or Gemm layers; Tanh or Gelu(approximate = tanh) hidden layers; one linear
output, the log density; Identity, and Reshape / Flatten / Squeeze / Unsqueeze / Cast before
the first layer or after the last. EMC2's evaluator runs tanh and gelu_tanh only, so a Relu,
an exact (erf) Gelu, a Sigmoid or any node that is not part of a plain chain is refused with
a message naming the first such node by position, op type and name; each new activation is
a C++ change in EMC2 first, then one line in the converter.

## Presets

A preset holds the parameters in input order, their transforms, the training box, the label
floor, the response coding (HSSM: -1 lower, +1 upper), the input layout, the mapping onto an
EMC2 model as text and structured, the defaults `register_nn_model()` needs
(`emc2_p_types`), the sha256 of each published file the preset was checked on, with its revision and box (`versions`: a source can replace its network under the same name, as HSSM's `ddm_uniform_st` did on 2026-10-05, 3 × 100 with st ≤ 0.25 → 5 × 100 with st ≤ 0.35; the converter takes the box of the file it is given and records `source_revision`; any other file draws a warning) and a `checked` block saying against what each value was verified. Today:
`ddm_uniform_st` (`a_EMC2 = 2 a`, `Z = z`, `t0 = t - st`, `st0 = 2 st`, `sv = SZ = 0`,
`s = 1`). Without a preset the same comes from `--params`, `--box`, `--ll-floor`,
`--response-values=-1,1`, `--layout` and `--transforms`. The card is marked
`calibration: "unknown; needs its own SBC cell before inference"` always.

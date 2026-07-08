# `port/` — flow likelihoods outside Python

Makes trained flows callable from R and C++ (destined for EMC2): given a
parameter vector and rt(s), return pdf and cdf (plus log-pdf and log-survivor,
which is what race-model likelihoods actually consume).

**Model-agnostic for race-character models** (LNR, RDM, and future models with
the same structure): a single spline flow over rt conditioned on parameters.
The network architecture is inferred from the checkpoint; only three
model-specific facts are registered per model (context column order,
training-region bounds, natural→sampled transforms). The DDM (spline flow +
choice classifier) will get a parallel workflow later; its spline-flow half
reuses this code.

## Pipeline

```
flow/scripts/export_flow.py   Python: checkpoint -> JSON + float64 golden vectors
        │ writes port/export/<model>_<budget>_*
        ▼
port/R/flow_race.R            R reference implementation (the readable spec)
port/src/flow_race.cpp        C++ (Rcpp) implementation, same API shape
port/tests/validate.R         both vs golden, box tests, benchmark
        ▼
(next) EMC2 dfun/pfun hookup
```

## Usage

```sh
# 1. export a trained checkpoint (run from flow/)
cd flow && uv run python scripts/export_flow.py --model LNR --budget small

# 2. validate R + C++ implementations and benchmark (run from repo root)
Rscript port/tests/validate.R LNR small
```

```r
source("port/R/flow_race.R")
Rcpp::sourceCpp("port/src/flow_race.cpp")
fl  <- flow_load("port/export/lnr_small_weights.json")
ptr <- flow_build(fl)                      # once per session
# parameter vector on the SAMPLED scale (for LNR: m, log s, log t0)
ev <- flow_eval_cpp(ptr, c(0.5, log(0.4), log(0.2)), rt = c(0.4, 0.7, 1.2))
ev$pdf; ev$cdf; ev$log_pdf; ev$log_sf
# trial-wise (EMC2-shaped): n x K parameter matrix + length-n rt;
# consecutive duplicate rows reuse the conditioner (cell-structured designs)
tv <- flow_eval_trials_cpp(ptr, theta_matrix, rt_vector)
```

## Adding a new race-character model

1. Train the flow (notebook) → checkpoint under
   `results/model_states/<budget>_budget/<HEAD>/` with `train_loader.pkl`.
2. Add an entry to `MODEL_REGISTRY` in `flow/scripts/export_flow.py`
   (context order as in the notebook's `CONTEXT`, bounds + transforms as in
   `simulation/designs.R` / `sim_code.R`).
3. Export, then `Rscript port/tests/validate.R <MODEL> <budget>`.

No R or C++ changes needed — architecture (depth, widths, LayerNorm, bins) is
read from the export. RDM is already registered (natural-scale parameters,
`transforms = NULL` in sim_code.R).

## Key facts

- **One spline inversion gives everything.** With u = log(rt) and
  z = spline⁻¹(u): pdf = dnorm(z)·|dz/du|/rt, cdf = pnorm(z),
  sf = pnorm(z, lower = FALSE). No integration.
- **Amortization.** The MLP conditioner depends only on parameters — it runs
  once per parameter vector (`flow_eval_cpp`) or once per run of identical
  trial rows (`flow_eval_trials_cpp`).
- **Bounding box.** Parameters outside the training region (`bounds_sampled`
  in the JSON) return pdf = 0 and cdf = 0 immediately (log_pdf = -Inf).
  Rejection sentinel, not extrapolation.
- **Scale convention.** Parameters are on the sampled scale used in training
  (LNR: m, log s, log t0; RDM: natural scale).
- **Precision.** Golden vectors are float64 recomputations of the exact
  distrax math, cross-checked in the exporter against the float32 JAX
  pipeline (~5e-5, float32 noise). R and C++ match the golden values at
  ~1e-13 and each other at ~1e-13; the pdf integrates to 1 within ~4e-8.
- **Speed** (LNR small, M-series CPU, pdf+cdf per call): C++ ≈ 15 µs at
  n=100 rts, ≈ 125 µs at n=1000 — about 2.5x the analytic
  `dlnorm`+`plnorm` cost. For models with expensive likelihoods (full DDM)
  this is the orders-of-magnitude win.

## Files

| File | Role |
|---|---|
| `../flow/scripts/export_flow.py` | model-agnostic exporter (arch inferred from checkpoint) + golden vectors |
| `export/<model>_<budget>_weights.json` | self-contained model (scaler, MLP, spline config, bounds) |
| `export/<model>_<budget>_golden.csv` | 25 parameter sets × 23 rts: z, log_pdf, cdf, log_sf |
| `export/<model>_<budget>_intermediates.json` | layer-by-layer values (port debugging) |
| `R/flow_race.R` | R reference: `flow_load`, `flow_knots`, `flow_eval` |
| `src/flow_race.cpp` | Rcpp port: `flow_build`, `flow_eval_cpp`, `flow_eval_trials_cpp` |
| `tests/validate.R` | golden + box + trial-wise + normalization checks, benchmark |

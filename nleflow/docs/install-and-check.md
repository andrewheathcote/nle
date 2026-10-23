# Install and check

What you need depends on the task (NLEFLOW_PLAN.md section 9.1): **using a card in
EMC2** needs R and EMC2 `dev-flow`; **training, exporting, ONNX and converting** need
the Python environment of this directory; the **simulator of a worked example** needs
both. Every step below has one check. Commands run from `nleflow/` unless said.

Order: R and EMC2 first, then the verification command, then Python.

**Which EMC2.** The steps name `dev-flow` of `ampl-psych/EMC2`: the nle work on `dev` after
the `joint_move` sampler merge; it reads a card's `fixed` field. The older `dev-nle-m0`
serves except for cards with fixed parameters ([START_HERE.md](../START_HERE.md), "Known limits").

## 1. R >= 4.5 and EMC2

R older than 4.5 has no `tools::sha256sum`; EMC2 then pins its shipped artefacts by md5
and a JSON card still registers, but use 4.5 or later.

1. R >= 4.5 from CRAN. Check: `R --version | head -1` shows 4.5 or later.
2. A C++17 compiler. macOS: Xcode command-line tools and Homebrew `libomp`; Linux:
   `g++` >= 11. Check: `R CMD config CXX17` prints a compiler.
3. EMC2's R packages and `WienR` >= 0.3-17 (not packaged for Ubuntu; from CRAN).
   Check: `Rscript -e 'stopifnot(packageVersion("WienR") >= "0.3.17")'` exits 0.
4. EMC2 branch `dev-flow` of `ampl-psych/EMC2`, into a dedicated library:

   ```sh
   git clone -b dev-flow https://github.com/ampl-psych/EMC2
   mkdir -p ~/R/nle-lib
   MAKEFLAGS=-j4 R CMD INSTALL --preclean --library=$HOME/R/nle-lib EMC2
   ```

   Check: `Rscript -e '.libPaths(c("~/R/nle-lib", .libPaths())); library(EMC2); cat(find.package("EMC2"), "\n")'`
   prints a path under `~/R/nle-lib`. If it prints another path, a second EMC2 is
   shadowing yours: put the library first with `R_LIBS=~/R/nle-lib` and run with
   `R_PROFILE_USER=/dev/null` (the dual-install trap, section 9.2 of the plan).
5. The verification command, which installs nothing:

   ```sh
   Rscript -e 'source(system.file("scripts", "nle_verify.R", package = "EMC2"))'
   ```

   Check: one `PASS` line per shipped artefact and `all checks passed`; the exit status
   is 1 on any `FAIL`. Cards are checked by naming them
   (`... nle_verify.R <card>`), each against the `<tag>_golden.csv` beside it.

## 2. The Python environment

1. **uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS also `brew install uv`).
   Check: `uv --version` prints 0.8 or later.
2. **Python 3.11 or 3.12**; uv fetches one when none is present (`uv python install 3.11`).
   Check: `uv python find 3.11` prints a path.
3. **The environment from the lock**: `uv sync --frozen --group dev`. It installs the
   pinned versions every shipped flow was trained and exported with (jax 0.5.3, flax
   0.10.6, distrax 0.1.5, optax 0.2.8, orbax-checkpoint 0.11.1, numpy 2.2.6, scipy 1.15.3)
   plus pyarrow, pyyaml, onnx, onnxruntime and pytest into `.venv/`. Check:
   `uv run python -c "import jax, flax, distrax, onnx, onnxruntime; print(jax.__version__)"`
   prints `0.5.3`.
4. **The command.** Check: `uv run nleflow --help` lists nine commands.
5. **ONNX runtime.** Check: `uv run python -c "import onnxruntime as ort; print(ort.__version__)"`
   prints 1.20 or later.
6. GPU is optional and untested (`uv sync --frozen --group dev --extra gpu`); every shipped
   flow was trained on CPU.

`nleflow train` and `export` limit themselves to `min(8, cores)` threads (`--cores` to
change), the plan's "at most 8 cores" rule.

## 3. The tests

```sh
export NLEFLOW_EMC2_LIB=$HOME/R/nle-lib NLEFLOW_RLIB=$HOME/R/nle-lib
export NLEFLOW_EMC2_SRC=/path/to/EMC2        # EMC2's source tree; default ../EMC2 beside the nle repo
R_PROFILE_USER=/dev/null uv run pytest -rs
```

Check: no failures. Without the three variables the R-dependent cases skip and say why
(`-rs`): `NLEFLOW_EMC2_LIB` (a library holding EMC2 `dev-flow`) turns on the cases that
register cards in R and compare the ONNX graph with EMC2, `NLEFLOW_RLIB` the R simulator
cases, and `NLEFLOW_EMC2_SRC` the LAN and z0 cases, which read EMC2's
`tests/testthat/golden/`.

Golden and ONNX comparisons are bounded per row by 1e-12 plus a multiple of the
rounding error of the spline knots over the width of the bin holding the point
(`tests/test_golden.py`, `tests/test_onnx.py`): the two narrowest bins (1e-4) amplify
the last bit of a matrix product, so the same card differs across BLAS libraries there.
The golden file of DDMnle differs from this machine's output by up to 3.6e-11 on x86_64
OpenBLAS and 1.8e-11 on Apple Accelerate, only where z lies in the top bin.

## 4. The container (no local R or Python needed)

`docker/Dockerfile` reproduces the cloud steps of the plan's section 9.2: Ubuntu 24.04,
R >= 4.5 from CRAN's Ubuntu repository, EMC2's R packages, `WienR` and the test suite's
`testthat`, `vdiffr`, `withr`, `DiagrammeR`, EMC2 in `/opt/emc2/lib` (its source in
`/opt/emc2/src`, `EMC2_SHA.txt` beside the library), `uv` and the lock, a UTF-8 locale.
From the root of the nle repository, either EMC2 from GitHub at `EMC2_SHA` (default
`dev-flow`, bb96fb03):

```sh
docker build -f nleflow/docker/Dockerfile -t nleflow .
```

or from a source tarball of a local EMC2 build, placed in
`nleflow/docker/emc2/` (git-ignored); `EMC2_SHA` is then only a label:

```sh
R CMD build <EMC2 worktree> && mv EMC2_3.4.1.9001.tar.gz nleflow/docker/emc2/
docker build -f nleflow/docker/Dockerfile --build-arg EMC2_TARBALL=EMC2_3.4.1.9001.tar.gz \
  --build-arg EMC2_SHA=bb96fb03 -t nleflow:dev-flow .
```

Then:

```sh
docker run --rm nleflow nle-verify     # EMC2's verification command; one PASS per card
docker run --rm nleflow pytest         # nleflow's tests with the R checks on
docker run --rm nleflow emc2-tests     # EMC2's testthat suite (NOT_CRAN=true, as devtools::test())
```

Check: each exits 0. `emc2-tests` prints `EMC2 tests: <passed> / 0 failed / <skipped>`.

**BLAS.** R and EMC2 link `libblas.so.3`, which the image points at the reference BLAS;
OpenBLAS is installed beside it. In a container (as root) `r-blas openblas` switches
both BLAS and LAPACK, `r-blas reference` switches back, `r-blas` prints the choice. A
card's fit is faster on OpenBLAS (the flows are matrix products); the analytic DDM is not
affected. Use `OPENBLAS_NUM_THREADS=1` when the chains run in parallel.

**Measured** (AH's Mac, Apple M2 Max, colima VM with 8 CPUs and 24 GB, linux/arm64;
2026-10-22, EMC2 `dev-flow` 9f2801b2 from its tarball, R 4.6.1): the image builds in
4 min 21 s and is 4.35 GB (1.09 GB content). EMC2's suite gives 1472 passed / 0 failed /
9 skipped in 75 s; the Mac gives 1474 / 0 / 8. The difference is two macOS-only
expectations: the Accelerate link-order test skips, and the `VECLIB_MAXIMUM_THREADS`
check in `test-nn-validate.R` does not run. `nle-verify` passes on the shipped cards,
DDMnle, the RDM A = 0 card and ddm_svst0_e299, with the same maxima as on the Mac.
EMC2's compiled likelihood on the inputs of `test-nn-reference.R`, exported from the
Mac, against the Mac's values: trial-wise max |diff| 3.1e-13 (ddm_cap256w_c4), 1.35e-12
(ddm_st0zero, one trial of 9600; the median is 6e-16), 9.5e-14 and 1.3e-13 (rdm_small
with 2 and 3 accumulators) and 4.1e-14 (ddm_svst0_e299); sums agree within 1.2e-14
relative. The reference and OpenBLAS builds agree with each other to 1.8e-15, so the
Mac-to-Linux difference comes from the compiler and libm, not the BLAS. In the image,
[TUTORIAL.md](../TUTORIAL.md) takes 243 s (Mac 208 s; simulation 52 s, training 163 s,
the rest 29 s) with every check as the page says; `pytest` gives 67 passed in 66 s;
`run_smoke.sh` takes 57 s; `lan_in` and `onnx_use` pass. The fits of EMC2's vignette
"Using a trained flow" (`inst/scripts/using_a_trained_flow_data.R`, 3 chains x 2 cores):

| BLAS | card fit | analytic DDM fit |
|---|---|---|
| reference (the image's default) | 17.3 s | 23.0 s |
| OpenBLAS | 11.0 s | 23.0 s |
| OpenBLAS, `OPENBLAS_NUM_THREADS=1` | 10.1 s | 23.0 s |
| Mac, Accelerate (native) | 5.9 s | 23.2 s |

The posterior tables on the two BLAS agree to 4e-15. CI
(`.github/workflows/nleflow.yml`) builds the default image on x86_64 on every push to
`nleflow-plan` and runs `nle-verify` and `pytest`.

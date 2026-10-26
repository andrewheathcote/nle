# Install and check

What you need depends on the task: **using a card in
EMC2** needs R and EMC2 with the neural-likelihood code; **training, exporting, ONNX and converting** need
the Python environment of this directory; the **simulator of a worked example** needs
both. Every step below has one check. Commands run from `nleflow/` unless said.

Order: R and EMC2 first, then the verification command, then Python.

**Which EMC2.** The steps use the EMC2 on the `main` branch of https://github.com/andrewheathcote/DMC2 (private; ask for
access). It contains the neural-likelihood code and reads a card's `fixed` field
([START_HERE.md](../START_HERE.md), "Known limits").

## 1. R >= 4.5 and EMC2

R older than 4.5 has no `tools::sha256sum`; EMC2 then pins its shipped cards by md5
and a JSON card still registers, but use 4.5 or later.

1. R >= 4.5 from CRAN. Check: `R --version | head -1` shows 4.5 or later.
2. A C++17 compiler. macOS: Xcode command-line tools and Homebrew `libomp`; Linux:
   `g++` >= 11. Check: `R CMD config CXX17` prints a compiler.
3. EMC2's R packages and `WienR` >= 0.3-17 (not packaged for Ubuntu; from CRAN).
   Check: `Rscript -e 'stopifnot(packageVersion("WienR") >= "0.3.17")'` exits 0.
4. EMC2 from https://github.com/andrewheathcote/DMC2, into a dedicated library:

   ```sh
   git clone https://github.com/andrewheathcote/DMC2 EMC2
   mkdir -p ~/R/nle-lib
   MAKEFLAGS=-j4 R CMD INSTALL --preclean --library=$HOME/R/nle-lib EMC2
   ```

   Check: `Rscript -e '.libPaths(c("~/R/nle-lib", .libPaths())); library(EMC2); cat(find.package("EMC2"), "\n")'`
   prints a path under `~/R/nle-lib`. If it prints another path, a second EMC2 is
   shadowing yours: put the library first with `R_LIBS=~/R/nle-lib` and run with
   `R_PROFILE_USER=/dev/null`.
5. The verification command, which installs nothing:

   ```sh
   Rscript -e 'source(system.file("scripts", "nle_verify.R", package = "EMC2"))'
   ```

   Check: one `PASS` line per shipped card and `all checks passed`; the exit status
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
change).

## 3. The tests

```sh
export NLEFLOW_EMC2_LIB=$HOME/R/nle-lib NLEFLOW_RLIB=$HOME/R/nle-lib
R_PROFILE_USER=/dev/null uv run pytest -rs
```

Check: no failures (68 passed). Without the two variables the R-dependent cases skip and
say why (`-rs`): `NLEFLOW_EMC2_LIB` (a library holding EMC2 (DMC2 main)) turns on the cases
that register cards in R and compare the ONNX graph with EMC2, `NLEFLOW_RLIB` the R
simulator cases. The published LAN, the LAN card and onnxruntime values EMC2's own tests
hold, and the other reference files the tests read are in `nleCards/`.

Golden and ONNX comparisons are bounded per row by 1e-12 plus a multiple of the
rounding error of the spline knots over the width of the bin holding the point
(`tests/test_golden.py`, `tests/test_onnx.py`): the two narrowest bins (1e-4) amplify
the last bit of a matrix product, so the same card differs across BLAS libraries there.
The golden file of the scaled DDM card differs from this machine's output by up to 3.6e-11 on x86_64
OpenBLAS and 1.8e-11 on Apple Accelerate, only where z lies in the top bin.

## 4. The container (no local R or Python needed)

`docker/Dockerfile` reproduces the steps below as an image: Ubuntu 24.04,
R >= 4.5 from CRAN's Ubuntu repository, EMC2's R packages, `WienR` and the test suite's
`testthat`, `vdiffr`, `withr`, `DiagrammeR`, EMC2 in `/opt/emc2/lib` (its source in
`/opt/emc2/src`, `EMC2_SHA.txt` beside the library), `uv` and the lock, a UTF-8 locale.
From the root of this repository, from a source tarball of your clone of EMC2
(step 1.4; the repository is private, so the image takes a tarball rather than cloning),
placed in `nleflow/docker/emc2/` (git-ignored); `EMC2_SHA` is then only a label:

```sh
R CMD build --no-build-vignettes <your EMC2 clone> && mv EMC2_*.tar.gz nleflow/docker/emc2/
docker build -f nleflow/docker/Dockerfile --build-arg EMC2_TARBALL=EMC2_3.4.1.tar.gz \
  --build-arg EMC2_SHA=<commit of your clone> -t nleflow:latest .
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
2026-10-26, DMC2 main 71b3261b from its tarball, R 4.6.1): the image builds in 4 min 53 s
and is 5.4 GB (1.31 GB content).

- `nle-verify`: all checks pass, on the four shipped cards and on the scaled DDM and RDM
  A = 0 copies in `nleCards/`. The largest golden differences are within a factor of three
  of the Mac's (the scaled DDM card 1.2e-10 against 1.0e-10, the CRDM card 7.4e-14 against
  2.8e-14, all within the cards' tolerances); compiled against R path 1.9e-13, the oracle
  8.3e-14.
- `pytest`: 67 passed and 1 skipped (R `arrow` is not in the image) in 72 s.
- `emc2-tests`: 5755 passed, 5 skipped, and 6 failures and 1 error in three files, in 9 min;
  the Mac gives 5772 passed / 0 failed / 4 skipped. The three files hold values or
  behaviour recorded on the Mac, and none tests the neural likelihoods: `test-ddm-precision.R` (the
  default-precision likelihood differs from its stored vector in the last bit of a few
  trials, up to 1.8e-15, a libm difference), `test-mtlnr.R` (its snapshot records the Mac's
  C++ random-number format, `libc++`, where Linux has `libstdc++`) and `test-hardfit.R` (a
  seeded short fit whose early stop the test expects at the second check; on Linux the same
  seed gives other chains and the stop does not fire). Every neural-likelihood test
  (`test-nn-*.R`) passes.

One log-likelihood evaluation of 400 trials with sv, SZ and st0 free, one thread (the mean
of 200 evaluations):

| | `DDMnn()` (the scaled DDM card) | analytic DDM |
|---|---|---|
| reference BLAS (the image's default) | 0.39 ms | 0.99 ms |
| OpenBLAS, `OPENBLAS_NUM_THREADS=1` | 0.30 ms | 1.06 ms |
| Mac, Accelerate (native) | 0.20 ms | 1.19 ms |

On the Mac against the same commit, `pytest` gives 68 passed, `nle_verify.R` passes on the
shipped cards and on the scaled DDM, RDM A = 0, `ddm_svst0_e299` and `lba_e299` cards in
`nleCards/`, and `examples/ddm_svst0/run_smoke.sh`,
`examples/lan_in/run.sh` and `examples/onnx_use/run.sh` pass.

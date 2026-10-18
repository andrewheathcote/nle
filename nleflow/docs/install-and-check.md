# Install and check

What you need depends on the task (NLEFLOW_PLAN.md section 9.1): **using a card in
EMC2** needs R and EMC2 `dev-nle-m0`; **training, exporting, ONNX and converting** need
the Python environment of this directory; the **simulator of a worked example** needs
both. Every step below has one check. Commands run from `nleflow/` unless said.

Order: R and EMC2 first, then the verification command, then Python.

## 1. R >= 4.5 and EMC2

R older than 4.5 has no `tools::sha256sum`; EMC2 then pins its shipped artefacts by md5
and a JSON card still registers, but use 4.5 or later.

1. R >= 4.5 from CRAN. Check: `R --version | head -1` shows 4.5 or later.
2. A C++17 compiler. macOS: Xcode command-line tools and Homebrew `libomp`; Linux:
   `g++` >= 11. Check: `R CMD config CXX17` prints a compiler.
3. EMC2's R packages and `WienR` >= 0.3-17 (not packaged for Ubuntu; from CRAN).
   Check: `Rscript -e 'stopifnot(packageVersion("WienR") >= "0.3.17")'` exits 0.
4. EMC2 branch `dev-nle-m0` of `ampl-psych/EMC2`, into a dedicated library:

   ```sh
   git clone -b dev-nle-m0 https://github.com/ampl-psych/EMC2
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
(`-rs`): `NLEFLOW_EMC2_LIB` (a library holding EMC2 `dev-nle-m0`) turns on the cases that
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
R >= 4.5 from CRAN's Ubuntu repository, the R packages, `WienR`, EMC2 `dev-nle-m0` in
`/opt/emc2/lib` with `EMC2_SHA.txt` beside it, `uv` and the lock. From the root of the
nle repository:

```sh
docker build -f nleflow/docker/Dockerfile -t nleflow .
docker run --rm nleflow nle-verify     # EMC2's verification command; one PASS per card
docker run --rm nleflow pytest         # nleflow's tests with the R checks on
```

Check: both commands exit 0. The three commands were not run on the Mac this was written
on (no Docker there); they run in `.github/workflows/nleflow.yml` on every push to
`nleflow-plan`, where the image built and `nle-verify` printed `all checks passed` on R 4.6.1
and `pytest` gave 51 passed.

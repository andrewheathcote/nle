# nleflow + EMC2 (a source tarball, or a commit of ampl-psych/EMC2), the cloud recipe of
# NLEFLOW_PLAN.md section 9.2 as an image.
#
#   docker build -f nleflow/docker/Dockerfile -t nleflow .     # from the root of the nle repository
#   docker run --rm nleflow nle-verify                          # EMC2's verification command, R >= 4.5
#   docker run --rm nleflow pytest                              # nleflow's tests with the R checks on
#   docker run --rm nleflow emc2-tests                          # EMC2's own test suite (testthat)
#
# Build arguments:
#   EMC2_TARBALL  the file name of an `R CMD build` tarball placed in nleflow/docker/emc2/ (git-ignored);
#                 when set, EMC2 is installed from it and nothing is cloned. This is how a branch that is
#                 not on GitHub (dev-flow) gets into the image:
#                   R CMD build <EMC2 worktree> && mv EMC2_*.tar.gz nleflow/docker/emc2/
#                   docker build -f nleflow/docker/Dockerfile --build-arg EMC2_TARBALL=EMC2_3.4.1.9001.tar.gz \
#                     --build-arg EMC2_SHA=9f2801b2 -t nleflow:dev-flow .
#   EMC2_SHA      without EMC2_TARBALL, the commit of ampl-psych/EMC2 to clone and build (default: branch
#                 dev-nle-m0); with it, only a label recorded in /opt/emc2/lib/EMC2_SHA.txt beside the
#                 tarball's sha256.
# Ubuntu 24.04 ships R 4.3.3, which cannot check the shipped artefacts by sha256, so R comes from
# CRAN's own Ubuntu repository (R >= 4.5); the R packages, WienR included, come from CRAN.
#
# BLAS: R and EMC2 link libblas.so.3 / liblapack.so.3, which Ubuntu's alternatives point at the
# reference BLAS (selected here) or OpenBLAS (installed beside it). `r-blas openblas` and
# `r-blas reference` switch them in a running container (as root); EMC2 links by soname, so the switch
# needs no rebuild.
FROM ubuntu:24.04

ARG EMC2_SHA=252e155f
ARG EMC2_TARBALL=
ARG DEBIAN_FRONTEND=noninteractive

# --- R >= 4.5 from CRAN's Ubuntu repository; reference BLAS selected, OpenBLAS beside it ----------
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git gnupg build-essential \
 && install -d -m 0755 /etc/apt/keyrings \
 && curl -fsSL https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc -o /etc/apt/keyrings/cran.asc \
 && echo "deb [signed-by=/etc/apt/keyrings/cran.asc] https://cloud.r-project.org/bin/linux/ubuntu noble-cran40/" \
      > /etc/apt/sources.list.d/cran.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends r-base-core r-base-dev libopenblas-dev \
 && Rscript -e 'stopifnot(getRversion() >= "4.5.0")' \
 && rm -rf /var/lib/apt/lists/*
COPY nleflow/docker/r-blas /usr/local/bin/r-blas
RUN chmod 0755 /usr/local/bin/r-blas && r-blas reference

# --- EMC2's Imports and its test suite's Suggests from CRAN (Posit's Ubuntu binaries where they exist) --
# The Ubuntu archive's r-cran-* are built for R 4.3 and did not load under R 4.5 (testthat/pkgload failed
# in the first CI run), so none are used.
RUN Rscript -e 'options(Ncpus = max(1L, parallel::detectCores()), \
        repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/noble/latest"), \
        HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version$platform, R.version$arch, R.version$os))); \
      need <- c("abind","coda","magic","MASS","matrixcalc","msm","mvtnorm","Matrix","Rcpp","RcppArmadillo", \
                "Brobdingnag","corrplot","colorspace","psych","loo","lpSolve","jsonlite","WienR", \
                "testthat","withr","vdiffr","DiagrammeR"); \
      install.packages(need); \
      stopifnot(all(vapply(need, requireNamespace, TRUE, quietly = TRUE)), packageVersion("WienR") >= "0.3.17")' \
 && (Rscript -e 'options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/noble/latest"), HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version$platform, R.version$arch, R.version$os))); install.packages("arrow"); stopifnot(requireNamespace("arrow", quietly = TRUE))' \
     || echo "arrow not installed: the R simulator writes CSV")

# --- EMC2 into /opt/emc2/lib, its source beside it (the tests read its golden files) ---------------
# From the tarball when EMC2_TARBALL is set (nleflow/docker/emc2/ is copied either way; it may hold
# only its README), else from a clone at EMC2_SHA.
COPY nleflow/docker/emc2 /opt/emc2/in
RUN set -e; mkdir -p /opt/emc2/lib; \
    if [ -n "${EMC2_TARBALL}" ]; then \
      tar -xzf "/opt/emc2/in/${EMC2_TARBALL}" -C /opt/emc2 && mv /opt/emc2/EMC2 /opt/emc2/src; \
      printf '%s tarball %s sha256 %s\n' "${EMC2_SHA}" "${EMC2_TARBALL}" \
        "$(sha256sum "/opt/emc2/in/${EMC2_TARBALL}" | cut -d' ' -f1)" > /opt/emc2/EMC2_SHA.txt; \
    else \
      git clone --quiet https://github.com/ampl-psych/EMC2 /opt/emc2/src; \
      git -C /opt/emc2/src checkout --quiet "${EMC2_SHA}"; \
      git -C /opt/emc2/src rev-parse HEAD > /opt/emc2/EMC2_SHA.txt; \
    fi; \
    MAKEFLAGS="-j$(nproc)" R_PROFILE_USER=/dev/null R CMD INSTALL --preclean --library=/opt/emc2/lib /opt/emc2/src; \
    mv /opt/emc2/EMC2_SHA.txt /opt/emc2/lib/EMC2_SHA.txt; rm -rf /opt/emc2/in; \
    R_PROFILE_USER=/dev/null Rscript -e 'library(EMC2, lib.loc = "/opt/emc2/lib"); cat("EMC2", as.character(packageVersion("EMC2")), "\n")'

# --- Python: uv and the lock ----------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_LINK_MODE=copy

# the package and the reference exports its tests read (REPO/handover/nle-methods/port/export)
COPY nleflow /opt/nle/nleflow
COPY handover/nle-methods/port/export /opt/nle/handover/nle-methods/port/export
WORKDIR /opt/nle/nleflow
RUN rm -rf /opt/nle/nleflow/docker/emc2 && uv sync --frozen --group dev

ENV NLEFLOW_EMC2_LIB=/opt/emc2/lib \
    NLEFLOW_RLIB=/opt/emc2/lib \
    NLEFLOW_EMC2_SRC=/opt/emc2/src \
    R_LIBS=/opt/emc2/lib \
    R_PROFILE_USER=/dev/null \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8
# a UTF-8 locale: without it R prints non-ASCII output escaped (<U+0001D441>) and EMC2's snapshot
# test of print(prior) (test-S3.R) fails

# the commands the image is checked by
RUN install -m 0755 /opt/nle/nleflow/docker/nle-verify /opt/nle/nleflow/docker/pytest \
      /opt/nle/nleflow/docker/emc2-tests /usr/local/bin/

CMD ["nle-verify"]

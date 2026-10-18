# nleflow + EMC2 dev-nle-m0, the cloud recipe of NLEFLOW_PLAN.md section 9.2 as an image.
#
#   docker build -f nleflow/docker/Dockerfile -t nleflow .     # from the root of the nle repository
#   docker run --rm nleflow nle-verify                          # EMC2's verification command, R >= 4.5
#   docker run --rm nleflow pytest                              # nleflow's tests with the R checks on
#
# Build arguments: EMC2_SHA is the commit of ampl-psych/EMC2 (branch dev-nle-m0) to build.
# Ubuntu 24.04 ships R 4.3.3, which cannot check the shipped artefacts by sha256, so R comes from
# CRAN's own Ubuntu repository (R >= 4.5); the R packages, WienR included, come from CRAN.
FROM ubuntu:24.04

ARG EMC2_SHA=252e155f
ARG DEBIAN_FRONTEND=noninteractive

# --- R >= 4.5 from CRAN's Ubuntu repository ------------------------------------------------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git gnupg build-essential \
 && install -d -m 0755 /etc/apt/keyrings \
 && curl -fsSL https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc -o /etc/apt/keyrings/cran.asc \
 && echo "deb [signed-by=/etc/apt/keyrings/cran.asc] https://cloud.r-project.org/bin/linux/ubuntu noble-cran40/" \
      > /etc/apt/sources.list.d/cran.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends r-base-core r-base-dev \
 && Rscript -e 'stopifnot(getRversion() >= "4.5.0")' \
 && rm -rf /var/lib/apt/lists/*

# --- EMC2's Imports from CRAN (Posit's Ubuntu binaries where they exist, source otherwise) ---------
# The Ubuntu archive's r-cran-* are built for R 4.3 and did not load under R 4.5 (testthat/pkgload failed
# in the first CI run), so none are used. testthat is not needed: the image runs nle_verify.R and pytest.
RUN Rscript -e 'options(Ncpus = max(1L, parallel::detectCores()), \
        repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/noble/latest"), \
        HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version$platform, R.version$arch, R.version$os))); \
      need <- c("abind","coda","magic","MASS","matrixcalc","msm","mvtnorm","Matrix","Rcpp","RcppArmadillo", \
                "Brobdingnag","corrplot","colorspace","psych","loo","lpSolve","jsonlite","WienR"); \
      install.packages(need); \
      stopifnot(all(vapply(need, requireNamespace, TRUE, quietly = TRUE)), packageVersion("WienR") >= "0.3.17")' \
 && (Rscript -e 'options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/noble/latest"), HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version$platform, R.version$arch, R.version$os))); install.packages("arrow"); stopifnot(requireNamespace("arrow", quietly = TRUE))' \
     || echo "arrow not installed: the R simulator writes CSV")

# --- EMC2 dev-nle-m0 into /opt/emc2/lib, its source beside it (the tests read its golden files) ---
RUN git clone --quiet https://github.com/ampl-psych/EMC2 /opt/emc2/src \
 && git -C /opt/emc2/src checkout --quiet "${EMC2_SHA}" \
 && mkdir -p /opt/emc2/lib \
 && MAKEFLAGS="-j$(nproc)" R_PROFILE_USER=/dev/null R CMD INSTALL --preclean --library=/opt/emc2/lib /opt/emc2/src \
 && git -C /opt/emc2/src rev-parse HEAD > /opt/emc2/lib/EMC2_SHA.txt \
 && R_PROFILE_USER=/dev/null Rscript -e 'library(EMC2, lib.loc = "/opt/emc2/lib"); cat("EMC2", as.character(packageVersion("EMC2")), "\n")'

# --- Python: uv and the lock ----------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_LINK_MODE=copy

# the package and the reference exports its tests read (REPO/handover/nle-methods/port/export)
COPY nleflow /opt/nle/nleflow
COPY handover/nle-methods/port/export /opt/nle/handover/nle-methods/port/export
WORKDIR /opt/nle/nleflow
RUN uv sync --frozen --group dev

ENV NLEFLOW_EMC2_LIB=/opt/emc2/lib \
    NLEFLOW_RLIB=/opt/emc2/lib \
    NLEFLOW_EMC2_SRC=/opt/emc2/src \
    R_LIBS=/opt/emc2/lib \
    R_PROFILE_USER=/dev/null

# the two commands the image is checked by
RUN install -m 0755 /opt/nle/nleflow/docker/nle-verify /opt/nle/nleflow/docker/pytest /usr/local/bin/

CMD ["nle-verify"]

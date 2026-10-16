#!/bin/bash
# Machine setup (run as root): R packages, then EMC2 dev-nle 142bd497 (final sampler aa66e11d; the build the
# laptop's ~/R/nle-lib and the eight-case analytic controls used) from the bundled source, built clean.
set -euo pipefail
cd "$(dirname "$0")"
Rscript -e 'need <- c("abind","coda","magic","MASS","matrixcalc","msm","mvtnorm","Matrix","Rcpp","RcppArmadillo","Brobdingnag",
  "corrplot","colorspace","psych","loo","lpSolve","WienR","jsonlite"); miss <- need[!vapply(need, requireNamespace, TRUE, quietly = TRUE)];
  cat("missing:", miss, "\n"); writeLines(miss, "/tmp/miss.txt")'
if [ -s /tmp/miss.txt ]; then apt-get install -y $(sed 's/.*/r-cran-\L&/' /tmp/miss.txt) >&2; fi
Rscript -e 'stopifnot(packageVersion("WienR") >= "0.3.17")'
LIB=/opt/emc2/lib-142bd497
if [ ! -f $LIB/EMC2_SHA.txt ]; then
  rm -rf /opt/emc2/src142 $LIB.tmp; mkdir -p /opt/emc2/src142 $LIB.tmp
  tar -xzf emc2_142bd497.tar.gz -C /opt/emc2/src142
  MAKEFLAGS=-j$(nproc) R CMD INSTALL --preclean --no-test-load --library=$LIB.tmp /opt/emc2/src142/EMC2 >&2
  echo 142bd497 > $LIB.tmp/EMC2_SHA.txt
  Rscript -e "library(EMC2, lib.loc='$LIB.tmp'); cat('EMC2', as.character(packageVersion('EMC2')), '\n')" >&2
  mv $LIB.tmp $LIB
fi
echo $LIB

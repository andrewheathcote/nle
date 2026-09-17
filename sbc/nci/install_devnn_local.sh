#!/bin/bash
# Install dev-nn EMC2 into a dedicated library on an ordinary Linux box
# (no environment modules, no scheduler). Gadi equivalent: install_devnn.sh
#
#   bash install_devnn_local.sh [git-url] [branch]
# defaults: ampl-psych/EMC2 dev-nn
#
# Override locations with EMC2NN_LIB / EMC2NN_SRC.
set -e
SRC=${1:-https://github.com/ampl-psych/EMC2.git}
BRANCH=${2:-dev-nn}
LIB=${EMC2NN_LIB:-$HOME/emc2nn/Rlib-devnn}
WORK=${EMC2NN_SRC:-$HOME/emc2nn/src}
export EMC2NN_LIB="$LIB"

command -v R >/dev/null || { echo "ABORT: R not on PATH"; exit 1; }
command -v git >/dev/null || { echo "ABORT: git not on PATH"; exit 1; }
mkdir -p "$LIB" "$WORK"

if [ -d "$SRC" ]; then
  PKG="$SRC"; echo "installing from local directory: $PKG"
else
  PKG="$WORK/EMC2-$BRANCH"
  if [ -d "$PKG/.git" ]; then
    git -C "$PKG" fetch --depth 1 origin "$BRANCH"
    git -C "$PKG" checkout -B "$BRANCH" FETCH_HEAD
  else
    rm -rf "$PKG"; git clone --depth 1 --branch "$BRANCH" "$SRC" "$PKG"
  fi
  echo "commit: $(git -C "$PKG" rev-parse --short HEAD)"
fi

ls "$PKG"/src/flow_ddm.cpp >/dev/null 2>&1 || {
  echo "ABORT: $PKG/src/flow_ddm.cpp not found - not the dev-nn tree."; exit 1; }

# dependencies come from the default library; only EMC2 goes in $LIB
R CMD INSTALL --library="$LIB" --preclean "$PKG"

Rscript -e '
  .libPaths(c(Sys.getenv("EMC2NN_LIB"), .libPaths()))
  library(EMC2)
  p <- find.package("EMC2")
  cat("installed:", p, as.character(packageVersion("EMC2")), "\n")
  if (!startsWith(normalizePath(p), normalizePath(Sys.getenv("EMC2NN_LIB"))))
    stop("EMC2 resolved to ", p, " not ", Sys.getenv("EMC2NN_LIB"))
  stopifnot("ddm_ens_eval_trials_cpp" %in% ls(getNamespace("EMC2")))
  stopifnot(exists("DDMnn", envir = asNamespace("EMC2")))
  cat("dev-nn install OK\n")'
echo
echo "Next:  bash run_sbc_c4_local.sh single"

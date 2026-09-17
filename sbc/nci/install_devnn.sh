#!/bin/bash
# Install the dev-nn EMC2 into a DEDICATED library on Gadi.
#
# Why a separate library: the dev-nn build reports version 3.4.1, exactly like
# the plain EMC2, so nothing stops one silently replacing the other in the
# default user library — which is what happened (the Sep 2 install wiped the
# dev-nn build that Phase 3 had run against, leaving results we could no longer
# reproduce). Keeping it in its own tree means the SBC jobs see dev-nn and
# everything else on the account keeps the plain EMC2.
#
# Usage:
#   bash install_devnn.sh https://github.com/<owner>/EMC2.git [branch]
#   bash install_devnn.sh /path/to/existing/EMC2/checkout
# Default branch: dev-nn
set -e
SRC=${1:?usage: install_devnn.sh <git-url-or-local-dir> [branch]}
BRANCH=${2:-dev-nn}
LIB=${EMC2NN_LIB:-/scratch/tg53/ajh575/emc2nn/Rlib-devnn}
export EMC2NN_LIB="$LIB"
WORK=${EMC2NN_SRC:-/scratch/tg53/ajh575/emc2nn/src}

module purge
module load gcc/12.2.0
module load intel-compiler/2021.10.0
module load R/4.5.0

mkdir -p "$LIB" "$WORK"

# R_LIBS_USER takes precedence over R_LIBS, so putting the dedicated library in
# R_LIBS is not enough - the plain EMC2 in the home library would shadow it.
# Put it FIRST in R_LIBS_USER and keep the home library after it, so package
# dependencies (Rcpp, etc.) still resolve from there.
HOMELIB=${R_LIBS_USER:-$HOME/R/x86_64-pc-linux-gnu-library/4.5}
export R_LIBS_USER="$LIB:$HOMELIB"


if [ -d "$SRC" ]; then
  PKG="$SRC"
  echo "installing from local directory: $PKG"
else
  PKG="$WORK/EMC2-$BRANCH"
  if [ -d "$PKG/.git" ]; then
    echo "updating existing clone: $PKG"
    git -C "$PKG" fetch --depth 1 origin "$BRANCH"
    git -C "$PKG" checkout -B "$BRANCH" FETCH_HEAD
  else
    rm -rf "$PKG"
    echo "cloning $SRC ($BRANCH) -> $PKG"
    git clone --depth 1 --branch "$BRANCH" "$SRC" "$PKG"
  fi
  echo "commit: $(git -C "$PKG" rev-parse --short HEAD)"
fi

# the flow sources are what make this build dev-nn; fail early if absent
ls "$PKG"/src/flow_ddm.cpp >/dev/null 2>&1 || {
  echo "ABORT: $PKG/src/flow_ddm.cpp not found — this is not the dev-nn tree."
  echo "       (branch '$BRANCH' may be wrong, or the flow code lives elsewhere)"; exit 1; }

# dependencies stay in the default library; only EMC2 goes in $LIB
R CMD INSTALL --library="$LIB" --preclean "$PKG"

Rscript -e '
  .libPaths(c(Sys.getenv("EMC2NN_LIB"), .libPaths()))
  library(EMC2)
  p <- find.package("EMC2")
  cat("installed:", p, as.character(packageVersion("EMC2")), "\n")
  if (!startsWith(normalizePath(p), normalizePath(Sys.getenv("EMC2NN_LIB"))))
    stop("EMC2 resolved to ", p, " not the dedicated library ",
         Sys.getenv("EMC2NN_LIB"), " - library precedence is wrong")
  stopifnot("ddm_ens_eval_trials_cpp" %in% ls(getNamespace("EMC2")))
  stopifnot(exists("DDMnn", envir = asNamespace("EMC2")))
  f <- system.file("extdata", "flownn", "ddm_ens5.rds", package = "EMC2")
  cat("bundled weights:", if (nzchar(f)) f else "NONE", "\n")
  if (nzchar(f)) { b <- readRDS(f); cat("  members:", paste(b$member_tags, collapse=", "), "\n") }
  cat("dev-nn install OK\n")'
echo
echo "Now:  export EMC2NN_LIB=$LIB"
echo "      bash launch_sbc_c4.sh single"

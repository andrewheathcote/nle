#!/bin/bash
# Phase 7: SBC of the corrected-classifier (c4) DDM likelihood.
#
# WHY: every G1 number we hold is contaminated. The classifier was fitted to an
# analytic P(R) target computed without the s-scaling (wrong by 0.055 mean /
# 0.83 max), and the joint score bias was partly flow/classifier cancellation
# rather than accuracy. c4 fixes the target (value error 0.0092 -> 0.000262) and
# removes the cancellation. See sbc/SBC_PLAN.md, Phase 7.
#
# CRITICAL: DDMnn reads <EMC2>/extdata/flownn/ddm_ens5.rds and that name is
# HARDCODED. The single-flow and 5-member configurations therefore CANNOT run
# at the same time against one EMC2 install - they would read each other's
# weights and the results would be silently wrong. Run one campaign, wait for
# it to finish, then swap and run the other.
#
# Usage:   bash launch_sbc_c4.sh single    # cap256w_c4 alone  (recommended first)
#          bash launch_sbc_c4.sh ens       # 5-member c4 ensemble
#
# Upload first (from the laptop repo root):
#   scp sbc/ddm_ens5_single_c4.rds sbc/ddm_ens5_c4.rds \
#       ajh575@gadi.nci.org.au:/scratch/tg53/ajh575/emc2nn/nci/
#   scp sbc/nci/SBCc4.R sbc/nci/SBCc4_N*.pbs sbc/nci/launch_sbc_c4.sh \
#       ajh575@gadi.nci.org.au:/scratch/tg53/ajh575/emc2nn/nci/
set -e
WHICH=${1:?usage: launch_sbc_c4.sh single|ens}
case "$WHICH" in
  single) SRC=ddm_ens5_single_c4.rds; OUT=results_c4single ;;
  ens)    SRC=ddm_ens5_c4.rds;        OUT=results_c4ens ;;
  *) echo "unknown: $WHICH (expected single|ens)"; exit 1 ;;
esac

# purge first: a login shell often already has an R module loaded, and
# "module load R/4.5.0" then fails with a conflict (which set -e turns into a
# silent-looking abort). This mirrors what the PBS scripts do.
# dev-nn EMC2 lives here, NOT in the default user library (see install_devnn.sh)
export EMC2NN_LIB=${EMC2NN_LIB:-/scratch/tg53/ajh575/emc2nn/Rlib-devnn}
[ -d "$EMC2NN_LIB" ] || { echo "ABORT: no dev-nn library at $EMC2NN_LIB"
  echo "  run:  bash install_devnn.sh <git-url-or-dir>"; exit 1; }
LIB="$EMC2NN_LIB"


module purge
module load gcc/12.2.0
module load intel-compiler/2021.10.0
module load R/4.5.0
command -v Rscript >/dev/null || { echo "Rscript not on PATH after module load"; exit 1; }

# R_LIBS_USER takes precedence over R_LIBS, so putting the dedicated library in
# R_LIBS is not enough - the plain EMC2 in the home library would shadow it.
# Put it FIRST in R_LIBS_USER and keep the home library after it, so package
# dependencies (Rcpp, etc.) still resolve from there.
HOMELIB=${R_LIBS_USER:-$HOME/R/x86_64-pc-linux-gnu-library/4.5}
export R_LIBS_USER="$LIB:$HOMELIB"
# set AFTER module load: the R module sets R_LIBS_USER itself, so exporting it
# beforehand is silently discarded (the dedicated library then vanishes from
# .libPaths entirely and the plain EMC2 in the home library wins).
echo "R_LIBS_USER=$R_LIBS_USER"


# Preflight: the EMC2 on the PATH must be the dev-nn build. A plain CRAN/master
# EMC2 3.4.1 has the same version number but no DDMnn, no ensemble evaluator and
# no bundled weights, so the swap target does not exist and DDMnn cannot run.
Rscript -e '.libPaths(c(Sys.getenv("EMC2NN_LIB"), .libPaths())); q(status = if (isTRUE(requireNamespace("EMC2", quietly=TRUE)) &&
    "ddm_ens_eval_trials_cpp" %in% ls(getNamespace("EMC2"))) 0 else 1)' || {
  echo "ABORT: the EMC2 on this R is NOT the dev-nn build."
  Rscript -e '.libPaths(c(Sys.getenv("EMC2NN_LIB"), .libPaths())); cat("  library paths:\n"); cat(paste("   ", .libPaths()), sep="\n");
    if (requireNamespace("EMC2", quietly=TRUE)) cat("  EMC2 found at:", find.package("EMC2"),
      "version", as.character(packageVersion("EMC2")), "\n")'
  echo "  It has no ddm_ens_eval_trials_cpp, so DDMnn does not exist and there"
  echo "  is nothing to swap. Install dev-nn into its own library first:"
  echo "    bash install_devnn.sh <git-url-or-local-dir>"
  echo "  then re-run this script."
  exit 1
}

DST=$(Rscript -e '.libPaths(c(Sys.getenv("EMC2NN_LIB"), .libPaths())); cat(system.file("extdata","flownn","ddm_ens5.rds",package="EMC2",mustWork=TRUE))')
[ -n "$DST" ] || { echo "could not locate EMC2 extdata"; exit 1; }

# Refuse only if OUR OWN SBCc4 jobs are queued/running: swapping the bundle
# under those would change the weights mid-run. Other jobs on the account are
# unaffected — they resolve EMC2 from the default library, and only SBCc4.R
# prepends the dedicated dev-nn one. That isolation is the whole point of
# installing into a separate library.
if qstat -u "$USER" 2>/dev/null | grep -q "SBCc4"; then
  echo "REFUSING: SBCc4 jobs are already queued or running. Swapping the bundle"
  echo "now would change the weights under them. Wait for them to finish:"
  qstat -u "$USER" | grep -E "Job|SBCc4"
  exit 1
fi

[ -f "$DST.orig" ] || { cp "$DST" "$DST.orig"; echo "backed up -> $DST.orig"; }
cp "$SRC" "$DST"
echo "installed $SRC -> $DST"
Rscript -e '.libPaths(c(Sys.getenv("EMC2NN_LIB"), .libPaths())); b <- readRDS(system.file("extdata","flownn","ddm_ens5.rds",package="EMC2"));
            cat("verify:", length(if (is.null(b$members)) list(b) else b$members),
                "member(s):", paste(b$member_tags, collapse=", "), "\n")'

for N in 100 200 400; do
  id=$(qsub -v OUT_DIR="$OUT",EMC2NN_LIB="$EMC2NN_LIB" "SBCc4_N$N.pbs")
  echo "submitted SBCc4_N$N.pbs (out_dir=$OUT) -> $id"
done
echo
echo "Results -> ./$OUT/DDMnn_N<trials>_R500.rds ; logs SBCc4_${OUT}_N*.log"
echo "Each job re-reads the bundle at startup and PRINTS its member tags, so"
echo "check the log header matches '$WHICH' before trusting any result."

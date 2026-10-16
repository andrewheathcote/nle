# EMC2 with the neural-likelihood evaluator (dev-nle) and the compiled DDMnle path

- `emc2_142bd497.tar.gz`: `git archive` of EMC2 branch `dev-nle` at 142bd497 (the final sampler; not on GitHub). Extract, then
  `R CMD INSTALL` into a **dedicated library** (it reports version 3.4.1, the same as the release, and the two overwrite each other silently).
- `emc2_dev-nle-scaled.patch`: the compiled DDMnle (scaled DDM) path, apply on top of the extracted tree with `patch -p1 < emc2_dev-nle-scaled.patch`
  (src/model_NN.h, src/flow_ddm.cpp, src/nle_native.h). It is uncommitted work and has not been pushed to the EMC2 repository.

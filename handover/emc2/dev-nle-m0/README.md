# EMC2 branch `dev-nle-m0`, as patches

The push of `dev-nle-m0` to `ampl-psych/EMC2` from the cloud instance was refused (HTTP 403: the Claude GitHub App
is not installed for the ampl-psych organisation), so the branch travels here as `git format-patch` output until AH
pushes it from the laptop. The three commits apply onto `origin/dev-nle` at 154f21ad (2026-10-02):

    git checkout -b dev-nle-m0 154f21ad
    git am handover/emc2/dev-nle-m0/000*.patch

1. `0001`: the compiled scaled-DDM path (the patch that used to live as `emc2_dev-nle-scaled.patch` beside this folder).
2. `0002`: the tree of the local 142bd497 (dev-sampler aa66e11d merged into dev-nle) on top of 154f21ad, so the
   branch carries the final sampler; the NN layer is identical in the two trees. On a laptop that has the local
   142bd497, this commit should be a no-op against it.
3. `0003`: M0 (NLEFLOW_PLAN.md section 11.1 as amended by section 13): flow_scaled cards, card format v1,
   nn_golden_check(), nn_sz_convert(), nle_verify.R, the advisory training-region rule, the rt_resolution warning,
   the md5 MANIFEST lines, the DDMnn t0 note, tests.

Once the branch is on GitHub, delete this folder and `emc2_dev-nle-scaled.patch`.

# Submission order, stage M6a cells

Written by stage M6a (part C). Costs from `handover/MALTE_STAGES.md`, "Stages M4 - M6,
re-scoped", cost table, and stage G's Gadi/laptop ratio (filled in once stage G has run;
until then the "laptop core-s at 2x" figure stands as the estimate the re-scoped section
registered, per its own note that it is "to be replaced by the check job's timing lines").

Submit in this order. Each row is one `qsub` of the bundle's `job.pbs` (or, for the chunked
Volterra controls, the 48-cpu job that runs all 16 chunks side by side and joins them).

| order | cell | replicates | cores | walltime requested | core-hours (laptop x2 estimate) |
|---|---|---|---|---|---|
| 1 | `rdm_a0_N400/O` | 250 | 48 | see below | ~0.6 |
| 2 | `rdm_a0_N400/HF` | 250 | 48 | see below | ~0.6 |
| 3 | `crdm_net_N400` | 500 | 48 | see below | ~9 |
| 4 | `crdm_ctl_dt2e-3` (16 chunks) | 250 | 48 | see below | ~54 |
| 5 | `crdm_ctl_dt5e-4` (16 chunks) | 50 | 48 | see below | ~139 |
| 6 | `rdm_o_a0_N100` | 500 net / 250 ctl | 48 | see below | ~1.2 |
| 7 | `rdm_o_std_N100` | 500 net / 250 ctl | 48 | see below | ~1.2 |
| 8 | `rdm_o_std_N400` | 500 net / 250 ctl | 48 | see below | ~1.2 |

Rationale for the order: `rdm_a0_N400` resolves M4's own "NOT SEPARATED" verdict and is the
cheapest network-only cell (no Volterra control), so it goes first. `crdm_net_N400` is the next
cheapest network cell. The two Volterra-only controls dominate the walltime budget (dt 5e-4
alone is close to the registered 48 h cap even chunked) and are the reason M6's Gadi bundles
exist at all, so they go next, in increasing cost. The M5 validation cells (`rdm_o_*`) close
the run: M4 already reads as "NOT SEPARATED: SBC cell needed" for O vs HF, so M5's own new-flow
validation is lower priority than resolving that verdict and the CRDM screens.

Core-hours: replicates x (laptop core-s per replicate) / 3600, doubled for the Gadi ratio
(stage G's measured ratio replaces "x2" once available; until then the re-scoped section's own
placeholder stands). `rdm_a0_N400` and `rdm_o_*` have no measured per-replicate cost in the
re-scoped section (only the CRDM cells were benchmarked, `sbc/malte_m6/cost_per_replicate.R`);
their core-hours above are a rough estimate from an RDM race's much cheaper likelihood (no
Volterra solve at all) and are not registered numbers.

Walltime per job: `run_sbc` fits replicates one after another per chunk of `cores_per_chain`;
at 48 cpus (`cores_per_chain = 48, cores_for_chains = 1`) a network cell's 250-500 replicates
complete inside the 40-minute check-job-style walltime many times over even doubled for safety
(the CRDM screens on the laptop, part A, measured whole-draw costs an order of magnitude larger
than one SBC replicate's network fit). The two chunked Volterra controls are the walltime-
dominant cells: each of the 16 chunks runs its own share of the replicates at 3 cores, so
walltime is (replicates / 16) x (per-replicate seconds at 3 cores) x 2 (safety) -- see the
re-scoped section's own table (dt 2e-3: about 1.2 h at 2x; dt 5e-4: about 3 h at 2x); both
job.pbs files request more, capped at 48 h, never exceeded.

**Measured, not just the cost table (this stage's smoke runs, one replicate, `cores_per_chain
= 3, cores_for_chains = 1` -- a single replicate never parallelises across `cores_per_chain`,
so this is close to single-threaded wall time per chain, 3 chains sequential):** dt 2e-3,
1 replicate, 400 trials: ~15 min wall. dt 5e-4, 1 replicate, 400 trials: ~76 min wall. On Gadi
with 15-16 replicates per chunk at `cores_per_chain = 3`, replicates DO parallelise (3 at a
time), so a chunk's wall time is about ceil(chunk_reps / 3) x the one-replicate figure above:
dt 2e-3 chunk (15-16 reps) ~ 6 x 15 min = 90 min; dt 5e-4 chunk (3-4 reps) ~ 2 x 76 min =
152 min. Both comfortably inside the 4 h / 8 h requested above; no change made to the PBS
walltimes since they were already conservative.

**P (posterior shift) does not gate these cells.** Stage M6a part A's screen also computes a
posterior-shift diagnostic (`sbc/malte_m6.R` mode `P`) that was found infeasible on the laptop
at the registered Volterra resolution (a 2-dataset smoke ran 4.5 h without finishing one
dataset; see `sbc/malte_m6/screen_final.out`). That diagnostic is unrelated code from these
cells' own SBC machinery (`nn_sbc_cell` / plain `run_sbc`, ordinary MCMC, already smoke-run
successfully for every cell here): it does not block or slow down anything in this file.

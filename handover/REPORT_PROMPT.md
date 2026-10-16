Run this in auto mode; if you are not in auto mode, stop at once and say so.

You are continuing the NLE project in /Users/andrew/Downloads/julian/nle with Andrew Heathcote (AH). The task is to
draft a report for Russ and Julian on the DDM neural-likelihood results. Write it as a new file,
`handover/REPORT_RUSS_JULIAN.md`. AH sends it himself; never send anything to anyone.

## Read first
- `handover/SCALED_DDM_PLAN.md`, from the "2026-10-15" entries to the end: the scaled network, the AWS SBC, the adoption
  of DDMnle, the compiled path and the LAN SBC.
- `handover/ZERO_BOX_RETRAIN_PLAN.md`, the "Eight-case review" section (the old per-case networks, for comparison).
- `handover/nle-methods/docs/METHODS.md`: the earlier methods write-up the report should follow in style. Its PDF
  `~/Downloads/julian/METHODS.pdf` is older (27 Sep).
- Your memory index; especially ddmnle-name, ddm-eight-variability-cases and aws-nle-runs.

## Names and facts to keep straight
- "DDMnle" = the adopted single DDM network, the scaled checkpoint sc_s1106_e245 (card
  `handover/nle-methods/port/export/ddm_sc_s1106_e245_weights.json`).
- The six cases: 2 st0+SZ (regime sv0), 3 st0+sv (SZ0), 4 sv+SZ (st00), 5 st0 (st0only), 7 SZ (SZonly), 8 all (full).
  Cases 1 and 6 were dropped because the analytic DDM is faster there.
- The DDMnle SBC is at N = 400 with 500 replicates per case (AWS run1). Table: `Rscript sbc/scaled_8case_table.R`.
  Summaries: `sbc/aws/scaled_sbc/run1/`. PDF: `sbc/aws/scaled_sbc/run1/sc_s1106_e245_SBC_6cases.pdf`.
- The compiled DDMnle is on EMC2-nle local branch `dev-nle-scaled`: uncommitted, not pushed, built into
  `~/R/nle-scaled-lib`. Do not commit or push it unless AH asks.

## Step 1: the speed table (do this first)
- The benchmark is `sbc/ddmnle_bench/bench.R` and writes `sbc/ddmnle_bench/bench.csv`.
  - If the CSV is missing and no `bench.R` R process is running (`ps -Ao pid,args | grep "[b]ench.R"`), rerun it in the
    background: `LIB=~/R/nle-scaled-lib R_PROFILE_USER=/dev/null Rscript sbc/ddmnle_bench/bench.R`.
  - It takes about 20–40 min.
- Report one table:
  - one row per case;
  - for N = 100 and N = 400: WienR at EMC2's default precision 5e-3, WienR at 1e-8, DDMnle compiled, and the two
    speed-ups.
  - Report the compiled DDMnle only; do not report the R-path DDMnle.
- Notes under the table:
  - Timings are single thread, ms per likelihood evaluation.
  - 1e-8 is the precision the project uses as ground truth.
  - EMC2's compiled DDM fixes 5e-3, so the 1e-8 column runs on EMC2's R path. The `wienr_5e3_Rpath` column shows how
    much the R path itself adds; say so.
  - The laptop was also running the 8-core LAN SBC during the timings.

## Step 2: methodology and results, along the lines of METHODS.md
Cover:
- **The scaled DDM network.** The network models the unit-scale time given κ = v·a/s², η = sv·a/s², Z and SZ. s and a
  enter by rescaling, t0 by a shift, and st0 by the exact CDF difference [G(u1) − G(u0)]/st0.
- **Selection.** Training was 6 seeds × 300 epochs. Every checkpoint was screened by score bias over 7 regimes, and
  the top 3 went to SBC.
- **SBC.** The six cases, compared with the old per-case networks and the analytic controls (that table is in
  SCALED_DDM_PLAN).
- **Adoption of DDMnle.** It was in the envelope in all cases bar one flag (a, case 5, 1 of 35 tests), and needs no
  t0 correction.
- **Speed** (Step 1).

Keep the project's main point from METHODS.md: calibration (SBC, score bias) is the test, not held-out likelihood.
Show AH the draft outline before writing long prose.

## Step 3: the LAN SBC (monitor, then integrate)
- **What it is.** HSSM's `ddm_uniform_st` LAN, case 5, SBC in the LAN's own training box, with the analytic DDM in
  the same parameterisation as control. Model and prior: `sbc/lan_case5/lan_model.R`. Cell script:
  `sbc/lan_case5/lan_cell.R`.
- **How it is running.** On the laptop under nohup, 8 cores, driven by `sbc/lan_case5/run2.sh`.
  - The control runs first (about 3 h from 10:08).
  - Then the LAN restarts at 4×2 cores, resuming from its saved replicates.
  - Expected end about 16:30 on 2026-10-16.
- **Monitoring.**
  - `tail sbc/lan_case5/run2.out`.
  - Count `rep_*.rds` in `sbc/lan_case5/results/results/sbc_{nn,control}_temp/`.
  - Logs are `sbc/lan_case5/{nn,control,summary}.log`.
  - It is done when run2.out says "lan case5 done". The summary is then in summary.log and
    `sbc/lan_case5/results/`.
- **Early signs.**
  - A 3-replicate smoke test showed v std bias +4.3 with 0/3 coverage. That is three replicates only; say so.
  - The LAN's replicates ran about 4x slower than the control's despite a 3x faster likelihood.
- **When done.**
  - Make a PDF like the DDMnle one: a cover table, then the ECDF pages of the LAN and the control. Reuse
    `sbc/aws/scaled_sbc/make_pdf_pages.R` (cover/title pages in R) and `pdfunite`.
  - Give AH the path and open it with `open`.
  - Add a LAN section to the report. Caveat: the LAN cell and DDMnle's case 5 use different boxes and priors, so
    compare each with its own analytic control, not with each other.
- **Log** the result in SCALED_DDM_PLAN.md and in memory.

## Rules
- Never launch anything on tux26. Ask before using Gadi or AWS (AWS rules: `~/.claude/plans/aws/nle_sbc_handover.md`).
- Laptop runs are limited to 8 cores, and the LAN SBC is using all 8, so start nothing heavy until it finishes. The
  benchmark is single-threaded and is fine.
- Do not commit unless AH asks. Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- At the end of your work, hand over ONE next stage: a model (Sonnet 4.6 or above) and a prompt, in auto mode.

# A published LAN into EMC2

HSSM's likelihood approximation network for the DDM with uniform non-decision-time
variability (`ddm_uniform_st`, [huggingface.co/Eitanm/ddm-st-lans](https://huggingface.co/Eitanm/ddm-st-lans),
MIT licence) brought into EMC2: downloaded at a pinned revision, converted to an
`mlp_joint` card, registered, checked against its own golden rows, and compared with
EMC2's analytic DDM. Nothing is trained here; [docs/lan-in.md](../../docs/lan-in.md)
describes the converter.

```sh
cd nleflow/examples/lan_in
export NLEFLOW_RLIB=<library holding EMC2 built from DMC2 main and its dependencies>   # default ~/R/nle-ondev-lib:~/R/nle-m4-lib
./run.sh                      # about 5 s; LAN_ONNX=<file> to use a file you have
```

The LAN's parameters are HSSM's, not EMC2's: `v`, `a` (half the boundary
separation), `z` (relative start), `t` (the centre of the uniform non-decision time)
and `st` (its half-width), with response −1 = lower and +1 = upper. On EMC2's DDM
that is `a = 2 a_lan`, `Z = z`, `t0 = t − st`, `st0 = 2 st`, `sv = SZ = 0`, `s = 1`,
response 2 = upper (the preset's `mapping`). The card keeps the LAN's own
parameters; a LAN design samples `v, a, z, t, st` on the natural scale.

## What `run.sh` does, and what it prints

| step | expected |
|---|---|
| download revision `40e0327c22a4`, check its sha256 | `out/ddm_uniform_st.onnx: sha256 4330d723... (revision 40e0327c22a4)` |
| `nleflow convert --list-presets` | the preset and its two versions: `40e0327c22a4` (5 × 100, st ≤ 0.35, current) and `8b1be7830fa5` (3 × 100, st ≤ 0.25, superseded) |
| `nleflow convert ... --preset ddm_uniform_st --verify 300` | `7-100-100-100-100-100-1 tanh MLP`; card vs float64 graph `1.07e-14` (gate 1e-10); vs the published float32 graph `4.80e-06` (gate 1e-04); `PASS`; the golden file written |
| `register_nn_model(card, p_types = card$p_types)` + `nn_golden_check()` | `PASS golden check ... (mlp_joint, 300 rows): max \|diff\| 4.80e-06 (log_pdf) against 1e-04` |
| `nle_verify.R <card>` (the card alone: it carries its `p_types`) | `all checks passed` |
| `check_in_emc2.R <card>` | below |

`check_in_emc2.R` on this laptop (Apple M2 Max):

```
per trial, 4000 rows over the box (dense: DDM log density > -5):
  all dense                       2639 rows: |log LAN - log DDM| median 0.0369, 95% 0.1729
  past the window                 2225 rows: |log LAN - log DDM| median 0.0313, 95% 0.1593
  inside the window                414 rows: |log LAN - log DDM| median 0.0757, 95% 0.2149
  st in (0.25, 0.35]               833 rows: |log LAN - log DDM| median 0.0346, 95% 0.1651
  wrong mapping: a not doubled        median 1.6560 (45x the right one)
  wrong mapping: t as the lower end   median 0.3690 (10x the right one)
  wrong mapping: responses swapped    median 2.2407 (61x the right one)
through a design, 400 trials, 200 draws around the generating values:
  LAN compiled vs R path: max |diff| 3.41e-13; all finite TRUE
  LAN vs DDM summed log lik: mean |diff| 6.50 (per trial 0.0162), cor 0.9852; mean -176.3 vs -176.1
  LAN - DDM across the 191 draws with every rt past t0: sd 7.32, IQR 8.48
  the 9 draws with t0 above the fastest rt (0.417 s): LAN - DDM median 11.1 (the LAN's density below its support)
  time for 200 draws x 400 trials: LAN 0.13 s, DDM 0.07 s
```

How to read it:

- **The mapping is right**, on the current network too: the LAN agrees with EMC2's
  DDM to a median 0.037 log units where the density is not tiny, and each of the three
  plausible wrong mappings is 10 to 61 times worse. The region only the current
  network was trained on (`st` in (0.25, 0.35]) is as good as the rest.
- **A LAN is an approximation.** Through a design the summed log likelihood of 400
  trials differs from the DDM's with sd 7.3 across draws near the generating values.
  The variation across draws, not the mean, is what moves a posterior, so a LAN needs
  its own calibration check before its posteriors are believed: the card says so
  (`validation.calibration`), and EMC2's `nn_cell()` / `run_sbc()` are the tools (see
  EMC2's vignette "Neural likelihoods").
- **Below its support a LAN is not zero.** Where `t0 = t − st` is above some rt the DDM
  density of that trial is 0 (EMC2 floors it at log 1e-10); the LAN returns a small
  positive density. Draws there look better to a LAN than they are.
- **It is not faster here**: with `sv = 0` EMC2's analytic DDM is cheap, and the
  5 × 100 network costs about twice as much per trial.

## The two published files

The source replaced `ddm_uniform_st.onnx` on 2026-10-05 under the same name. The
preset knows both by sha256 and takes the box of the file it is given: the current
5 × 100 network (`st` up to 0.35) and the first 3 × 100 one (`st` up to 0.25; the copy
in `handover/nle-methods/port/export/`). Any other file converts
with a warning; check its parameters, box and coding against its model card before
trusting the mapping.

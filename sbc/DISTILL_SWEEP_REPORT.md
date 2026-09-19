# Distillation beta sweep — report

Generated 2026-09-19 06:12. Settings identical to the existing
dist_* arms (batch 1024, 300 epochs, [256,128], 64 bins, c4clf classifier),
so these points join the same curve. g at H=0.01, 200 SBC prior draws.

## Stage 1-3: the sweep
```
arm                     g_a  (ckpt sd)       g_t0  (ckpt sd)      n
distb20             +0.0342     0.0067    -0.0836     0.0057     10
distb30             +0.0231     0.0103    +0.0159     0.0092     10  <- min |g_t0|
distb50             +0.0146     0.0049    +0.0216     0.0167     10

best single checkpoints by |g_t0|:
  distb30_e293               g_t0 +0.0107   g_a +0.0171
  distb30_e297               g_t0 +0.0107   g_a +0.0188
  distb30_e299               g_t0 +0.0108   g_a +0.0188
  distb30_e298               g_t0 +0.0108   g_a +0.0188
  distb30_e292               g_t0 +0.0113   g_a +0.0162
```

Selected **distb30 (beta = 30)** on smallest mean |g_t0|.

## Stage 4-5: three fresh seeds at the selected beta
```
arm                     g_a  (ckpt sd)       g_t0  (ckpt sd)      n
repdistb30_s301     -0.0645     0.0024    -0.0879     0.0037     10
repdistb30_s302     +0.0204     0.0055    -0.0823     0.0060     10  <- min |g_t0|
repdistb30_s303     -0.0645     0.0044    -0.1417     0.0087     10

best single checkpoints by |g_t0|:
  repdistb30_s302_e258       g_t0 -0.0718   g_a +0.0270
  repdistb30_s302_e247       g_t0 -0.0745   g_a +0.0256
  repdistb30_s301_e279       g_t0 -0.0778   g_a -0.0650
  repdistb30_s302_e256       g_t0 -0.0807   g_a +0.0260
  repdistb30_s302_e268       g_t0 -0.0811   g_a +0.0178
```

## Reference points (measured earlier, same estimator)

| arm | g_a | g_t0 |
|---|---|---|
| plain MLE (5 seeds) | -0.0259 | -0.0696 |
| cap256w_c4 (production) | +0.0118 | -0.0620 |
| beta = 1 | -0.0114 | -0.0624 |
| beta = 10 | +0.0137 | -0.0317 |
| beta = pure | +0.0461 | +0.0265 |
| direct regression | +0.0080 | +0.0063 |

Caveat: mean g_t0 near zero does NOT imply SBC passes. The spread of
g_t0 across theta is ~0.03 for every distilled arm measured so far and
does not shrink with beta, so a region-dependent residual survives a
zero mean (Phase 9b). SBC remains the verdict.

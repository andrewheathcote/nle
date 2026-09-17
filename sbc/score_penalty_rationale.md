Subject: Why the neural likelihoods need a change to how we train them

SUMMARY

Our neural (normalising-flow) likelihoods for the DDM are miscalibrated in
st0, and we now know why. The cause is not that the flow fits the RT density
badly — it fits it well. The cause is that maximum-likelihood training puts no
constraint on the *derivative* of the density error with respect to the
parameters, and it is that derivative, not the error itself, which pushes a
posterior off the truth. We are adding a term to the training loss that
targets it directly. This note explains the evidence, because the change is
not cosmetic and it probably affects LNR as well as the DDM.


WHAT WE OBSERVED

Simulation-based calibration (500 replicates per cell, run on Gadi) of the
production 5-flow DDM ensemble gives 95% interval coverage for st0 of:

    N = 100    0.940
    N = 200    0.836
    N = 400    0.666      (should be 0.95)

It gets worse with more data, which is the signature of a systematic bias
rather than noise: more trials buy more confidence in the wrong place. The
other parameters are in good shape — in particular t0 and a, which used to be
catastrophic (coverage 0.52 and 0.61 at N=100), were fixed by a wider
conditioner network and are now ~0.94-0.96.


WHAT WE TRIED THAT DID NOT WORK

Over several months we ruled out the obvious explanations, each with its own
study:

  - More training data. Budgets of 50k / 200k / 1M simulations. The bias does
    not shrink with budget; it is a capacity/objective problem, not a data
    problem.
  - Bigger/finer networks. More spline bins did nothing. A wider conditioner
    fixed t0 and a but left st0 untouched (in fact slightly worse, because the
    residual error redistributed).
  - Censored training (ignore RTs beyond 10 s), on the theory that the flow
    was wasting capacity on an unsampled tail. No effect: only 0.34% of trials
    are out there.
  - Narrowing the spline support. No effect on st0, and it made Z worse.
  - Ensembling five independently trained flows. This removed essentially all
    the seed-to-seed noise and fixed the remaining wobble in a and Z — but not
    st0, which is the definition of a systematic error.

We also checked directly whether the flow's density error is large in the
region st0 controls. It is large — about +5 log units out at 1-3 s — but that
region carries almost no probability mass, and once you control for where a
draw sits in parameter space, the size of the density error carries no
information at all about which draws are miscalibrated. So "the flow is
inaccurate somewhere" turned out to be the wrong frame.


WHAT IS ACTUALLY WRONG

The right frame is the score: the gradient of the log-likelihood with respect
to the parameters. For a correct likelihood, the expected score at the true
parameter is exactly zero — that is what makes the truth the peak. If a
likelihood has a non-zero expected score at the truth, it exerts a constant
force pushing every posterior off in the same direction, and the force
accumulates with trials.

We measured it. The flow's expected score in st0 is +0.029 per trial,
essentially identical at N = 100, 200 and 400 (as it must be — it is a
property of the likelihood, not of the sample size), and confirmed by two
completely independent estimators (+0.029 by numerical integration against
WienR, +0.025 by Monte Carlo simulation). Feeding that single number through
the standard posterior-displacement formula predicts:

    N        predicted     observed
    100        0.34          0.29
    200        0.59          0.98
    400        1.04          1.53
             (displacement in posterior SDs)

So one constant, independently measured, reproduces the whole degradation
curve. The residual gap is parameter coupling, which we can see separately.

Crucially, this also explains why the censoring experiment failed. A uniform
offset in log-density — however big — shifts the log-likelihood without moving
its peak. Only a *tilt* moves the peak. We had been targeting the wrong
quantity.


WHY TRAINING ALLOWS THIS

Maximum-likelihood training minimises cross-entropy, which controls how close
the flow's density is to the truth on average. It says nothing whatsoever
about the gradient of the residual error in parameter space. A flow can
therefore have excellent held-out likelihood — ours does — and still carry a
systematic score bias. Nothing in the current objective penalises it, so
nothing prevents it.


THE CHANGE

We add a penalty on the flow's squared score bias. It can be estimated without
bias because our simulator writes exactly two trials per parameter draw: for
two conditionally independent RTs from the same parameters, the inner product
of their scores is an unbiased estimate of the squared bias. So the extra cost
is one backward pass, and the training data needs no regeneration. Setting the
penalty weight to zero recovers the current objective exactly, so the change
is safe by construction and the comparison is clean.


DOES THIS AFFECT THE OTHER MODELS

LNR: yes, probably. Measured score biases are 11-23x their Monte Carlo error,
the largest being +0.032 in s. Independently, the Gadi SBC campaign had flagged
exactly one unexplained anomaly in LNR — the s parameter's ranks were
non-uniform with a small positive bias. The score measurement predicts that
anomaly, in the right parameter and the right direction, without having been
shown it. That is the strongest external check we have.

RDM: not established. The signals are only 2-3x their error and our estimator
does not validate cleanly for RDM's boundary parameters, so we are not making
a claim yet. RDM passed SBC, so any bias there is small.

Worth noting: if st0 is fixed at zero, the analytic DDM is both exactly
calibrated and about as fast as the flow (6.3 vs 3.9 microseconds/trial at
N=5000, and the analytic is *faster* at N=100). For that case the flow is not
needed at all. The neural likelihood earns its keep when st0 is estimated —
which is precisely the case this fix targets.


WHAT HAPPENS NEXT

Train the DDM flow with the penalty, sweep its weight, export and
golden-validate as usual, then re-run the laptop precision gates and, if those
pass, an SBC cell. If the account is right, the score bias should fall towards
zero and st0 coverage should return towards 0.95 with no cost to the
parameters that are already correct.

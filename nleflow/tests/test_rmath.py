"""R's pnorm and qnorm in NumPy (rmath.py) against scipy: the two algorithms
the ONNX graph carries, so that it computes the normal CDF and the probit
as R (and so EMC2) does."""
import numpy as np
from scipy.special import ndtr, ndtri

from nleflow.rmath import pnorm, qnorm


def test_pnorm_matches_scipy():
    """Relative 2e-14 over the range the flows use (z > -8); Cody's own tail
    accuracy, about 2e-13, down to -30 (R's pnorm has the same)."""
    x = np.linspace(-30, 8, 380001)
    rel = np.abs(pnorm(x) - ndtr(x)) / ndtr(x)
    assert rel[x > -8].max() < 2e-14
    assert rel.max() < 2e-13
    assert pnorm(np.array([0.0])) == 0.5
    assert np.all(pnorm(np.array([-40.0, 40.0])) == np.array([0.0, 1.0]))


def test_qnorm_matches_scipy_to_2e_15():
    p = np.concatenate([np.linspace(1e-12, 1 - 1e-12, 200001), [0.075, 0.925, 0.5]])
    assert np.abs(qnorm(p) - ndtri(p)).max() < 2e-15
    assert np.abs(pnorm(qnorm(p)) - p).max() < 1e-15

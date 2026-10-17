"""R's normal CDF and quantile, as R computes them, in NumPy.

`pnorm()` is Cody's rational Chebyshev approximation (R's nmath/pnorm.c,
pnorm_both with log_p = FALSE) and `qnorm()` is Wichura's AS 241 (R's
nmath/qnorm.c), transliterated so that the ONNX graph of a scaled DDM card
(onnx.py), which must agree with EMC2's compiled evaluator to 1e-11, computes
the normal CDF of the spline coordinate and the probit of Z with R's own
algorithms: onnxruntime has no double-precision Erf, and R::pnorm is what
EMC2 calls. Each function is written with the elementwise operations the
graph has (Where, Floor, Sign, Exp, Log, Sqrt), in the same order, so the two
implementations are one algorithm. Tested against scipy to 1e-15.
"""
from __future__ import annotations

import numpy as np

M_SQRT_32 = 5.656854249492380195206754896838
M_1_SQRT_2PI = 0.398942280401432677939946059934

_A = [2.2352520354606839287, 161.02823106855587881, 1067.6894854603709582, 18154.981253343561249,
      0.065682337918207449113]
_B = [47.20258190468824187, 976.09855173777669322, 10260.932208618978205, 45507.789335026729956]
_C = [0.39894151208813466764, 8.8831497943883759412, 93.506656132177855979, 597.27027639480026226,
      2494.5375852903726711, 6848.1904505362823326, 11602.651437647350124, 9842.7148383839780218,
      1.0765576773720192317e-8]
_D = [22.266688044328115691, 235.38790178262499861, 1519.377599407554805, 6485.558298266760755,
      18615.571640885098091, 34900.952721145977266, 38912.003286093271411, 19685.429676859990727]
_P = [0.21589853405795699, 0.1274011611602473639, 0.022235277870649807, 0.001421619193227893466,
      2.9112874951168792e-5, 0.02307344176494017303]
_Q = [1.28426009614491121, 0.468238212480865118, 0.0659881378689285515, 0.00378239633202758244,
      7.29751555083966205e-5]


def _trunc16(x):
    """ldexp(trunc(ldexp(x, 4)), -4): x rounded towards zero to a multiple of 1/16."""
    return np.sign(x) * np.floor(np.abs(x) * 16.0) / 16.0


def pnorm(x):
    """The lower-tail standard normal CDF, R's pnorm(x) (Cody)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.abs(x)
    # region 1: |x| <= qnorm(3/4)
    xsq = x * x
    xnum = _A[4] * xsq
    xden = xsq
    for i in range(3):
        xnum = (xnum + _A[i]) * xsq
        xden = (xden + _B[i]) * xsq
    temp1 = x * (xnum + _A[3]) / (xden + _B[3])
    cum1 = 0.5 + temp1
    # region 2: qnorm(3/4) < |x| <= sqrt(32)
    xnum = _C[8] * y
    xden = y
    for i in range(7):
        xnum = (xnum + _C[i]) * y
        xden = (xden + _D[i]) * y
    temp2 = (xnum + _C[7]) / (xden + _D[7])
    xsq2 = _trunc16(y)
    del2 = (y - xsq2) * (y + xsq2)
    upper2 = np.exp(-xsq2 * (xsq2 / 2.0)) * np.exp(-del2 / 2.0) * temp2    # P(|X| > y) one-sided
    cum2 = np.where(x > 0.0, 1.0 - upper2, upper2)
    # region 3: sqrt(32) < |x| < 38.4674 (lower) : same with the asymptotic ratio
    safe = np.where(y > 1e-300, x, 1.0)              # region 3 only; x = 0 is region 1
    xsq3 = 1.0 / (safe * safe)
    xnum = _P[5] * xsq3
    xden = xsq3
    for i in range(4):
        xnum = (xnum + _P[i]) * xsq3
        xden = (xden + _Q[i]) * xsq3
    temp3 = xsq3 * (xnum + _P[4]) / (xden + _Q[4])
    temp3 = (M_1_SQRT_2PI - temp3) / np.where(y > 1e-300, y, 1.0)
    xsq3b = _trunc16(x)
    del3 = (x - xsq3b) * (x + xsq3b)
    upper3 = np.exp(-xsq3b * (xsq3b / 2.0)) * np.exp(-del3 / 2.0) * temp3
    cum3 = np.where(x > 0.0, 1.0 - upper3, upper3)
    far = np.where(x > 0.0, 1.0, 0.0)
    out = np.where(y <= 0.67448975, cum1,
                   np.where(y <= M_SQRT_32, cum2,
                            np.where((x > -38.4674) & (x < 8.2924), cum3, far)))
    return out


def qnorm(p):
    """The standard normal quantile, R's qnorm(p) (AS 241), for p in (0, 1)
    not closer than exp(-729) to either end."""
    p = np.asarray(p, dtype=np.float64)
    q = p - 0.5
    r = 0.180625 - q * q
    val1 = (q * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r
                     + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r
                  + 133.14166789178437745) * r + 3.387132872796366608)
            / (((((((r * 5226.495278852854561 + 28729.085735721942674) * r + 39307.89580009271061) * r
                   + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r
                + 42.313330701600911252) * r + 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        pm = np.where(q > 0.0, 1.0 - p, p)
        rr = np.sqrt(-np.log(np.where(pm > 0.0, pm, 0.5)))
    r5 = rr - 1.6
    val2 = ((((((((r5 * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r5 + 0.24178072517745061177) * r5
                 + 1.27045825245236838258) * r5 + 3.64784832476320460504) * r5 + 5.7694972214606914055) * r5
              + 4.6303378461565452959) * r5 + 1.42343711074968357734)
            / (((((((r5 * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r5 + 0.0151986665636164571966) * r5
                   + 0.14810397642748007459) * r5 + 0.68976733498510000455) * r5 + 1.6763848301838038494) * r5
                + 2.05319162663775882187) * r5 + 1.0))
    r27 = rr - 5.0
    val3 = ((((((((r27 * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r27 + 0.0012426609473880784386) * r27
                 + 0.026532189526576123093) * r27 + 0.29656057182850489123) * r27 + 1.7848265399172913358) * r27
              + 5.4637849111641143699) * r27 + 6.6579046435011037772)
            / (((((((r27 * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r27 + 1.8463183175100546818e-5) * r27
                   + 7.868691311456132591e-4) * r27 + 0.0148753612908506148525) * r27 + 0.13692988092273580531) * r27
                + 0.59983220655588793769) * r27 + 1.0))
    tail = np.where(rr <= 5.0, val2, val3)
    tail = np.where(q < 0.0, -tail, tail)
    return np.where(np.abs(q) <= 0.425, val1, tail)

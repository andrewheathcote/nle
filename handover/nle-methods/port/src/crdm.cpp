// Conflict-pulse diffusion (CRDM building block): one accumulator whose mean
// path is  M(t) = v t + amp (e t / tau) exp(-t / tau)  (gamma shape 2, peak
// height amp), noise s, absorbing boundary b, start 0.
//
// Direct translation of Malte Lueken's eamax (accumulators/pulse.py,
// diffusion.py, volterra.py, v0.1.1 6d7f2ff). Standalone: no dependence on
// flow_race.cpp. Everything is scalar-parameter; the caller passes all trials
// that share a parameter row in one call.
//
//   crdm_mean(t, v, amp, tau)        -> list(mean = M(t), drift = M'(t))
//   rcrdm_acc(n, v, amp, tau, s, b, dt, t_max)
//        first-passage times: exact mean path + Brownian increments on the
//        grid dt, 2dt, ..., round(t_max/dt) dt; Brownian-bridge crossing test
//        between grid points; crossing time (index + u) dt, u ~ U(0,1)
//        (within-step dequantisation); Inf when no crossing by the last grid
//        point. Uses R's RNG (set.seed works). Streams one path at a time and
//        stops at the crossing, so memory is O(1) not O(n * steps).
//   crdm_volterra_grid(v, amp, tau, s, b, dt, t_max)
//        Fortet recursion on the grid: list(t, pdf, cdf).
//   dcrdm_volterra(t, v, amp, tau, s, b, dt, t_max)
//        the same solved once, then linearly interpolated at t. Like
//        jnp.interp in eamax, t outside [dt, t_max] holds the end value
//        (t < dt returns pdf(dt), cdf(dt); t > t_max returns the last values),
//        so t_max must cover the data.
//
// The recursion is O(steps^2) and sequential; the kernel is evaluated on the
// fly (no steps x steps matrix), and terms whose Gaussian factor is below
// e^-700 are skipped.

#include <Rcpp.h>
#include <cmath>
#include <vector>
using namespace Rcpp;

static const double INV_SQRT_2PI = 0.3989422804014327;

static inline double pulse_C(double t, double amp, double tau) {
  return amp * std::exp(-t / tau) * (M_E * t / tau);
}
// dC/dt = C (1/t - 1/tau)   (shape 2)
static inline double pulse_dC(double t, double amp, double tau) {
  return pulse_C(t, amp, tau) * (1.0 / t - 1.0 / tau);
}

static void check_pulse(double tau, double s, double b, double dt, double t_max) {
  if (!(tau > 0) || !(s > 0) || !(b > 0)) stop("tau, s and b must be positive");
  if (!(dt > 0) || !(t_max >= dt)) stop("need 0 < dt <= t_max");
}

// [[Rcpp::export]]
List crdm_mean(NumericVector t, double v, double amp, double tau) {
  if (!(tau > 0)) stop("tau must be positive");
  int n = t.size();
  NumericVector m(n), d(n);
  for (int i = 0; i < n; i++) {
    m[i] = v * t[i] + pulse_C(t[i], amp, tau);
    d[i] = v + pulse_dC(t[i], amp, tau);
  }
  return List::create(_["mean"] = m, _["drift"] = d);
}

// [[Rcpp::export]]
NumericVector rcrdm_acc(int n, double v, double amp, double tau, double s,
                        double b, double dt, double t_max) {
  check_pulse(tau, s, b, dt, t_max);
  const int N = (int)std::llround(t_max / dt);
  const double sq = std::sqrt(dt);
  const double c = 2.0 / (s * s * dt);
  NumericVector out(n);
  RNGScope scope;
  for (int i = 0; i < n; i++) {
    double W = 0.0, xprev = 0.0, res = R_PosInf;
    for (int k = 1; k <= N; k++) {
      W += norm_rand() * sq;
      double tk = k * dt;
      double x = v * tk + pulse_C(tk, amp, tau) + s * W;
      bool cross = x >= b;
      if (!cross && xprev < b) {
        double p = std::exp(-c * (b - xprev) * (b - x));
        cross = unif_rand() < p;
      }
      if (cross) { res = ((k - 1) + unif_rand()) * dt; break; }
      xprev = x;
    }
    out[i] = res;
  }
  return out;
}

static void volterra_solve(double v, double amp, double tau, double s, double b,
                           double dt, int N, std::vector<double>& g,
                           std::vector<double>& G) {
  std::vector<double> m(N), vi(N), h0(N), sqlag(N + 1);
  for (int k = 0; k < N; k++) {
    double tk = (k + 1) * dt;
    m[k] = v * tk + pulse_C(tk, amp, tau);
    vi[k] = v + pulse_dC(tk, amp, tau);
    double st = s * std::sqrt(tk);
    double z = (b - m[k]) / st;
    h0[k] = INV_SQRT_2PI * std::exp(-0.5 * z * z) / st * (vi[k] + (b - m[k]) / tk);
  }
  for (int l = 1; l <= N; l++) sqlag[l] = std::sqrt(l * dt);
  g.assign(N, 0.0);
  for (int n = 0; n < N; n++) {
    double acc = 0.0;
    for (int j = 0; j < n; j++) {
      if (g[j] == 0.0) continue;
      int lag = n - j;
      double dtd = lag * dt;
      double md = m[n] - m[j];
      double sd = s * sqlag[lag];
      double z = md / sd;
      double z2 = z * z;
      if (z2 > 1400.0) continue;
      double phi = INV_SQRT_2PI * std::exp(-0.5 * z2) / sd;
      acc += g[j] * (0.5 * phi * (-vi[n] + md / dtd));
    }
    double val = h0[n] + 2.0 * dt * acc;
    g[n] = val > 0.0 ? val : 0.0;
  }
  G.resize(N);
  double cs = 0.0;
  for (int k = 0; k < N; k++) { cs += g[k]; G[k] = dt * (cs - 0.5 * g[k]); }
}

// [[Rcpp::export]]
List crdm_volterra_grid(double v, double amp, double tau, double s, double b,
                        double dt, double t_max) {
  check_pulse(tau, s, b, dt, t_max);
  const int N = (int)std::llround(t_max / dt);
  std::vector<double> g, G;
  volterra_solve(v, amp, tau, s, b, dt, N, g, G);
  NumericVector t(N);
  for (int k = 0; k < N; k++) t[k] = (k + 1) * dt;
  return List::create(_["t"] = t, _["pdf"] = wrap(g), _["cdf"] = wrap(G));
}

static inline double interp_hold(double x, double dt, int N, const std::vector<double>& f) {
  double u = x / dt - 1.0;           // fractional index into the grid dt, 2dt, ...
  if (!(u > 0.0)) return f[0];       // also catches NaN -> f[0]; callers pass finite t
  if (u >= N - 1) return f[N - 1];
  int i = (int)u;
  double w = u - i;
  return f[i] + w * (f[i + 1] - f[i]);
}

// [[Rcpp::export]]
List dcrdm_volterra(NumericVector t, double v, double amp, double tau, double s,
                    double b, double dt, double t_max) {
  check_pulse(tau, s, b, dt, t_max);
  const int N = (int)std::llround(t_max / dt);
  std::vector<double> g, G;
  volterra_solve(v, amp, tau, s, b, dt, N, g, G);
  int n = t.size();
  NumericVector pdf(n), cdf(n);
  for (int i = 0; i < n; i++) {
    pdf[i] = interp_hold(t[i], dt, N, g);
    cdf[i] = interp_hold(t[i], dt, N, G);
  }
  return List::create(_["pdf"] = pdf, _["cdf"] = cdf);
}

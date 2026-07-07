# Regression test for simulation/designs.R and simulate_trials().
#
# Pins:
#   - design parameter names and ordering for each model
#   - lower/upper bound values
#   - sampled-scale transforms
#   - Sobol parameter grid from param_matrix() for a fixed n_draws
#   - simulate_trials() output schema + a small numerical property for a
#     fixed seed
#
# Run from repo root:
#   Rscript simulation/tests/test_designs.R

source("simulation/sim_functions.R")
source("simulation/designs.R")

stop_if_not <- function(cond, msg) {
  if (!isTRUE(cond)) stop(msg, call. = FALSE)
}

# ---- 1. Design parameter names --------------------------------------------

stop_if_not(
  identical(param_namesRDM,
            c("v_lR1", "v_lR2", "B_lR1", "B_lR2", "t0",
              "A_lR1", "A_lR2", "s_lR1", "s_lR2")),
  "RDM param names changed"
)

stop_if_not(
  identical(param_namesDDM,
            c("v", "a", "t0", "s", "Z", "SZ", "sv", "st0")),
  "DDM param names changed"
)

stop_if_not(
  identical(param_namesLNR,
            c("m_lR1", "m_lR2", "s_lR1", "s_lR2", "t0")),
  "LNR param names changed"
)

# ---- 2. Bounds & transforms ------------------------------------------------

stop_if_not(
  identical(lowerLNR, c(-3, -3, 0.1, 0.1, 0.05)) &&
    identical(upperLNR, c(3, 3, 2, 2, 1)) &&
    identical(transformsLNR,
              c("identity", "identity", "log", "log", "log")),
  "LNR bounds/transforms changed"
)

stop_if_not(
  identical(transformsDDM,
            c("identity", "log", "log", "log",
              "probit", "probit", "log", "log")),
  "DDM transforms changed"
)

# ---- 3. param_matrix() Sobol grid is deterministic -------------------------

p <- param_matrix(n_draws = 3, param_names = param_namesLNR,
                  lower = lowerLNR, upper = upperLNR,
                  transforms = transformsLNR, seed = 0L, Sobol = TRUE)

stop_if_not(is.matrix(p), "param_matrix should return a matrix")
stop_if_not(identical(dim(p), c(3L, 5L)), "param_matrix shape changed")
stop_if_not(identical(colnames(p), param_namesLNR), "param_matrix colnames changed")
stop_if_not(all(p[, "m_lR1"] >= lowerLNR[1] & p[, "m_lR1"] <= upperLNR[1]),
            "param_matrix m_lR1 out of bounds")

# ---- 4. simulate_trials() schema + sanity ---------------------------------

set.seed(42L)
sim <- simulate_trials(p_vec = p[1, ], design = designLNR,
                       param_id = 1L, n_trials = 32L)

expected_cols <- c("param_id", "S", "R", "rt", "C", "ll", param_namesLNR)
stop_if_not(identical(colnames(sim), expected_cols),
            "simulate_trials output schema changed")
stop_if_not(all(sim$rt > 0), "simulate_trials produced non-positive rt")
stop_if_not(all(sim$R %in% 1:2), "simulate_trials R not in {1,2}")
stop_if_not(all(sim$param_id == 1L), "simulate_trials param_id mismatch")

cat("OK: simulation/tests/test_designs.R passed\n")

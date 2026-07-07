# Design objects, parameter bounds, and sampled-scale transforms for the
# evidence accumulation models (RDM, DDM, LNR; plus single-accumulator
# variants used as targets for the flow / KDE).
#
# Sourced by:
#   - simulation/sim_code.R                       (driver for data simulation)
#   - simulation/tests/test_designs.R             (design checks)
#   - benchmark/scripts_julian/get_data_params.R  (timing eval datasets + params)
#   - benchmark/scripts_julian/get_kde_train_at_eval_params.R
#   - results/figures_paper/plotting.R            (designs/bounds for plots)
#
# Requires EMC2 to be loaded (handled by simulation/sim_functions.R).

###############################################
##### Design objects and parameter bounds #####
###############################################

# RDM
designRDM <- design(
  model = RDM,
  factors = list(subjects = 1, S = 1:2),
  Rlevels = 1:2,
  formula = list(v ~ 0+lR, B ~ 0+lR, t0 ~ 1, A ~ 0+lR, s ~ 0+lR), # shared NDT
)
param_namesRDM <- names(EMC2::sampled_pars(designRDM))
param_namesRDM # "v_lR1" "v_lR2" "B_lR1" "B_lR2" "t0"    "A_lR1" "A_lR2" "s_lR1" "s_lR2"

lowerRDM <- c(
  1e-3, 1e-3, # drift rate (min in EMC2)
  0.1, 0.1,   # threshold gap
  0.05,       # non-decision time (min in EMC2)
  1e-4, 1e-4, # width of uniform start-point (min in EMC2)
  0.1, 0.1    # moment-to-moment variation
)

upperRDM <- c(
  5, 5,
  3, 3,
  1,
  2, 2,
  2, 2
)

# RDM (single accumulator)
designRDM_single <- design(
  model = RDM,
  Rlevels = 1,
  formula = list(v ~ 1, B ~ 1, t0 ~ 1, A ~ 1, s ~ 1)
)

param_namesRDM_single <- names(EMC2::sampled_pars(designRDM_single))
param_namesRDM_single # "v"  "B"  "t0" "A"  "s"

lowerRDM_single <- c(
  1e-3, # drift rate (min in EMC2)
  0.1,   # threshold gap
  0.05,       # non-decision time (min in EMC2)
  1e-4,  # width of uniform start-point (min in EMC2)
  0.1    # moment-to-moment variation
)

upperRDM_single <- c(
  5,
  3,
  1,
  2,
  2
)

# DDM
designDDM <- design(
  model = DDM,
  factors = list(subjects = 1, S = 1:2),
  Rlevels = 1:2,
  formula = list(v ~ 1, a ~ 1, t0 ~ 1, s ~ 1, Z ~ 1, SZ ~ 1, sv ~ 1, st0 ~ 1)
)
param_namesDDM <- names(EMC2::sampled_pars(designDDM))
param_namesDDM # "v"   "a"   "t0"  "s"   "Z"   "SZ"  "sv"  "st0"

lowerDDM <- c(
  -6,
  0.05,
  0.05,  # EMC2 min
  0.1,
  0.1,
  0.05,  # SZ: SZ_mod = 2*SZ*min(Z,1-Z) > 0.01; worst case min(Z,1-Z)=0.1 -> SZ > 0.01/(2*0.1) = 0.05
  0.01,  # EMC2 min
  0.01   # EMC2 min is 0, but log(0) = -Inf; use small positive value
)

upperDDM <- c(
  6,
  5,
  1,
  2,
  0.9,
  0.99,  # SZ: SZ_mod = 2*SZ*min(Z,1-Z) < 0.99; worst case min(Z,1-Z)=0.5 -> SZ < 0.99/(2*0.5) = 0.99
  4,
  0.49   # EMC2 max is 0.5 (strict), stay below
)

# Per-parameter transforms (inverse of what model$transform applies):
#   v   -> identity  (unbounded drift rate, EMC2 uses no transform)
#   a   -> log       (positive threshold)
#   t0  -> log       (positive non-decision time)
#   s   -> log       (positive diffusion coefficient)
#   Z   -> probit     (relative start point, proportion in (0,1))
#   SZ  -> probit     (start-point variability, proportion in (0,1))
#   sv  -> log       (drift variability, positive)
#   st0 -> log       (NDT variability, positive)
transformsDDM <- c("identity", "log", "log", "log", "probit", "probit", "log", "log")

# LNR
designLNR <- design(
  model = LNR,
  factors = list(subjects = 1, S = 1:2),
  Rlevels = 1:2,
  formula = list(m ~ 0+lR, s ~ 0+lR, t0 ~ 1)
)
param_namesLNR <- names(EMC2::sampled_pars(designLNR))
param_namesLNR # "m_lR1" "m_lR2" "s_lR1" "s_lR2" "t0"

lowerLNR <- c(
  -3, -3, # m_lR1, m_lR2 (log-mean, log-seconds scale)
  0.1, 0.1, # s_lR1, s_lR2 (log-SD)
  0.05 # t0 (EMC2 min)
)

upperLNR <- c(
  3, 3,   # m_lR1, m_lR2
  2, 2,   # s_lR1, s_lR2
  1       # t0
)

# m is already on log scale (unbounded) -> identity; s and t0 are positive -> log
transformsLNR <- c("identity", "identity", "log", "log", "log")


# LNR (one accumulator)
designLNR_single <- design(
  model = LNR,
  Rlevels = 1,
  formula = list(m ~ 1, s ~ 1, t0 ~ 1)
)
param_namesLNR_single <- names(EMC2::sampled_pars(designLNR_single))
param_namesLNR_single # "m"  "s"  "t0"

lowerLNR_single <- c(
  -3, # m
  0.1, # s
  0.05 # t0 (EMC2 min)
)

upperLNR_single <- c(
  3,   # m
  2,   # s
  1       # t0
)

# m is already on log scale (unbounded) -> identity; s and t0 are positive -> log
transformsLNR_single <- c("identity", "log", "log")

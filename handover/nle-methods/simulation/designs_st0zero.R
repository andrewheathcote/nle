# DDM design and training box with st0 = 0 (7 sampled parameters).
#
# Differs from simulation/designs.R ONLY in dropping st0. EMC2 gives st0 a
# default of -Inf on the sampled scale with an exp transform, so OMITTING it
# from the formula sets st0 to exactly 0 -- no -Inf has to be passed by hand,
# and nothing else about the design changes.
#
#   DDM()$p_types  ->  st0 default -Inf ; transform func st0 = "exp"
#
# Everything else (bounds, transforms, ordering) is copied verbatim from
# simulation/designs.R so the two builds differ in one factor only.
suppressPackageStartupMessages(library(EMC2))

designDDM0 <- design(
  model = DDM,
  factors = list(subjects = 1, S = 1:2),
  Rlevels = 1:2,
  formula = list(v ~ 1, a ~ 1, t0 ~ 1, s ~ 1, Z ~ 1, SZ ~ 1, sv ~ 1)
)
param_namesDDM0 <- names(EMC2::sampled_pars(designDDM0))
stopifnot(!("st0" %in% param_namesDDM0), length(param_namesDDM0) == 7L)

#                     v     a     t0    s    Z    SZ    sv
lowerDDM0 <- c(      -6,  0.05,  0.05, 0.1, 0.1, 0.05, 0.01)
upperDDM0 <- c(       6,  5.00,  1.00, 2.0, 0.9, 0.99, 4.00)
transformsDDM0 <- c("identity", "log", "log", "log", "probit", "probit", "log")

# guard against a silent reordering: sampled_pars order must match the vectors
stopifnot(identical(param_namesDDM0, c("v","a","t0","s","Z","SZ","sv")),
          length(lowerDDM0) == 7L, length(upperDDM0) == 7L,
          length(transformsDDM0) == 7L)
cat("st0=0 DDM design OK:", paste(param_namesDDM0, collapse = ", "), "\n")

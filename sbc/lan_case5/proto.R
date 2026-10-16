.libPaths(c(path.expand("~/R/nle-lib"), .libPaths())); suppressPackageStartupMessages(library(EMC2))
source("sbc/lan_case5/lan_model.R")
mk <- lan_model()
cell <- nn_cell(mk, mean = PM, sd = PS, control = DDMlan, factors = list(S = c("left", "right")),
                formula = lapply(names(PM), function(p) as.formula(paste(p, "~ 1"))), Rlevels = c("left", "right"))
print(cell)
set.seed(1); dat <- make_data(cell$mean, cell$design, n_trials = 200); print(summary(dat$rt)); print(table(dat$R))
# log-likelihood of the same data under network and control at the true and perturbed parameters
pv <- cell$mean
f <- function(des, p) { dadm <- EMC2:::design_model(dat, des, verbose = FALSE)
  sum(EMC2:::calc_ll_manager(t(as.matrix(p)), dadm, des$model)) }
for (k in 1:5) { p <- pv + rnorm(5, 0, PS); cat(sprintf("LAN %9.2f  control %9.2f\n", f(cell$design, p), f(cell$control_design, p))) }

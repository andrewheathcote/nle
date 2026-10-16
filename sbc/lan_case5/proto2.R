.libPaths(c(path.expand("~/R/nle-lib"), .libPaths())); suppressPackageStartupMessages(library(EMC2))
source("sbc/lan_case5/lan_model.R")
cell <- nn_cell(lan_model(), mean = PM, sd = PS, control = DDMlan, factors = list(S = c("left", "right")),
                formula = lapply(names(PM), function(p) as.formula(paste(p, "~ 1"))), Rlevels = c("left", "right"))
set.seed(2); p <- cell$mean
dat <- make_data(p, cell$design, n_trials = 200)
f <- function(des, p) { dadm <- EMC2:::design_model(dat, des, verbose = FALSE)
  sum(EMC2:::calc_ll_manager(t(as.matrix(p)), dadm, des$model)) }
n <- c(v = p[["v"]], a = 2 * exp(p[["a"]]), z = pnorm(p[["z"]]), t0 = exp(p[["t"]]) - exp(p[["st"]]), st0 = 2 * exp(p[["st"]]))
w <- WienR::dWDM(dat$rt, response = ifelse(dat$R == "right", "upper", "lower"), a = n[["a"]], v = n[["v"]], w = n[["z"]],
                 t0 = n[["t0"]], st0 = n[["st0"]], precision = 1e-8)$logvalue
cat(sprintf("at truth: LAN %.3f  control %.3f  WienR direct %.3f\n", f(cell$design, p), f(cell$control_design, p), sum(w)))
for (des in list(LAN = cell$design, control = cell$control_design)) {
  dadm <- EMC2:::design_model(dat, des, verbose = FALSE); P <- t(as.matrix(p))[rep(1, 100), ]
  cat(sprintf("%.3f ms per particle (400 trials)\n", system.time(EMC2:::calc_ll_manager(P, dadm, des$model))[3] * 10)) }

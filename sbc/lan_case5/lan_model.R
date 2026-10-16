# HSSM's ddm_uniform_st LAN (huggingface.co/Eitanm/ddm-st-lans) in EMC2 dev-nle, and its analytic
# control in the same (LAN) parameterisation. Card: EMC2-nle tests/testthat/golden/lan_ddm_uniform_st.rds,
# converted by inst/scripts/onnx_to_card.py and checked there against onnxruntime.
# LAN inputs: v, a (half the boundary separation), z, t (centre of the uniform t0), st (its half-width).
# EMC2 DDM: a = 2 a_lan, Z = z, t0 = t - st, st0 = 2 st, sv = SZ = 0, s = 1.
DDMlan <- function() {
  d <- DDM()
  list(type = "DDM",
       p_types = c(v = 0, a = log(.75), z = 0, t = log(.55), st = log(.075)),
       transform = list(func = c(v = "identity", a = "exp", z = "pnorm", t = "exp", st = "exp")),
       bound = list(minmax = cbind(v = c(-20, 20), a = c(0, 5), z = c(.01, .99), t = c(0, Inf), st = c(0, .5))),
       Ttransform = function(pars, dadm) {
         n <- nrow(pars)
         dp <- cbind(v = pars[, "v"], a = 2 * pars[, "a"], sv = 0, t0 = pars[, "t"] - pars[, "st"],
                     st0 = 2 * pars[, "st"], s = 1, Z = pars[, "z"], SZ = 0)
         out <- d$Ttransform(dp, dadm)
         attr(out, "ok") <- attr(pars, "ok")
         out
       },
       rfun = function(data = NULL, pars) rDDM(data$R, pars, attr(pars, "ok")),
       dfun = function(rt, R, pars) { out <- numeric(length(rt)); ok <- pars[, "t0"] > 0
         out[ok] <- dDDM(rt[ok], R[ok], pars[ok, , drop = FALSE]); out },
       pfun = function(rt, R, pars) pDDM(rt, R, pars),
       log_likelihood = function(pars, dadm, model, min_ll = log(1e-10))
         log_likelihood_ddm(pars = pars, dadm = dadm, model = model, min_ll = min_ll))
}
environment(DDMlan) <- asNamespace("EMC2")
LANCARD <- "sbc/lan_case5/lan_ddm_uniform_st.rds"
lan_model <- function() register_nn_model(LANCARD, twin = DDMlan)
# Prior on the sampled scale, inside the LAN's training box at +-4 sd:
#   v [-1.9, 2.9] in [-3, 3]; a_lan [0.34, 1.67] in [0.3, 2.5]; z [.316, .684] in [.3, .7];
#   t [0.30, 1.00] in [0.25, 2.25]; st [0.028, 0.20] in [0.001, 0.25].
PM <- c(v = 0.5, a = log(.75), z = 0, t = log(.55), st = log(.075))
PS <- c(v = 0.6, a = 0.2, z = 0.12, t = 0.15, st = 0.25)

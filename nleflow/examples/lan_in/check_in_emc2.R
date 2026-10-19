# The converted LAN inside EMC2, against EMC2's analytic DDM, under the preset's
# mapping (a_EMC2 = 2 a, Z = z, t0 = t - st, st0 = 2 st, sv = SZ = 0, s = 1;
# response 2 = upper). Usage:
#   R_LIBS=<library with EMC2> Rscript check_in_emc2.R <card>
# 1. per trial: the card's density vs DDM's over the box, past and inside the
#    non-decision window, and the three wrong mappings (each far worse);
# 2. through a design: the same data and draws in a LAN design and in a DDM
#    design (draws mapped), the summed log likelihoods, compiled vs R path, timing.
suppressMessages(library(EMC2)); nle <- function(f) getFromNamespace(f, "EMC2")
card <- commandArgs(TRUE)[1]
p_types <- unlist(jsonlite::fromJSON(card)$p_types)
if (is.null(p_types)) p_types <- c(v = 0, a = 1, z = .5, t = .5, st = .05)   # a card converted without a preset
lan <- suppressMessages(register_nn_model(card, p_types = p_types))
box <- jsonlite::fromJSON(card)$bounds_natural; names(box$lower) <- names(box$upper) <- jsonlite::fromJSON(card)$context_names
cat(sprintf("card %s: inputs %s; st box [%g, %g]\n", basename(card), paste(names(box$lower), collapse = ", "),
            box$lower[["st"]], box$upper[["st"]]))

# --- 1. per trial ---------------------------------------------------------------------------------
d <- DDM()
ddm_lp <- function(rt, R, th, a = 2 * th[, "a"], t0 = th[, "t"] - th[, "st"], R2 = R)
  log(d$dfun(rt, R2, d$Ttransform(cbind(v = th[, "v"], a = a, sv = 0, t0 = t0, st0 = 2 * th[, "st"], s = 1,
                                        Z = th[, "z"], SZ = 0), NULL)))
set.seed(31)
n <- 4000
th <- cbind(v = runif(n, -2.5, 2.5), a = runif(n, .4, 2.2), z = runif(n, .35, .65), t = runif(n, .3, 1.5),
            st = runif(n, .01, box$upper[["st"]] - .001))
th[, "st"] <- pmin(th[, "st"], th[, "t"] - .05)                # t0 = t - st > 0 (EMC2's DDM refuses t0 < 0)
inside <- seq_len(n) <= n / 4                                   # a quarter inside the window [t - st, t + st]
rt <- ifelse(inside, th[, "t"] + th[, "st"] * runif(n, -.9, .9), th[, "t"] + th[, "st"] + rexp(n, 1.2) + .05)
R <- sample(1:2, n, TRUE)
lp <- log(lan()$dfun(rt, R, th))
ld <- ddm_lp(rt, R, th)
dense <- is.finite(ld) & ld > -5                                # where the density is not tiny
err <- abs(lp - ld)
say <- function(lab, g) cat(sprintf("  %-30s %5d rows: |log LAN - log DDM| median %.4f, 95%% %.4f\n",
                                    lab, sum(g), median(err[g]), quantile(err[g], .95)))
cat(sprintf("per trial, %d rows over the box (dense: DDM log density > -5):\n", n))
say("all dense", dense)
say("past the window", dense & !inside)
say("inside the window", dense & inside)
say("st in (0.25, 0.35]", dense & th[, "st"] > .25)
wrong <- list(`a not doubled` = ddm_lp(rt, R, th, a = th[, "a"]),
              `t as the lower end` = ddm_lp(rt, R, th, t0 = th[, "t"]),
              `responses swapped` = ddm_lp(rt, R, th, R2 = 3L - R))
for (w in names(wrong)) {
  m <- median(abs(lp - wrong[[w]])[dense], na.rm = TRUE)
  cat(sprintf("  wrong mapping: %-20s median %.4f (%.0fx the right one)\n", w, m, m / median(err[dense])))
}

# --- 2. through a design --------------------------------------------------------------------------
# Data from the analytic DDM; the drift depends on the stimulus (v ~ S). The
# LAN design samples (v, v_Sright, a, z, t, st) on the natural scale (identity
# transforms); the DDM design (v, v_Sright, a, t0, Z, st0) on EMC2's scales.
fac <- list(subjects = 1, S = c("left", "right"))
des_lan <- suppressMessages(design(factors = fac, Rlevels = c("left", "right"), model = lan,
                                   formula = list(v ~ S, a ~ 1, z ~ 1, t ~ 1, st ~ 1), report_p_vector = FALSE))
des_ddm <- suppressMessages(design(factors = fac, Rlevels = c("left", "right"), model = DDM,
                                   formula = list(v ~ S, a ~ 1, t0 ~ 1, Z ~ 1, st0 ~ 1), report_p_vector = FALSE))
to_ddm <- function(X) cbind(v = X[, "v"], v_Sright = X[, "v_Sright"], a = log(2 * X[, "a"]),
                            t0 = log(X[, "t"] - X[, "st"]), Z = qnorm(X[, "z"]), st0 = log(2 * X[, "st"]))
centre <- c(v = -1, v_Sright = 2, a = .6, z = .5, t = .45, st = .1)   # LAN scale: a = half the separation
set.seed(7)
dat <- suppressMessages(make_data(to_ddm(t(centre))[1, ], des_ddm, n_trials = 200))
nd <- 200
X <- sapply(names(centre), function(p) centre[[p]] + rnorm(nd, 0, c(v = .3, v_Sright = .3, a = .05, z = .03,
                                                                     t = .02, st = .03)[[p]]))
X[, "st"] <- pmin(pmax(X[, "st"], .01), .34)
Xd <- to_ddm(X)
dadm_lan <- nle("nn_dadm")(dat, des_lan)
dadm_ddm <- nle("nn_dadm")(dat, des_ddm)
t_lan <- system.time(ll_lan <- as.numeric(nle("calc_ll_manager")(X, dadm_lan, des_lan$model)))[["elapsed"]]
t_ddm <- system.time(ll_ddm <- as.numeric(nle("calc_ll_manager")(Xd, dadm_ddm, des_ddm$model)))[["elapsed"]]
r_path <- function() { ml <- des_lan$model(); ml$c_name <- NULL; ml }
ll_r <- as.numeric(nle("calc_ll_manager")(X, dadm_lan, r_path))
cat(sprintf("through a design, %d trials, %d draws around the generating values:\n", nrow(dat), nd))
cat(sprintf("  LAN compiled vs R path: max |diff| %.2e; all finite %s\n", max(abs(ll_lan - ll_r)), all(is.finite(ll_lan))))
cat(sprintf("  LAN vs DDM summed log lik: mean |diff| %.2f (per trial %.4f), cor %.4f; mean %.1f vs %.1f\n",
            mean(abs(ll_lan - ll_ddm)), mean(abs(ll_lan - ll_ddm)) / nrow(dat), cor(ll_lan, ll_ddm),
            mean(ll_lan), mean(ll_ddm)))
# What moves a posterior is how the error varies across draws, not its mean.
# Where t0 = t - st exceeds the fastest rt the DDM gives those trials zero
# density (floored at log 1e-10) while the LAN leaks density below its support:
# such draws are reported apart.
e <- ll_lan - ll_ddm
supp <- X[, "t"] - X[, "st"] < min(dat$rt)
cat(sprintf("  LAN - DDM across the %d draws with every rt past t0: sd %.2f, IQR %.2f\n",
            sum(supp), sd(e[supp]), IQR(e[supp])))
cat(sprintf("  the %d draws with t0 above the fastest rt (%.3f s): LAN - DDM median %.1f (the LAN's density below its support)\n",
            sum(!supp), min(dat$rt), median(e[!supp])))
cat(sprintf("  time for %d draws x %d trials: LAN %.2f s, DDM %.2f s\n", nd, nrow(dat), t_lan, t_ddm))

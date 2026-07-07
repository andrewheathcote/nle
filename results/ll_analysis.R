library(dplyr)
library(readr)
library(ggplot2)

##### Anomalous ll analysis #####
read_preds <- function(col_names, method = "flow", model = "DDM", budgets = c("small", "medium", "large"), results_path = "results") {
  df <- data.frame()
  res_path <- results_path
  if (model == "DDM" && method == "flow") {
    col_names <- c(col_names, "logZ_converged")
    res_path <- file.path(results_path, "truncation")
  }
  
  for (b in budgets) {
    if (method == "flow") {
      p <- file.path(res_path, model, b, "flow_ll_pred.csv")
      df_temp <- read_csv(p, col_select = all_of(col_names), show_col_types = FALSE) |>
        mutate(model = model, budget = b)
      
      if (model != "DDM") 
        df_temp <- df_temp |> mutate(logZ_converged = TRUE)
      
    } else {
      p <- file.path(res_path, model, b, "kde_ll_pred.csv")
      df_temp <- read_csv(p, show_col_types = FALSE) |>
        mutate(model = model, budget = b)
    }
    df <- rbind.data.frame(df, df_temp)
  }
  return(df)
}

param_names_DDM <- c("v", "a", "t0", "s", "Z", "SZ", "sv", "st0")

# Reading and summarising data
flow_dat <- read_preds(col_names = c("param_id", "R", "rt", "ll", "flow_ll_pred", param_names_DDM), method = "flow", model = "DDM")
kde_dat <- read_preds(col_names = c("param_id", "R", "rt", "ll", "flow_ll_pred", param_names_DDM), method = "kde", model = "DDM")

# to make left_join find matching rows
flow_dat <- flow_dat |> mutate(rt_key = round(rt, 12)) 
kde_dat  <- kde_dat  |> mutate(rt_key = round(rt, 12))

full_dat <- flow_dat |>
  left_join(kde_dat |> select(param_id, R, model, budget, rt_key, est),
            by = c("param_id", "R", "model", "budget", "rt_key")) |>
  rename(kde_ll_pred = est)
anyNA(full_dat)

# check out RT dist
ggplot(full_dat, aes(x = rt)) +
  geom_histogram()


# extract anomalous samples
anom_ll <- full_dat |>
  filter(abs(flow_ll_pred - ll) > 10 & budget == "large") # only select samples that persist in large budget
any(!anom_ll$logZ_converged)

ggplot(anom_ll, aes(x = ll)) +
  geom_point(aes(y = flow_ll_pred)) + 
  geom_point(aes(y = kde_ll_pred), color = "red")

unique(anom_ll$param_id)
unique(anom_ll$budget)


params <- anom_ll |> select(all_of(param_names_DDM)) |> unique()



transform_DDM_params <- function(dat) {
  # transform params
  dat[, c("a", "t0", "sv", "st0", "s")] <- exp(dat[, c("a", "t0", "sv", "st0", "s")])
  dat[, c("Z", "SZ")] <- apply(dat[, c("Z", "SZ")], 2, pnorm)
  dat[, "SZ"] <- 2 * dat[, "SZ"] * pmin(dat[, "Z"], 1 - dat[, "Z"])
  
  return(dat)
}



dat <- transform_DDM_params(as.data.frame(anom_ll))
levels(dat$R) <- c("lower", "upper")
params <- transform_DDM_params(as.data.frame(anom_ll |> select(all_of(param_names_DDM))))


a   <- params[, "a"] / params[, "s"]
v   <- params[, "v"] / params[, "s"]
w   <- params[, "Z"]
t0  <- params[, "t0"]
sv  <- params[, "sv"] / params[, "s"]
sw  <- params[, "SZ"]
st0 <- params[, "st0"]
response <- dat$R



new_preds <- WienR::dWDM(
  t = dat$rt, 
  response = response, 
  a = a, 
  v = v, 
  w = w, 
  t0 = t0, 
  sv = sv, 
  sw = sw, 
  st0 = st0, 
  precision = 1e-20,
  n.evals = 50000
)
ggplot(anom_ll, aes(x = new_preds$logvalue)) +
  geom_point(aes(y = flow_ll_pred)) +
  geom_point(aes(y = kde_ll_pred), color = "red") +
  geom_abline(intercept = 0, slope = 1)


##### Inspect the likelihood function around anomalous parameters #####
# The vectors a, v, w, t0, sv, sw, st0 (computed above) are the WienR-scale
# parameters, row-aligned with `dat` / `anom_ll`. Collect them per observation.
# Build a proper response factor (dat$R is numeric, so the `levels<-` on
# line 67 only attaches an attribute and does not create a factor).
response_lab <- factor(dat$R)
levels(response_lab) <- c("lower", "upper")

wien_params <- data.frame(
  param_id     = dat$param_id,
  budget       = dat$budget,
  rt           = dat$rt,
  response     = response_lab,
  ll           = dat$ll,
  flow_ll_pred = dat$flow_ll_pred,
  kde_ll_pred  = dat$kde_ll_pred,
  a = a, v = v, w = w, t0 = t0, sv = sv, sw = sw, st0 = st0
)

# One row per unique parameter set (the density curve depends only on params).
param_sets <- wien_params |>
  distinct(param_id, a, v, w, t0, sv, sw, st0)

# rt range to inspect per param_id: from just above t0 to a bit past the
# largest observed anomalous rt.
rt_ranges <- wien_params |>
  group_by(param_id) |>
  summarise(rt_hi = max(rt), .groups = "drop")
param_sets <- param_sets |> left_join(rt_ranges, by = "param_id")

# Evaluate the WienR log-density over a grid of rt for one param set / response.
eval_grid <- function(ps, response, n = 1000,
                      precision = 1e-20, n.evals = 50000) {
  rt_grid <- seq(ps$t0 + 1e-4, 1.5, length.out = n)
  res <- WienR::dWDM(
    t = rt_grid, response = response,
    a = ps$a, v = ps$v, w = ps$w, t0 = ps$t0,
    sv = ps$sv, sw = ps$sw, st0 = ps$st0,
    precision = precision, n.evals = n.evals
  )
  data.frame(param_id = ps$param_id, response = response,
             rt = rt_grid, logvalue = res$logvalue, precision = precision)
}

# Shared plot style, matching results/figures_paper/plotting.R ----------------
# (defined locally rather than sourced, so plotting.R's file I/O / ggsave side
# effects are not triggered here)
theme_results <- function() {
  theme_minimal(base_size = 13) +
    theme(
      legend.position     = "bottom",
      plot.title          = element_text(face = "bold"),
      plot.title.position = "plot",
      axis.line  = element_line(),
      axis.text  = element_text(colour = "black"),
      axis.ticks = element_line()
    )
}

# Okabe-Ito colours, consistent with the paper figures
response_pal <- c(lower = "#0072B2", upper = "#D55E00")
sweep_pal    <- c("#E69F00", "#56B4E9", "#009E73", "#CC79A7")

# Likelihood curves for both responses, with the observed anomalous points
# overlaid so the anomaly can be located on the curve.
grid_curves <- do.call(rbind, lapply(seq_len(nrow(param_sets)), function(i) {
  ps <- param_sets[i, ]
  rbind(eval_grid(ps, "lower"), eval_grid(ps, "upper"))
}))

ggplot(grid_curves, aes(rt, logvalue, color = response)) +
  geom_line(linewidth = 0.8) +
  geom_point(data = wien_params,
             aes(x = rt, y = ll, color = response), shape = 4, size = 2) +
  facet_wrap(~ param_id, scales = "free") +
  scale_color_manual(name = "Response", values = response_pal) +
  labs(title = "WienR DDM log-density around anomalous observations",
       subtitle = "lines: density over rt grid | x: observed anomalous (rt, ll)",
       x = "Response Time (s)", y = "Log Density") +
  theme_results() +
  theme(strip.text = element_text(face = "bold"))

# Precision sweep: re-evaluate the same grids at increasing precision to test
# whether the anomaly reflects numerical precision of the WienR solver.
precisions <- c(1e-4, 1e-8, 1e-12, 1e-20)
prec_curves <- do.call(rbind, lapply(precisions, function(pr) {
  do.call(rbind, lapply(seq_len(nrow(param_sets)), function(i) {
    ps <- param_sets[i, ]
    rbind(eval_grid(ps, "lower", precision = pr),
          eval_grid(ps, "upper", precision = pr))
  }))
}))
prec_curves$precision <- factor(format(prec_curves$precision, scientific = TRUE))

# One plot per parameter set, two panels (lower / upper) per response.
prec_plots <- lapply(unique(prec_curves$param_id), function(pid) {
  ggplot(prec_curves |> filter(param_id == pid),
         aes(rt, logvalue, color = precision)) +
    geom_line(linewidth = 0.8, alpha = 0.8) +
    geom_point(data = wien_params |> filter(param_id == pid),
               aes(x = rt, y = ll), color = "black", shape = 4, size = 1.5,
               inherit.aes = FALSE) +
    facet_wrap(~ response, ncol = 2, 
               labeller = labeller(response = c(upper = "Response 2", lower = "Response 1"))) +
    scale_color_manual(name = "Precision", values = sweep_pal) +
    labs(# title = paste("Precision sweep — param_id", pid),
         # subtitle = "divergence between curves indicates a precision issue",
         x = "Reaction Time (s)", y = "Log-likelihood") +
    coord_cartesian(ylim = c(-50, 5), xlim = c(0, 0.6)) +
    theme_results() +
    theme(strip.text = element_text(face = "bold"))
})
for (p in prec_plots) print(p)
fig_path <- "results/figures_paper"
ggsave(file.path(fig_path, "ddm_precision.pdf"), prec_plots[[1]], device = "pdf",
       width = 6.5, height = 4, units = "in")


# n.evals sweep at the highest precision: tests whether truncation of the
# infinite series (rather than `precision` itself) drives the anomaly.
n_evals <- c(100, 1000, 10000, 50000)
neval_curves <- do.call(rbind, lapply(n_evals, function(ne) {
  do.call(rbind, lapply(seq_len(nrow(param_sets)), function(i) {
    ps <- param_sets[i, ]
    out <- rbind(eval_grid(ps, "lower", precision = 1e-20, n.evals = ne),
                 eval_grid(ps, "upper", precision = 1e-20, n.evals = ne))
    out$n.evals <- ne
    out
  }))
}))
neval_curves$n.evals <- factor(neval_curves$n.evals)

# One plot per parameter set, two panels (lower / upper) per response.
neval_plots <- lapply(unique(neval_curves$param_id), function(pid) {
  ggplot(neval_curves |> filter(param_id == pid),
         aes(rt, logvalue, color = n.evals)) +
    geom_line(linewidth = 0.8, alpha = 0.8) +
    geom_point(data = wien_params |> filter(param_id == pid),
               aes(x = rt, y = ll), color = "black", shape = 4, size = 2,
               inherit.aes = FALSE) +
    facet_wrap(~ response, ncol = 2) +
    scale_color_manual(name = "n.evals", values = sweep_pal) +
    labs(title = paste("n.evals sweep (precision = 1e-20) — param_id", pid),
         subtitle = "divergence between curves indicates series-truncation error",
         x = "Response Time (s)", y = "Log Density") +
    coord_cartesian(ylim = c(-100, 5)) +
    theme_results() +
    theme(strip.text = element_text(face = "bold"))
})
for (p in neval_plots) print(p)
# reasonably high values for n.eval lead to the same result


### LNR
param_names_LNR <- c("m_lR1", "m_lR2", "s_lR1", "s_lR2", "t0")

flow_dat <- read_preds(col_names = c("param_id", "R", "rt", "ll", "flow_ll_pred", param_names_LNR), method = "flow",  model = "LNR")
kde_dat <- read_preds(col_names = c("param_id", "R", "rt", "ll", "flow_ll_pred", param_names_LNR), method = "kde",  model = "LNR")

# to make left_join find matching rows
flow_dat <- flow_dat |> mutate(rt_key = round(rt, 12)) 
kde_dat  <- kde_dat  |> mutate(rt_key = round(rt, 12))

full_dat <- flow_dat |>
  left_join(kde_dat |> select(param_id, R, model, budget, rt_key, est),
            by = c("param_id", "R", "model", "budget", "rt_key")) |>
  rename(kde_ll_pred = est)
anyNA(full_dat)

#### Problems with approximating extreme RTs
anom_ll <- full_dat |>
  filter(abs(ll - flow_ll_pred) > 10 & budget == "large") 

ggplot(anom_ll, aes(x = ll)) +
  geom_point(aes(y = flow_ll_pred)) + 
  #geom_point(aes(y = kde_ll_pred), color = "red") +
  geom_abline(intercept = 0, slope = 1)
range(anom_ll$rt) # these are all extreme RTs

# transform params to natural scale
anom_ll[, c("s_lR1", "s_lR2", "t0")] <- exp(anom_ll[, c("s_lR1", "s_lR2", "t0")])
unique(anom_ll[, param_names_LNR])
# characterized by large means, sds and t0s

# plot RTs
lnr_rts <- ggplot(anom_ll, aes(x = rt, y = after_stat(density))) +
  geom_histogram(bins = 20, fill = "grey", color = "black") +
  theme_results() +
  coord_cartesian(xlim = c(100, 400)) +
  labs(x = "Reaction Time (s)", y = "Density")

ggsave(file.path(fig_path, "lnr_likelihood_rts.pdf"), lnr_rts, device = "pdf",
       width = 6.5, height = 4, units = "in")



#### Problems with approximating the leading edge of the distribution
# extract anomalous samples
anom_ll <- full_dat |>
  filter(abs(ll - flow_ll_pred) > 5 & abs(ll - flow_ll_pred) < 10) |>
  filter(ll > -7 & ll < 0)

ggplot(anom_ll, aes(x = ll)) +
  geom_point(aes(y = flow_ll_pred)) + 
  geom_point(aes(y = kde_ll_pred), color = "red") +
  geom_abline(intercept = 0, slope = 1)
range(anom_ll$rt)
unique(anom_ll$param_id)


##### Diagnose the anomalous LNR flow predictions #####
# The LNR likelihood is analytic, so `ll` is exact ground truth and the signed
# error is simply prediction - ll (positive = over-predicts density).
anom_diag <- anom_ll |>
  mutate(flow_err = flow_ll_pred - ll,
         kde_err  = kde_ll_pred  - ll)

# (a) Flow vs KDE error at the anomalous rows: is the flow uniquely wrong here,
#     or does KDE struggle at the same points?
anom_diag |>
  summarise(
    n             = n(),
    flow_mae      = mean(abs(flow_err)),
    kde_mae       = mean(abs(kde_err)),
    flow_worse_n  = sum(abs(flow_err) > abs(kde_err)),  # rows where flow is worse
    kde_worse_n   = sum(abs(kde_err)  > abs(flow_err))
  ) |> print()

# (b) Direction of the flow error: does the flow over- or under-predict density?
anom_diag |>
  summarise(over_predict  = sum(flow_err > 0),
            under_predict = sum(flow_err < 0),
            mean_signed   = mean(flow_err)) |> print()

ggplot(anom_diag, aes(x = flow_err)) +
  geom_histogram(bins = 30) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(title = "LNR: signed flow error at anomalous rows",
       subtitle = "flow_ll_pred - ll  ( >0 over-predicts, <0 under-predicts density )",
       x = "flow_ll_pred - ll", y = "count")

# (c) Error vs rt quantile across ALL rows, to confirm the errors concentrate in
#     the right tail. Bin rows into rt-quantile bins and take the mean error per
#     bin — far faster than geom_smooth over the full dataset, and clearer.
err_by_rt <- full_dat |>
  mutate(rt_bin = ntile(rt, 50),
         flow = abs(flow_ll_pred - ll),
         kde  = abs(kde_ll_pred  - ll)) |>
  group_by(rt_bin) |>
  summarise(rt_quantile = mean(rt),
            flow = mean(flow),
            kde  = mean(kde),
            .groups = "drop") |>
  tidyr::pivot_longer(c(flow, kde), names_to = "method", values_to = "mean_abs_err")

ggplot(err_by_rt, aes(x = rt_bin / 50, y = mean_abs_err, color = method)) +
  geom_line() +
  geom_point(size = 1) +
  labs(title = "LNR: mean absolute likelihood error by rt quantile bin",
       subtitle = "errors rising toward quantile 1 indicate a right-tail problem",
       x = "rt quantile (50 bins)", y = "mean |prediction - ll|")


##### Inspect the analytic LNR density around the anomalous points #####
# Reconstruct the exact analytic likelihood used in the simulation
# (simulation/sim_functions.R): m raw, s = exp(s_lR), t0 = exp(t0),
# ll = log(dLNR(rt, winner)) + log(1 - pLNR(rt, loser)).  Returns the total ll
# and its two components so we can see which term creates the feature the flow
# misses. `response` follows the data coding R = winning accumulator (1 or 2).
lnr_ll_curve <- function(rt, response, p) {
  s1 <- exp(p$s_lR1); s2 <- exp(p$s_lR2); t0 <- exp(p$t0)
  if (response == 1) { mw <- p$m_lR1; sw <- s1; mlo <- p$m_lR2; slo <- s2
  } else             { mw <- p$m_lR2; sw <- s2; mlo <- p$m_lR1; slo <- s1 }
  n  <- length(rt)
  pw <- cbind(m = rep(mw,  n), s = rep(sw,  n), t0 = rep(t0, n))
  pl <- cbind(m = rep(mlo, n), s = rep(slo, n), t0 = rep(t0, n))
  ll_win <- log(EMC2:::dLNR(rt, pw))        # winner log-pdf
  ll_los <- log(1 - EMC2:::pLNR(rt, pl))    # loser log-survival
  data.frame(rt = rt, response = response,
             ll = ll_win + ll_los, ll_win = ll_win, ll_los = ll_los)
}

anom_pids <- unique(anom_ll$param_id)

# (1) Analytic curve per anomalous param_id, with flow & KDE predictions (which
#     exist for every row) overlaid across the full observed rt range.
lnr_curves <- do.call(rbind, lapply(anom_pids, function(pid) {
  p      <- full_dat |> filter(param_id == pid) |> slice(1)
  rt_rng <- range(full_dat$rt[full_dat$param_id == pid])
  rt_grid <- seq(max(exp(p$t0) + 1e-4, rt_rng[1]), rt_rng[2], length.out = 800)
  rbind(lnr_ll_curve(rt_grid, 1, p), lnr_ll_curve(rt_grid, 2, p))
}))
lnr_curves$param_id <- rep(anom_pids, each = 1600)

pred_pts <- full_dat |>
  filter(param_id %in% anom_pids) |>
  transmute(param_id, response = R, rt, flow = flow_ll_pred, kde = kde_ll_pred) |>
  tidyr::pivot_longer(c(flow, kde), names_to = "method", values_to = "pred")

anom_pts <- anom_ll |>
  filter(param_id %in% anom_pids) |>
  transmute(param_id, response = R, rt, flow = flow_ll_pred, kde = kde_ll_pred) |>
  tidyr::pivot_longer(c(flow, kde), names_to = "method", values_to = "pred")

lnr_plots <- lapply(anom_pids, function(pid) {
  ggplot(lnr_curves |> filter(param_id == pid), aes(rt, ll)) +
    geom_line(linewidth = 0.8) +                                   # analytic ll
    geom_point(data = pred_pts |> filter(param_id == pid),
               aes(rt, pred, color = method), size = 0.7, alpha = 0.4,
               inherit.aes = FALSE) +
    geom_point(data = anom_pts |> filter(param_id == pid),         # the anomalies
               aes(rt, pred, color = method), shape = 1, size = 3, stroke = 1,
               inherit.aes = FALSE) +
    facet_wrap(~ response, ncol = 2, labeller = label_both) +
    labs(title = paste("LNR analytic ll vs predictions — param_id", pid),
         subtitle = "black: analytic ll | points: flow/KDE predictions | rings: anomalous rows",
         x = "rt", y = "log likelihood") +
    coord_cartesian(ylim = c(-15, 2))
})
for (p in lnr_plots) print(p)

# (2) Decompose the analytic ll into winner-pdf and loser-survival terms to see
#     which one produces the feature the flow fails to capture (e.g. a dip where
#     the winner pdf falls off while the loser is still likely to be running).
decomp <- lnr_curves |>
  tidyr::pivot_longer(c(ll, ll_win, ll_los),
                      names_to = "term", values_to = "value")

decomp_plots <- lapply(anom_pids, function(pid) {
  ggplot(decomp |> filter(param_id == pid), aes(rt, value, color = term)) +
    geom_line() +
    geom_point(data = anom_ll |> filter(param_id == pid),
               aes(rt, ll), color = "black", shape = 4, size = 3,
               inherit.aes = FALSE) +
    facet_wrap(~ response, ncol = 2, labeller = label_both) +
    labs(title = paste("LNR ll decomposition — param_id", pid),
         subtitle = "ll = ll_win (winner log-pdf) + ll_los (loser log-survival)",
         x = "rt", y = "log value") +
    coord_cartesian(ylim = c(-15, 2))
})
for (p in decomp_plots) print(p)

# (3) Characterise the anomalous rows: parameter values + signed errors, to spot
#     a shared parameter regime (e.g. well-separated accumulators, small s).
anom_table <- anom_ll |>
  transmute(param_id, response = R, rt, ll,
            flow_ll_pred, kde_ll_pred,
            flow_err = flow_ll_pred - ll, kde_err = kde_ll_pred - ll,
            m_lR1, m_lR2, s_lR1, s_lR2, t0) |>
  arrange(desc(abs(flow_err)))
print(anom_table, width = Inf)


##### Paper figure: the NLE leading-edge failure heals with training budget #####
# Styling mirrors results/figures_paper/plotting.R so this block is self-contained.
# For one anomalous parameter set with well-separated accumulators we overlay the
# exact analytic LNR likelihood (line) with the NLE and KDE predictions (points)
# across all three budgets and both responses. The minority response (top row)
# shows the NLE density collapse at the fast leading edge shrinking as budget
# grows, while KDE tracks the analytic curve throughout.
method_pal <- c(Analytical = "#009E73", NLE = "#0072B2", KDE = "#D55E00")
theme_results <- function() {
  theme_minimal(base_size = 13) +
    theme(legend.position     = "bottom",
          plot.title          = element_text(face = "bold"),
          plot.title.position  = "plot",
          axis.line           = element_line(),
          axis.text           = element_text(colour = "black"),
          axis.ticks          = element_line())
}

fig_param <- 708    # an anomalous param_id (see anom_table); change to explore others
xmax      <- 1.6    # focus the x-axis on the leading edge / bulk

fig_dat <- full_dat |> filter(param_id == fig_param)
p1      <- fig_dat |> slice(1)   # parameters are identical across budgets

# analytic curve per response (depends only on the parameters)
rt_grid   <- seq(exp(p1$t0) + 1e-4, xmax, length.out = 600)
fig_curve <- bind_rows(lapply(c(1, 2), function(r)
  data.frame(R = r, rt = rt_grid, ll = lnr_ll_curve(rt_grid, r, p1)$ll)))

lab_budget <- function(x) factor(x, levels = c("small", "medium", "large"),
                                 labels = c("Small", "Medium", "Large"))
lab_R      <- function(x) factor(paste("Response", x), levels = paste("Response", c(1, 2)))

# replicate the (budget-independent) curve across budget facets
fig_curve_e <- fig_curve |> tidyr::crossing(budget = c("small", "medium", "large")) |>
  mutate(budget = lab_budget(budget), R = lab_R(R))

fig_pred <- fig_dat |>
  transmute(budget, R, rt, NLE = flow_ll_pred, KDE = kde_ll_pred) |>
  tidyr::pivot_longer(c(NLE, KDE), names_to = "method", values_to = "ll") |>
  mutate(method = factor(method, levels = c("NLE", "KDE")),
         budget = lab_budget(budget), R = lab_R(R))

p_edge_budget <- ggplot() +
  geom_line(data = fig_curve_e, aes(rt, ll, colour = "Analytical"), linewidth = 0.9) +
  geom_line(data = fig_pred, aes(rt, ll, colour = method)) +
  facet_grid(R ~ budget) +
  coord_cartesian(xlim = c(0, xmax), ylim = c(-13, 3)) +
  scale_colour_manual(name = "Method", values = method_pal,
                      breaks = c("Analytical", "NLE", "KDE")) +
  labs(x = "Response Time", y = "Log-likelihood",
       title = sprintf("LNR param_id %s: NLE leading-edge failure heals with budget", fig_param),
       subtitle = "Minority response (top): NLE collapses at fast RTs at Small budget, recovering the analytical curve as budget grows") +
  theme_results() +
  theme(strip.text = element_text(face = "bold")) 
p_edge_budget

ggsave("results/figures_paper/lnr_leading_edge_budget.pdf", p_edge_budget,
       device = "pdf", width = 8.5, height = 5.2, units = "in")



### RDM
param_names_RDM <- c("v_lR1", "v_lR2", "s_lR1", "s_lR2", "t0", 
                     "B_lR1", "B_lR2", "A_lR1", "A_lR2")

flow_dat <- read_preds(col_names = c("param_id", "R", "rt", "ll", "flow_ll_pred", param_names_RDM), method = "flow",  model = "RDM")
kde_dat <- read_preds(col_names = c("param_id", "R", "rt", "ll", "flow_ll_pred", param_names_RDM), method = "kde",  model = "RDM")

# to make left_join find matching rows
flow_dat <- flow_dat |> mutate(rt_key = round(rt, 12)) 
kde_dat  <- kde_dat  |> mutate(rt_key = round(rt, 12))

full_dat <- flow_dat |>
  left_join(kde_dat |> select(param_id, R, model, budget, rt_key, est),
            by = c("param_id", "R", "model", "budget", "rt_key")) |>
  rename(kde_ll_pred = est)
anyNA(full_dat)

#### Problems with approximating extreme RTs
anom_ll <- full_dat |>
  filter(abs(ll - flow_ll_pred) > 10 & budget == "large") 
nrow(anom_ll) # 0 anomalies found


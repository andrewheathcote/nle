##### Plots for thesis paper #####
library(ggplot2)
library(stringr)
library(jsonlite)
library(tidyr)
library(dplyr)
library(readr)

fig_path <- "results/figures_paper"

## colours
method_pal <- c(
  NLE             = "#0072B2",
  KDE              = "#D55E00",
  Numerical         = "#009E73", 
  `DDM flow`       = "#0072B2",
  `DDM classifier` = "#CC79A7"
)

model_pal <- c(
  LNR = "#E69F00",
  RDM = "#56B4E9",
  DDM = "#117733"
)

## theme
theme_results <- function() {
  theme_minimal(base_size = 13) +
    theme(
      legend.position     = "bottom",
      plot.title          = element_text(face = "bold"),
      plot.title.position = "plot",
      axis.line = element_line(),
      axis.text = element_text(colour = "black"),
      axis.ticks = element_line()
    ) 
}

## reading training logs
read_train_logs <- function(
    budgets = c("small_budget", "medium_budget", "large_budget"),
    models = c("DDM_flow", "DDM_classifier", "RDM", "LNR"),
    path = "results/model_states"
) {
  out <- list()
  for (b in budgets) {
    for(m in models) {
      p <- file.path(path, b, m, "training_log.json")
      out[[m]][[sub("_budget", "", b)]] <- fromJSON(p)
    }
  }
  return(out)
}

extract_train_times <- function(
    l,
    budgets = c("small", "medium", "large"),
    models = c("DDM_flow", "DDM_classifier", "RDM", "LNR")
) {
  df <- data.frame()
  for (b in budgets) {
    for(m in models) {
      l_temp <- l[[m]][[b]]
      min_idx <- which.min(l_temp$val_losses)
      t <- sum(c(l_temp$train_times_s[1:min_idx], l_temp$val_times_s[1:min_idx]))
      df_temp <- data.frame(model = m, budget = b, time = t)
      df <- rbind.data.frame(df, df_temp)
    }
  }
  return(df)
}

extract_losses <- function(
    l,
    budgets = c("small", "medium", "large"),
    models = c("DDM_flow", "DDM_classifier", "RDM", "LNR")
) {
  df <- data.frame()
  for (b in budgets) {
    for(m in models) {
      l_temp <- l[[m]][[b]]
      df_temp <- data.frame(model = m, budget = b, val_loss = l_temp$val_losses, 
                            train_loss = l_temp$train_losses)
      df <- rbind.data.frame(df, df_temp)
    }
  }
  return(df)
}

read_eval_times <- function(
    method = "NLE",
    n_params = c("N=100", "N=1000", "N=10000"),
    models = c("LNR", "RDM", "DDM"),
    path = "results"
) {
  df <- data.frame()
  for (m in models) {
    for (b in n_params) {
      if (method == "NLE") {
        p <- file.path(path, m, "timing", b, "flow_timing.json")
        l <- fromJSON(p)
        times_mat <- do.call(rbind, lapply(l, function(x) x$times))
        df_temp <- as.data.frame(times_mat)
      }
      
      if (method == "KDE") {
        p <- file.path(path, m, "timing", "kde_eval", b, "times_vec.csv")
        df_temp <- read.csv(p)
      }
      
      if (method == "Numerical") {
        p <- file.path(path, m, "timing", b, "times.csv")
        df_temp <- read.csv(p)
      }
      
      df_temp$method <- method
      df_temp$model <- m
      df_temp$n_params <- b
      df <- rbind(df, df_temp)
    }
  }
  return(df)
}

read_fit_times <- function(models = c("LNR", "RDM", "DDM"), path = "results") {
  df <- data.frame()
  for (m in models) {
    p <- file.path(path, m, "timing", "kde_fit", "eval_params", "times.csv")
    d <- read.csv(p)
    rowmed <- apply(d[, paste0("V", 1:10)], 1, median)
    df <- rbind(df, data.frame(model = m, fit_ms = median(rowmed) / 1e6))
  }
  df
}

read_sim_times <- function(models = c("LNR", "RDM", "DDM"),
                           n_sets = 1000,
                           path = "benchmark/scripts_julian") {
  df <- data.frame()
  for (m in models) {
    p <- file.path(path, m, "kde_train", "sim_time.RDS")
    elapsed <- as.numeric(readRDS(p)["elapsed"])  # seconds, total over n_sets
    df <- rbind(df, data.frame(model = m, sim_ms = elapsed / n_sets * 1000))
  }
  df
}


# simulation times
read_times <- function(model, budgets = c("small", "medium", "large"), path = "data/train") {
  time <- c()
  for (b in budgets) {
    p <- file.path(path, b, model, "sim_time.RDS")
    time <- c(time, readRDS(p)["elapsed"])
  }
  
  method <- ifelse(grepl("kde", path), "KDE", "NLE")
  return(data.frame(time = time, budget = budgets, 
                    model = rep(sub("_single", "", model), length(time)),
                    method = rep(method, length(time))))
}
df_sim_times_NLE <- rbind(read_times("LNR_single"), read_times("RDM_single"), read_times("DDM"))
df_sim_times_KDE <- rbind(read_times("LNR", path = "data/kde_fit"), 
                          read_times("RDM", path = "data/kde_fit"), 
                          read_times("DDM", path = "data/kde_fit"))
df_sim_times <- rbind.data.frame(df_sim_times_NLE, df_sim_times_KDE)
df_sim_times$model <- factor(df_sim_times$model, levels = c("LNR", "RDM", "DDM"))
df_sim_times$budget <- factor(df_sim_times$budget, levels = c("small", "medium", "large"))
df_sim_times$method <- factor(df_sim_times$method, levels = c("NLE", "KDE"))

p_sim_time <- ggplot(df_sim_times, aes(y = time, x = budget, fill = method)) + 
  geom_bar(stat = "identity", position = "dodge") + 
  facet_wrap(~model) +
  labs(x = "Budget", y = "Time (log10 sec)") +
  scale_fill_discrete(name = "Simulation Target", palette = method_pal) +
  scale_y_continuous(transform = "log10") +
  scale_x_discrete(labels = str_to_title) +
  theme_results() +
  theme(panel.grid.major.x = element_blank(),
        strip.text = element_text(face = "bold"))

ggsave(file.path(fig_path, "sim_time.pdf"), p_sim_time, device = "pdf",
       width = 6.5, height = 4, units = "in")

## Evalutation times
times_kde_ddm <- read.csv('results/DDM/timing/kde_eval/N=10000/times.csv')
times_kde <- unlist(times_kde_ddm)
times_nle_rdm <- fromJSON('results/DDM/timing/N=10000/flow_timing.json', simplifyVector = TRUE)
times_nle <- as.vector(vapply(times_nle_rdm, function(e) e$times, numeric(10)))

times_numerical <- unlist(read.csv('results/DDM/timing/N=10000/times.csv'))


times_df <- data.frame(time = c(times_kde, times_nle, times_numerical),
                       method = rep(c("KDE", "NLE", "Approx"), each = length(times_kde)))

ggplot(times_df, aes(y = time, fill = method)) +
  geom_boxplot() +
  scale_y_continuous(transform = "log10") +
  theme_results() + 
  scale_fill_discrete(name = "Method", palette = method_pal) +
  theme(axis.ticks.x = element_blank(),
        axis.text.x = element_blank())

## scatter plots
source('results/tables_paper/tables.R')
set.seed(1)

# base pool of converged trials, keyed for matching across budgets
scatter_base <- full_dat |>
  filter(logZ_converged == TRUE) |>
  select(all_of(c("param_id", "R", "rt_key", "ll",
                  "flow_ll_pred", "kde_ll_pred", "model", "budget")))

# sample the same 20,000 trials per model (converged in all budgets) so that
# the small/medium/large scatters show identical test points
scatter_keys <- scatter_base |>
  group_by(model, param_id, R, rt_key) |>
  filter(n_distinct(budget) == 3) |>
  ungroup() |>
  distinct(model, param_id, R, rt_key) |>
  group_by(model) |>
  sample_n(20000) |>
  ungroup()

scatter_dat <- scatter_base |>
  semi_join(scatter_keys, by = c("model", "param_id", "R", "rt_key")) |>
  select(all_of(c("ll", "flow_ll_pred", "kde_ll_pred", "model", "budget"))) |>
  pivot_longer(cols = c("flow_ll_pred", "kde_ll_pred"), values_to = "ll_pred",
               names_to = "method") |>
  mutate(method = recode(method,
                         flow_ll_pred = "NLE",
                         kde_ll_pred = "KDE")) |>
  mutate(budget = factor(budget, levels = c("small", "medium", "large"), labels = c("Small", "Medium", "Large"))) |>
  mutate(method = factor(method, levels = c("NLE", "KDE")))

# loop over models for separate plots
scatter_plts <- list()
for (m in c("DDM", "RDM", "LNR")) {
  scatter_plts[[m]] <- ggplot(scatter_dat |> filter(model == m), aes(x = ll, y = ll_pred, color = method)) + 
    geom_point(shape = 20, alpha = 0.5, size = 1) +
    geom_abline(slope = 1, intercept = 0) + 
    facet_grid(vars(method), vars(budget)) +
    labs(x = ifelse(m == "DDM", "Numerical Approximation", "Analytical Log-likelihood"), 
         y = "Predicted Log-likelihood") +
    theme_results() +
    scale_color_discrete(name = "Method", palette = method_pal) +
    theme(strip.text = element_text(face = "bold"))

  file_name <- paste0("scatter_", m, ".pdf")
  ggsave(file.path(fig_path, file_name), scatter_plts[[m]], device = "pdf",
         width = 6.5, height = 5, units = "in")
}

# Training times
train_logs <- read_train_logs()

train_times <- extract_train_times(train_logs) |> 
  separate(model, into = c("model", "head"), sep = "_", fill = "right") |>
  mutate(head = factor(str_to_title(head), levels = c("Flow", "Classifier"))) |> 
  mutate(time = time / 60) |> # minutes 
  mutate(model = factor(model, levels = c("LNR", "RDM", "DDM"))) |>
  mutate(budget = factor(budget, levels = c("small", "medium", "large")))
  

p_train_times <- ggplot(train_times, aes(x = budget, y = time, fill = head)) +
  geom_bar(stat = "identity", position = "stack") +
  facet_wrap(~model) + 
  theme_results() + 
  scale_x_discrete(labels = str_to_title) + 
  scale_fill_discrete(name = "Head", palette = c(
    Flow = "#0072B2",
    Classifier = "#B20018"
  ), na.value = "#0072B2", breaks = c("Flow", "Classifier")) +
  labs(x = "Budget", y = "Time (min)") + 
  theme(panel.grid.major.x = element_blank(),
        strip.text = element_text(face = "bold"))
  
ggsave(file.path(fig_path, "train_time.pdf"), p_train_times, device = "pdf",
       width = 6.5, height = 4, units = "in")


# Loss curves
losses <- extract_losses(train_logs) |> 
  group_by(model, budget) |>
  mutate(min_val_loss_idx = which.min(val_loss)) |>
  ungroup() |>
  pivot_longer(
    cols = c("val_loss", "train_loss"),
    names_to = "loss_type",
    values_to = "loss"
  ) |>
  mutate(loss_type = recode(loss_type,
    train_loss = "Training",
    val_loss = "Validation"
  )) |>
  group_by(model, budget, loss_type) |>
  mutate(epoch = seq_along(loss)) |>
  ungroup() |>
  mutate(model = factor(model, levels = c("LNR", "RDM", "DDM_classifier", "DDM_flow"),
                        labels = c("LNR", "RDM", "DDM Classifier", "DDM Flow"))) |>
  mutate(budget = factor(budget, levels = c("small", "medium", "large"),
                         labels = c("Small", "Medium", "Large"))) |>
  mutate(
      model_color = factor(
        case_when(
          model %in% c("DDM Flow", "DDM Classifier") ~ "DDM",
          TRUE ~ model
        ),
      levels = c("LNR", "RDM", "DDM")
    )
  )

p_loss <- ggplot(losses, aes(x = epoch, y = loss, linetype = loss_type, color = model_color)) +
  geom_line() + 
  geom_vline(aes(xintercept = min_val_loss_idx), linetype = "dashed", color = "grey") +
  facet_grid(vars(model), vars(budget), scales = "free_y") +
  theme_results() + 
  scale_color_discrete(name = "Model", palette = model_pal) + 
  scale_linetype_discrete(name = "Set") + 
  labs(x = "Epoch", y = "Loss") + 
  theme(strip.text = element_text(face = "bold"))
  
ggsave(file.path(fig_path, "losses.pdf"), p_loss, device = "pdf",
       width = 6.5, height = 7.5, units = "in")


# Evaluation times
times_nle <- read_eval_times(method = "NLE")
times_nle$rowmed <- apply(times_nle[, paste0("V", 1:10)], 1, median)

times_kde <- read_eval_times(method = "KDE")
times_kde$rowmed <- apply(times_kde[, paste0("V", 1:10)], 1, median)

times_numapprox <- read_eval_times(method = "Numerical")
times_numapprox$rowmed <- apply(times_numapprox[, paste0("V", 1:10)], 1, median)

eval_times <- rbind.data.frame(times_nle, times_kde, times_numapprox) |>
  mutate(n_params = factor(n_params, levels = c("N=100", "N=1000", "N=10000"),
                           labels = c("100", "1000", "10000"))) |>
  mutate(model = factor(model, levels = c("LNR", "RDM", "DDM"))) |> 
  mutate(method = factor(method, levels = c("Numerical", "NLE", "KDE"))) |>
  mutate(rowmed = rowmed / 1e6) # nano- to milliseconds 

# fit_times <- read_fit_times() |>
#   mutate(model = factor(model, levels = c("LNR", "RDM", "DDM")))
# 
# kde_plus_fit <- eval_times |>
#   filter(method == "KDE") |>
#   left_join(fit_times, by = "model") |>
#   mutate(rowmed = rowmed + fit_ms, method = "KDE + fit") |>
#   select(-fit_ms)
# 
# eval_times_aug <- bind_rows(eval_times, kde_plus_fit) |>
#   mutate(method = factor(method, levels = c("Numerical", "NLE", "KDE", "KDE + fit")))

sim_times <- read_sim_times() |>
  mutate(model = factor(model, levels = c("LNR", "RDM", "DDM")))

kde_plus_sim <- eval_times |>
  filter(method == "KDE") |>
  left_join(sim_times, by = "model") |>
  mutate(rowmed = rowmed + sim_ms, method = "KDE + Sim") |>
  select(-sim_ms)

eval_times_aug <- bind_rows(eval_times, kde_plus_sim) |>
  mutate(method = factor(method, levels = c("Numerical", "NLE", "KDE", "KDE + Sim")))

# old boxplot
# ggplot(eval_times, aes(x = method, y = rowmed, fill = method)) +
#   geom_violin() + 
#   facet_grid(vars(model), vars(n_params)) + 
#   theme_results() + 
#   scale_y_continuous(transform = "log10") + 
#   scale_fill_manual(name = "Method", values = method_pal) + 
#   theme(strip.text = element_text(face = "bold")) + 
#   labs(x = "Method", y = "Evaluation Time (log10 ms)")

# ggplot(eval_times_aug, aes(x = n_params, y = rowmed, color = method, group = method)) +
#   stat_summary(fun = median, geom = "line", linewidth = 1) +
#   stat_summary(fun = median, geom = "point", size = 2) +
#   stat_summary(fun.min = function(x) quantile(x, 0.05),
#                fun.max = function(x) quantile(x, 0.95),
#                geom = "ribbon", aes(fill = method), colour = NA, alpha = 0.2) +
#   # geom_point(position = "jitter", size = 1, alpha = .3) +
#   scale_y_continuous(transform = "log10") +
#   facet_wrap(~ model) + 
#   theme_results() + 
#   scale_color_manual(name = "Method",
#                      values = c(method_pal, `KDE + Sim` = "#8C3B00")) + 
#   scale_fill_manual(name = "Method",
#                      values = c(method_pal, `KDE + Sim` = "#8C3B00")) + 
#   labs(x = "Number of Parameter Sets", y = "Evaluation Time (log10 ms)") + 
#   theme(strip.text = element_text(face = "bold"))


p_eval_times <- ggplot(eval_times_aug, aes(x = n_params, y = rowmed, color = method, group = method)) +
  stat_summary(fun = median, geom = "line", linewidth = 0.8) +
  geom_boxplot(aes(group = interaction(method, n_params)),
               width = 0.3, position = "identity",
               outlier.size = 0.3, alpha = 0.25, show.legend = FALSE) +
  scale_y_continuous(transform = "log10") +
  facet_wrap(~ model) +
  theme_results() +
  scale_color_manual(name = "Method", labels = c("Numerical" = "Analytical/Numerical"),
                     values = c(method_pal, `KDE + Sim` = "#8C3B00")) +
  labs(x = "Number of Trials", y = "Evaluation Time (log10 ms)") +
  theme(strip.text = element_text(face = "bold"))

ggsave(file.path(fig_path, "eval_times.pdf"), p_eval_times, device = "pdf",
       width = 6.5, height = 5, units = "in")


## RT distributions with overlaid likelihood approximations -------------------
# For a single test-set parameter set, plot the empirical RT distribution per
# response and overlay the analytical, KDE and NLE likelihoods. The stored
# values (ll / est / flow_ll_pred) are log defective densities f(rt, R), so we
# exponentiate and normalise by the choice proportion p(R) to obtain the
# within-response conditional density f(rt | R) that matches the per-response
# density histogram.


# Parameter columns in the prediction CSVs are on the sampled (unconstrained)
# scale. These maps give the forward transform applied at simulation time
# (see simulation/designs.R); we invert them for display. RDM is simulated with
# transforms = NULL, i.e. log on every parameter (simulation/sim_functions.R).
param_transforms <- list(
  LNR = c(m_lR1 = "identity", m_lR2 = "identity",
          s_lR1 = "log", s_lR2 = "log", t0 = "log"),
  RDM = c(v_lR1 = "log", v_lR2 = "log", B_lR1 = "log", B_lR2 = "log", t0 = "log",
          A_lR1 = "log", A_lR2 = "log", s_lR1 = "log", s_lR2 = "log"),
  DDM = c(v = "identity", a = "log", t0 = "log", s = "log",
          Z = "probit", SZ = "probit", sv = "log", st0 = "log")
)

# inverse of the sampling transform: sampled scale -> natural scale
back_transform <- function(x, transform) {
  switch(transform,
         identity = x,
         log      = exp(x),
         probit   = pnorm(x),
         stop(paste("Unknown transform:", transform)))
}

read_param_preds <- function(model, budget, param, path = "results") {
  # DDM likelihoods are stored renormalised for truncation (the 20 s cap), so
  # the overlays for DDM must come from results/truncation/DDM rather than the
  # uncorrected results/DDM directory.
  base <- if (model == "DDM") file.path(path, "truncation", model)
          else file.path(path, model)

  # read all flow columns so the model-specific parameter columns (which sit
  # between `ll` and `flow_ll_pred` and are constant per param_id) are retained
  flow <- read_csv(file.path(base, budget, "flow_ll_pred.csv"),
                   show_col_types = FALSE) |>
    filter(param_id == param) |>
    mutate(rt_key = round(rt, 12))

  kde <- read_csv(file.path(base, budget, "kde_ll_pred.csv"),
                  col_select = c("param_id", "R", "rt", "est"),
                  show_col_types = FALSE) |>
    filter(param_id == param) |>
    mutate(rt_key = round(rt, 12))

  flow |>
    left_join(kde |> select(param_id, R, rt_key, est),
              by = c("param_id", "R", "rt_key")) |>
    rename(kde_ll_pred = est)
}

plot_rt_overlay <- function(model, budget, param) {
  dat <- read_param_preds(model, budget, param)

  # model-specific parameter values (constant within a param_id): everything in
  # the flow file that is not an id/data/likelihood column. Stored on the
  # sampled scale, so back-transform to the natural scale for the label.
  non_param <- c("param_id", "S", "R", "rt", "C", "ll", "flow_ll_pred",
                 "kde_ll_pred", "rt_key", "logZ", "logZ_converged")
  param_cols <- setdiff(names(dat), non_param)
  tfs <- param_transforms[[model]][param_cols]
  param_vals <- mapply(back_transform, unlist(dat[1, param_cols, drop = FALSE]), tfs)
  param_label <- paste(sprintf("%s = %.2f", param_cols, param_vals),
                       collapse = ", ")
  print(param_label)

  # choice proportions p(R) over the observed trials of this parameter set
  props <- dat |>
    count(R, name = "n") |>
    mutate(p = n / sum(n))

  # exponentiate to densities and normalise by the choice proportion
  dens <- dat |>
    left_join(props, by = "R") |>
    mutate(
      Analytical = exp(ll)           / p,
      NLE        = exp(flow_ll_pred) / p,
      KDE        = exp(kde_ll_pred)  / p
    ) |>
    arrange(R, rt)

  lines_long <- dens |>
    select(R, rt, Analytical, NLE, KDE) |>
    pivot_longer(c(Analytical, NLE, KDE),
                 names_to = "method", values_to = "density") |>
    mutate(method = factor(method, levels = c("Analytical", "NLE", "KDE")))

  overlay_pal <- c(Analytical = unname(method_pal["Numerical"]),
                   NLE        = unname(method_pal["NLE"]),
                   KDE        = unname(method_pal["KDE"]))

  lab <- ifelse(model == "DDM", "Numerical", "Analytical")
  p <- ggplot() +
    geom_histogram(data = dat, aes(x = rt, y = after_stat(density)),
                   bins = 60, fill = "grey80", colour = NA) +
    geom_line(data = lines_long, aes(x = rt, y = density, colour = method),
              linewidth = 0.8)
    
    if (length(unique(dat$R)) > 1)
      p <- p + facet_wrap(~R, labeller = labeller(R = function(x) paste("Response", x)))
    
    p <- p + 
      scale_colour_manual(name = "Method", values = overlay_pal,
                          labels = c(lab, "NLE", "KDE")) +
      labs(x = "Reaction Time (s)", y = "Density") +
      theme_results() +
      theme(strip.text = element_text(face = "bold"))
    return(p)
}

# Example: pick a parameter set from the test set and save the figure
rt_overlay_model  <- "RDM"
rt_overlay_budget <- "large"
rt_overlay_param  <- 481

p_rt_overlay <- plot_rt_overlay(rt_overlay_model, rt_overlay_budget, rt_overlay_param)
p_rt_overlay <- p_rt_overlay + coord_cartesian(xlim = c(0, 10))
p_rt_overlay
ggsave(file.path(fig_path,
                 sprintf("rt_overlay_%s_%s_param%s.pdf",
                         rt_overlay_model, rt_overlay_budget, rt_overlay_param)),
       p_rt_overlay, device = "pdf", width = 7, height = 4, units = "in")


# Multiple panels for DDM
library(patchwork)

p1 <- plot_rt_overlay("DDM", "large", 392)
p2 <- plot_rt_overlay("DDM", "large", 942)

combined <- (p1 / p2) +                      # stack vertically; use | for side-by-side
  plot_layout(guides = "collect", axis_titles = "collect", )  &        # '&' = apply to ALL subplots
  theme(legend.position = "bottom",
        strip.text = element_text(face = "bold"))

ggsave(file.path(fig_path, "DDM_overlay_combined.pdf"),
       combined, device = "pdf", width = 7, height = 8, units = "in")


## Truncation Figure 2: error vs RT by stratum and method --------------------
# Ported from results/truncation_plots.R (Figure 2). The KDE boundary bias near
# the 20 s cap shows up as the binned-mean error drifting away from zero in the
# affected stratum. Reuses fig_path, method_pal and theme_results() defined
# above; pulls in the upstream pieces that Figure 2 depends on (logZ stratum
# table and the paired flow/KDE predictions).
CAP     <- 20
THRESH  <- 0.99
MODEL   <- "DDM"
BUDGETS <- c("small", "medium", "large")
BASE    <- file.path("results", "truncation", MODEL)

logZ <- read_csv(file.path(BASE, sprintf("logZ_by_param_cap%g.csv", CAP)),
                 show_col_types = FALSE) |>
  select(param_id, P_below_cap, converged)

# Load both methods PAIRED on the same trial. flow and KDE are stored in
# separate files whose row order is not guaranteed to align (KDE order is
# non-deterministic), so we inner-join on (param_id, R, rt_key) exactly.
load_paired <- function() {
  out <- list()
  for (b in BUDGETS) {
    flow <- read_csv(file.path(BASE, b, "flow_ll_pred.csv"),
                     show_col_types = FALSE) |>
      transmute(param_id, R, rt, rt_key = round(rt, 12),
                ll, NLE = flow_ll_pred)
    kde <- read_csv(file.path(BASE, b, "kde_ll_pred.csv"),
                    show_col_types = FALSE) |>
      transmute(param_id, R, rt_key = round(rt, 12), KDE = est)
    out[[b]] <- flow |>
      inner_join(kde, by = c("param_id", "R", "rt_key")) |>
      mutate(budget = b)
  }
  bind_rows(out) |>
    left_join(logZ, by = "param_id") |>
    filter(converged) |>
    mutate(stratum = if_else(P_below_cap > THRESH, "clean", "affected"),
           budget  = factor(budget, levels = BUDGETS,
                            labels = c("Small", "Medium", "Large")))
}

paired <- load_paired()

# samples from two DDM parameter sets are missing as they did not converge
# 2 (params) * 3 (budgets) * 2000 samples per parameter set
nrow(paired)


dat <- paired |> filter(budget == "Small") |> 
  pivot_longer(c(NLE, KDE), names_to = "method", values_to = "pred") |>
  mutate(error  = pred - ll,
         method = factor(method, levels = c("NLE", "KDE")))

# Sample trial keys from the paired (wide) table grouped by stratum only, then
# melt -- so the scattered points are the identical trials in both method
# panels. The trend (binned mean) uses the full long `dat`.

label_clean <- sprintf("p(RT < %i, c ~ '|' ~ theta) > %.2f", CAP, THRESH)
label_affected <- sprintf("p(RT < %i, c ~ '|' ~ theta) <= %.2f", CAP, THRESH)

set.seed(1)
pts <- paired |> filter(budget == "Small") |>
  group_by(stratum) |>
  sample_n(10000) |>
  ungroup() |>
  pivot_longer(c(NLE, KDE), names_to = "method", values_to = "pred") |>
  mutate(error   = pred - ll,
         method  = factor(method, levels = c("NLE", "KDE")),
         stratum = factor(stratum, levels = c("clean", "affected"),
                          labels = c(label_clean, label_affected)))

p_trunc_err <- ggplot(dat |>
                  mutate(stratum = factor(stratum, levels = c("clean", "affected"),
                                          labels = c(label_clean, label_affected))),
                aes(x = rt, y = error, colour = method)) +
  geom_point(data = pts, shape = 20, alpha = 0.5, size = 1) +
  geom_hline(yintercept = 0, colour = "grey40") +
  stat_summary_bin(fun = mean, geom = "line", bins = 50, colour = "black",
                   linewidth = 0.9) +
  geom_vline(xintercept = CAP, linetype = "dashed", colour = "grey50") +
  facet_grid(vars(stratum), vars(method), labeller = label_parsed) +
  scale_colour_manual(name = "Method", values = method_pal) +
  coord_cartesian(ylim = c(log(1e-10), 5)) +
  labs(x = "Reaction Time (s)",
       y = expression(Error ~ (hat(ll) - ll))) +
  theme_results() +
  theme(strip.text = element_text(face = "bold"))
p_trunc_err

ggsave(file.path(fig_path, "trunc_error_vs_rt_DDM.pdf"), p_trunc_err,
       device = "pdf", width = 7, height = 5.5, units = "in")



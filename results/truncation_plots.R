# Step 5 - Visual confirmation of the truncation correction and KDE boundary bias.
# -----------------------------------------------------------------------------
# Two figures, mirroring results/figures_paper/plotting.R style:
#   (1) pred vs ll_trunc scatter, faceted method x budget (DDM, large budget
#       shown; reference is the truncation-consistent ll_trunc).
#   (2) error-vs-rt, error = pred - ll_trunc, faceted stratum x method, so the
#       KDE boundary bias accumulating near the 20 s cap is visible.
#
# Reads the augmented tables from results/truncation_correction.R. Writes only
# NEW files under results/figures_paper/. Run:
#   Rscript results/truncation_plots.R

suppressMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

CAP     <- 20
THRESH  <- 0.99
MODEL   <- "DDM"
BUDGETS <- c("small", "medium", "large")
BASE    <- file.path("results", "truncation", MODEL)
FIG     <- "results/figures_paper"
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)

method_pal <- c(NLE = "#0072B2", KDE = "#D55E00")

theme_results <- function() {
  theme_minimal(base_size = 13) +
    theme(legend.position = "bottom",
          plot.title = element_text(face = "bold"),
          plot.title.position = "plot",
          axis.line = element_line(),
          axis.text = element_text(colour = "black"),
          axis.ticks = element_line(),
          strip.text = element_text(face = "bold"))
}

logZ <- read_csv(file.path(BASE, sprintf("logZ_by_param_cap%g.csv", CAP)),
                 show_col_types = FALSE) %>%
  select(param_id, P_below_cap, converged)

# Load both methods PAIRED on the same trial. flow and KDE are stored in
# separate files whose row order is not guaranteed to align (KDE order is
# non-deterministic), so we inner-join on (param_id, R, rt_key) exactly as
# tables.R does. This yields one row per trial carrying *both* methods'
# predictions, so any subsample drawn from it shows the identical trials in
# both method panels.
load_paired <- function() {
  out <- list()
  for (b in BUDGETS) {
    flow <- read_csv(file.path(BASE, b, "flow_ll_pred.csv"),
                     show_col_types = FALSE) %>%
      transmute(param_id, R, rt, rt_key = round(rt, 12),
                ll, NLE = flow_ll_pred)
    kde <- read_csv(file.path(BASE, b, "kde_ll_pred.csv"),
                    show_col_types = FALSE) %>%
      transmute(param_id, R, rt_key = round(rt, 12), KDE = est)
    out[[b]] <- flow %>%
      inner_join(kde, by = c("param_id", "R", "rt_key")) %>%
      mutate(budget = b)
  }
  paired <- bind_rows(out) %>%
    left_join(logZ, by = "param_id") %>%
    filter(converged) %>%
    mutate(stratum = if_else(P_below_cap > THRESH, "clean", "affected"),
           budget  = factor(budget, levels = BUDGETS,
                            labels = c("Small", "Medium", "Large")))
  message(sprintf("paired rows: %d (any NA join: %s)",
                  nrow(paired), anyNA(paired$KDE)))
  paired
}

# Wide (one row per trial, both methods) -> long for plotting. Because we melt a
# single paired table, every (param_id, R, rt) appears once per method, i.e. the
# *same* trials in both panels.
paired <- load_paired()

dat <- paired %>%
  pivot_longer(c(NLE, KDE), names_to = "method", values_to = "pred") %>%
  mutate(error  = pred - ll,
         method = factor(method, levels = c("NLE", "KDE")))

# --- Figure 1: pred vs ll_trunc scatter (method x budget) ---------------------
# Sample trial keys from the paired (wide) table grouped by budget only, then
# melt -- so the same trials appear in the NLE and KDE rows.
set.seed(1)
scatter_dat <- paired %>%
  group_by(budget) %>%
  sample_n(10000) %>%
  ungroup() %>%
  pivot_longer(c(NLE, KDE), names_to = "method", values_to = "pred") %>%
  mutate(method = factor(method, levels = c("NLE", "KDE")))

p_scatter <- ggplot(scatter_dat, aes(x = ll, y = pred, colour = method)) +
  geom_point(shape = 20, alpha = 0.4, size = 1) +
  geom_abline(slope = 1, intercept = 0) +
  facet_grid(vars(method), vars(budget)) +
  scale_colour_manual(name = "Method", values = method_pal) +
  labs(x = "Truncation-consistent reference (ll_trunc)",
       y = "Predicted Log-likelihood",
       title = "DDM: predictions vs truncation-consistent reference") +
  theme_results()

ggsave(file.path(FIG, "trunc_scatter_DDM.pdf"), p_scatter,
       device = "pdf", width = 6.5, height = 5, units = "in")

# --- Figure 2: error vs rt, faceted stratum x method --------------------------
# Sample TRIAL KEYS from the paired (wide) table grouped by stratum only, then
# melt -- so the scattered points are the identical trials in both method
# panels. The trend (binned mean) uses the full long `dat`.
set.seed(1)
pts <- paired %>%
  group_by(stratum) %>%
  sample_n(10000) %>%
  ungroup() %>%
  pivot_longer(c(NLE, KDE), names_to = "method", values_to = "pred") %>%
  mutate(error   = pred - ll,
         method  = factor(method, levels = c("NLE", "KDE")),
         stratum = factor(stratum, levels = c("clean", "affected"),
                          labels = c("Clean (P>0.99)", "Affected (P<=0.99)")))

p_err <- ggplot(dat %>%
                  mutate(stratum = factor(stratum, levels = c("clean", "affected"),
                                          labels = c("Clean (P>0.99)", "Affected (P<=0.99)"))),
                aes(x = rt, y = error, colour = method)) +
  geom_point(data = pts, shape = 20, alpha = 0.12, size = 0.5) +
  geom_hline(yintercept = 0, colour = "grey40") +
  stat_summary_bin(fun = mean, geom = "line", bins = 40, colour = "black",
                   linewidth = 0.9) +
  geom_vline(xintercept = CAP, linetype = "dashed", colour = "grey50") +
  facet_grid(vars(stratum), vars(method)) +
  scale_colour_manual(name = "Method", values = method_pal, guide = "none") +
  coord_cartesian(ylim = c(log(1e-10), 5)) +
  labs(x = "Response time (s)",
       y = "Error  (pred - ll_trunc)",
       title = "DDM error vs RT by stratum and method",
       subtitle = "black = binned mean error | dashed = 20 s cap") +
  theme_results()

ggsave(file.path(FIG, "trunc_error_vs_rt_DDM.pdf"), p_err,
       device = "pdf", width = 7, height = 5.5, units = "in")

message("Wrote:")
message("  ", file.path(FIG, "trunc_scatter_DDM.pdf"))
message("  ", file.path(FIG, "trunc_error_vs_rt_DDM.pdf"))

# Quick numeric backup of the visual: mean error in rt bins, affected stratum.
bins <- dat %>%
  filter(stratum == "affected") %>%
  mutate(rt_bin = cut(rt, breaks = c(0, 2, 5, 10, 15, 18, 20))) %>%
  group_by(method, rt_bin) %>%
  summarise(mean_err = round(mean(error), 3), n = n(), .groups = "drop") %>%
  pivot_wider(names_from = method, values_from = c(mean_err, n))
cat("\nMean error by RT bin (affected stratum):\n")
print(as.data.frame(bins), row.names = FALSE)

###### Numerical results for tables #######
library(readr)
library(dplyr)
library(tidyr)


## Helpers
read_preds <- function(method = "flow", models = c("LNR", "RDM", "DDM"), budgets = c("small", "medium", "large"), results_path = "results") {
  df <- data.frame()
  for (m in models) {
    col_names <- c("param_id", "R", "rt", "ll", "flow_ll_pred")
    res_path <- results_path
    if (m == "DDM" && method == "flow") {
      col_names <- c(col_names, "logZ_converged")
      res_path <- file.path(results_path, "truncation")
    }
    for (b in budgets) {
      if (method == "flow") {
        p <- file.path(res_path, m, b, "flow_ll_pred.csv")
        df_temp <- read_csv(p, col_select = all_of(col_names), show_col_types = FALSE) |>
          mutate(model = m, budget = b)
        
        if (m != "DDM") 
          df_temp <- df_temp |> mutate(logZ_converged = TRUE)
        
      } else {
        p <- file.path(res_path, m, b, "kde_ll_pred.csv")
        df_temp <- read_csv(p, show_col_types = FALSE) |>
          mutate(model = m, budget = b)
      }
      df <- rbind.data.frame(df, df_temp)
    }
  }
  return(df)
}

MSE <- function(y, x) {
  mean((y - x)^2)
}

KLD <- function(y, x) {
  mean(y - x)
}

# Reading and summarising data
flow_dat <- read_preds(method = "flow")
kde_dat <- read_preds(method = "kde")

# to make left_join find matching rows
flow_dat <- flow_dat |> mutate(rt_key = round(rt, 12)) 
kde_dat  <- kde_dat  |> mutate(rt_key = round(rt, 12))

full_dat <- flow_dat |>
  left_join(kde_dat |> select(param_id, R, model, budget, rt_key, est),
            by = c("param_id", "R", "model", "budget", "rt_key")) |>
  rename(kde_ll_pred = est)
anyNA(full_dat)

# compute stats
tab_stats <- full_dat |>
  filter(logZ_converged == TRUE) |>
  group_by(model, budget) |>
  summarise(
    cor_flow = cor(flow_ll_pred, ll),
    cor_kde = cor(kde_ll_pred, ll),
    MSE_flow = MSE(flow_ll_pred, ll),
    MSE_kde = MSE(kde_ll_pred, ll),
    KLD_flow = KLD(ll, flow_ll_pred),
    KLD_kde = KLD(ll, kde_ll_pred)
  )
write.csv(tab_stats, "results/tables_paper/approx_stats.csv", row.names = FALSE)


library(arrow)
library(dplyr)

transform_LNR_params <- function(dat) {
  # transform params
  dat[, c("s_lR1", "s_lR2", "t0")] <- exp(dat[, c("s_lR1", "s_lR2", "t0")])
  return(dat)
}

get_wl_dfs <- function(dat, params) {
  param_names <- colnames(params)
  params <- as.numeric(params)
  names(params) <- param_names
  
  winner_df <- dat |>
    mutate(m = case_when(
      R == 1 ~ params["m_lR1"],
      R == 2 ~ params["m_lR2"]
    )) |>
    mutate(s = case_when(
      R == 1 ~ params["s_lR1"],
      R == 2 ~ params["s_lR2"]
    )) |>
    mutate(rt = rt - t0) |>
    select(c("rt", "m", "s"))
  
  loser_df <- dat |>
    mutate(m = case_when(
      R == 1 ~ params["m_lR2"],
      R == 2 ~ params["m_lR1"]
    )) |>
    mutate(s = case_when(
      R == 1 ~ params["s_lR2"],
      R == 2 ~ params["s_lR1"]
    )) |>
    mutate(rt = rt - t0) |>
    select(c("rt", "m", "s"))
  
  return(list(
    "winner" = winner_df,
    "loser" = loser_df
  ))
}

dat <- open_dataset('benchmark/scripts_julian/LNR/N=1000/trials') |> collect()
dat <- transform_LNR_params(dat)
levels(dat$R) <- c("lower", "upper")

# parameters to be evaluated are the same across budgets
params <- read.csv("benchmark/scripts_julian/LNR/eval_params.csv")
params <- transform_LNR_params(params)

# sanity check
wl_dfs <- get_wl_dfs(dat, dat[1, c("m_lR1", "m_lR2", "s_lR1", "s_lR2", "t0")])
winner_df <- wl_dfs$winner
loser_df <- wl_dfs$loser

ll_d <- stats::dlnorm(winner_df$rt, winner_df$m, winner_df$s, log = TRUE)
ll_p <- log(1 - stats::plnorm(loser_df$rt, loser_df$m, loser_df$s))
ll_pred <- ll_d + ll_p 

all(near(ll_pred, dat$ll)) # TRUE
plot(ll_pred, dat$ll)
abline(0, 1, col = 'red')


# Timing of the function call
N_EVAL <- 10

budgets <- c(
  "N=100" = 100,
  "N=1000" = 1000,
  "N=10000" = 10000
)
budget_names <- names(budgets)

for (bname in budget_names) {

  dat <- open_dataset(
    file.path('benchmark/scripts_julian/LNR', bname, 'trials')
  ) |> collect()
  dat <- transform_LNR_params(dat)
  levels(dat$R) <- c("lower", "upper")
  rts <- dat$rt

  times <- matrix(nrow = nrow(params), ncol = N_EVAL)
  ll_preds <- matrix(nrow = nrow(params), ncol = nrow(dat))

  for (i in 1:nrow(params)) {
    print(i)
    wl_dfs <- get_wl_dfs(dat, params[i, ])
    winner_df <- wl_dfs$winner
    loser_df <- wl_dfs$loser
    
    m_winner <- winner_df$m
    s_winner <- winner_df$s
    m_loser <- loser_df$m
    s_loser <- loser_df$s
    
    mb <- microbenchmark::microbenchmark(
      ll_preds[i, ] <- 
        stats::dlnorm(rts, m_winner, s_winner, log = TRUE) +
        log(1 - stats::plnorm(rts, m_loser, s_loser)),
      times = N_EVAL
    )
    times[i, ] <- mb$time # in nanoseconds
  } 
  
  out_dir <- file.path("results/LNR/timing", bname)
  dir.create(out_dir, recursive = TRUE)
  
  ll_preds <- ifelse(ll_preds < log(1e-10), log(1e-10), ll_preds)
  write.csv(times, file.path(out_dir, "times.csv"), row.names = FALSE)
  write.csv(ll_preds, file.path(out_dir, "lls.csv"), row.names = FALSE)
}





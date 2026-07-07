library(EMC2)
library(arrow)
library(dplyr)

get_wl_dfs_RDM <- function(dat, params) {
  param_names <- colnames(params)
  params <- as.numeric(params)
  names(params) <- param_names
  
  winner_df <- dat |>
    mutate(v = case_when(
      R == 1 ~ params["v_lR1"] / params["s_lR1"],
      R == 2 ~ params["v_lR2"] / params["s_lR2"]
    )) |>
    mutate(B = case_when(
      R == 1 ~ params["B_lR1"] / params["s_lR1"],
      R == 2 ~ params["B_lR2"] / params["s_lR2"]
    )) |>
    mutate(A = case_when(
      R == 1 ~ params["A_lR1"] / params["s_lR1"],
      R == 2 ~ params["A_lR2"] / params["s_lR2"]
    )) |>
    select(c("rt", "v", "B", "A", "t0"))
  
  loser_df <- dat |>
    mutate(v = case_when(
      R == 2 ~ params["v_lR1"] / params["s_lR1"],
      R == 1 ~ params["v_lR2"] / params["s_lR2"]
    )) |>
    mutate(B = case_when(
      R == 2 ~ params["B_lR1"] / params["s_lR1"],
      R == 1 ~ params["B_lR2"] / params["s_lR2"]
    )) |>
    mutate(A = case_when(
      R == 2 ~ params["A_lR1"] / params["s_lR1"],
      R == 1 ~ params["A_lR2"] / params["s_lR2"]
    )) |>
    select(c("rt", "v", "B", "A", "t0"))
  
  return(list(
    "winner" = winner_df,
    "loser" = loser_df
  ))
}

dat <- open_dataset('benchmark/scripts_julian/RDM/N=1000/trials') |> collect()
dat[, c("v_lR1", "v_lR2", "B_lR1", "B_lR2", "t0", "A_lR1", "A_lR2", "s_lR1", "s_lR2")] <- 
  exp(dat[, c("v_lR1", "v_lR2", "B_lR1", "B_lR2", "t0", "A_lR1", "A_lR2", "s_lR1", "s_lR2")])
levels(dat$R) <- c("lower", "upper")

# parameters to be evaluated are the same across budgets
params <- read.csv("benchmark/scripts_julian/RDM/eval_params.csv")
params <- exp(params)

# sanity check
wl_dfs <- get_wl_dfs_RDM(dat, dat[1, c("v_lR1", "v_lR2", "B_lR1", "B_lR2", "t0", "A_lR1", "A_lR2", "s_lR1", "s_lR2")])
winner_df <- wl_dfs$winner
loser_df <- wl_dfs$loser

rts <- as.numeric(dat$rt)
ll_d <- log(.Call(EMC2:::`_EMC2_dWald`, 
              rts + 0, # forces a new copy
              as.numeric(winner_df$v),
              as.numeric(winner_df$B),
              as.numeric(winner_df$A),
              as.numeric(winner_df$t0)))
ll_p <- log(pmax(1 - .Call(EMC2:::`_EMC2_pWald`,
                           rts + 0,
                           as.numeric(loser_df$v),
                           as.numeric(loser_df$B),
                           as.numeric(loser_df$A),
                           as.numeric(loser_df$t0)), 1e-10))
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
    file.path('benchmark/scripts_julian/RDM', bname, 'trials')
  ) |> collect()
  dat[, c("v_lR1", "v_lR2", "B_lR1", "B_lR2", "t0", "A_lR1", "A_lR2", "s_lR1", "s_lR2")] <- 
    exp(dat[, c("v_lR1", "v_lR2", "B_lR1", "B_lR2", "t0", "A_lR1", "A_lR2", "s_lR1", "s_lR2")])
  levels(dat$R) <- c("lower", "upper")
  rts <- as.numeric(dat$rt)

  times <- matrix(nrow = nrow(params), ncol = N_EVAL)
  ll_preds <- matrix(nrow = nrow(params), ncol = nrow(dat))
  
  for (i in 1:nrow(params)) {
    print(i)
    wl_dfs <- get_wl_dfs_RDM(dat, params[i, ])
    winner_df <- wl_dfs$winner
    loser_df <- wl_dfs$loser
    
    v_winner <- as.numeric(winner_df$v)
    B_winner <- as.numeric(winner_df$B)
    A_winner <- as.numeric(winner_df$A)
    t0_winner <- as.numeric(winner_df$t0)
    
    v_loser <- as.numeric(loser_df$v)
    B_loser <- as.numeric(loser_df$B)
    A_loser <- as.numeric(loser_df$A)
    t0_loser <- as.numeric(loser_df$t0)
    
    mb <- microbenchmark::microbenchmark(
      ll_preds[i, ] <- 
        log(.Call(EMC2:::`_EMC2_dWald`, 
                  rts + 0, # forces a new copy
                  v_winner,
                  B_winner,
                  A_winner,
                  t0_winner)) + 
        log(pmax(1 - .Call(EMC2:::`_EMC2_pWald`,
                           rts + 0,
                           v_loser,
                           B_loser,
                           A_loser,
                           t0_loser), 1e-10)),
      times = N_EVAL
    )
    times[i, ] <- mb$time # in nanoseconds
  } 
  
  out_dir <- file.path("results/RDM/timing", bname)
  dir.create(out_dir, recursive = TRUE)
  
  ll_preds <- ifelse(ll_preds < log(1e-10), log(1e-10), ll_preds)
  write.csv(times, file.path(out_dir, "times.csv"), row.names = FALSE)
  write.csv(ll_preds, file.path(out_dir, "lls.csv"), row.names = FALSE)
}





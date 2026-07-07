library(arrow)
library(dplyr)
library(readr)  # write_csv: full round-trip double precision (write.csv truncates)

# fit_kde and eval_kde live in kde_functions.R so the timing script can reuse them
source("kde/kde_functions.R")

evaluate_bws <- function(train_data, test_data, bws) {

  df <- data.frame(bw = bws)
  KDE <- list()
  for (i in seq_along(bws)) {
    #print(i)
    # KDE fitting
    KDE[[i]] <- lapply(train_data, fit_kde, bw = bws[i])

    # KDE evaluation
    est <- apply(test_data, 1, eval_kde, kdes = KDE[[i]])

    df[i, 2] <- cor(est, test_data$ll)
    df[i, 3] <- mean((test_data$ll - est)^2) # MSE
    df[i, 4] <- mean(test_data$ll - est) # KL divergence
  }
  colnames(df) <- c("bw", "cor", "MSE", "KL")

  cor_ind <- which.max(df$cor)
  MSE_ind <- which.min(df$MSE)
  KL_ind  <- which.min(abs(df$KL)) # closest to 0

  message(
    sprintf("Maximum correlation (%.3f) found for bw = %.3f.\nMinimum MSE (%.3f) found for bw = %.3f.\nBest KL divergence (%.3f) for bw = %.3f.",
            df$cor[cor_ind], df$bw[cor_ind],
            df$MSE[MSE_ind], df$bw[MSE_ind],
            df$KL[KL_ind], df$bw[KL_ind])
  )

  # return KDE with best KL divergence
  best_kde <- KDE[[KL_ind]]

  return(list(df, best_kde))
}


# validation sets
data_val_DDM <- open_dataset('data/kde_fit/val/DDM/trials/') |>
  select(all_of(c("param_id", "R", "rt", "ll"))) |>
  collect() |> as.data.frame()

data_val_LNR <- open_dataset('data/kde_fit/val/LNR/trials/') |>
  select(all_of(c("param_id", "R", "rt", "ll"))) |>
  collect() |> as.data.frame()

data_val_RDM <- open_dataset('data/kde_fit/val/RDM/trials/') |>
  select(all_of(c("param_id", "R", "rt", "ll"))) |>
  collect() |> as.data.frame()

# test sets
data_test_DDM <- open_dataset('data/test/DDM/trials') |>
  select(all_of(c("param_id", "R", "rt", "ll"))) |> 
  as.data.frame()

data_test_LNR <- open_dataset('data/test/LNR/trials/') |>
  select(all_of(c("param_id", "R", "rt", "ll"))) |> 
  as.data.frame()

data_test_RDM <- open_dataset('data/test/RDM/trials/') |>
  select(all_of(c("param_id", "R", "rt", "ll"))) |> 
  as.data.frame()

budget_names <- c("small", "medium", "large")

read_existing_KDE <- TRUE

if (!read_existing_KDE) {
  for (bname in budget_names) {
    print(bname)
    #### DDM
    # read data
    # training
    data <- open_dataset(file.path("data/kde_fit", bname, "DDM/trials")) |>
      select(all_of(c("param_id", "R", "rt", "ll"))) |>
      collect() |> as.data.frame()
    
    # split based on param_id
    data_split <- split(data, data$param_id)
    
    # KDE_fits <- lapply(data_split, fit_kde)
    # apply(data, 1, eval_kde, kdes = KDE_fits)
    
    # find optimal bandwidth
    bw_eval <- evaluate_bws(data_split, data_val_DDM, bws = c(0.1, 0.01, 0.001))
    bw_eval[[1]]
    
    # Plots
    # KDE evaluation
    est <- apply(data_test_DDM, 1, eval_kde, kdes = bw_eval[[2]])
    
    # create dir
    dir.create(file.path("results/DDM", bname), recursive = TRUE)
    saveRDS(bw_eval, file.path("results/DDM", bname, "bw_eval.rds"))
    
    plot(est[1:10000], data_test_DDM$ll[1:10000]) # Shows potential bias
    abline(a = 0, b = 1, col = "red", lw = 2)
    # Save est WITH its identifying keys so the est<->truth pairing is
    # self-documenting and independent of the (non-deterministic) row order
    # open_dataset() returns. Downstream code can join on (param_id, R, rt).
    readr::write_csv(cbind(data_test_DDM[, c("param_id", "R", "rt", "ll")], est = est),
                     file.path("results/DDM", bname, "kde_ll_pred.csv"))
    
    # rt_sub <- data$rt[data$param_id == 1 & data$R == 1]
    # kde_fun <- bw_eval[[2]][[1]][[1]]
    # 
    # hist(rt_sub, freq = FALSE, main = "param_id=1, R=1")
    # xs <- seq(min(rt_sub), max(rt_sub), length.out = 500)
    # lines(xs, exp(kde_fun(xs)), col = "red")
    
    # mean((est - test$ll)^2)
    # cor(est, test$ll)
    
    ### LNR
    # read data
    # training
    data <- open_dataset(file.path("data/kde_fit", bname, "LNR/trials")) |>
      select(all_of(c("param_id", "R", "rt", "ll"))) |>
      collect() |> as.data.frame()
    
    # split based on param_id
    data_split <- split(data, data$param_id)
    
    # KDE_fits <- lapply(data_split, fit_kde)
    # apply(data, 1, eval_kde, kdes = KDE_fits)
    
    # find optimal bandwidth
    bw_eval <- evaluate_bws(data_split, data_val_LNR, bws = c(0.1, 0.01, 0.001))
    bw_eval[[1]]
    
    # Plots
    # KDE evaluation
    est <- apply(data_test_LNR, 1, eval_kde, kdes = bw_eval[[2]])
    
    # create dir
    dir.create(file.path("results/LNR", bname), recursive = TRUE)
    saveRDS(bw_eval, file.path("results/LNR", bname, "bw_eval.rds"))
    
    plot(est[1:10000], data_test_LNR$ll[1:10000]) # Shows potential bias
    abline(a = 0, b = 1, col = "red", lw = 2)
    readr::write_csv(cbind(data_test_LNR[, c("param_id", "R", "rt", "ll")], est = est),
                     file.path("results/LNR", bname, "kde_ll_pred.csv"))
    
    
    ### RDM
    # read data
    # training
    data <- open_dataset(file.path("data/kde_fit", bname, "RDM/trials")) |>
      select(all_of(c("param_id", "R", "rt", "ll"))) |>
      collect() |> as.data.frame()
    
    # split based on param_id
    data_split <- split(data, data$param_id)
    
    # KDE_fits <- lapply(data_split, fit_kde)
    # apply(data, 1, eval_kde, kdes = KDE_fits)
    
    # find optimal bandwidth
    bw_eval <- evaluate_bws(data_split, data_val_RDM, bws = c(0.1, 0.01, 0.001))
    bw_eval[[1]]
    
    # Plots
    # KDE evaluation
    est <- apply(data_test_RDM, 1, eval_kde, kdes = bw_eval[[2]])
    
    # create dir
    dir.create(file.path("results/RDM", bname), recursive = TRUE)
    saveRDS(bw_eval, file.path("results/RDM", bname, "bw_eval.rds"))
    
    plot(est[1:10000], data_test_RDM$ll[1:10000]) # Shows potential bias
    abline(a = 0, b = 1, col = "red", lw = 2)
    readr::write_csv(cbind(data_test_RDM[, c("param_id", "R", "rt", "ll")], est = est),
                     file.path("results/RDM", bname, "kde_ll_pred.csv"))
    
  }
} else {
  for (bname in budget_names) {
    print(bname)
    #### DDM
    print("DDM")
    bw_eval <- readRDS(file.path("results/DDM", bname, "bw_eval.rds"))
    
    # Plots
    # KDE evaluation
    est <- apply(data_test_DDM, 1, eval_kde, kdes = bw_eval[[2]])
    
    # create dir
    dir.create(file.path("results/DDM", bname), recursive = TRUE)
    
    plot(est[1:10000], data_test_DDM$ll[1:10000]) # Shows potential bias
    abline(a = 0, b = 1, col = "red", lw = 2)
    # Save est WITH its identifying keys so the est<->truth pairing is
    # self-documenting and independent of the (non-deterministic) row order
    # open_dataset() returns. Downstream code can join on (param_id, R, rt).
    readr::write_csv(cbind(data_test_DDM[, c("param_id", "R", "rt", "ll")], est = est),
                     file.path("results/DDM", bname, "kde_ll_pred.csv"))
    
    ### LNR
    print("LNR")
    bw_eval <- readRDS(file.path("results/LNR", bname, "bw_eval.rds"))
    # Plots
    # KDE evaluation
    est <- apply(data_test_LNR, 1, eval_kde, kdes = bw_eval[[2]])
    
    # create dir
    dir.create(file.path("results/LNR", bname), recursive = TRUE)
    saveRDS(bw_eval, file.path("results/LNR", bname, "bw_eval.rds"))
    
    plot(est[1:10000], data_test_LNR$ll[1:10000]) # Shows potential bias
    abline(a = 0, b = 1, col = "red", lw = 2)
    readr::write_csv(cbind(data_test_LNR[, c("param_id", "R", "rt", "ll")], est = est),
                     file.path("results/LNR", bname, "kde_ll_pred.csv"))
    
    
    ### RDM
    print("RDM")
    bw_eval <- readRDS(file.path("results/RDM", bname, "bw_eval.rds"))
    
    # Plots
    # KDE evaluation
    est <- apply(data_test_RDM, 1, eval_kde, kdes = bw_eval[[2]])
    
    # create dir
    dir.create(file.path("results/RDM", bname), recursive = TRUE)
    saveRDS(bw_eval, file.path("results/RDM", bname, "bw_eval.rds"))
    
    plot(est[1:10000], data_test_RDM$ll[1:10000]) # Shows potential bias
    abline(a = 0, b = 1, col = "red", lw = 2)
    readr::write_csv(cbind(data_test_RDM[, c("param_id", "R", "rt", "ll")], est = est),
                     file.path("results/RDM", bname, "kde_ll_pred.csv"))
    
  }
}




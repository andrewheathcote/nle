library(WienR)
library(arrow)
library(dplyr)

transform_DDM_params <- function(dat) {
  # transform params
  dat[, c("a", "t0", "sv", "st0", "s")] <- exp(dat[, c("a", "t0", "sv", "st0", "s")])
  dat[, c("Z", "SZ")] <- apply(dat[, c("Z", "SZ")], 2, pnorm)
  dat[, "SZ"] <- 2 * dat[, "SZ"] * pmin(dat[, "Z"], 1 - dat[, "Z"])
  
  return(dat)
}


dat <- open_dataset('benchmark/scripts_julian/DDM/N=1000/trials') |> collect()
params <- read.csv("benchmark/scripts_julian/DDM/eval_params.csv")

dat <- transform_DDM_params(as.data.frame(dat))
levels(dat$R) <- c("lower", "upper")
params <- transform_DDM_params(as.data.frame(params))


a = params[, "a"] / params[, "s"]
v = params[, "v"] / params[, "s"]
w = params[, "Z"]
t0 = params[, "t0"]
sv = params[, "sv"] / params[, "s"]
sw = params[, "SZ"]
st0 = params[, "st0"]
response <- dat$R

# dWDM defaults
precision = NULL
K = NULL
n.threads = FALSE
n.evals = 6000
resps <- ifelse(response == "lower" | response == 1, 0, 1)
n.threads = FALSE
if(is.logical(n.threads)) n.threads <- ifelse(n.threads == TRUE, 99999, 0)
if(is.null(precision)) PRECISION_FLAG <- FALSE
# dDiffusion7 reads precision[0] and K[0] in C; NULL -> length-0 vector -> OOB read -> crash
if(is.null(K)) K <- 0
if(is.null(precision)) precision <- 0

i <- 1

# sanity check: Compare to exported R function dWDM
l <- .Call(WienR:::dDiffusion7, 
           as.numeric(dat$rt), 
           as.numeric(rep(a[i], nrow(dat))), 
           as.numeric(rep(v[i], nrow(dat))),
           as.numeric(rep(t0[i], nrow(dat))), 
           as.numeric(rep(w[i], nrow(dat))), 
           as.numeric(rep(sw[i], nrow(dat))), 
           as.numeric(rep(sv[i], nrow(dat))), 
           as.numeric(rep(st0[i], nrow(dat))), 
           as.numeric(precision), 
           as.integer(resps), 
           as.integer(K), 
           as.integer(length(a)), 
           as.integer(n.threads), 
           as.integer(0),
           as.integer(n.evals),
           as.logical(PRECISION_FLAG)
)


l_dWDM <- WienR::dWDM(
  t = dat$rt,
  response = as.character(dat$R),
  a = a[i],
  v = v[i],
  w = w[i],
  t0 = t0[i],
  sv = sv[i],
  sw = sw[i],
  st0 = st0[i]
)$logvalue



out <- pmax(log(1e-10), l$logpdf)
out_dWDM <- pmax(log(1e-10), l_dWDM)

plot(out, out_dWDM)
abline(a = 0, b = 1, col = "red")
all(dplyr::near(out, out_dWDM)) # TRUE


# Timing of the C call
N_EVAL <- 10

budgets <- c(
  "N=100" = 100,
  "N=1000" = 1000,
  "N=10000" = 10000
)
budget_names <- names(budgets)

for (bname in budget_names) {

  dat <- open_dataset(
    file.path('benchmark/scripts_julian/DDM', bname, 'trials')
  ) |> collect()
  dat <- transform_DDM_params(as.data.frame(dat))
  levels(dat$R) <- c("lower", "upper")
  rts <- dat$rt
  resps <- ifelse(dat$R == "lower" | dat$R == 1, 0, 1)

  times <- matrix(nrow = length(a), ncol = N_EVAL)
  ll_preds <- matrix(nrow = length(a), ncol = nrow(dat))

  for (i in seq_along(a)) {
    print(i)
    as <- rep(a[i], nrow(dat))
    vs <- rep(v[i], nrow(dat))
    t0s <- rep(t0[i], nrow(dat))
    ws <- rep(w[i], nrow(dat))
    sws <- rep(sw[i], nrow(dat))
    svs <- rep(sv[i], nrow(dat))
    st0s <- rep(st0[i], nrow(dat))

    # TODO perhaps move the as.numeric calls out of the C call
    mb <- microbenchmark::microbenchmark(
      ll_preds[i, ] <- .Call(
        WienR:::dDiffusion7,
        as.numeric(rts),
        as.numeric(as),
        as.numeric(vs),
        as.numeric(t0s),
        as.numeric(ws),
        as.numeric(sws),
        as.numeric(svs),
        as.numeric(st0s),
        as.numeric(precision),
        as.integer(resps),
        as.integer(K),
        as.integer(length(as)),
        as.integer(n.threads),
        as.integer(0),
        as.integer(n.evals),
        as.logical(PRECISION_FLAG)
      )$logpdf,
      times = N_EVAL
    )
    times[i, ] <- mb$time # in nanoseconds
  }

  out_dir <- file.path("results/DDM/timing", bname)
  dir.create(out_dir, recursive = TRUE)

  ll_preds <- ifelse(ll_preds < log(1e-10), log(1e-10), ll_preds)
  write.csv(times, file.path(out_dir, "times.csv"), row.names = FALSE)
  write.csv(ll_preds, file.path(out_dir, "lls.csv"), row.names = FALSE)
}

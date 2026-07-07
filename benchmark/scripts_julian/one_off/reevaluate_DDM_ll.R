# ============================================================================
# ARCHIVED ONE-OFF — ALREADY APPLIED; DO NOT RE-RUN.
# This script recomputed the DDM analytic `ll` and OVERWROTE
# data/test/DDM/trials/trials_000{1,2}.parquet *in place* (see the write_parquet
# calls below). The test set data/test/DDM already carries the corrected `ll`,
# so re-running would needlessly re-write live test data. It also reads a private
# pre-fix backup under ~/Downloads only for a post-hoc sanity plot, so it is not
# portable — kept here as a record of the repair, not maintained.
# See benchmark/README.md.
if (!interactive()) stop("one-off DDM ll repair; already applied — do not re-run")
# ============================================================================

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


dat <- open_dataset('data/test/DDM/trials/') |> collect()
params <- as.matrix(dat |> select(all_of(c(names(EMC2::DDM()$transform$func)))))

dat <- transform_DDM_params(as.data.frame(dat))
levels(dat$R) <- c("lower", "upper")
params <- transform_DDM_params(as.data.frame(params))


a   <- params[, "a"] / params[, "s"]
v   <- params[, "v"] / params[, "s"]
w   <- params[, "Z"]
t0  <- params[, "t0"]
sv  <- params[, "sv"] / params[, "s"]
sw  <- params[, "SZ"]
st0 <- params[, "st0"]
response <- dat$R

# dWDM defaults
precision = NULL
K = NULL
n.evals = 6000
resps <- ifelse(response == "lower" | response == 1, 0, 1)
# dDiffusion7 reads precision[0] and K[0] in C; NULL -> length-0 vector -> OOB read -> crash
if(is.null(K)) K <- 0
PRECISION_FLAG <- !is.null(precision)
if(is.null(precision)) precision <- 0

# This WienR build has no OpenMP, so the C-level n.threads argument is ignored
# (always serial). Instead we fan out across processes with mclapply: each worker
# runs dDiffusion7 single-threaded on a contiguous slice of the rows.
library(parallel)
n_cores <- 6L
n <- length(a)

ddm_logpdf_chunk <- function(idx) {
  .Call(WienR:::dDiffusion7,
        as.numeric(dat$rt[idx]),
        as.numeric(a[idx]),
        as.numeric(v[idx]),
        as.numeric(t0[idx]),
        as.numeric(w[idx]),
        as.numeric(sw[idx]),
        as.numeric(sv[idx]),
        as.numeric(st0[idx]),
        as.numeric(precision),
        as.integer(resps[idx]),
        as.integer(K),
        as.integer(length(idx)),
        as.integer(0),          # n.threads: serial within each worker
        as.integer(0),
        as.integer(n.evals),
        as.logical(PRECISION_FLAG))$logpdf
}

# split(... cut ...) yields contiguous ascending chunks, so unlist restores row order
chunks <- split(seq_len(n), cut(seq_len(n), n_cores, labels = FALSE))
log_ll <- unlist(mclapply(chunks, ddm_logpdf_chunk, mc.cores = n_cores),
                 use.names = FALSE)

log_ll_cap <- pmax(log(1e-10), log_ll)
cor(log_ll_cap, dat$ll)
plot(log_ll_cap, dat$ll)
abline(0,1, col = 'red')

# make sure that these are aligned
dat1 <- open_dataset('data/test/DDM/trials/trials_0001.parquet') |> collect()
all(dat1 == dat[1:nrow(dat1), ])

dat2 <- open_dataset('data/test/DDM/trials/trials_0002.parquet') |> collect()
all(dat2 == dat[(nrow(dat1)+1): (nrow(dat2) + nrow(dat1)), ])

dat$ll <- log_ll_cap
write_parquet(dat[1:nrow(dat1), ], 'data/test/DDM/trials/trials_0001.parquet')
write_parquet(dat[(nrow(dat1)+1): (nrow(dat2) + nrow(dat1)), ], 'data/test/DDM/trials/trials_0002.parquet')


dat <- open_dataset('/Users/julian/Downloads/test/DDM/trials/') |> collect()
dat_new <- open_dataset('data/test/DDM/trials') |> collect()
cor(dat$ll, dat_new$ll) # should not be 1 anymore
plot(dat$ll, dat_new$ll)
abline(0,1, col = 'red')

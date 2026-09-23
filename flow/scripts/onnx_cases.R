if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
source("port/R/flow_race.R"); source("port/R/flow_ddm.R")
fl <- ddm_load("port/export/ddm_cap256w_c4_weights.json")
set.seed(99); N <- 400
PM <- c(v=1, a=log(.8), t0=log(.3), s=0, Z=qnorm(.5), SZ=qnorm(.2), sv=log(.1), st0=log(.05))
PS <- c(.6, .5, .35, 0, .4, .5, .7, .6)
rows <- NULL
for (i in 1:N) {
  th <- PM + PS * rnorm(8); th[["s"]] <- 0
  rt <- exp(th[["t0"]]) + exp(runif(1, log(0.01), log(6)))
  R <- sample(1:2, 1)
  lp <- ddm_eval(fl, th, rt, R)$log_pdf
  rows <- rbind(rows, c(th, rt = rt, choice = if (R == 2) 1 else -1, logp = lp))
}
write.csv(rows, file.path(Sys.getenv("SP"), "onnx_cases.csv"), row.names = FALSE)
cat("cases:", nrow(rows), " finite logp:", sum(is.finite(rows[,"logp"])), "\n")

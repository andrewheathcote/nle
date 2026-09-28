# Validate the R reference evaluator (port/R/flow_race.R) on cards converted from
# Malte Lueken's flows, against float64 golden vectors from his own evaluate_pdf_sf,
# and confirm an existing card (rdm_small) evaluates exactly as before.
#
# Fixtures: malte/make_fixtures.py (random-weight conditioners in three of his
# configurations) -> malte/fixtures/<name>/{card.json, golden.csv}.
#
# Run from the repo root:  Rscript port/tests/validate_malte.R
# Prints one PASS/FAIL line per check and a final ACCEPTANCE line.

source("port/R/flow_race.R")

tol <- 1e-10
fails <- 0L
line <- function(label, diff, tol) {
  ok <- is.finite(diff) && diff < tol
  cat(sprintf("%-4s %-38s max|diff| %.3e (tol %g)\n",
              if (ok) "PASS" else "FAIL", label, diff, tol))
  if (!ok) fails <<- fails + 1L
}

for (name in c("rdm_final", "rdm_plain", "crdm_final")) {
  dir <- file.path("malte", "fixtures", name)
  fl <- flow_load(file.path(dir, "card.json"))
  g <- read.csv(file.path(dir, "golden.csv"))
  th <- as.matrix(g[, fl$context_names])
  # one conditioner evaluation per unique theta, as the amortised path does
  key <- do.call(paste, c(as.data.frame(th), sep = "|"))
  d_lp <- d_lsf <- d_lsf_tail <- d_z <- 0
  for (k in unique(key)) {
    i <- which(key == k)
    ev <- flow_eval(fl, th[i[1L], ], g$dt[i])
    stopifnot(ev$in_box)
    d_lp  <- max(d_lp,  abs(ev$log_pdf - g$log_pdf[i]))
    # jax's norm.logsf is only good to ~4e-9 for z >= 20 (make_fixtures.py); there the port
    # is held to the exact log-survival recorded beside his, and his deviation is reported
    band <- g$z[i] < 20
    d_lsf <- max(d_lsf, abs(ev$log_sf - g$log_sf[i])[band])
    d_lsf_tail <- max(d_lsf_tail, abs(ev$log_sf - g$log_sf_exact[i])[!band], 0)
    d_z   <- max(d_z,   abs(ev$z - g$z[i]))
  }
  cat(sprintf("%s: %d points, %d parameter vectors, affine=%s, input_scaling=%s, %d hidden\n",
              name, nrow(g), length(unique(key)), fl$affine,
              !is.null(fl$input_scaling), length(fl$mlp$layers) - 1L))
  line(paste0(name, " log_pdf"), d_lp, tol)
  line(paste0(name, " log_sf (his, z < 20)"),  d_lsf, tol)
  line(paste0(name, " log_sf (exact, z >= 20)"), d_lsf_tail, tol)
  cat(sprintf("     info: %d points with z >= 20; his log_sf is off by up to %.1e there\n",
              sum(g$z >= 20), max(abs(g$log_sf - g$log_sf_exact))))
  line(paste0(name, " z (info)"), d_z, 1e-8)
}

# Just outside the box the evaluator refuses (pdf 0), for the natural-scale cards too
fl <- flow_load(file.path("malte", "fixtures", "crdm_final", "card.json"))
lo <- fl$bounds_natural$lower; hi <- fl$bounds_natural$upper
ok_box <- in_box(fl, lo) && in_box(fl, hi) && in_box(fl, (lo + hi) / 2)
for (j in seq_along(lo)) for (side in 1:2) {
  th <- (lo + hi) / 2
  th[j] <- if (side == 1) lo[j] - 1e-9 else hi[j] + 1e-9
  ok_box <- ok_box && !in_box(fl, th) && !flow_eval(fl, th, 1)$in_box
}
cat(sprintf("%-4s box: corners in, every face just outside refused\n",
            if (ok_box) "PASS" else "FAIL"))
if (!ok_box) fails <- fails + 1L

# rdm_small: (a) against its golden file, (b) unchanged to 0 against the outputs of the
# evaluator as it was before the Malte extension (port/tests/ref, written from HEAD~ code)
tag <- "rdm_rdm_small_e207"
fl <- flow_load(file.path("port", "export", paste0(tag, "_weights.json")))
g  <- read.csv(file.path("port", "export", paste0(tag, "_golden.csv")))
ref <- readRDS(file.path("port", "tests", "ref", paste0(tag, "_pre_malte.rds")))
stopifnot(nrow(ref) == nrow(g))
d_gold <- 0; d_ref <- 0
for (pid in unique(g$param_id)) {
  i <- which(g$param_id == pid)
  ev <- flow_eval(fl, as.numeric(g[i[1L], fl$context_names]), g$rt[i])
  d_gold <- max(d_gold, abs(ev$log_pdf - g$log_pdf[i]), abs(ev$log_sf - g$log_sf[i]))
  d_ref <- max(d_ref, abs(ev$z - ref$z[i]), abs(ev$log_pdf - ref$log_pdf[i]),
               abs(ev$cdf - ref$cdf[i]), abs(ev$log_sf - ref$log_sf[i]))
}
line("rdm_small vs its golden file", d_gold, 1e-9)
cat(sprintf("%-4s %-38s max|diff| %.3e (required exactly 0)\n",
            if (d_ref == 0) "PASS" else "FAIL", "rdm_small vs pre-extension evaluator", d_ref))
if (d_ref != 0) fails <- fails + 1L

cat(if (fails == 0L) "ACCEPTANCE PASS" else sprintf("ACCEPTANCE FAIL (%d)", fails), "\n")
quit(status = if (fails == 0L) 0L else 1L)

# Join the 16 seed-chunked replicates of a Volterra-control SBC cell (crdm_ctl_dt2e-3,
# crdm_ctl_dt5e-4) into one run_sbc()-shaped object and summarise with nn_sbc_summary().
# Each chunk_<seed>/ directory holds chunk.rds, the raw run_sbc() output of that chunk
# (list(rank, med, bias, coverage), each a $alpha matrix/data.frame of replicates x parameters).
#
#   Rscript join_control_chunks.R <cell_dir> [LIB]
a <- commandArgs(trailingOnly = TRUE)
DIR <- a[1]; LIB <- path.expand(if (length(a) > 1) a[2] else "~/R/nle-lib")
.libPaths(c(LIB, .libPaths()))
suppressPackageStartupMessages(library(EMC2))
stopifnot(startsWith(normalizePath(find.package("EMC2")), normalizePath(LIB)))

chunks <- sort(list.files(DIR, pattern = "^chunk_[0-9]+$", full.names = TRUE))
stopifnot(length(chunks) > 0)
files <- file.path(chunks, "chunk.rds")
missing <- !file.exists(files)
if (any(missing)) stop("missing chunk.rds in: ", paste(basename(chunks)[missing], collapse = ", "))
xs <- lapply(files, readRDS)
n_rep <- sapply(xs, function(x) nrow(x$rank$alpha))
cat(sprintf("joining %d chunks, replicates each: %s (total %d)\n", length(xs), paste(n_rep, collapse = ", "), sum(n_rep)))

rbind_field <- function(name, is_df) {
  parts <- lapply(xs, function(x) x[[name]]$alpha)
  out <- if (is_df) do.call(rbind, parts) else do.call(rbind, parts)
  rownames(out) <- as.character(seq_len(nrow(out)))
  out
}
joined <- list(rank = list(alpha = rbind_field("rank", FALSE)),
               med = list(alpha = rbind_field("med", FALSE)),
               bias = list(alpha = rbind_field("bias", TRUE)),
               coverage = list(alpha = rbind_field("coverage", FALSE)))
saveRDS(joined, file.path(DIR, "joined.rds"))

summ <- tryCatch(nn_sbc_summary(joined), error = function(e) { cat("nn_sbc_summary() failed:", conditionMessage(e), "\n"); NULL })
if (!is.null(summ)) {
  print(summ, digits = 3)
  saveRDS(summ, file.path(DIR, "summary.rds"))
  gate <- max(abs(summ$std_bias), na.rm = TRUE)
  cat(sprintf("\nmax |standardised bias| %.3f (gate: <= 0.5)\n", gate))
} else {
  # fallback: standardised bias, coverage, KS by hand (same definitions nn_sbc_summary documents)
  b <- joined$bias$alpha; r <- joined$rank$alpha; cvg <- joined$coverage$alpha
  sb <- colMeans(b) / apply(b, 2, sd)
  ks <- sapply(colnames(r), function(p) suppressWarnings(ks.test(r[, p], "punif"))$p.value)
  out <- data.frame(parameter = colnames(b), standardised_bias = sb, mean_bias = colMeans(b),
                     coverage = colMeans(cvg), ks_p = ks, row.names = NULL)
  print(out, digits = 3)
  saveRDS(out, file.path(DIR, "summary.rds"))
  cat(sprintf("\nmax |standardised bias| %.3f (gate: <= 0.5)\n", max(abs(sb))))
}

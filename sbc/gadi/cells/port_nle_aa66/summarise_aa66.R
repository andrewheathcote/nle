# Summary of the Port-NLE (EMC2 dev-nle 142bd497 = dev-sampler aa66e11d) RDM SBC cells, Gadi 2026-10-06.
.libPaths(c(path.expand("~/R/nle-lib"), .libPaths())); suppressPackageStartupMessages(library(EMC2))
root <- "~/Downloads/julian/nle/sbc/gadi/cells"
cells <- c("rdm_a0_N400/O", "rdm_a0_N400/HF", "rdm_o_a0_N100", "rdm_o_std_N100", "rdm_o_std_N400")
out <- do.call(rbind, lapply(cells, function(cl) do.call(rbind, lapply(c(network = "sbc_nn", control = "sbc_control"), function(f) {
  p <- file.path(path.expand(root), cl, "results_aa66", paste0(f, ".RData")); if (!file.exists(p)) return(NULL)
  e <- new.env(); load(p, e); s <- nn_sbc_summary(e$SBC)
  cbind(cell = cl, model = if (f == "sbc_nn") "network" else "control", s)
}))))
out$gate <- ifelse(abs(out$std_bias) <= .5 & out$in_envelope, "ok", "FLAG")
num <- sapply(out, is.numeric); out[num] <- lapply(out[num], signif, 3)
print(out, row.names = FALSE)
write.csv(out, file.path(path.expand(root), "port_nle_aa66", "summary_aa66.csv"), row.names = FALSE)

# Figures for the methods document.
#
#   1. screen_vs_sbc.png  -- the posterior-shift screen's prediction against the
#      SBC verdict, every parameter of every DDM flow for which both exist.
#   2. rank_ecdf_t0.png   -- the t0 rank distribution of the shipped flow against
#      its analytic control, with a pointwise 95% band.
#
#   Rscript evaluation/make_figures.R
if (dir.exists(".Rlibs")) .libPaths(c(normalizePath(".Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(ggplot2))
OUT <- "docs/figures"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

SURFACE <- "#fcfcfb"; INK <- "#1a1a19"; MUTED <- "#6b6b68"; GRID <- "#e4e4e1"
SERIES <- c("cap256w_c4 (8 par)" = "#2a78d6",      # slot 1 blue
            "z0_s503_e282 (st0 = 0)" = "#eb6834",  # slot 2 orange
            "dt_s703_e167 (decision time)" = "#1baf7a")  # slot 3 aqua
SHAPES <- c("cap256w_c4 (8 par)" = 16, "z0_s503_e282 (st0 = 0)" = 17,
            "dt_s703_e167 (decision time)" = 15)

base_theme <- theme_minimal(base_size = 11) +
  theme(panel.background = element_rect(fill = SURFACE, colour = NA),
        plot.background = element_rect(fill = SURFACE, colour = NA),
        panel.grid.major = element_line(colour = GRID, linewidth = 0.3),
        panel.grid.minor = element_blank(),
        axis.text = element_text(colour = MUTED),
        axis.title = element_text(colour = INK),
        plot.title = element_text(colour = INK, face = "bold", size = 12),
        plot.subtitle = element_text(colour = MUTED, size = 9.5),
        plot.caption = element_text(colour = MUTED, size = 8, hjust = 0),
        legend.position = "top", legend.title = element_blank(),
        legend.text = element_text(colour = INK, size = 9))

std_bias <- function(f) { x <- readRDS(f)$SBC; b <- x$bias$alpha
  sapply(colnames(b), function(p) { v <- b[, p][is.finite(b[, p])]; mean(v) / sd(v) }) }

## ---- figure 1: screen vs SBC ----------------------------------------------
cells <- list(
  list(lab = "cap256w_c4 (8 par)",
       pred = "results/score_bias/postshift_cap256w_c4.rds",
       obs  = "results/sbc/ddm_c4single/DDMnn_N400_R500.rds"),
  list(lab = "z0_s503_e282 (st0 = 0)",
       pred = "results/score_bias/postshift_z0_s503_e282_emcsim.rds",
       obs  = "results/sbc/ddm7_st0fix/DDMnn7_N400_R500.rds"),
  list(lab = "dt_s703_e167 (decision time)",
       pred = "results/score_bias/postshift_dt_s703_e167_emcsim.rds",
       obs  = "results/sbc/ddm6_st0fix/DDMnn6_N400_R500.rds"))

d <- do.call(rbind, lapply(cells, function(cc) {
  p <- readRDS(cc$pred)$sb_fl; o <- std_bias(cc$obs)
  k <- intersect(names(p), names(o))
  data.frame(flow = cc$lab, par = k, predicted = as.numeric(p[k]),
             observed = as.numeric(o[k]), stringsAsFactors = FALSE) }))
d$flow <- factor(d$flow, levels = names(SERIES))

lim <- range(c(d$predicted, d$observed)) + c(-0.05, 0.08)
lab <- d[d$par == "t0", ]
fit <- cor(d$predicted, d$observed)
mae <- mean(abs(d$predicted - d$observed))

p1 <- ggplot(d, aes(predicted, observed, colour = flow, shape = flow)) +
  geom_abline(slope = 1, intercept = 0, colour = MUTED, linewidth = 0.4,
              linetype = "22") +
  geom_point(size = 2.6, stroke = 0) +
  geom_text(data = lab, aes(label = "t0"), nudge_x = -0.035, nudge_y = 0.03,
            hjust = 1, size = 3, colour = INK, show.legend = FALSE) +
  scale_colour_manual(values = SERIES) + scale_shape_manual(values = SHAPES) +
  coord_equal(xlim = lim, ylim = lim) +
  labs(title = "A one-minute screen tracks the SBC verdict",
       subtitle = sprintf("Standardised bias, every parameter of three DDM flows,\nN = 400 and 500 replicates (r = %.2f, mean absolute error %.2f)", fit, mae),
       x = "predicted by the posterior-shift screen",
       y = "observed by SBC",
       caption = paste("Dashed line is equality. t0, the parameter that drives every",
                       "DDM verdict, is labelled.\nThe screen costs ~1 min per",
                       "7-parameter flow against ~3 h for an SBC cell.")) +
  base_theme
ggsave(file.path(OUT, "screen_vs_sbc.png"), p1, width = 6.6, height = 5.2, dpi = 200)

## ---- figure 2: t0 rank ECDF ------------------------------------------------
ranks <- function(f, p = "t0") { x <- readRDS(f)$SBC; r <- x$rank$alpha[, p]
  r <- r[is.finite(r)]; sort(r / max(r)) }
fl <- ranks("results/sbc/ddm_c4single/DDMnn_N400_R500.rds")
an <- ranks("results/sbc/results_analytic_N400/DDM_N400_R500.rds")
grid <- seq(0, 1, length.out = 201)
ec <- function(v) sapply(grid, function(g) mean(v <= g))
band <- data.frame(u = grid,
                   lo = qbinom(.025, length(fl), grid) / length(fl),
                   hi = qbinom(.975, length(fl), grid) / length(fl))
e <- rbind(data.frame(u = grid, F = ec(fl), which = "flow cap256w_c4"),
           data.frame(u = grid, F = ec(an), which = "analytic control"))
e$which <- factor(e$which, levels = c("flow cap256w_c4", "analytic control"))

p2 <- ggplot() +
  geom_ribbon(data = band, aes(u, ymin = lo, ymax = hi), fill = GRID, alpha = .9) +
  geom_abline(slope = 1, intercept = 0, colour = MUTED, linewidth = .4, linetype = "22") +
  geom_step(data = e, aes(u, F, colour = which), linewidth = .7) +
  scale_colour_manual(values = c("flow cap256w_c4" = unname(SERIES[1]),
                                 "analytic control" = MUTED)) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(title = "Where the DDM flow's t0 calibration actually fails",
       subtitle = "Rank ECDF at N = 400, 500 replicates; the grey band is the\npointwise 95% interval under uniformity",
       x = "normalised rank of the truth in its posterior", y = "empirical CDF",
       caption = paste("The flow's ranks bow above the diagonal: the truth sits low in",
                       "the posterior,\ni.e. t0 is overestimated (KS p < 1e-6). The exact",
                       "likelihood on the same\ndesign stays inside the band (p = 0.40).",
                       "Coverage is nominal for both --\nintervals are valid; the point",
                       "estimate carries a 0.28% offset.")) +
  base_theme
ggsave(file.path(OUT, "rank_ecdf_t0.png"), p2, width = 6.6, height = 5.2, dpi = 200)

cat(sprintf("screen vs SBC: n = %d points, r = %.3f, mean |error| = %.3f, max |error| = %.3f\n",
            nrow(d), fit, mae, max(abs(d$predicted - d$observed))))
cat(sprintf("KS p: flow %.2g, analytic %.2g\n",
            suppressWarnings(ks.test(fl, "punif")$p.value),
            suppressWarnings(ks.test(an, "punif")$p.value)))
cat("wrote", list.files(OUT, full.names = TRUE), sep = "\n  ")
cat("\n")

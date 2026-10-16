# Cover and per-case title pages for sc_s1106_e245_SBC_6cases.pdf (2026-10-16). Usage: Rscript make_pdf_pages.R <outdir>;
# then pdfunite cover.pdf t2.pdf <sv0 ecdf_nn.pdf> t3.pdf <SZ0 ...> ... t8.pdf <full ...> out.pdf
S <- commandArgs(TRUE)[1]
L <- readRDS("sbc/aws/scaled_sbc/run1/table_8case.rds")$long; L <- L[L$card == "sc_s1106_e245", ]
C <- c(`2` = "sv0|st0 + SZ free (sv = 0)", `3` = "SZ0|st0 + sv free (SZ = 0)", `4` = "st00|sv + SZ free (st0 = 0)",
       `5` = "st0only|st0 free (sv = SZ = 0)", `7` = "SZonly|SZ free (sv = st0 = 0)", `8` = "full|all free")
pdf(file.path(S, "cover.pdf"), 11, 8.5)
plot.new(); text(.5, .95, "Scaled DDM network sc_s1106_e245: SBC, six variability cases", cex = 1.6, font = 2)
text(.5, .89, "EMC2 dev-nle 142bd497, N = 400 trials, 500 replicates per case, AWS run1 2026-10-15", cex = 1)
text(.5, .85, "Each case: a title page with the summary, then the rank-ECDF difference plots (95% envelope shaded)", cex = .9)
y <- .76; text(.05, y, "case", adj = 0, font = 2); P <- c("v","a","t0","Z","sv","SZ","st0")
for (j in seq_along(P)) text(.25 + .09 * j, y, P[j], font = 2)
text(.05, y - .03, "std bias (* = outside envelope)", adj = 0, cex = .8)
for (i in seq_along(C)) { k <- names(C)[i]; d <- L[L$case == k, ]; yy <- y - .08 * i
  text(.05, yy, paste0(k, ": ", sub(".*\\|", "", C[[k]])), adj = 0, cex = .9)
  for (j in seq_along(P)) { r <- d[d$parameter == P[j], ]
    text(.25 + .09 * j, yy, if (nrow(r)) sprintf("%+.2f%s", r$std_bias, if (!r$in_envelope) "*" else "") else "—", cex = .9) } }
text(.5, .14, sprintf("Coverage %.3f-%.3f; min KS p %.3f; 1 of 35 parameter tests outside the envelope (a, case 5)",
     min(L$coverage), max(L$coverage), min(L$ks_p)), cex = .9)
text(.5, .09, "Old networks for comparison: case 2 a, st0; 3 a, st0; 4 a, t0; 5 t0, st0; 7 Z; 8 t0, Z (worst |std bias| .15-.50)", cex = .8)
dev.off()
for (k in names(C)) { d <- L[L$case == k, ]; pdf(file.path(S, paste0("t", k, ".pdf")), 11, 8.5); plot.new()
  text(.5, .7, sprintf("Case %s: %s", k, sub(".*\\|", "", C[[k]])), cex = 2, font = 2)
  text(.5, .6, sprintf("regime %s", sub("\\|.*", "", C[[k]])), cex = 1.2)
  o <- d$parameter[!d$in_envelope]
  text(.5, .45, sprintf("outside envelope: %s", if (length(o)) paste(o, collapse = ", ") else "none"), cex = 1.3)
  text(.5, .37, sprintf("worst |std bias| %.2f (%s); coverage %.3f-%.3f; min KS p %.3f",
       max(abs(d$std_bias)), d$parameter[which.max(abs(d$std_bias))], min(d$coverage), max(d$coverage), min(d$ks_p)), cex = 1.1)
  dev.off() }

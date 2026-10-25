#!/bin/sh
# Build and open handover/report_nle.pdf (the procedure after every edit of the report).
cd "$(dirname "$0")" || exit 1
pandoc -f markdown-implicit_figures report_nle.md -o report_nle.pdf --pdf-engine=xelatex \
  -V mainfont="STIXGeneral" -V monofont="Menlo" -V geometry:margin=2.2cm -V fontsize=11pt -V colorlinks=true \
  -V header-includes='\usepackage{longtable,booktabs}\renewcommand{\arraystretch}{1.15}\small' && open report_nle.pdf

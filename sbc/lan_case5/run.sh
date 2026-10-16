#!/bin/bash
# LAN case-5 SBC on the laptop, 8 cores: network 1x2, control 2x3, then the joint summary.
cd /Users/andrew/Downloads/julian/nle
export R_PROFILE_USER=/dev/null OUT=sbc/lan_case5/results
CPC=1 CFC=2 Rscript sbc/lan_case5/lan_cell.R nn > sbc/lan_case5/nn.log 2>&1 &
CPC=2 CFC=3 Rscript sbc/lan_case5/lan_cell.R control > sbc/lan_case5/control.log 2>&1 &
wait
Rscript sbc/lan_case5/lan_cell.R summary > sbc/lan_case5/summary.log 2>&1
echo "lan case5 done $(date)"

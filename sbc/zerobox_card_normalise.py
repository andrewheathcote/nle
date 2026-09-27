#!/usr/bin/env python3
"""Bring zero-box export cards to dev-nle's card contract, idempotently:
context_transforms 'asinh(x/c)' -> 'zerobox' (the constant lives in zerobox_c),
bounds_natural.lower -> 0 for the zero-box inputs (must agree with
bounds_sampled, whose lower is 0). Exports made after 2026-09-27 already
comply (export_flow_ddm.py); this repairs the ones made before.
    python3 sbc/zerobox_card_normalise.py port/export/ddm_zb_*_weights.json
"""
import json, sys
n_fixed = 0
for f in sys.argv[1:]:
    d = json.load(open(f))
    if d.get("context_encoding") != "zerobox":
        continue
    zc = d["zerobox_c"]; ctx = d["context_names"]; changed = False
    for k in zc:
        if d["context_transforms"].get(k) != "zerobox":
            d["context_transforms"][k] = "zerobox"; changed = True
        j = ctx.index(k)
        if d["bounds_natural"]["lower"][j] != 0.0:
            d["bounds_natural"]["lower"][j] = 0.0; changed = True
        assert d["bounds_sampled"]["lower"][j] == 0.0, (f, k)
    if changed:
        json.dump(d, open(f, "w"), allow_nan=False); n_fixed += 1
print(f"normalised {n_fixed} of {len(sys.argv) - 1} cards")

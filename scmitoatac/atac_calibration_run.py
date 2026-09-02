#!/usr/bin/env python
"""
scMitoATAC Phase 3.4 — CALIBRATION BACKBONE.

Takes the Phase-3.3 scored spike-in matrix (per-(cell,site) rows carrying the
injection ground truth `is_carrier` and the fused posterior `phase2_score`) and
assesses whether the posterior is CALIBRATED: does confidence x correspond to
measured precision x? Reports it SEPARATELY for clean vs artifact sites, because
the whole ATAC question is whether the site-aware fusion abstains (stays honest)
at the systematic-artifact sites rather than emitting false-confident calls on the
~1% NUMT/mapping background.

Outputs:
  * atac_calibration_summary.json  — global + per-class ECE/MCE, pre/post recalibration
  * atac_reliability.csv           — reliability-diagram bins per class (for the figure)
  * atac_calibration_stratified.csv — ECE/MCE per (site_class, tier, vaf)
Reuses scmitoatac/calibration.py unchanged.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import (reliability_diagram, ece_mce, stratified_ece,  # noqa
                         fit_per_stratum_isotonic)

VAF_BANDS = [(0.0, 0.03, "lo"), (0.03, 0.08, "mid"), (0.08, 1.01, "hi")]


def _label(long):
    """Ground-truth binary carrier label, restricted to LICENSED cells (the ones the
    tool is allowed to call): a carrier below its depth floor is routed to 'unknown'
    and is NOT a false negative. Non-carriers are always in the scoring set."""
    car = long["is_carrier"].values
    lic = long["licensed"].values
    scored = (~car) | (car & lic)
    y = (car & lic).astype(int)
    return scored, y


def _cal_for(sub, tag):
    scored, y = _label(sub)
    s = sub.loc[scored].copy()
    ys = y[scored.values] if hasattr(scored, "values") else y[scored]
    p = s["phase2_score"].values.astype(float)
    if len(p) < 20 or ys.sum() < 2 or (len(ys) - ys.sum()) < 2:
        return None
    em = ece_mce(ys, p, n_bins=10)
    rd = reliability_diagram(ys, p, n_bins=10)
    # per-stratum isotonic recalibration fit on a held-out half
    s2 = s.reset_index(drop=True); ys2 = ys.copy()
    rng = np.random.default_rng(7)
    tr = rng.random(len(s2)) < 0.5
    dtr = s2.loc[tr].copy(); dtr["_y"] = ys2[tr]
    dte = s2.loc[~tr].copy(); dte["_y"] = ys2[~tr]
    post = {"ece": None, "mce": None}
    try:
        predict, info = fit_per_stratum_isotonic(dtr, "_y", "phase2_score", "tier",
                                                 vaf_col="vaf", vaf_bands=VAF_BANDS)
        p_re = predict(dte)
        emp = ece_mce(dte["_y"].values, p_re, n_bins=10)
        post = {"ece": emp["ece"], "mce": emp["mce"]}
    except Exception as e:
        post = {"ece": None, "mce": None, "err": str(e)}
    return {"tag": tag, "n_scored": int(len(p)), "n_licensed_carriers": int(ys.sum()),
            "ece": em["ece"], "mce": em["mce"], "post_ece": post["ece"], "post_mce": post["mce"],
            "reliability": rd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True, help="atac_spiked_matrix.csv.gz (scored)")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    long = pd.read_csv(a.matrix)
    assert "phase2_score" in long.columns, "matrix has no phase2_score — rerun 3.3 benchmark"

    out = {"columns": list(long.columns), "n_rows": int(len(long)),
           "site_class_counts": long["site_class"].value_counts().to_dict()}
    rel_rows = []
    per_class = {}
    for tag in ("all", "clean", "artifact"):
        sub = long if tag == "all" else long[long["site_class"] == tag]
        r = _cal_for(sub, tag)
        if r is None:
            per_class[tag] = {"note": "insufficient licensed carriers"}
            continue
        rd = r.pop("reliability")
        for mp, of, ct in zip(rd["mean_pred"], rd["obs_freq"], rd["count"]):
            rel_rows.append({"site_class": tag, "mean_pred": mp, "obs_freq": of, "count": ct})
        per_class[tag] = r
    out["calibration"] = per_class
    pd.DataFrame(rel_rows).to_csv(os.path.join(a.outdir, "atac_reliability.csv"), index=False)

    # stratified ECE by (site_class, tier, vaf)
    strat = []
    for tag in ("clean", "artifact"):
        sub = long[long["site_class"] == tag].copy()
        scored, y = _label(sub)
        s = sub.loc[scored].copy(); s["_y"] = y[scored.values] if hasattr(scored, "values") else y[scored]
        for row in stratified_ece(s, "tier", "vaf", "_y", "phase2_score"):
            row["site_class"] = tag
            strat.append(row)
    pd.DataFrame(strat).to_csv(os.path.join(a.outdir, "atac_calibration_stratified.csv"), index=False)

    json.dump(out, open(os.path.join(a.outdir, "atac_calibration_summary.json"), "w"),
              indent=2, default=float)
    print(json.dumps({tag: {k: per_class[tag].get(k) for k in
                            ("n_scored", "n_licensed_carriers", "ece", "mce", "post_ece", "post_mce")}
                      for tag in per_class if "note" not in per_class[tag]}, indent=1))


if __name__ == "__main__":
    main()
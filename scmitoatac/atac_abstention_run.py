#!/usr/bin/env python
"""
scMitoATAC Phase 3.5 — CALLABILITY / PRINCIPLED ABSTENTION.

Given the Phase-3.4 recalibrated posterior, set an abstention threshold so that the
RETAINED calls achieve a target precision, and report the callable fraction and the
achieved precision/recall SEPARATELY for clean vs artifact sites.

The ATAC test: honest abstention must RETAIN real low-VAF carriers at clean sites
while ABSTAINING on the systematic-artifact background — i.e. the callable fraction
at artifact sites should be low precisely because those high-posterior 'calls' are
the ~1% NUMT/mapping background, not real heteroplasmy. A tool that keeps calling at
artifact sites at the same rate as clean sites would be repeating scMitoHet's error.

Method:
  * fit the per-stratum isotonic recalibration on a held-out-fit half (as 3.4),
    apply it to the eval half -> recalibrated posterior p_cal on the eval set.
  * sweep a threshold t on p_cal; a call is RETAINED (non-abstained) if p_cal >= t.
  * for the smallest t that achieves precision >= target on the EVAL set (pooled
    over classes), report per-class: callable_fraction (retained/total scored),
    precision, recall-among-licensed.
Reuses scmitoatac/calibration.py.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import fit_per_stratum_isotonic  # noqa

VAF_BANDS = [(0.0, 0.03, "lo"), (0.03, 0.08, "mid"), (0.08, 1.01, "hi")]


def _label(sub):
    car = sub["is_carrier"].values
    lic = sub["licensed"].values
    scored = (~car) | (car & lic)
    y = (car & lic).astype(int)
    return scored, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    long = pd.read_csv(a.matrix)
    scored_mask, y_all = _label(long)
    s = long.loc[scored_mask].reset_index(drop=True).copy()
    s["_y"] = y_all[scored_mask.values] if hasattr(scored_mask, "values") else y_all[scored_mask]

    # split: fit recalibration on half, evaluate abstention on the other half
    rng = np.random.default_rng(a.seed)
    fit = rng.random(len(s)) < 0.5
    dfit = s.loc[fit].copy(); dev = s.loc[~fit].reset_index(drop=True).copy()
    predict, info = fit_per_stratum_isotonic(dfit, "_y", "phase2_score", "tier",
                                             vaf_col="vaf", vaf_bands=VAF_BANDS)
    dev["p_cal"] = predict(dev)

    # sweep threshold to hit target precision on the pooled eval set
    ts = np.unique(np.round(np.concatenate([np.linspace(0, 1, 201), dev["p_cal"].values]), 4))
    chosen = None
    for t in ts:
        ret = dev["p_cal"].values >= t
        if ret.sum() == 0:
            continue
        prec = dev["_y"].values[ret].mean()
        if prec >= a.target_precision:
            chosen = float(t); break
    if chosen is None:
        chosen = float(ts[-1])

    def report(sub):
        ret = sub["p_cal"].values >= chosen
        y = sub["_y"].values
        n = len(sub); n_ret = int(ret.sum()); n_pos = int(y.sum())
        prec = float(y[ret].mean()) if n_ret else float("nan")
        rec = float(y[ret].sum() / max(n_pos, 1))  # recall among licensed carriers
        return {"n_scored": n, "n_licensed_carriers": n_pos, "n_retained": n_ret,
                "callable_fraction": n_ret / max(n, 1), "precision": prec,
                "recall_among_licensed": rec}

    def class_threshold(sub):
        # smallest t achieving target precision WITHIN this class
        tt = np.unique(np.round(np.concatenate([np.linspace(0, 1, 201), sub["p_cal"].values]), 4))
        y = sub["_y"].values
        for t in tt:
            ret = sub["p_cal"].values >= t
            if ret.sum() == 0:
                continue
            if y[ret].mean() >= a.target_precision:
                return {"threshold": float(t), "n_scored": len(sub),
                        "n_licensed_carriers": int(y.sum()), "n_retained": int(ret.sum()),
                        "callable_fraction": ret.sum() / max(len(sub), 1),
                        "precision": float(y[ret].mean()),
                        "recall_among_licensed": float(y[ret].sum() / max(int(y.sum()), 1))}
        return {"threshold": 1.01, "n_scored": len(sub), "n_licensed_carriers": int(y.sum()),
                "n_retained": 0, "callable_fraction": 0.0, "precision": float("nan"),
                "recall_among_licensed": 0.0}
    per_class_thr = {sc: class_threshold(dev[dev.site_class == sc]) for sc in ("clean", "artifact")}

    res = {"target_precision": a.target_precision, "threshold": chosen,
           "overall": report(dev),
           "clean": report(dev[dev.site_class == "clean"]),
           "artifact": report(dev[dev.site_class == "artifact"]),
           "per_class_threshold": per_class_thr}
    # by VAF within each class (retained-carrier recovery vs abstention)
    res["by_vaf"] = {}
    for sc in ("clean", "artifact"):
        d = dev[dev.site_class == sc]
        res["by_vaf"][sc] = {str(v): report(d[d.vaf == v]) for v in sorted(d.vaf.unique())}
    json.dump(res, open(os.path.join(a.outdir, "atac_abstention_summary.json"), "w"),
              indent=2, default=float)
    # tidy CSV for the figure
    rows = []
    for sc in ("clean", "artifact"):
        for v, r in res["by_vaf"][sc].items():
            rows.append({"site_class": sc, "vaf": float(v), **r})
    pd.DataFrame(rows).to_csv(os.path.join(a.outdir, "atac_abstention_by_vaf.csv"), index=False)
    print(json.dumps({k: res[k] for k in ("threshold", "overall", "clean", "artifact")}, indent=1))


if __name__ == "__main__":
    main()
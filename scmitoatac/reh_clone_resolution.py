#!/usr/bin/env python
"""
scMitoATAC Phase 4.6 — single-cell clone-resolution consistency (REH B-ALL, WGS-anchored).

SCOPE, stated up front and honestly: REH is a MONOCLONAL cell line, so it cannot test
separation BETWEEN distinct subclones (that requires a multi-clonal substrate — ccRCC /
myeloma — deferred, needs heavy staging). What REH CAN validate is the single-cell
resolution MACHINERY against a real population truth: at each WGS-confirmed heteroplasmic
site, does the per-cell carrier posterior recover a carrier FRACTION consistent with the
WGS population VAF, and how does per-cell detection power scale with depth?

For each true het site we:
  1. build the site null + mixture model from the per-cell (alt, dp) using the ported
     fusion/site_null engine (the same one used everywhere in this project),
  2. fuse a per-cell carrier posterior,
  3. compare the recovered carrier fraction (mean posterior, and threshold-called fraction)
     to the WGS truth VAF — the population-consistency check,
  4. report per-cell detection power vs depth.

Inputs: --atac-pileup, --truth (reh_sites.csv w/ wgs_vaf), --floor (error_floor.json), --outdir.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

BASES = "ACGTN"
CONTROL = set(range(1, 577)) | set(range(16024, 16570))
NSPACER = {3106, 3107}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atac-pileup", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--floor", required=True)
    ap.add_argument("--pkg", default="scmitoatac")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--vaf-lo", type=float, default=0.02)
    ap.add_argument("--vaf-hi", type=float, default=0.95)
    ap.add_argument("--k-min", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.pkg)) or ".")
    from scmitoatac import site_null as SN, fusion as FU

    floor = json.load(open(a.floor)); prior_dict = {
        "mu0": floor["cons"]["prior_mu0"], "s0": floor["cons"]["prior_s0"],
        "single_err_floor": floor["cons"]["measured_floor"]}

    truth = pd.read_csv(a.truth)
    het = truth[(truth["wgs_vaf"] >= a.vaf_lo) & (truth["wgs_vaf"] <= a.vaf_hi)
                & ~truth["pos"].isin(CONTROL) & ~truth["pos"].isin(NSPACER)]
    het_pos = sorted(het["pos"].tolist())
    truth_map = dict(zip(het["pos"], het["wgs_vaf"]))

    df = pd.read_csv(a.atac_pileup, sep="\t", compression="gzip")
    cons = [f"cons_{b}" for b in BASES if f"cons_{b}" in df.columns]
    df["cons_dp"] = df[cons].sum(axis=1)
    df = df[df["pos"].isin(het_pos)]

    rows = []
    for pos, sub in df.groupby("pos"):
        pooled = sub[cons].sum().sort_values(ascending=False)
        alt_b = pooled.index[1].replace("cons_", "") if len(pooled) > 1 else "N"
        alt = sub[f"cons_{alt_b}"].values.astype(float)
        dp = sub["cons_dp"].values.astype(float)
        keep = dp > 0
        alt, dp = alt[keep], dp[keep]
        if len(dp) < 20:
            continue
        # site null + mixture + fused posterior (the project's engine)
        sn = SN.fit_site_null(alt, dp, prior_dict)
        mix = FU.mixture_lr(alt, dp, sn["mu"], sn["s"])
        post = FU.fuse_posterior(alt, dp, sn, mix, k_min=a.k_min)["posterior"]
        power = FU.detection_power(dp, max(mix.get("mu_c", 0.05), 0.02), k_min=a.k_min)
        # recovered carrier fraction two ways: mean posterior, and threshold>0.5 fraction
        rec_frac_post = float(np.mean(post))
        rec_frac_call = float(np.mean(post > 0.5))
        # pooled VAF (population estimate) as a third recovery estimate
        pooled_vaf = float(alt.sum() / dp.sum())
        rows.append({"pos": int(pos), "n_cells": int(len(dp)), "truth_wgs_vaf": float(truth_map[pos]),
                     "pooled_atac_vaf": pooled_vaf,
                     "mean_posterior": rec_frac_post, "called_carrier_frac": rec_frac_call,
                     "median_detection_power": float(np.median(power)),
                     "median_cell_dp": float(np.median(dp))})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.outdir, "reh_clone_sites.csv"), index=False)

    # population-consistency: does pooled ATAC VAF track WGS truth? (already r~0.99 in 4.2,
    # here the point is the SINGLE-CELL posterior fraction tracking too)
    def corr(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else None
    summary = {
        "n_het_sites": int(len(res)),
        "corr_pooled_vaf_vs_truth": corr(res["pooled_atac_vaf"], res["truth_wgs_vaf"]) if len(res) else None,
        "corr_called_carrierfrac_vs_truth": corr(res["called_carrier_frac"], res["truth_wgs_vaf"]) if len(res) else None,
        "median_detection_power": float(res["median_detection_power"].median()) if len(res) else None,
        "scope_note": "REH monoclonal: single-cell resolution machinery validated against population WGS VAF; between-clone separation deferred to a multi-clonal substrate (ccRCC/myeloma).",
    }
    json.dump(summary, open(os.path.join(a.outdir, "reh_clone_summary.json"), "w"),
              indent=2, default=float)
    print(json.dumps(summary, indent=1, default=float)[:1500])


if __name__ == "__main__":
    main()
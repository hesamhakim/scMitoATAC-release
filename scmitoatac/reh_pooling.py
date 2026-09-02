#!/usr/bin/env python
"""
scMitoATAC Phase 4.5 — real-data pooling-depth recovery (REH B-ALL, WGS truth).

The Phase-4.5 PRECURSOR was a molecule-count projection: it showed the LICENSING floor
descends as k/(pooled depth) with slope ~-1 (20/20 climb, no plateau). This is the REAL-DATA
upgrade: instead of projecting from per-cell depth, we pool REAL cells into synthetic clones
of increasing size N, measure the POOLED CONSENSUS VAF at true heteroplasmic sites, and show
the estimate converges on the WGS truth with variance shrinking ~1/N and NO plateau — the
DNA-readout behaviour the project hypothesised, on real calls.

Central hypothesis (opposite the scMitoHet RNA plateau): on a DNA readout, pooling keeps
climbing toward the diagonal. We test it by measuring, at each pool size N and each true het
site, the spread of pooled-VAF estimates across many random cell subsets, and fit the
log-log variance-vs-N slope (ideal -1).

Inputs: --atac-pileup (fragment_pileup TSV.gz), --truth (reh_sites.csv with wgs_vaf per pos),
        --outdir. Truth het sites = WGS VAF in [vaf_lo, vaf_hi] and not control/N-spacer.
"""
import argparse, json, os
import numpy as np, pandas as pd

BASES = "ACGTN"
CONTROL = set(range(1, 577)) | set(range(16024, 16570))
NSPACER = {3106, 3107}


def load_cellsite(path):
    df = pd.read_csv(path, sep="\t", compression="gzip")
    # keep the columns we need: cell, pos, consensus base counts
    cons = [f"cons_{b}" for b in BASES if f"cons_{b}" in df.columns]
    df["cons_dp"] = df[cons].sum(axis=1)
    return df, cons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atac-pileup", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--vaf-lo", type=float, default=0.02)
    ap.add_argument("--vaf-hi", type=float, default=0.95)
    ap.add_argument("--pool-sizes", default="1,5,25,100,400,1600")
    ap.add_argument("--n-subsets", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    pool_sizes = [int(x) for x in a.pool_sizes.split(",")]

    truth = pd.read_csv(a.truth)
    # true het sites: WGS VAF in band, non-control, non-N-spacer, and (for REH) real variants
    het = truth[(truth["wgs_vaf"] >= a.vaf_lo) & (truth["wgs_vaf"] <= a.vaf_hi)
                & ~truth["pos"].isin(CONTROL) & ~truth["pos"].isin(NSPACER)]
    het_pos = sorted(het["pos"].tolist())
    truth_map = dict(zip(het["pos"], het["wgs_vaf"]))
    if not het_pos:
        # fall back: use mid-VAF sites the ATAC itself calls (no external truth band hit)
        raise SystemExit("no truth het sites in band")

    df, cons = load_cellsite(a.atac_pileup)
    df = df[df["pos"].isin(het_pos)].copy()
    cells = df["cell_barcode"].unique()
    n_cells = len(cells)

    # For each site, precompute per-cell alt/dp against the WGS-consistent alt base.
    # Determine alt base per site = the non-ref consensus base most abundant pooled.
    rows = []
    per_site = {}
    for pos, sub in df.groupby("pos"):
        pooled = sub[cons].sum()
        # ref = pooled majority; alt = pooled second
        order = pooled.sort_values(ascending=False)
        ref_b = order.index[0].replace("cons_", "")
        alt_b = order.index[1].replace("cons_", "") if len(order) > 1 else "N"
        # per-cell alt and dp arrays aligned to `cells`
        cell_alt = sub.set_index("cell_barcode")[f"cons_{alt_b}"].reindex(cells).fillna(0).values
        cell_dp = sub.set_index("cell_barcode")["cons_dp"].reindex(cells).fillna(0).values
        per_site[pos] = (cell_alt.astype(float), cell_dp.astype(float))

    # pooling experiment: for each pool size N, draw n_subsets random cell subsets,
    # compute pooled VAF = sum(alt)/sum(dp) per site, record spread + bias vs truth.
    for N in pool_sizes:
        if N > n_cells:
            continue
        idx_draws = [rng.choice(n_cells, size=N, replace=False) for _ in range(a.n_subsets)]
        for pos in het_pos:
            alt, dp = per_site[pos]
            vafs = []
            for ix in idx_draws:
                d = dp[ix].sum()
                vafs.append(alt[ix].sum() / d if d > 0 else np.nan)
            vafs = np.array(vafs, float); vafs = vafs[~np.isnan(vafs)]
            if len(vafs) < 5:
                continue
            rows.append({"pos": int(pos), "N": N, "truth_vaf": float(truth_map[pos]),
                         "mean_pooled_vaf": float(vafs.mean()),
                         "median_pooled_vaf": float(np.median(vafs)),
                         "var_pooled_vaf": float(vafs.var()),
                         "mean_abs_err": float(np.abs(vafs - truth_map[pos]).mean()),
                         "n_draws": int(len(vafs))})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.outdir, "reh_pooling.csv"), index=False)

    # fit log-log variance-vs-N slope per site (climb test); count climb vs plateau
    slopes = {}
    for pos, sub in res.groupby("pos"):
        sub = sub[sub["var_pooled_vaf"] > 0].sort_values("N")
        if len(sub) >= 3:
            sl = np.polyfit(np.log(sub["N"]), np.log(sub["var_pooled_vaf"]), 1)[0]
            slopes[int(pos)] = float(sl)
    sl_arr = np.array(list(slopes.values()))
    # plateau = slope shallower than -0.5 (variance stops falling); climb = <= -0.5
    summary = {
        "n_cells": int(n_cells), "n_het_sites": len(het_pos),
        "pool_sizes": pool_sizes, "n_subsets": a.n_subsets,
        "median_loglog_var_slope": float(np.median(sl_arr)) if len(sl_arr) else None,
        "n_climb": int((sl_arr <= -0.5).sum()), "n_plateau": int((sl_arr > -0.5).sum()),
        "slopes_by_site": slopes,
        # error decay: median mean_abs_err at smallest vs largest pool
        "err_decay": {str(N): float(res[res.N == N]["mean_abs_err"].median())
                      for N in pool_sizes if (res.N == N).any()},
    }
    json.dump(summary, open(os.path.join(a.outdir, "reh_pooling_summary.json"), "w"),
              indent=2, default=float)
    print(json.dumps(summary, indent=1, default=float)[:2000])


if __name__ == "__main__":
    main()
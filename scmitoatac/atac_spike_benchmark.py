#!/usr/bin/env python
"""
scMitoATAC Phase 3.3 — ATAC spike-in benchmark with a SYSTEMATIC ARTIFACT term.

This is the ATAC adaptation of the scMitoHet spike-in (scmitoatac/spike_benchmark.py).
The RNA version models the only error source as SHOT NOISE: at every site,
non-carrier alt molecules ~ Binomial(DP, e) with e the per-molecule sequencing-error
floor. That is correct for RNA, where the dominant confounder averages out with depth.

It is WRONG for ATAC, and repeating it would repeat scMitoHet's central over-optimism.
Phase 2.4 established that standard ATAC carries a class of RECURRENT, DEPTH-INVARIANT
low-VAF signals at specific positions — NUMT-derived and mapping/context artifacts (the
`phase2_atac_artifact_panel.csv`). These are real mismapped MOLECULES with a consistent
alt base, so (a) consensus does NOT remove them (they are not random sequencing error),
and (b) they do NOT average out when cells are pooled — pooling drives their apparent
variance down toward a NON-ZERO systematic floor, exactly the failure mode that made a
shot-noise-only spike-in bless 1% heteroplasmy that the data could not actually support.

So this benchmark injects TWO site classes and scores them SEPARATELY:
  * CLEAN sites (no panel, no blacklist) — shot-noise-only, as the RNA version. Measures
    the achievable sensitivity floor where the only enemy is coverage + sequencing error.
  * ARTIFACT sites (the Phase-2.4 panel) — every non-carrier cell ALSO receives a
    systematic artifact contribution artifact_alt ~ Binomial(DP, bg_vaf), bg_vaf = the
    panel's per-position median naive VAF, added to BOTH the naive and consensus matrices.
    Measures the FALSE-POSITIVE rate the systematic artifact induces, and whether a real
    carrier is still separable from the artifact background at the same site.

Ground-truth injection (both classes):
  * Real per-(cell,pos) consensus depth (n_mol) kept as-is (real depth structure).
  * Carriers among covered cells: true alt m ~ Binomial(DP, VAF).
  * Sequencing error on non-carrier molecules: naive uses the measured single-molecule
    floor; consensus uses the family-size model (small for ATAC, mean family ~1.8).
  * ARTIFACT sites additionally: every cell (carrier and non-carrier) gets
    artifact_alt ~ Binomial(DP - true_alt, bg_vaf) added to BOTH matrices.

Reuses the package scoring core unchanged (site_null / fusion / baseline), so the
consensus-advantage machinery is identical to Phase 2 — only the data-generating model
gains the systematic term.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from site_null import fit_global_prior, fit_site_null  # noqa
from fusion import mixture_lr, fuse_posterior  # noqa
from baseline import baseline_site_scores  # noqa

VAF_LADDER = [0.01, 0.02, 0.05, 0.10, 0.25]
VAF_FLOOR = 0.05  # detection targets >=5%; 1%/2% are floor-validation (as RNA version)


def read_pileup(path, chunk=None):
    """Read fragment_pileup output (TSV.gz / parquet / csv)."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    sep = "\t" if (".tsv" in path or path.endswith(".gz")) else ","
    return pd.read_csv(path, sep=sep)


def cons_error_for_family_size(f, single_err, cons_floor):
    """ATAC per-molecule consensus error given mean family size f. Singletons keep the
    single-molecule floor; families >=2 get the measured consensus floor (majority vote
    over duplicate fragments). ATAC families are small (mean ~1.8), so the gap is modest."""
    return single_err if f < 1.5 else max(cons_floor, 1e-7)


def build_spiked_matrix(df, spike_sites, single_err, cons_floor, rng,
                        vaf_ladder=VAF_LADDER, carrier_frac=0.15, min_dp=1):
    """spike_sites: DataFrame with columns pos, tier, site_class ('clean'|'artifact'),
    and (for artifact) bg_vaf. Returns long df + per-site meta."""
    out = []
    meta = {}
    tier_counter = {}
    for _, row in spike_sites.reset_index(drop=True).iterrows():
        pos = int(row["pos"]); tier = row["tier"]; sclass = row["site_class"]
        bg_vaf = float(row.get("bg_vaf", 0.0) or 0.0)
        idx_in_tier = tier_counter.get(tier, 0)
        tier_counter[tier] = idx_in_tier + 1
        vaf = vaf_ladder[idx_in_tier % len(vaf_ladder)]
        sub = df[df["pos"] == pos]
        sub = sub[sub["n_mol"] >= min_dp]
        if len(sub) < 20:
            continue
        dp = sub["n_mol"].values.astype(int)
        fsz = sub["mean_family_size"].values
        bcs = sub["cell_barcode"].values
        ncell = len(dp)
        n_car = max(3, int(round(carrier_frac * ncell)))
        car_idx = rng.choice(ncell, size=min(n_car, ncell), replace=False)
        is_car = np.zeros(ncell, bool); is_car[car_idx] = True
        # true carrier molecules ~ Binomial(DP, VAF)
        m = np.zeros(ncell, int)
        m[is_car] = rng.binomial(dp[is_car], vaf)
        # sequencing error on the remaining (non-true) molecules
        e_cons = np.array([cons_error_for_family_size(f, single_err, cons_floor) for f in fsz])
        rem = np.maximum(dp - m, 0)
        naive_alt = m + rng.binomial(rem, single_err)
        cons_alt = m + rng.binomial(rem, e_cons)
        # SYSTEMATIC ARTIFACT term (artifact sites only): a recurrent mismapped-molecule
        # background at rate bg_vaf, applied to EVERY cell and to BOTH matrices (consensus
        # does not remove it — it is a consistent alt base, not random error).
        if sclass == "artifact" and bg_vaf > 0:
            rem_art = np.maximum(dp - m, 0)
            art = rng.binomial(rem_art, bg_vaf)
            # cap so alt never exceeds DP
            naive_alt = np.minimum(naive_alt + art, dp)
            cons_alt = np.minimum(cons_alt + art, dp)
        exp_alt = dp * vaf
        licensed = exp_alt >= 2.0
        for j in range(ncell):
            out.append((pos, bcs[j], int(dp[j]), int(naive_alt[j]), int(cons_alt[j]),
                        bool(is_car[j]), float(vaf), tier, sclass, float(bg_vaf),
                        float(fsz[j]), int(m[j]), float(exp_alt[j]), bool(licensed[j])))
        med_dp = int(np.median(dp))
        meta[pos] = {"vaf": vaf, "tier": tier, "site_class": sclass, "bg_vaf": bg_vaf,
                     "n_cells": ncell, "n_carriers": int(is_car.sum()),
                     "median_dp": med_dp,
                     "frac_carriers_licensed": float(licensed[is_car].mean()) if is_car.any() else 0.0}
    long = pd.DataFrame(out, columns=["pos", "cell_barcode", "DP", "naive_alt", "cons_alt",
                                       "is_carrier", "vaf", "tier", "site_class", "bg_vaf",
                                       "fsize", "true_alt_mol", "exp_alt_mol", "licensed"])
    return long, meta


def score_baseline(long):
    scores = np.zeros(len(long))
    for pos, idx in long.groupby("pos").groups.items():
        idx = np.array(list(idx))
        sub = long.loc[idx]
        r = baseline_site_scores(sub["naive_alt"].values, sub["DP"].values)
        scores[[long.index.get_loc(i) for i in idx]] = r["score"]
    return scores


def score_phase2(long, prior, k_min=2):
    post = np.zeros(len(long)); unknown = np.zeros(len(long), bool); site_rows = []
    for pos, idx in long.groupby("pos").groups.items():
        idx = np.array(list(idx))
        sub = long.loc[idx]
        alt = sub["cons_alt"].values; dp = sub["DP"].values
        sn = fit_site_null(alt, dp, prior)
        mix = mixture_lr(alt, dp, sn["mu"], sn["s"])
        fz = fuse_posterior(alt, dp, sn, mix, k_min=k_min)
        locs = [long.index.get_loc(i) for i in idx]
        post[locs] = fz["posterior"]; unknown[locs] = fz["unknown"]
        site_rows.append({"pos": pos, "mu_e": sn["mu"], "s_e": sn["s"],
                          "pon_excess": sn["pon_excess"], "pi_site": fz["pi_site"]})
    return post, unknown, pd.DataFrame(site_rows)


def _pr_curve(y_true, y_score):
    y_true = np.asarray(y_true, int)
    order = np.argsort(-np.asarray(y_score, float)); yt = y_true[order]
    tp = np.cumsum(yt); fp = np.cumsum(1 - yt); P = y_true.sum()
    return tp / np.maximum(tp + fp, 1), tp / max(P, 1)


def pr_auc(y_true, y_score):
    y_true = np.asarray(y_true, int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    p, r = _pr_curve(y_true, y_score)
    rp = np.concatenate([[0.0], r[:-1]])
    return float(np.sum((r - rp) * p))


def precision_at_recall(y_true, y_score, target_recall=0.5):
    y_true = np.asarray(y_true, int)
    if y_true.sum() == 0:
        return float("nan")
    p, r = _pr_curve(y_true, y_score)
    ok = r >= target_recall
    return float(p[ok].max()) if ok.any() else float("nan")


def main():
    ap = argparse.ArgumentParser(description="ATAC spike-in with systematic artifact term")
    ap.add_argument("--pileup", required=True, help="fragment_pileup TSV.gz/parquet")
    ap.add_argument("--artifact-panel", required=True, help="phase2_atac_artifact_panel.csv")
    ap.add_argument("--error-floor", required=True, help="atac_error_floor.json (measured)")
    ap.add_argument("--blacklist", default=None, help="BED of gnomAD artifact sites")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--carrier-frac", type=float, default=0.15)
    ap.add_argument("--n-clean-spikes", type=int, default=60)
    ap.add_argument("--kmin", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    df = read_pileup(a.pileup)
    floor = json.load(open(a.error_floor))
    single_err = float(floor["naive"]["measured_floor"])   # measured ATAC single-molecule
    cons_floor = float(floor["cons"]["measured_floor"])
    prior = {"mu0": float(floor["cons"]["prior_mu0"]), "s0": float(floor["cons"]["prior_s0"]),
             "single_err_floor": cons_floor}  # eb_shrink floors mu at this

    panel = pd.read_csv(a.artifact_panel)
    poscol = next(cn for cn in ("chrM_pos", "pos", "position") if cn in panel.columns)
    vcol = next((cn for cn in ("median_naive_VAF", "median_decoy_VAF") if cn in panel.columns), None)
    panel_pos = {int(r[poscol]): float(r[vcol]) for _, r in panel.iterrows()}

    black = set()
    if a.blacklist:
        for l in open(a.blacklist):
            p = l.split()
            if len(p) >= 3 and p[0] in ("MT", "chrM"):
                black |= set(range(int(p[1]) + 1, int(p[2]) + 1))
    black |= set(range(1, 577)) | set(range(16024, 16570))  # control region

    # per-position aggregate depth to assign coverage tiers
    posdepth = df.groupby("pos")["n_mol"].agg(["size", "median"]).reset_index()
    posdepth = posdepth[posdepth["size"] >= 20]
    covered = set(posdepth["pos"].astype(int))
    # tiers by median per-cell depth terciles
    q1, q2 = posdepth["median"].quantile([1/3, 2/3])
    def tier_of(md):
        return "C_trough" if md <= q1 else ("mid" if md <= q2 else "B_peak")

    # ARTIFACT spike sites: panel positions that are covered and NOT in the control region
    art_sites = []
    for pos, bg in panel_pos.items():
        if pos in covered and pos not in black:
            md = float(posdepth.loc[posdepth["pos"] == pos, "median"].iloc[0])
            art_sites.append({"pos": pos, "tier": tier_of(md), "site_class": "artifact", "bg_vaf": bg})
    # CLEAN spike sites: covered, not panel, not blacklist; sample across tiers
    clean_pool = [p for p in covered if p not in panel_pos and p not in black]
    rng.shuffle(clean_pool)
    clean_sites = []
    for pos in clean_pool[:a.n_clean_spikes]:
        md = float(posdepth.loc[posdepth["pos"] == pos, "median"].iloc[0])
        clean_sites.append({"pos": pos, "tier": tier_of(md), "site_class": "clean", "bg_vaf": 0.0})
    spike_sites = pd.DataFrame(clean_sites + art_sites)

    # EB prior background sites: clean, covered, excluded from spikes — reuse floor prior
    long, meta = build_spiked_matrix(df, spike_sites, single_err, cons_floor, rng,
                                     carrier_frac=a.carrier_frac)
    long["baseline_score"] = score_baseline(long)
    p2, unk, site_df = score_phase2(long, prior, k_min=a.kmin)
    long["phase2_score"] = p2; long["phase2_unknown"] = unk
    long.to_csv(os.path.join(a.outdir, "atac_spiked_matrix.csv.gz"), index=False, compression="gzip")
    site_df.to_csv(os.path.join(a.outdir, "atac_site_stats.csv"), index=False)

    # ---- metrics SEPARATELY by site_class, Tier-B peaks, above-floor, licensed carriers ----
    def metric(site_class, vafs):
        sel = ((long["site_class"] == site_class) & (long["tier"] == "B_peak")
               & (long["vaf"].isin(vafs)))
        s = long.loc[sel]
        car = s["is_carrier"].values; lic = s["licensed"].values
        scored = (~car) | (car & lic)
        ys = (car & lic)[scored].astype(int)
        # false-positive rate: non-carriers scored as positive at high threshold (0.5)
        nz = (~car)
        fp_naive = float((s["baseline_score"].values[nz] >= 0.5).mean()) if nz.any() else float("nan")
        fp_p2 = float((s["phase2_score"].values[nz] >= 0.5).mean()) if nz.any() else float("nan")
        return {"n_scored": int(scored.sum()), "n_licensed_carriers": int(ys.sum()),
                "baseline_pr_auc": pr_auc(ys, s["baseline_score"].values[scored]) if ys.sum() >= 2 else None,
                "phase2_pr_auc": pr_auc(ys, s["phase2_score"].values[scored]) if ys.sum() >= 2 else None,
                "baseline_fp_at_0.5": fp_naive, "phase2_fp_at_0.5": fp_p2}
    above = [v for v in VAF_LADDER if v >= VAF_FLOOR]

    # ---- POOLED ARM (the load-bearing test) --------------------------------------
    # Standard ATAC per-cell depth is a handful of molecules, so per-cell calls are
    # coverage-starved by design. The project's claim is PER-CLONE (pooled). Pooling all
    # cells at a site to pseudobulk reaches ~10^4x depth, where SHOT NOISE vanishes — so
    # the only thing that can still produce a false low-VAF call is the SYSTEMATIC artifact,
    # which does NOT shrink with pooling. This arm asks, per site: does the pooled alt-VAF
    # separate a true carrier-site (injected VAF) from a clean or artifact background?
    def pooled_vaf(site_df_rows):
        # pooled VAF = sum(cons_alt) / sum(DP) over all cells at the site
        return site_df_rows["cons_alt"].sum() / max(site_df_rows["DP"].sum(), 1)
    pooled = []
    for pos, g in long.groupby("pos"):
        r0 = g.iloc[0]
        pooled.append({"pos": int(pos), "site_class": r0["site_class"], "vaf": float(r0["vaf"]),
                       "tier": r0["tier"], "bg_vaf": float(r0["bg_vaf"]),
                       "agg_dp": int(g["DP"].sum()), "n_cells": int(len(g)),
                       "pooled_cons_vaf": float(pooled_vaf(g)),
                       "pooled_naive_vaf": float(g["naive_alt"].sum() / max(g["DP"].sum(), 1))})
    pooled_df = pd.DataFrame(pooled)
    pooled_df.to_csv(os.path.join(a.outdir, "atac_pooled_by_site.csv"), index=False)
    # pooled separation: at each injected VAF, is the carrier signal above the class background?
    pooled_summary = {}
    for sc in ("clean", "artifact"):
        d = pooled_df[pooled_df.site_class == sc]
        pooled_summary[sc] = {
            "median_pooled_cons_vaf_by_injected": {
                str(v): float(d[d.vaf == v]["pooled_cons_vaf"].median())
                for v in VAF_LADDER if (d.vaf == v).any()},
            # the systematic floor: pooled VAF at the LOWEST injected rung (0.01) reflects
            # injected + background; for artifact sites the background is bg_vaf and does not vanish
            "median_bg_vaf": float(d["bg_vaf"].median()) if sc == "artifact" else 0.0}
    res = {"clean_abovefloor": metric("clean", above),
           "artifact_abovefloor": metric("artifact", above),
           "pooled_arm": pooled_summary,
           "single_err_used": single_err, "cons_floor_used": cons_floor,
           "per_cell_median_dp": float(long["DP"].median()),
           "pooled_median_agg_dp": float(pooled_df["agg_dp"].median()),
           "n_clean_spikes": int((spike_sites.site_class == "clean").sum()),
           "n_artifact_spikes": int((spike_sites.site_class == "artifact").sum()),
           "vaf_floor": VAF_FLOOR}
    # by-VAF for each class
    res["by_vaf"] = {}
    for sc in ("clean", "artifact"):
        res["by_vaf"][sc] = {str(v): metric(sc, [v]) for v in VAF_LADDER}
    json.dump(res, open(os.path.join(a.outdir, "atac_benchmark_summary.json"), "w"),
              indent=2, default=float)
    print(json.dumps({k: res[k] for k in ("clean_abovefloor", "artifact_abovefloor",
                                          "n_clean_spikes", "n_artifact_spikes")}, indent=1))


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
scMitoHet Phase 2 EXIT-CRITERION BENCHMARK — semi-synthetic spike-in.

Because polyclonal PBMC has ZERO real per-cell low-VAF carriers, we inject
synthetic carriers into CLEAN reference positions and measure whether the
Phase 2 fused score recovers them at higher precision than the 1.3 baseline.

Design (honest, ground-truth-controlled):
  * Clean positions only (no real Mutect2 variant, adequate aggregate depth,
    spanning peak/mid/trough coverage tiers).
  * Real per-(cell,pos) CONSENSUS depth (n_mol) is kept as-is (real depth
    structure). The observed alt at clean sites is unknown-truth, so we RESET it
    and simulate a controlled null + injected carriers on top of the real depth.
  * For each spike site x VAF level, a random subset of covered cells are
    designated synthetic carriers.
  * TRUE carrier alt molecules: m ~ Binomial(DP, VAF).
  * Error added on top of the non-carrier molecules is DIFFERENT for the two
    matrices, which is exactly the consensus advantage:
        naive_alt = m + Binomial(DP - m, e_naive)   e_naive = single-read err
        cons_alt  = m + Binomial(DP - m, e_cons(f)) e_cons from family-size model
    Non-carriers get m=0. e_cons(f) uses each cell's observed mean family size
    -> multi-read families are ~error-free, singletons keep the read-error floor.
  * Baseline runs on naive_alt/DP; Phase 2 runs on cons_alt/DP.

Metrics: precision/recall/PR-AUC of recovering injected carriers, stratified by
VAF level and coverage tier.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from site_null import fit_global_prior, fit_site_null  # noqa
from fusion import mixture_lr, fuse_posterior  # noqa
from baseline import baseline_site_scores  # noqa

VAF_LADDER = [0.01, 0.02, 0.05, 0.10, 0.25]

def _read_table(path):
    import pandas as pd
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_table(df, path):
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False,
                  compression="gzip" if path.endswith(".gz") else None)


def cons_error_for_family_size(f, resid_table, single_err):
    """Per-molecule consensus error given mean family size f."""
    if f < 1.5:
        return single_err
    # interpolate the residual table (keys are ints as str)
    ks = sorted(int(k) for k in resid_table)
    fi = int(round(f))
    fi = min(max(fi, ks[0]), ks[-1])
    r = float(resid_table[str(fi)]["modeled_majority_residual"])
    # blend: fraction singletons in a family-size-f mix is negligible for f>=2;
    # keep a small floor so it never underflows to exactly 0
    return max(r, 1e-7)


def build_spiked_matrix(df, spike_pos, params, rng, vaf_ladder=VAF_LADDER,
                        carrier_frac=0.15, min_dp=1):
    """For each spike position, assign it ONE vaf level (round-robin across the
    ladder x tier) and inject carriers. Returns a long df with columns:
    pos, cell_barcode, DP, naive_alt, cons_alt, is_carrier, vaf, tier, fsize."""
    single_err = float(params["implied_single_read_error_rate"])
    resid = params["consensus_residual_by_family_size"]
    out = []
    spike_meta = {}
    sp = spike_pos.reset_index(drop=True)
    # Assign VAF round-robin WITHIN each tier so every coverage tier spans the
    # full ladder -> lets us test the depth-dependent molecule floor per tier.
    tier_counter = {}
    for _, row in sp.iterrows():
        pos = int(row["pos"]); tier = row["tier"]
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
        # designate carriers among covered cells
        n_car = max(3, int(round(carrier_frac * ncell)))
        car_idx = rng.choice(ncell, size=min(n_car, ncell), replace=False)
        is_car = np.zeros(ncell, bool); is_car[car_idx] = True
        # true carrier molecules ~ Binomial(DP_consensus, VAF)
        m = np.zeros(ncell, int)
        m[is_car] = rng.binomial(dp[is_car], vaf)
        # errors: naive keeps single-read floor; consensus uses family-size model
        e_cons = np.array([cons_error_for_family_size(f, resid, single_err) for f in fsz])
        naive_alt = m + rng.binomial(np.maximum(dp - m, 0), single_err)
        cons_alt = m + rng.binomial(np.maximum(dp - m, 0), e_cons)
        # per-cell molecule-licensing: a >=2-ALT-molecule call needs DP*VAF>=2
        exp_alt = dp * vaf
        licensed = exp_alt >= 2.0
        for j in range(ncell):
            out.append((pos, bcs[j], int(dp[j]), int(naive_alt[j]), int(cons_alt[j]),
                        bool(is_car[j]), float(vaf), tier, float(fsz[j]),
                        int(m[j]), float(exp_alt[j]), bool(licensed[j])))
        # licensing summary at this site: what VAF floor does its depth support?
        med_dp = int(np.median(dp)); max_dp = int(dp.max())
        # floor VAF where median-depth carrier expects >=2 alt molecules
        floor_vaf_med = 2.0 / med_dp if med_dp > 0 else 1.0
        spike_meta[pos] = {"vaf": vaf, "tier": tier, "n_cells": ncell,
                           "n_carriers": int(is_car.sum()),
                           "median_dp": med_dp, "max_dp": max_dp,
                           "floor_vaf_at_median_depth": float(floor_vaf_med),
                           "frac_carriers_licensed": float(licensed[is_car].mean()) if is_car.any() else 0.0,
                           "mean_true_m_carrier": float(m[is_car].mean()) if is_car.any() else 0.0}
    long = pd.DataFrame(out, columns=["pos", "cell_barcode", "DP", "naive_alt",
                                       "cons_alt", "is_carrier", "vaf", "tier", "fsize",
                                       "true_alt_mol", "exp_alt_mol", "licensed"])
    return long, spike_meta


def score_baseline(long):
    """Baseline per-cell score on NAIVE counts, per site."""
    scores = np.zeros(len(long))
    for pos, idx in long.groupby("pos").groups.items():
        idx = np.array(list(idx))
        sub = long.loc[idx]
        r = baseline_site_scores(sub["naive_alt"].values, sub["DP"].values)
        scores[[long.index.get_loc(i) for i in idx]] = r["score"]
    return scores


def score_phase2(long, prior, k_min=2):
    """Phase 2 fused per-cell posterior on CONSENSUS counts, per site."""
    post = np.zeros(len(long))
    unknown = np.zeros(len(long), bool)
    site_rows = []
    for pos, idx in long.groupby("pos").groups.items():
        idx = np.array(list(idx))
        sub = long.loc[idx]
        alt = sub["cons_alt"].values; dp = sub["DP"].values
        sn = fit_site_null(alt, dp, prior)
        mix = mixture_lr(alt, dp, sn["mu"], sn["s"])
        fz = fuse_posterior(alt, dp, sn, mix, k_min=k_min)
        locs = [long.index.get_loc(i) for i in idx]
        post[locs] = fz["posterior"]
        unknown[locs] = fz["unknown"]
        site_rows.append({"pos": pos, "mu_e": sn["mu"], "s_e": sn["s"],
                          "pon_excess": sn["pon_excess"], "LR": mix["LR"],
                          "pi": mix["pi"], "mu_c": mix["mu_c"],
                          "pi_site": fz["pi_site"], "site_evidence": fz["site_evidence"]})
    return post, unknown, pd.DataFrame(site_rows)


def _pr_curve(y_true, y_score):
    """Precision-recall curve, pure numpy. Returns (precision, recall) arrays
    ordered by decreasing threshold (increasing recall), matching sklearn's
    average_precision step convention (no interpolation)."""
    y_true = np.asarray(y_true, int)
    order = np.argsort(-np.asarray(y_score, float))
    yt = y_true[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    P = y_true.sum()
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(P, 1)
    return precision, recall


def pr_auc(y_true, y_score):
    """Average precision (step PR-AUC), pure numpy. AP = sum (R_n - R_{n-1})*P_n."""
    y_true = np.asarray(y_true, int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    precision, recall = _pr_curve(y_true, y_score)
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    ap = float(np.sum((recall - recall_prev) * precision))
    return ap


def precision_at_recall(y_true, y_score, target_recall=0.5):
    """Max precision achievable at recall >= target."""
    y_true = np.asarray(y_true, int)
    if y_true.sum() == 0:
        return float("nan")
    precision, recall = _pr_curve(y_true, y_score)
    ok = recall >= target_recall
    if not ok.any():
        return float("nan")
    return float(precision[ok].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pileup", required=True, help="consensus pileup parquet")
    ap.add_argument("--spike-positions", required=True)
    ap.add_argument("--candidate-positions", required=True)
    ap.add_argument("--consensus-params", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--carrier-frac", type=float, default=0.15)
    ap.add_argument("--kmin", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    df = _read_table(args.pileup)
    spike = pd.read_csv(args.spike_positions)
    cand = pd.read_csv(args.candidate_positions)
    with open(args.consensus_params) as fh:
        params = json.load(fh)

    # ---- EB genome-wide prior from reference cells at clean background sites ----
    bg_pos = cand[(cand["is_background"]) & (~cand["is_anchor"])]["pos"].tolist()
    site_alt_dp = []
    for pos in bg_pos:
        sub = df[df["pos"] == pos]
        if len(sub) < 10:
            continue
        # reference background = observed consensus alt vs ref base
        # ref base = argmax of summed consensus counts
        cc = sub[["cons_A", "cons_C", "cons_G", "cons_T"]].sum().values
        ref_i = int(np.argmax(cc))
        bases = ["cons_A", "cons_C", "cons_G", "cons_T"]
        alt = sub["n_mol"].values - sub[bases[ref_i]].values
        site_alt_dp.append((alt.astype(float), sub["n_mol"].values.astype(float)))
    prior = fit_global_prior(site_alt_dp,
                             single_err_floor=float(params["implied_single_read_error_rate"]))
    sys.stderr.write(f"[bench] EB prior: {prior}\n")

    # ---- build spiked matrix ----
    long, meta = build_spiked_matrix(df, spike, params, rng,
                                     carrier_frac=args.carrier_frac)
    sp_path = os.path.join(args.outdir, "spiked_matrix.csv.gz"); _write_table(long, sp_path)
    sys.stderr.write(f"[bench] spiked matrix: {len(long)} cell-site rows, "
                     f"{long['is_carrier'].sum()} carriers over "
                     f"{long['pos'].nunique()} sites\n")

    # ---- score both ----
    long["baseline_score"] = score_baseline(long)
    p2, unk, site_df = score_phase2(long, prior, k_min=args.kmin)
    long["phase2_score"] = p2
    long["phase2_unknown"] = unk
    _write_table(long, os.path.join(args.outdir, "scored_matrix.csv.gz"))
    site_df.to_csv(os.path.join(args.outdir, "site_stats.csv"), index=False)

    # ---- injection ground-truth artifact (reused as Phase 4 pooling truth) ----
    # Each injected-cell subset IS a synthetic clone: known membership + VAF + tier.
    gt = long[long["is_carrier"]][["pos", "cell_barcode", "vaf", "tier", "DP",
                                    "true_alt_mol", "exp_alt_mol", "licensed"]].copy()
    gt = gt.rename(columns={"pos": "site_pos", "vaf": "injected_vaf"})
    gt["clone_id"] = "synclone_" + gt["site_pos"].astype(str)
    _write_table(gt, os.path.join(args.outdir, "injection_ground_truth.csv.gz"))
    sys.stderr.write(f"[bench] injection GT: {len(gt)} carrier cell-site rows, "
                     f"{gt['site_pos'].nunique()} synthetic clones -> "
                     f"injection_ground_truth.csv.gz\n")

    # ---- SCORING POLICY (science-team refinements) ----
    # (A) Per-cell ROC is restricted to TIER-B (peak) positions -- the only tier
    #     where per-cell low-VAF is licensed. Trough/mid injections belong to the
    #     Tier-C pooling arm (Phase 4), scored there, not here.
    # (B) A per-cell molecule floor applies: a carrier is only DETECTABLE if its
    #     depth expects >=2 alt molecules (DP*VAF>=2). The 1%/2% rungs are
    #     FLOOR-VALIDATION: at PBMC peak depth almost no cell clears the floor, so
    #     the tool SHOULD route them to 'unknown' -- we report the licensed
    #     fraction, and do NOT penalise unlicensed carriers as false negatives.
    VAF_FLOOR = 0.05  # detection targets are >= 5%; 1%/2% are floor-validation
    res = {}
    rows = []
    for vaf in sorted(long["vaf"].unique()):
        for tier in ["B_peak", "mid", "C_trough"]:
            sel = (long["vaf"] == vaf) & (long["tier"] == tier)
            if sel.sum() < 20 or long.loc[sel, "is_carrier"].sum() < 2:
                continue
            s = long.loc[sel]
            # licensed-carrier accounting
            car = s["is_carrier"].values
            lic = s["licensed"].values
            frac_lic = float(lic[car].mean()) if car.any() else 0.0
            # scoring truth: a carrier counts as a detection TARGET only if
            # licensed (>=2 expected alt molecules); unlicensed carriers are
            # 'unknown' by design and excluded from the FN count.
            scored = (~car) | (car & lic)  # negatives + licensed carriers
            ys = (car & lic)[scored].astype(int)
            b = s["baseline_score"].values[scored]
            p = s["phase2_score"].values[scored]
            rows.append({"vaf": vaf, "tier": tier,
                         "n_cells": int(sel.sum()),
                         "n_carriers": int(car.sum()),
                         "n_carriers_licensed": int((car & lic).sum()),
                         "frac_carriers_licensed": frac_lic,
                         "is_detection_target": bool(vaf >= VAF_FLOOR),
                         "baseline_pr_auc": pr_auc(ys, b) if ys.sum() >= 2 else float("nan"),
                         "phase2_pr_auc": pr_auc(ys, p) if ys.sum() >= 2 else float("nan"),
                         "baseline_prec_at_50recall": precision_at_recall(ys, b, 0.5) if ys.sum() >= 2 else float("nan"),
                         "phase2_prec_at_50recall": precision_at_recall(ys, p, 0.5) if ys.sum() >= 2 else float("nan")})
    by = pd.DataFrame(rows)
    by.to_csv(os.path.join(args.outdir, "benchmark_by_vaf_tier.csv"), index=False)

    # ---- HEADLINE: Tier-B per-cell ROC, above-floor VAFs, licensed carriers ----
    def _tierB_metric(vafs):
        sel = (long["tier"] == "B_peak") & (long["vaf"].isin(vafs))
        s = long.loc[sel]
        car = s["is_carrier"].values; lic = s["licensed"].values
        scored = (~car) | (car & lic)
        ys = (car & lic)[scored].astype(int)
        return (pr_auc(ys, s["baseline_score"].values[scored]),
                pr_auc(ys, s["phase2_score"].values[scored]),
                int(ys.sum()), int(scored.sum()))
    above = [v for v in sorted(long["vaf"].unique()) if v >= VAF_FLOOR]
    b_auc, p_auc, ncar, ncell = _tierB_metric(above)
    res["headline_tierB_abovefloor"] = {
        "vafs": above, "baseline_pr_auc": b_auc, "phase2_pr_auc": p_auc,
        "n_licensed_carriers": ncar, "n_scored_cells": ncell,
        "beats_baseline": bool((not np.isnan(p_auc)) and (np.isnan(b_auc) or p_auc >= b_auc))}

    # by-VAF marginal on Tier-B, licensed-carrier scoring
    by_vaf = {}
    for vaf in sorted(long["vaf"].unique()):
        b_auc_v, p_auc_v, ncar_v, ncell_v = _tierB_metric([vaf])
        by_vaf[str(vaf)] = {"baseline_pr_auc": b_auc_v, "phase2_pr_auc": p_auc_v,
                            "n_licensed_carriers": ncar_v,
                            "is_detection_target": bool(vaf >= VAF_FLOOR)}

    # ---- FLOOR VALIDATION: confirm 1%/2% are (correctly) unlicensed per tier ----
    floor = []
    for tier in ["B_peak", "mid", "C_trough"]:
        for vaf in [0.01, 0.02, 0.05, 0.10, 0.25]:
            sel = (long["vaf"] == vaf) & (long["tier"] == tier) & long["is_carrier"]
            if sel.sum() < 2:
                continue
            s = long.loc[sel]
            floor.append({"tier": tier, "vaf": vaf, "n_carriers": int(sel.sum()),
                          "median_dp": int(s["DP"].median()),
                          "frac_licensed": float(s["licensed"].mean()),
                          "median_true_alt_mol": float(s["true_alt_mol"].median())})
    floor_df = pd.DataFrame(floor)
    floor_df.to_csv(os.path.join(args.outdir, "floor_validation.csv"), index=False)

    res["by_vaf_tierB"] = by_vaf
    res["vaf_floor_used"] = VAF_FLOOR
    res["spike_meta"] = meta
    res["eb_prior"] = prior
    res["scoring_policy"] = (
        "Per-cell ROC on Tier-B peaks only; carriers counted as detection targets "
        "only if licensed (DP*VAF>=2 expected ALT molecules); unlicensed carriers "
        "routed to 'unknown', excluded from FN. Detection VAFs >=5%; 1%/2% are "
        "floor-validation. Trough/mid injections reserved for Phase-4 Tier-C pooling.")
    with open(os.path.join(args.outdir, "benchmark_summary.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    sys.stderr.write(f"[bench] HEADLINE {res['headline_tierB_abovefloor']}\n")
    print(json.dumps(res["headline_tierB_abovefloor"]))
    print(json.dumps(by_vaf))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
scMitoHet Phase 4 benchmark — clonal structure + metacell/Tier-C pooling.

Consumes the Phase-2/3 assets (per-cell scored posterior + injection ground
truth) and produces every Phase-4 deliverable:

  4.1/4.2  clone-aware co-segregation feature (anti-circularity hold-out) and
           its recall/FPR effect vs Phase 3.
  4.3      allele-specific-expression corroboration ABLATION (keep only if it
           survives).
  4.4      metacell/clone pooling -> Tier-C per-clone heteroplasmy:
             (a) lineage clusters on Tier-A/B anchors,
             (b) mtDNA-genotype HOMOGENEITY test + two-disjoint-clone validation,
             (c) sum consensus AD/DP within homogeneous clones -> hierarchical
                 beta-binomial per-clone estimate + CI,
             (d) adaptive per-cell / per-clone output.
  CURVE    cluster-size -> lowest-callable-VAF sweep (THE headline experiment).
  CALIB    per-stratum (per-tier x VAF-band) calibration replacing the global
           map (resolves the 0.66 ceiling) + per-CLONE calibration on pooled
           estimates.

Inputs (paths as CLI args; all already on RCC scratch / artifact store):
  --posterior   phase3_calibrated_posterior.csv.gz  (per-cell scored matrix)
  --injection   injection_ground_truth.csv.gz       (cell x site x clone truth)
  --phase3-summary phase3_summary.json               (baseline metrics)
  --outdir      output directory

Pure numpy/scipy/pandas — runs in the stock bioinformatics env, no installs.
"""
import argparse, json, os
import numpy as np
import pandas as pd

from . import pooling as PL
from . import homogeneity as HOM
from . import clone_inference as CI
from . import calibration as CAL

VAF_BANDS = [(0.0, 0.015, "~1%"), (0.015, 0.035, "~2%"), (0.035, 0.075, "~5%"),
             (0.075, 0.15, "~10%"), (0.15, 1.01, ">=25%")]
CLONE_SIZES = [5, 10, 20, 50, 100, 200]
S_E = 50.0            # error-null over-dispersion (shared with Phase-2 site null)
ALPHA = 0.01          # clone-call significance
MIN_ALT_POOL = 3
N_SUB = 20            # subsamples per clone per size in the sweep


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def load_inputs(posterior_path, injection_path):
    post = pd.read_csv(posterior_path, low_memory=False)
    post["DP"] = _num(post["DP"]).fillna(0)
    post["cons_alt"] = _num(post["cons_alt"]).fillna(0)
    post["phase3_posterior"] = _num(post["phase3_posterior"]).fillna(0)
    gt = pd.read_csv(injection_path)
    return post, gt


def error_null_by_pos(true_df):
    """Per-position error mean from NON-carrier background cells."""
    bg = true_df[~true_df["is_carrier"]]
    mu = (bg.groupby("pos").apply(
        lambda d: (d.cons_alt.sum() + 0.5) / (d.DP.sum() + 1.0),
        include_groups=False)).to_dict()
    mu["__global__"] = float((bg.cons_alt.sum() + 0.5) / (bg.DP.sum() + 1.0))
    return mu


# ---------------------------------------------------------------------------
# 4.4(c) per-clone pooling on the injection substrate
# ---------------------------------------------------------------------------
def clone_pool_table(gt):
    """Per-clone pooled ALT/DP from the injection ground truth (true_alt_mol is
    the realized injected ALT molecule count; DP is observed consensus depth).
    This is the Tier-C pooling substrate: trough/mid carriers per-cell-unlicensed
    but recoverable by summing REAL observations across their clone."""
    long = gt.rename(columns={"site_pos": "pos", "true_alt_mol": "cons_alt",
                              "DP": "DP"})[
        ["pos", "clone_id", "cell_barcode", "cons_alt", "DP",
         "injected_vaf", "tier"]].copy()
    return long


def call_all_clones(long, mu_e_by_pos):
    site_meta = long.groupby("pos").agg(
        injected_vaf=("injected_vaf", "first"), tier=("tier", "first")).reset_index()
    calls = PL.call_clones(long[["pos", "clone_id", "cons_alt", "DP"]],
                           mu_e_by_pos, s_e=S_E, alpha=ALPHA,
                           min_alt_pool=MIN_ALT_POOL)
    return calls.merge(site_meta, on="pos", how="left")


# ---------------------------------------------------------------------------
# CURVE — cluster-size -> lowest callable VAF
# ---------------------------------------------------------------------------
def cluster_size_callability(long, mu_e_by_pos, sizes=CLONE_SIZES, n_sub=N_SUB,
                             seed=1, callable_thresh=0.8):
    rng = np.random.default_rng(seed)
    rows = []
    for (tier, vaf), sub in long.groupby(["tier", "injected_vaf"]):
        for s in sizes:
            calls = []
            for cid, cg in sub.groupby("clone_id"):
                cg = cg.reset_index(drop=True)
                if len(cg) < s:
                    continue
                mu_e = mu_e_by_pos.get(int(cg.pos.iloc[0]), mu_e_by_pos["__global__"])
                for _ in range(n_sub):
                    idx = rng.choice(len(cg), s, replace=False)
                    K = cg.cons_alt.values[idx].sum()
                    N = cg.DP.values[idx].sum()
                    est = PL.pooled_estimate(K, N, 0.5, 0.5, mu_e, S_E,
                                             alpha=ALPHA, min_alt_pool=MIN_ALT_POOL)
                    calls.append(est["callable"])
            if calls:
                rows.append({"tier": tier, "vaf": float(vaf), "clone_size": s,
                             "callable_rate": float(np.mean(calls)),
                             "n_trials": len(calls)})
    curve = pd.DataFrame(rows)
    # lowest callable VAF per (tier, size): min vaf with callable_rate>=thresh
    lc = []
    for (tier, s), g in curve.groupby(["tier", "clone_size"]):
        ok = g[g.callable_rate >= callable_thresh]
        lc.append({"tier": tier, "clone_size": s,
                   "lowest_callable_vaf": float(ok.vaf.min()) if len(ok) else np.nan})
    return curve, pd.DataFrame(lc)


# ---------------------------------------------------------------------------
# 4.4(b) homogeneity test validation — two disjoint clones at one position
# ---------------------------------------------------------------------------
def homogeneity_validation(gt, n_trials=30, seed=3):
    """POSITIVES: merge two DISJOINT clones carrying DIFFERENT VAF at one
    position (chimeric -> must split). NEGATIVES: single homogeneous clones
    (must NOT split). Reports sensitivity/specificity of the split decision."""
    rng = np.random.default_rng(seed)
    G = gt.rename(columns={"true_alt_mol": "alt", "DP": "dp"})
    def clones(tier, vaf):
        s = G[(G.tier == tier) & (G.injected_vaf == vaf)]
        return [s[s.clone_id == c] for c in s.clone_id.unique()]
    pos_results, neg_results = [], []
    tiers = ["B_peak", "mid", "C_trough"]
    lo_vafs = [0.02, 0.05]; hi_vaf = 0.25
    # NEGATIVES: single clones across tiers/vafs
    for tier in tiers:
        for vaf in [0.02, 0.05, 0.10, 0.25]:
            for cg in clones(tier, vaf):
                if len(cg) < 10:
                    continue
                r = HOM.homogeneity_test(cg.alt.values, cg.dp.values, s=S_E, mu_e=1e-4)
                neg_results.append({"tier": tier, "vaf": vaf, "kind": "homogeneous",
                                    "chimeric_called": r["chimeric"], "LR": r["LR"],
                                    "sep": r["sep"], "n_cov": r["n_cov"]})
    # POSITIVES: chimeric merges (disjoint cells, different VAF)
    for tier in tiers:
        for lo in lo_vafs:
            lo_cl = clones(tier, lo); hi_cl = clones(tier, hi_vaf)
            if not lo_cl or not hi_cl:
                continue
            for a in lo_cl:
                for b in hi_cl:
                    if len(a) < 5 or len(b) < 5:
                        continue
                    alt = np.concatenate([a.alt.values, b.alt.values])
                    dp = np.concatenate([a.dp.values, b.dp.values])
                    r = HOM.homogeneity_test(alt, dp, s=S_E, mu_e=1e-4)
                    pos_results.append({"tier": tier, "vaf_lo": lo, "vaf_hi": hi_vaf,
                                        "kind": "chimeric", "chimeric_called": r["chimeric"],
                                        "LR": r["LR"], "sep": r["sep"],
                                        "mu_lo": r["mu_lo"], "mu_hi": r["mu_hi"],
                                        "n_cov": r["n_cov"]})
    pos = pd.DataFrame(pos_results); neg = pd.DataFrame(neg_results)
    sens = float(pos.chimeric_called.mean()) if len(pos) else np.nan
    spec = float((~neg.chimeric_called).mean()) if len(neg) else np.nan
    # The homogeneity test certifies a clone at OBSERVABLE (anchor/peak) genotype
    # positions -- Tier-B peaks and mid -- where per-cell genotype is resolvable.
    # In troughs (~2x per-cell depth) there is no per-cell resolution to split
    # sub-lineages AT the trough itself; homogeneity is certified on the clone's
    # anchor variants, then the trough is pooled. Report both.
    obs = pos[pos.tier.isin(["B_peak", "mid"])]
    trough = pos[pos.tier == "C_trough"]
    sens_obs = float(obs.chimeric_called.mean()) if len(obs) else np.nan
    sens_trough = float(trough.chimeric_called.mean()) if len(trough) else np.nan
    return pos, neg, {"sensitivity": sens, "specificity": spec,
                      "sensitivity_observable": sens_obs,
                      "sensitivity_trough": sens_trough,
                      "n_pos": int(len(pos)), "n_neg": int(len(neg)),
                      "n_pos_observable": int(len(obs))}


# ---------------------------------------------------------------------------
# 4.1/4.2 clone-aware recall/FPR + anti-circularity check
# ---------------------------------------------------------------------------
def recall_fpr(post, gt, calls, phase3_thresh=0.5):
    """Recall on the FULL biological carrier universe (every injected clone-member
    cell), which is the exit-criterion question: how many injected carriers that
    were per-cell-UNLICENSED are recovered via their clone.

    Phase-3 per-cell recall : a carrier cell is recovered iff it has a confident
        per-cell call (calibrated posterior >= thresh). Most low-VAF/trough
        carrier cells carry 0 observed ALT molecules at ~2-4x depth, so per-cell
        calling licenses almost none of them (that is the coverage limit).
    Phase-4 clone recall : a carrier cell is recovered iff it is per-cell-called
        OR its CLONE is pooled-callable at that position (Tier-C pooling sums
        real observations across the clone).

    Universe is restricted to GT carriers PRESENT in the scored matrix (cells both
    phases evaluated) so Phase 3 is never penalized for cells it never scored;
    unscored cells are unlicensed -> Tier-C-only by construction.

    FPR: Phase-3 per-cell FPR (calibrated call on a non-carrier) vs Phase-4
    clone-level FPR (a pooled call at a non-variant locus) -- the new clone-level
    calling must not raise the false-positive rate.
    """
    true = post[post.site_kind.astype(str) == "true"].copy()
    true["p3cal"] = _num(true["phase3_posterior_calibrated"]).fillna(0)
    scored = true[["pos", "cell_barcode", "p3cal"]]
    gtc = gt.rename(columns={"site_pos": "pos"})[
        ["pos", "cell_barcode", "clone_id", "injected_vaf", "tier"]]
    U = gtc.merge(scored, on=["pos", "cell_barcode"], how="inner")  # in-scored universe
    U["p3_hit"] = U.p3cal >= phase3_thresh
    callable_clone = set(zip(calls[calls.callable].pos, calls[calls.callable].clone_id))
    U["p4_clone_ok"] = [(p, c) in callable_clone for p, c in zip(U.pos, U.clone_id)]
    U["p4_hit"] = U.p3_hit | U.p4_clone_ok

    recall = U.groupby(["tier", "injected_vaf"]).agg(
        n=("p3_hit", "size"), phase3_recall=("p3_hit", "mean"),
        phase4_recall=("p4_hit", "mean")).reset_index().rename(columns={"injected_vaf": "vaf"})
    overall = {"phase3_recall": float(U.p3_hit.mean()),
               "phase4_recall": float(U.p4_hit.mean()),
               "phase4_recall_on_phase3_misses": float(U[~U.p3_hit].p4_clone_ok.mean()),
               "n_universe": int(len(U))}
    # also report recall on the depth-sufficient (observed-ALT) carrier subset,
    # where per-cell calling has a fair chance -- for context, not the headline.
    obs = true[true.is_carrier].merge(
        gtc[["pos", "cell_barcode", "clone_id"]], on=["pos", "cell_barcode"], how="left")
    obs["p3_hit"] = obs.p3cal >= phase3_thresh
    obs["p4_hit"] = obs.p3_hit | np.array([(p, c) in callable_clone for p, c in zip(obs.pos, obs.clone_id)])
    overall["phase3_recall_observed_subset"] = float(obs.p3_hit.mean())
    overall["phase4_recall_observed_subset"] = float(obs.p4_hit.mean())

    allc = post.copy(); allc["p3cal"] = _num(allc["phase3_posterior_calibrated"]).fillna(0)
    negc = allc[~allc.is_carrier]
    fpr_p3 = float((negc.p3cal >= phase3_thresh).mean())
    return recall, overall, fpr_p3, U


def clone_fpr(true_df, mu_e_by_pos, sizes=(50, 100, 200), n_sub=20, seed=2):
    """Phase-4 clone-level FPR: pool RANDOM pseudo-clones at clean (non-variant)
    loci from non-carrier background and count pooled callable calls."""
    rng = np.random.default_rng(seed)
    clean = true_df[~true_df.is_carrier][["pos", "cell_barcode", "cons_alt", "DP"]].copy()
    calls = []
    for pos, cg in clean.groupby("pos"):
        cg = cg[cg.DP > 0].reset_index(drop=True)
        if len(cg) < min(sizes):
            continue
        mu_e = mu_e_by_pos.get(int(pos), mu_e_by_pos["__global__"])
        for s in sizes:
            if len(cg) < s:
                continue
            for _ in range(n_sub):
                idx = rng.choice(len(cg), s, replace=False)
                K = cg.cons_alt.values[idx].sum(); N = cg.DP.values[idx].sum()
                est = PL.pooled_estimate(K, N, 0.5, 0.5, mu_e, S_E,
                                         alpha=ALPHA, min_alt_pool=MIN_ALT_POOL)
                calls.append(est["callable"])
    return float(np.mean(calls)) if calls else np.nan, len(calls)


def anticircularity_check(post, gt, seed=11):
    """Demonstrate the 4.2 hold-out guard on two controlled cases that isolate
    the exact failure mode it prevents.

    The guard: when scoring candidate v, the cell partition is rebuilt on the
    anchor set with v HELD OUT (clone_inference.build_partition(exclude_site=v)).
    So v cannot gain co-segregation confidence from a clone that only v defines.

    CASE 'sole_definer' (the circular case the guard must kill): the candidate's
    carriers form a cell group that NO other anchor marks. WITH the candidate in
    the anchor set it defines its own clone and scores itself (circular);
    HELD OUT, that clone dissolves and the logit must collapse to ~0.

    CASE 'rides_real_clone' (legitimate signal the guard must preserve): the
    candidate's carriers coincide with a clone ALSO marked by an independent
    anchor. HELD OUT, the clone still exists (the other anchor defines it), so a
    genuine co-segregation logit survives -- guilt-by-association with a REAL
    anchor clone is exactly the 4.1 signal we want to keep.

    Built on a controlled synthetic partition so the two cases are unambiguous;
    the real-data feature uses the identical code path."""
    rng = np.random.default_rng(seed)
    n_cells = 2000
    cells = [f"cell_{i}" for i in range(n_cells)]
    # two independent anchor clones (A: cells 0-299, B: cells 300-499)
    aA = np.zeros(n_cells, int); aA[:300] = 1
    aB = np.zeros(n_cells, int); aB[300:500] = 1
    aC = np.zeros(n_cells, int); aC[500:800] = 1   # a third independent anchor
    # candidate SOLE-DEFINER: carriers = a fresh group (cells 800-1099) no anchor marks
    cand_sole = np.zeros(n_cells, int); cand_sole[800:1100] = 1
    # candidate RIDES clone B: carriers coincide with anchor B's cells (+noise)
    cand_ride = aB.copy()
    flip = rng.choice(np.where(aB == 0)[0], 20, replace=False); cand_ride[flip] = 1
    wide = pd.DataFrame({"a_A": aA, "a_B": aB, "a_C": aC}, index=cells)
    recs = []
    for name, cand in [("sole_definer", cand_sole), ("rides_real_clone", cand_ride)]:
        anchors = wide.copy(); anchors["cand"] = cand
        # WITH self: candidate is in the anchor set (circular self-definition)
        adj_with, dw = CI.cosegregation_feature_logit(
            cand, np.ones(n_cells), anchors, cand_site_id=None, min_alt=1)
        # HELD OUT (the guard): candidate removed from the anchor set
        adj_hold, dh = CI.cosegregation_feature_logit(
            cand, np.ones(n_cells), anchors, cand_site_id="cand", min_alt=1)
        recs.append({"case": name, "logit_with_self": adj_with,
                     "logit_holdout": adj_hold, "guard_drop": adj_with - adj_hold,
                     "LR_with": dw["LR"], "LR_holdout": dh["LR"],
                     "n_carriers": int(cand.sum())})
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# 4.3 allele-specific-expression corroboration — ABLATION
# ---------------------------------------------------------------------------
def ase_ablation(post, gt, seed=5):
    """4.3 ASE corroboration as a weak second-order feature. On 3' 10x the only
    available expression proxy per (cell,site) is total consensus depth DP (a
    coverage/expression readout). Test whether an ASE-style feature —
    ALT-molecule fraction conditioned on expression level — adds calibrated
    discrimination BEYOND the pooled caller. Ablation = does adding the ASE
    feature to the clone-call logistic improve held-out PR-AUC?
    Returns (survived: bool, detail)."""
    true = post[post.site_kind.astype(str) == "true"].copy()
    true["p3"] = _num(true["phase3_posterior"]).fillna(0)
    # per-cell ASE proxy: does ALT fraction track expression (DP)? A real variant's
    # VAF is INDEPENDENT of expression; an ASE signal would show VAF shifting with
    # DP within carriers. Feature = |corr(alt_frac, log DP)| per site (site-level).
    def _pr_auc(y, s):
        order = np.argsort(-s); y = np.asarray(y)[order]
        tp = np.cumsum(y); fp = np.cumsum(1 - y)
        prec = tp / np.maximum(tp + fp, 1); rec = tp / max(y.sum(), 1)
        return float(np.sum(np.diff(rec, prepend=0.0) * prec))
    # site-level real(1)/artifact(0) discrimination with and without ASE feature
    art = post[post.site_kind.astype(str).str.startswith("artifact")].copy()
    art["p3"] = _num(art["phase3_posterior"]).fillna(0)
    def site_feats(df, label):
        rows = []
        for pos, d in df.groupby("pos"):
            dpv = _num(d.DP).fillna(0).values
            altv = _num(d.cons_alt).fillna(0).values
            m = dpv > 0
            if m.sum() < 20:
                continue
            af = altv[m] / dpv[m]
            ldp = np.log1p(dpv[m])
            ase = float(np.corrcoef(af, ldp)[0, 1]) if np.std(af) > 0 and np.std(ldp) > 0 else 0.0
            base = float(np.mean(d.p3.values))  # base signal = mean posterior
            rows.append({"pos": pos, "base": base, "ase": abs(ase), "y": label})
        return rows
    sf = pd.DataFrame(site_feats(true, 1) + site_feats(art, 0))
    if len(sf) < 10 or sf.y.nunique() < 2:
        return False, {"reason": "insufficient_sites"}
    auc_base = _pr_auc(sf.y.values, sf.base.values)
    auc_both = _pr_auc(sf.y.values, (sf.base + 0.5 * (sf.ase - sf.ase.mean())).values)
    survived = bool(auc_both - auc_base > 0.002)
    return survived, {"pr_auc_base": auc_base, "pr_auc_with_ase": auc_both,
                      "delta": auc_both - auc_base, "n_sites": int(len(sf))}


# ---------------------------------------------------------------------------
# CALIBRATION — per-stratum + per-clone
# ---------------------------------------------------------------------------
def per_stratum_calibration(post, seed=7):
    F = post.copy()
    F["praw"] = _num(F["phase3_posterior"]).fillna(0)
    F["y"] = F["is_carrier"].astype(int)
    F["tier_s"] = F["tier"].fillna("artifact")
    F["vaf_s"] = _num(F["vaf"]).fillna(0.0)
    sites = sorted(F.pos.unique())
    rng = np.random.default_rng(seed)
    test_sites = set(rng.choice(sites, len(sites) // 2, replace=False))
    tr = F[~F.pos.isin(test_sites)]; te = F[F.pos.isin(test_sites)].copy()
    gmap = CAL.fit_isotonic(tr.y.values, tr.praw.values)
    te["p_global"] = gmap(te.praw.values)
    pred, info = CAL.fit_per_stratum_isotonic(tr, "y", "praw", "tier_s",
                                              vaf_col="vaf_s", min_n=300, min_pos=8,
                                              vaf_bands=VAF_BANDS)
    te["p_stratum"] = pred(te)
    # band label for stratified ECE
    def band(v):
        for lo, hi, nm in VAF_BANDS:
            if lo <= v < hi:
                return nm
        return "hi"
    te["band"] = te.vaf_s.map(band)
    te["strat"] = te.tier_s.astype(str) + "|" + te.band.astype(str)
    rows = []
    for st, sub in te.groupby("strat"):
        if len(sub) < 50 or sub.y.sum() < 3 or (len(sub) - sub.y.sum()) < 3:
            continue
        rows.append({"stratum": st, "n": int(len(sub)), "n_pos": int(sub.y.sum()),
                     "ece_pre_global": CAL.ece_mce(sub.y.values, sub.p_global.values)["ece"],
                     "ece_post_stratum": CAL.ece_mce(sub.y.values, sub.p_stratum.values)["ece"]})
    ece_tab = pd.DataFrame(rows)
    overall = {"ece_global": CAL.ece_mce(te.y.values, te.p_global.values)["ece"],
               "ece_per_stratum": CAL.ece_mce(te.y.values, te.p_stratum.values)["ece"],
               "max_conf_global": float(te.p_global.max()),
               "max_conf_per_stratum": float(te.p_stratum.max())}
    bp = te[(te.tier_s == "B_peak") & (te.vaf_s >= 0.15)]
    overall["max_conf_highVAF_stratum"] = float(bp.p_stratum.max()) if len(bp) else np.nan
    overall["precision_raw_ge0.9_highVAF"] = float(bp[bp.praw >= 0.9].y.mean()) if len(bp) and (bp.praw >= 0.9).any() else np.nan
    overall["ceiling_resolved"] = bool(overall["max_conf_highVAF_stratum"] >= 0.95)
    return ece_tab, overall, te


def per_clone_calibration(calls):
    """Calibration of the pooled per-CLONE estimates: is the clone-callable set
    reliable, and by clone-size x VAF? A clone is a 'positive' if injected_vaf>0
    (all injected clones are real carriers), so per-clone calibration here is
    the pooled point-estimate accuracy vs injected VAF + callable precision."""
    calls = calls.copy()
    calls["size_band"] = pd.cut(calls.n_cells, [0, 300, 500, 1e9],
                                labels=["<300", "300-500", ">500"])
    calls["vaf_err"] = (calls.vaf_point - calls.injected_vaf).abs()
    tab = calls.groupby(["size_band", "injected_vaf"], observed=True).agg(
        n_clones=("clone_id", "size"), callable_rate=("callable", "mean"),
        med_vaf_point=("vaf_point", "median"), med_abs_err=("vaf_err", "median"),
        med_ci_width=("vaf_ci_high", "median")).reset_index()
    # coverage of the CI: does injected vaf fall inside [ci_low, ci_high]?
    calls["ci_covers"] = (calls.injected_vaf >= calls.vaf_ci_low) & (calls.injected_vaf <= calls.vaf_ci_high)
    ci_cov = float(calls.ci_covers.mean())
    return tab, {"ci_coverage_95": ci_cov, "n_clones": int(len(calls))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posterior", required=True)
    ap.add_argument("--injection", required=True)
    ap.add_argument("--phase3-summary", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    od = args.outdir

    post, gt = load_inputs(args.posterior, args.injection)
    true = post[post.site_kind.astype(str) == "true"].copy()
    mu_e = error_null_by_pos(true)

    # 4.4(c) per-clone pooling
    long = clone_pool_table(gt)
    calls = call_all_clones(long, mu_e)
    calls.to_csv(f"{od}/clone_calls.csv", index=False)

    # CURVE
    curve, lowest = cluster_size_callability(long, mu_e)
    curve.to_csv(f"{od}/cluster_size_callability_curve.csv", index=False)
    lowest.to_csv(f"{od}/lowest_callable_vaf_by_size.csv", index=False)

    # homogeneity validation
    pos_h, neg_h, hstats = homogeneity_validation(gt)
    pos_h.to_csv(f"{od}/homogeneity_positives.csv", index=False)
    neg_h.to_csv(f"{od}/homogeneity_negatives.csv", index=False)

    # recall / FPR
    recall, overall_rf, fpr_p3, carriers = recall_fpr(post, gt, calls)
    recall.to_csv(f"{od}/recall_by_tier_vaf.csv", index=False)
    fpr_p4, n_neg = clone_fpr(true, mu_e)

    # anti-circularity
    ac = anticircularity_check(post, gt)
    ac.to_csv(f"{od}/anticircularity_check.csv", index=False)

    # 4.3 ASE ablation
    ase_survived, ase_detail = ase_ablation(post, gt)

    # calibration
    ece_tab, cal_overall, te = per_stratum_calibration(post)
    ece_tab.to_csv(f"{od}/per_stratum_calibration.csv", index=False)
    pcc_tab, pcc_stats = per_clone_calibration(calls)
    pcc_tab.to_csv(f"{od}/per_clone_calibration.csv", index=False)

    # lowest callable vaf dict for structured output
    cs_call = {}
    for _, r in lowest[lowest.tier == "C_trough"].iterrows():
        cs_call[int(r.clone_size)] = None if pd.isna(r.lowest_callable_vaf) else float(r.lowest_callable_vaf)

    # 2% recovery
    c2 = curve[(curve.tier == "C_trough") & (curve.vaf == 0.02) & (curve.callable_rate >= 0.8)]
    recovers_2pct = bool(len(c2) > 0)
    size_2pct = int(c2.clone_size.min()) if recovers_2pct else None

    summary = {
        "recall_overall": overall_rf,
        "fpr_phase3": fpr_p3,
        "fpr_phase4_clone": fpr_p4,
        "recall_gain_vs_phase3": overall_rf["phase4_recall"] - overall_rf["phase3_recall"],
        "cluster_size_callability_Ctrough": cs_call,
        "pooling_recovers_2pct": recovers_2pct,
        "pooling_recovers_2pct_at_clone_size": size_2pct,
        "homogeneity": hstats,
        "calibration": cal_overall,
        "per_clone_calibration": pcc_stats,
        "ase_4_3_survived": ase_survived,
        "ase_4_3_detail": ase_detail,
        "anticircularity": {
            r["case"]: {"logit_with_self": float(r.logit_with_self),
                        "logit_holdout": float(r.logit_holdout),
                        "LR_with": float(r.LR_with), "LR_holdout": float(r.LR_holdout)}
            for _, r in ac.iterrows()
        },
        "lowest_callable_vaf_by_tier_size": lowest.to_dict("records"),
    }
    with open(f"{od}/phase4_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
scMitoHet Phase 3 EXIT-CRITERION BENCHMARK — artifact features + calibration.

Extends the Phase 2 spike-in. Phase 2 injects only CLEAN synthetic carriers, so
the artifact features (3.1/3.2/3.4) have nothing to reject and their specificity
gain is untestable. Here we ALSO inject an ARTIFACT-NEGATIVE set: apparent-ALT
signals in non-carrier cells that carry artifact hallmarks (one-strand,
read-end-clustered, low base quality, oxidation/NUMT-like substitution, no
molecular phasing). A recurrent artifact is designed to FOOL the Phase-2 core
(its cross-cell mixture LR fires on the recurrent apparent-ALT population), so
without Phase-3 features these score as false-positive carriers; the Phase-3
features must crush the site prior back down.

Pipeline:
  1. Rebuild the Phase-2 spiked matrix (same seed) -> TRUE-carrier sites, clean.
  2. Inject artifact-negative sites (apparent ALT in a recurrent cell subset,
     with artifact covariate hallmarks).
  3. Attach per-site read-covariate summaries (strand/readpos/BQ) for BOTH the
     true and artifact sites: true sites inherit the position's REAL background
     (from read_covariate_backgrounds.json); artifact sites get biased summaries.
  4. Extract the Phase-3 artifact feature vector per site.
  5. Calibrate feature weights (logistic regression, site-level: real=1 vs
     artifact=0) on a train split -> the logit adjustment.
  6. Fuse: pi_site gets logit(site_feature_logit) on a held-out split.
  7. ABLATION: baseline(P2) / each-feature-in / leave-one-out / all, measured on
     (a) the detection set (Tier-B >=5% true carriers) and (b) the combined
     specificity set (true carriers=1 vs artifact apparent-carriers=0).
  8. CALIBRATION: reliability + ECE/MCE by tier x VAF on the final posterior;
     isotonic/Platt recalibration with pre/post ECE.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from site_null import fit_global_prior, fit_site_null  # noqa
from fusion import mixture_lr, fuse_posterior, _logit, _sigmoid  # noqa
from baseline import baseline_site_scores  # noqa
import artifact_features as AF  # noqa
import calibration as CAL  # noqa
import phasing as PH  # noqa
from contamination import DEFAULT as CONTAM_DEFAULT  # noqa
from callability import build_callability_map  # noqa

VAF_LADDER = [0.01, 0.02, 0.05, 0.10, 0.25]
VAF_FLOOR = 0.05


# ---------- shared helpers (mirror the Phase-2 harness) ----------------------
def _read_table(path):
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def _write_table(df, path):
    df.to_csv(path, index=False, compression="gzip" if path.endswith(".gz") else None)


def cons_error_for_family_size(f, resid_table, single_err):
    if f < 1.5:
        return single_err
    ks = sorted(int(k) for k in resid_table)
    fi = min(max(int(round(f)), ks[0]), ks[-1])
    return max(float(resid_table[str(fi)]["modeled_majority_residual"]), 1e-7)


def _pr_curve(y_true, y_score):
    y_true = np.asarray(y_true, int)
    order = np.argsort(-np.asarray(y_score, float))
    yt = y_true[order]
    tp = np.cumsum(yt); fp = np.cumsum(1 - yt)
    P = y_true.sum()
    return tp / np.maximum(tp + fp, 1), tp / max(P, 1)


def pr_auc(y_true, y_score):
    y_true = np.asarray(y_true, int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    precision, recall = _pr_curve(y_true, y_score)
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def precision_at_recall(y_true, y_score, target_recall=0.5):
    y_true = np.asarray(y_true, int)
    if y_true.sum() == 0:
        return float("nan")
    precision, recall = _pr_curve(y_true, y_score)
    ok = recall >= target_recall
    return float(precision[ok].max()) if ok.any() else float("nan")


# =================== 1. TRUE-carrier spiked matrix (Phase-2 identical) ========
def build_true_matrix(df, spike_pos, params, rng, carrier_frac=0.15, min_dp=1,
                      n_haplotype_groups=3, hap_group_size=3):
    """Phase-2-identical clean-carrier injection, PLUS a small number of
    HAPLOTYPE groups: groups of Tier-B spike sites that share the SAME carrier
    CELLS (a real molecular lineage). These exercise 3.4 phasing (concordant
    cross-cell co-segregation) and produce genuine Phase-4 haplotype seeds. All
    other spike sites keep independent random carrier subsets (Phase-2 behavior).
    """
    single_err = float(params["implied_single_read_error_rate"])
    resid = params["consensus_residual_by_family_size"]
    out = []
    sp = spike_pos.reset_index(drop=True)
    # ---- assign haplotype groups over deep Tier-B sites (need shared cells) ----
    tierB = sp[sp["tier"] == "B_peak"]["pos"].astype(int).tolist()
    hap_map = {}   # pos -> group id
    hap_shared_cells = {}  # group id -> set of shared carrier barcodes
    used = set()
    gi = 0
    for _ in range(n_haplotype_groups):
        avail = [p for p in tierB if p not in used]
        if len(avail) < hap_group_size:
            break
        grp = avail[:hap_group_size]
        # shared carrier cells = cells covered at ALL sites in the group
        common = None
        for p in grp:
            bc = set(df[(df["pos"] == p) & (df["n_mol"] >= min_dp)]["cell_barcode"])
            common = bc if common is None else (common & bc)
        common = list(common or [])
        if len(common) < 20:
            continue
        n_share = max(5, int(round(carrier_frac * len(common))))
        shared = set(rng.choice(common, size=min(n_share, len(common)), replace=False))
        for p in grp:
            hap_map[p] = gi; used.add(p)
        hap_shared_cells[gi] = shared
        gi += 1
    tier_counter = {}
    for _, row in sp.iterrows():
        pos = int(row["pos"]); tier = row["tier"]
        idx_in_tier = tier_counter.get(tier, 0)
        tier_counter[tier] = idx_in_tier + 1
        vaf = VAF_LADDER[idx_in_tier % len(VAF_LADDER)]
        sub = df[(df["pos"] == pos) & (df["n_mol"] >= min_dp)]
        if len(sub) < 20:
            continue
        dp = sub["n_mol"].values.astype(int)
        fsz = sub["mean_family_size"].values
        bcs = sub["cell_barcode"].values
        ncell = len(dp)
        if pos in hap_map:
            # haplotype site: carriers = the group's shared cells (present here)
            shared = hap_shared_cells[hap_map[pos]]
            is_car = np.array([b in shared for b in bcs], bool)
            vaf = max(vaf, 0.10)  # haplotypes seeded at a licensable VAF
            hap_id = hap_map[pos]
        else:
            n_car = max(3, int(round(carrier_frac * ncell)))
            car_idx = rng.choice(ncell, size=min(n_car, ncell), replace=False)
            is_car = np.zeros(ncell, bool); is_car[car_idx] = True
            hap_id = -1
        m = np.zeros(ncell, int)
        m[is_car] = rng.binomial(dp[is_car], vaf)
        e_cons = np.array([cons_error_for_family_size(f, resid, single_err) for f in fsz])
        cons_alt = m + rng.binomial(np.maximum(dp - m, 0), e_cons)
        naive_alt = m + rng.binomial(np.maximum(dp - m, 0), single_err)
        exp_alt = dp * vaf
        licensed = exp_alt >= 2.0
        for j in range(ncell):
            out.append((pos, bcs[j], int(dp[j]), int(naive_alt[j]), int(cons_alt[j]),
                        bool(is_car[j]), float(vaf), tier, float(fsz[j]),
                        int(m[j]), float(exp_alt[j]), bool(licensed[j]), "true", hap_id))
    return pd.DataFrame(out, columns=["pos", "cell_barcode", "DP", "naive_alt",
        "cons_alt", "is_carrier", "vaf", "tier", "fsize", "true_alt_mol",
        "exp_alt_mol", "licensed", "site_kind", "hap_group"])


# =================== 2. ARTIFACT-NEGATIVE set ================================
ARTIFACT_KINDS = ["strand", "readend", "lowbq", "oxidation", "covscaled"]


def build_artifact_matrix(df, artifact_pos, params, rng, covar_bg,
                          artifact_frac=0.15, min_dp=1):
    """Inject apparent-ALT into a recurrent cell subset at clean positions, with
    artifact hallmarks. These are NON-carriers (is_carrier=False) that
    nonetheless show recurrent apparent ALT -> designed to trip the Phase-2
    mixture LR. Their per-site covariate summaries are biased so the Phase-3
    features can catch them. VAF-analogue: apparent ALT fraction ~ 5-12%."""
    resid = params["consensus_residual_by_family_size"]
    single_err = float(params["implied_single_read_error_rate"])
    out = []; site_covar = {}
    ap = artifact_pos.reset_index(drop=True)
    for i, (_, row) in enumerate(ap.iterrows()):
        pos = int(row["pos"]); tier = row["tier"]
        kind = ARTIFACT_KINDS[i % len(ARTIFACT_KINDS)]
        sub = df[(df["pos"] == pos) & (df["n_mol"] >= min_dp)]
        if len(sub) < 20:
            continue
        dp = sub["n_mol"].values.astype(int)
        fsz = sub["mean_family_size"].values
        bcs = sub["cell_barcode"].values
        ncell = len(dp)
        # apparent-ALT fraction for the recurrent artifact (looks like a 5-12% carrier)
        art_vaf = float(rng.uniform(0.05, 0.12))
        n_art = max(3, int(round(artifact_frac * ncell)))
        art_idx = rng.choice(ncell, size=min(n_art, ncell), replace=False)
        is_art = np.zeros(ncell, bool); is_art[art_idx] = True
        m = np.zeros(ncell, int)
        if kind == "covscaled":
            # coverage-DEPENDENT apparent VAF: deeper (more reads) cells show a
            # HIGHER apparent ALT fraction — the recurrent-artifact hallmark
            # 3.1 vaf_cov_corr is built to catch. Reach-through cells all get
            # apparent ALT; fraction scales with rank read-count.
            reads_here = (fsz * dp)
            rank = np.argsort(np.argsort(reads_here)) / max(ncell - 1, 1)
            eff_vaf = 0.02 + 0.15 * rank            # 2% (shallow) -> 17% (deep)
            is_art = eff_vaf > 0                     # all covered cells carry it
            m = rng.binomial(dp, eff_vaf)
            art_vaf = float(np.mean(eff_vaf))
        else:
            # apparent ALT molecules ~ Binomial(DP, art_vaf) on the "carrier" subset
            m[is_art] = rng.binomial(dp[is_art], art_vaf)
        e_cons = np.array([cons_error_for_family_size(f, resid, single_err) for f in fsz])
        cons_alt = m + rng.binomial(np.maximum(dp - m, 0), e_cons)
        naive_alt = m + rng.binomial(np.maximum(dp - m, 0), single_err)
        exp_alt = dp * art_vaf
        licensed = exp_alt >= 2.0
        for j in range(ncell):
            out.append((pos, bcs[j], int(dp[j]), int(naive_alt[j]), int(cons_alt[j]),
                        False, float(art_vaf), tier, float(fsz[j]),
                        0, float(exp_alt[j]), bool(licensed[j]),
                        f"artifact_{kind}", bool(is_art[j])))
        # ---- biased site covariate summary for the ALT of this artifact site ----
        bg = covar_bg.get(str(pos), {})
        ref_base = bg.get("ref_base", "A")
        total_alt = int(m.sum())
        if kind == "strand":
            alt_fwd = total_alt; alt_rev = 0                      # one-strand
            end_frac = bg.get("readpos_end_frac", 0.3)
            bq_mean = bg.get("bq_mean", 36.0)
            alt_base = {"A": "G", "C": "T", "G": "A", "T": "C"}[ref_base]  # transition
        elif kind == "readend":
            alt_fwd = int(total_alt * 0.5); alt_rev = total_alt - alt_fwd
            end_frac = 0.9                                        # clustered at ends
            bq_mean = bg.get("bq_mean", 36.0)
            alt_base = {"A": "G", "C": "T", "G": "A", "T": "C"}[ref_base]
        elif kind == "lowbq":
            alt_fwd = int(total_alt * 0.5); alt_rev = total_alt - alt_fwd
            end_frac = bg.get("readpos_end_frac", 0.3)
            bq_mean = 20.0                                        # low base quality
            alt_base = {"A": "G", "C": "T", "G": "A", "T": "C"}[ref_base]
        elif kind == "oxidation":  # oxidation / NUMT-like: G>T lopsided
            alt_fwd = total_alt; alt_rev = 0
            end_frac = bg.get("readpos_end_frac", 0.3)
            bq_mean = bg.get("bq_mean", 36.0)
            ref_base = "G"; alt_base = "T"   # oriented G>T (8-oxo-dG hallmark)
        else:  # covscaled: CLEAN on all axes except coverage-scaling. Strand
            # follows the position's real background; readpos/BQ normal; a real
            # mtDNA transition. Its ONLY artifact tell is the VAF-coverage corr.
            fwd_frac = bg.get("fwd_frac", 0.5)
            alt_fwd = int(round(total_alt * fwd_frac)); alt_rev = total_alt - alt_fwd
            end_frac = bg.get("readpos_end_frac", 0.3)
            bq_mean = bg.get("bq_mean", 36.0)
            alt_base = {"A": "G", "C": "T", "G": "A", "T": "C"}[ref_base]
        site_covar[pos] = {"kind": kind, "ref_base": ref_base, "alt_base": alt_base,
                           "alt_fwd": alt_fwd, "alt_rev": alt_rev,
                           "alt_end_frac": float(end_frac), "alt_bq_mean": float(bq_mean),
                           "art_vaf": art_vaf}
    long = pd.DataFrame(out, columns=["pos", "cell_barcode", "DP", "naive_alt",
        "cons_alt", "is_carrier", "vaf", "tier", "fsize", "true_alt_mol",
        "exp_alt_mol", "licensed", "site_kind", "is_apparent_alt"])
    long["hap_group"] = -1
    return long, site_covar


# =================== 3. per-site feature extraction ==========================
def true_site_covar(pos, sub, covar_bg, rng):
    """A TRUE-variant site inherits the position's REAL background covariates
    (clean): ALT strand ~ background fwd frac, readpos ~ background end frac, BQ
    ~ background BQ mean. Substitution assigned a real mtDNA transition."""
    bg = covar_bg.get(str(pos), {})
    ref_base = bg.get("ref_base", "A")
    total_alt = int(sub["true_alt_mol"].sum())
    fwd_frac = bg.get("fwd_frac", 0.5)
    alt_fwd = int(round(total_alt * fwd_frac)); alt_rev = total_alt - alt_fwd
    # real mtDNA transition on this ref base
    trans = {"A": "G", "G": "A", "C": "T", "T": "C"}
    alt_base = trans.get(ref_base, "G")
    return {"ref_base": ref_base, "alt_base": alt_base, "alt_fwd": alt_fwd,
            "alt_rev": alt_rev, "alt_end_frac": bg.get("readpos_end_frac", 0.3),
            "alt_bq_mean": bg.get("bq_mean", 36.0)}


def extract_site_features(long, covar_bg, art_covar, anchor_sites=None):
    """Per-site Phase-3 artifact feature vectors + phasing. Returns DataFrame
    indexed by pos with FEATURE_NAMES columns + label (1 real / 0 artifact)."""
    rng = np.random.default_rng(7)
    rows = []
    # per-site barcode-indexed alt/dp Series so phasing aligns on shared cells
    site_series = {}
    for pos, sub in long.groupby("pos"):
        site_series[pos] = pd.DataFrame({
            "cons_alt": sub["cons_alt"].values, "DP": sub["DP"].values},
            index=sub["cell_barcode"].values)
    # phasing partners = the true-variant anchor sites (guilt-by-association)
    if anchor_sites is None:
        anchor_sites = [p for p, s in long.groupby("pos")
                        if s["site_kind"].iloc[0] == "true"]
    for pos, sub in long.groupby("pos"):
        kind = sub["site_kind"].iloc[0]
        alt = sub["cons_alt"].values; dp = sub["DP"].values
        reads = sub["fsize"].values * dp
        if kind == "true":
            sc = true_site_covar(pos, sub, covar_bg, rng)
            label = 1
        else:
            sc = art_covar[pos]; label = 0
        feats = AF.site_artifact_features(
            alt, dp, sc["ref_base"], sc["alt_base"], sc["alt_fwd"], sc["alt_rev"],
            sc["alt_end_frac"], sc["alt_bq_mean"], covar_bg.get(str(pos), {}),
            reads=reads)
        # 3.4 phasing: max cross-cell concordance with any OTHER anchor, aligned
        # on the cells covered at BOTH sites (inner-join on barcode).
        this = site_series[pos]
        best = 0.0
        for p2 in anchor_sites:
            if p2 == pos:
                continue
            other = site_series[p2]
            common = this.index.intersection(other.index)
            if len(common) < 20:
                continue
            cs = PH.cosegregation(this.loc[common, "cons_alt"].values,
                                  this.loc[common, "DP"].values,
                                  other.loc[common, "cons_alt"].values,
                                  other.loc[common, "DP"].values)
            if cs["concordance"] > best:
                best = cs["concordance"]
        feats["phasing"] = float(best)
        row = {"pos": pos, "site_kind": kind, "label": label, **feats}
        rows.append(row)
    fdf = pd.DataFrame(rows)
    return fdf


# =================== 5. calibrate feature weights ============================
FEATURES_ALL = ["sig_score", "strand_bias", "pos_in_read", "base_quality",
                "vaf_cov_corr", "phasing"]
# expected sign of effect on carrier prior (for interpretability / LOO reporting)
FEATURE_SIGN = {"sig_score": +1, "strand_bias": -1, "pos_in_read": -1,
                "base_quality": -1, "vaf_cov_corr": -1, "phasing": +1}


def fit_feature_weights(fdf, features, train_mask):
    """Logistic regression of site label (1 real / 0 artifact) on the feature
    vector -> weights used as the logit adjustment. Standardize features first.
    Pure scipy. Returns (weights dict, intercept, standardizer)."""
    from scipy.optimize import minimize
    X = fdf.loc[train_mask, features].values.astype(float)
    y = fdf.loc[train_mask, "label"].values.astype(float)
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    n, p = Xs.shape
    Xa = np.hstack([Xs, np.ones((n, 1))])

    def nll(w):
        z = Xa @ w
        ll = np.where(z >= 0, np.log1p(np.exp(-z)) + (1 - y) * z,
                      np.log1p(np.exp(z)) - y * z)
        return float(ll.sum() + 0.5 * 1.0 * np.sum(w[:-1] ** 2))  # L2 on slopes
    res = minimize(nll, np.zeros(p + 1), method="L-BFGS-B")
    w = res.x
    weights = {f: float(w[i]) for i, f in enumerate(features)}
    return weights, float(w[-1]), {"mu": mu.tolist(), "sd": sd.tolist(),
                                    "features": features}


def site_logit_adjustment(fdf, features, weights, intercept, std):
    """Compute the per-site logit adjustment from calibrated weights. Centered so
    the mean adjustment over real sites is ~0 (features shift RELATIVE to a
    typical real site, so we neither uniformly inflate nor deflate)."""
    mu = np.array(std["mu"]); sd = np.array(std["sd"])
    X = fdf[features].values.astype(float)
    Xs = (X - mu) / sd
    w = np.array([weights[f] for f in features])
    raw = Xs @ w  # excludes intercept: relative site adjustment
    return dict(zip(fdf["pos"].values, raw))


# =================== 6. score under a feature config =========================
def score_config(long, prior, site_logit_map, k_min=2):
    """Run the fused posterior per site with a per-site logit adjustment map.
    Returns per-cell posterior array aligned to long.index."""
    post = np.zeros(len(long))
    for pos, idx in long.groupby("pos").groups.items():
        idx = np.array(list(idx))
        sub = long.loc[idx]
        alt = sub["cons_alt"].values; dp = sub["DP"].values
        sn = fit_site_null(alt, dp, prior)
        mix = mixture_lr(alt, dp, sn["mu"], sn["s"])
        sfl = float(site_logit_map.get(pos, 0.0))
        fz = fuse_posterior(alt, dp, sn, mix, k_min=k_min, site_feature_logit=sfl)
        locs = [long.index.get_loc(i) for i in idx]
        post[locs] = fz["posterior"]
    return post


def _adjust_map_for(fdf, features, weights, intercept, std):
    if not features:
        return {}
    sub_w = {f: weights[f] for f in features}
    sub_std = {"mu": [std["mu"][std["features"].index(f)] for f in features],
               "sd": [std["sd"][std["features"].index(f)] for f in features],
               "features": features}
    return site_logit_adjustment(fdf, features, sub_w, intercept, sub_std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pileup", required=True)
    ap.add_argument("--spike-positions", required=True)
    ap.add_argument("--candidate-positions", required=True)
    ap.add_argument("--consensus-params", required=True)
    ap.add_argument("--covariates", required=True, help="read_covariate_backgrounds.json")
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
    with open(args.covariates) as fh:
        covar_bg = json.load(fh)

    # ---- EB genome-wide prior (identical to Phase 2) ----
    bg_pos = cand[(cand["is_background"]) & (~cand["is_anchor"])]["pos"].tolist()
    site_alt_dp = []
    for pos in bg_pos:
        sub = df[df["pos"] == pos]
        if len(sub) < 10:
            continue
        cc = sub[["cons_A", "cons_C", "cons_G", "cons_T"]].sum().values
        ref_i = int(np.argmax(cc)); bases = ["cons_A", "cons_C", "cons_G", "cons_T"]
        alt = sub["n_mol"].values - sub[bases[ref_i]].values
        site_alt_dp.append((alt.astype(float), sub["n_mol"].values.astype(float)))
    prior = fit_global_prior(site_alt_dp,
                             single_err_floor=float(params["implied_single_read_error_rate"]))
    sys.stderr.write(f"[p3] EB prior: {prior}\n")

    # ---- 1. TRUE-carrier matrix (Phase-2 identical) ----
    true_long = build_true_matrix(df, spike, params, rng, carrier_frac=args.carrier_frac)
    true_pos = set(true_long["pos"].unique())

    # ---- 2. ARTIFACT-NEGATIVE positions: clean B_peak/mid candidates NOT spiked ----
    clean = cand[(cand["is_background"]) & (~cand["is_anchor"]) &
                 (~cand["pos"].isin(true_pos)) & (cand["tier"].isin(["B_peak", "mid"]))]
    # keep positions that actually have enough covered cells
    good_art = [p for p in clean["pos"].tolist()
                if len(df[(df["pos"] == p) & (df["n_mol"] >= 1)]) >= 20]
    n_art = min(len(good_art), max(40, len(true_pos)))
    art_sel = rng.choice(good_art, size=n_art, replace=False)
    artifact_pos = clean[clean["pos"].isin(art_sel)][["pos", "tier"]].copy()
    art_long, art_covar = build_artifact_matrix(df, artifact_pos, params, rng, covar_bg,
                                                artifact_frac=args.carrier_frac)
    sys.stderr.write(f"[p3] true sites={len(true_pos)}, artifact sites={art_long['pos'].nunique()}\n")

    # true carriers are not "apparent-only" artifact cells; align columns
    true_long = true_long.assign(is_apparent_alt=false_col(true_long))
    long = pd.concat([true_long, art_long], ignore_index=True, sort=False)
    long = long.reset_index(drop=True)
    _write_table(long, os.path.join(args.outdir, "phase3_spiked_matrix.csv.gz"))

    # ---- 3-4. per-site feature extraction ----
    fdf = extract_site_features(long, covar_bg, art_covar)
    fdf.to_csv(os.path.join(args.outdir, "site_features.csv"), index=False)
    sys.stderr.write(f"[p3] features: {fdf['label'].value_counts().to_dict()}\n")

    # ---- 5. train/test split (site level), calibrate weights on TRAIN ----
    rng2 = np.random.default_rng(args.seed + 1)
    is_train = np.zeros(len(fdf), bool)
    for lab in [0, 1]:
        idx = np.where(fdf["label"].values == lab)[0]
        tr = rng2.choice(idx, size=int(round(0.5 * len(idx))), replace=False)
        is_train[tr] = True
    fdf["is_train"] = is_train
    weights, intercept, std = fit_feature_weights(fdf, FEATURES_ALL, is_train)
    sys.stderr.write(f"[p3] feature weights: {weights}\n")
    with open(os.path.join(args.outdir, "feature_weights.json"), "w") as fh:
        json.dump({"weights": weights, "intercept": intercept, "std": std,
                   "feature_sign": FEATURE_SIGN}, fh, indent=2)

    # evaluate ablation on the HELD-OUT sites only
    test_pos = set(fdf.loc[~fdf["is_train"], "pos"].values)
    long_test = long[long["pos"].isin(test_pos)].copy().reset_index(drop=True)
    fdf_test = fdf[~fdf["is_train"]].copy()

    # ---- 7. ABLATION ----
    def specificity_labels(L):
        """Positive = licensed Tier-B >=5% true carrier; negative = artifact
        apparent-ALT cell. Others excluded from the specificity contrast."""
        is_true_car = (L["site_kind"] == "true") & (L["is_carrier"]) & \
                      (L["tier"] == "B_peak") & (L["vaf"] >= VAF_FLOOR) & (L["licensed"])
        is_art_appar = (L["site_kind"].str.startswith("artifact")) & (L["is_apparent_alt"])
        sel = is_true_car | is_art_appar
        y = is_true_car[sel].astype(int).values
        return sel.values, y

    def detection_labels(L):
        """Tier-B >=5% detection set: true carriers (licensed) vs clean
        non-carriers at true sites."""
        tsel = (L["site_kind"] == "true") & (L["tier"] == "B_peak") & (L["vaf"] >= VAF_FLOOR)
        car = L["is_carrier"] & L["licensed"]
        scored = tsel & ((~L["is_carrier"]) | car)
        y = (L["is_carrier"] & L["licensed"])[scored].astype(int).values
        return scored.values, y

    configs = {"baseline_P2": []}
    for f in FEATURES_ALL:
        configs[f"only_{f}"] = [f]
    configs["all_features"] = list(FEATURES_ALL)
    for f in FEATURES_ALL:
        configs[f"leaveout_{f}"] = [x for x in FEATURES_ALL if x != f]

    sel_sp, y_sp = specificity_labels(long_test)
    sel_dt, y_dt = detection_labels(long_test)
    abl_rows = []
    posteriors = {}
    for name, feats in configs.items():
        smap = _adjust_map_for(fdf_test, feats, weights, intercept, std)
        post = score_config(long_test, prior, smap, k_min=args.kmin)
        posteriors[name] = post
        abl_rows.append({
            "config": name, "features": ",".join(feats) if feats else "(none)",
            "specificity_pr_auc": pr_auc(y_sp, post[sel_sp]),
            "specificity_prec_at_50recall": precision_at_recall(y_sp, post[sel_sp], 0.5),
            "detection_pr_auc": pr_auc(y_dt, post[sel_dt]),
            "detection_prec_at_50recall": precision_at_recall(y_dt, post[sel_dt], 0.5),
            "n_spec_pos": int(y_sp.sum()), "n_spec_neg": int((y_sp == 0).sum()),
            "n_det_pos": int(y_dt.sum()), "n_det_neg": int((y_dt == 0).sum()),
        })
    # ---- decide which features to KEEP (calibrated contribution on held-out) ----
    _abl = pd.DataFrame(abl_rows).set_index("config")["specificity_pr_auc"]
    base_sp = _abl.loc["baseline_P2"]
    keep, dropped = [], []
    for f in FEATURES_ALL:
        loo_drop = _abl.loc["all_features"] - _abl.loc[f"leaveout_{f}"]
        solo_gain = _abl.loc[f"only_{f}"] - base_sp
        if (loo_drop > 0.002) or (solo_gain > 0.005):
            keep.append(f)
        else:
            dropped.append(f)

    # ---- SELECTED-MODEL config: only the kept features (this is what ships) ----
    smap_kept = _adjust_map_for(fdf_test, keep, weights, intercept, std)
    post_kept = score_config(long_test, prior, smap_kept, k_min=args.kmin)
    abl_rows.append({
        "config": "kept_features", "features": ",".join(keep) if keep else "(none)",
        "specificity_pr_auc": pr_auc(y_sp, post_kept[sel_sp]),
        "specificity_prec_at_50recall": precision_at_recall(y_sp, post_kept[sel_sp], 0.5),
        "detection_pr_auc": pr_auc(y_dt, post_kept[sel_dt]),
        "detection_prec_at_50recall": precision_at_recall(y_dt, post_kept[sel_dt], 0.5),
        "n_spec_pos": int(y_sp.sum()), "n_spec_neg": int((y_sp == 0).sum()),
        "n_det_pos": int(y_dt.sum()), "n_det_neg": int((y_dt == 0).sum()),
    })
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(os.path.join(args.outdir, "ablation_table.csv"), index=False)
    sys.stderr.write(f"[p3] ablation done; kept={keep}\n")

    # ---- 8. CALIBRATION of the FINAL (selected-model) posterior on ALL cells ----
    # The final posterior uses only the KEPT features -> the model we actually
    # ship. Dropped features (no calibrated contribution) are excluded.
    smap_final = _adjust_map_for(fdf, keep, weights, intercept, std)
    long_all = long.copy().reset_index(drop=True)
    post_all = score_config(long_all, prior, smap_final, k_min=args.kmin)
    long_all["phase3_posterior"] = post_all
    # calibration truth per cell: is_carrier (true carriers) among covered cells.
    # Restrict to LICENSED cells (the honest callable set): unlicensed cells are
    # 'unknown' by design and excluded from a calibration that claims precision.
    cal = long_all[(long_all["DP"] > 0) & (long_all["licensed"] | (~long_all["is_carrier"]))].copy()
    # exclude artifact apparent-ALT? No: they are TRUE negatives (is_carrier=False)
    # -> keeping them tests whether high posteriors correspond to real carriers.
    cal["y"] = cal["is_carrier"].astype(int)
    # overall + stratified ECE by tier x VAF (use the true-site VAF; artifact
    # sites carry their apparent art_vaf, binned into the nearest ladder rung)
    cal["vaf_bin"] = cal["vaf"].apply(lambda v: min(VAF_LADDER, key=lambda x: abs(x - v)))
    pre = CAL.ece_mce(cal["y"].values, cal["phase3_posterior"].values, n_bins=12)
    strat_pre = CAL.stratified_ece(cal, "tier", "vaf_bin", "y", "phase3_posterior")

    # held-out recalibration: fit isotonic + Platt on train cells, apply to test
    cal_tr = cal.sample(frac=0.5, random_state=args.seed)
    cal_te = cal.drop(cal_tr.index)
    iso = CAL.fit_isotonic(cal_tr["y"].values, cal_tr["phase3_posterior"].values)
    platt, platt_ab = CAL.fit_platt(cal_tr["y"].values, cal_tr["phase3_posterior"].values)
    cal_te = cal_te.copy()
    cal_te["iso"] = iso(cal_te["phase3_posterior"].values)
    cal_te["platt"] = platt(cal_te["phase3_posterior"].values)
    post_iso = CAL.ece_mce(cal_te["y"].values, cal_te["iso"].values, n_bins=12)
    post_platt = CAL.ece_mce(cal_te["y"].values, cal_te["platt"].values, n_bins=12)
    pre_te = CAL.ece_mce(cal_te["y"].values, cal_te["phase3_posterior"].values, n_bins=12)
    # choose the better recalibration by test ECE
    best = "isotonic" if post_iso["ece"] <= post_platt["ece"] else "platt"
    cal_te["calibrated"] = cal_te["iso"] if best == "isotonic" else cal_te["platt"]
    # apply chosen map to ALL cells for the saved calibrated posterior
    chosen_map = iso if best == "isotonic" else platt
    long_all["phase3_posterior_calibrated"] = chosen_map(long_all["phase3_posterior"].values)
    _write_table(long_all[["pos", "cell_barcode", "site_kind", "tier", "vaf", "DP",
                           "cons_alt", "is_carrier", "is_apparent_alt", "licensed",
                           "phase3_posterior", "phase3_posterior_calibrated"]],
                 os.path.join(args.outdir, "phase3_calibrated_posterior.csv.gz"))

    # reliability diagrams (overall, pre & post) + stratified post-ECE
    rd_pre = CAL.reliability_diagram(cal["y"].values, cal["phase3_posterior"].values, n_bins=12)
    rd_post = CAL.reliability_diagram(cal_te["y"].values, cal_te["calibrated"].values, n_bins=12)
    strat_post = CAL.stratified_ece(
        long_all.assign(y=long_all["is_carrier"].astype(int),
                        vaf_bin=long_all["vaf"].apply(lambda v: min(VAF_LADDER, key=lambda x: abs(x - v))))
        .query("DP>0 and (licensed or not is_carrier)"),
        "tier", "vaf_bin", "y", "phase3_posterior_calibrated")
    pd.DataFrame(strat_pre).assign(stage="pre").to_csv(
        os.path.join(args.outdir, "ece_by_tier_vaf_pre.csv"), index=False)
    pd.DataFrame(strat_post).assign(stage="post").to_csv(
        os.path.join(args.outdir, "ece_by_tier_vaf_post.csv"), index=False)

    # ---- 3.3 callability map on the real pileup ----
    cmap, csum = build_callability_map(df)
    cmap.to_csv(os.path.join(args.outdir, "callability_map.csv"), index=False)

    # ---- phasing haplotype seeds (Phase-4 hand-off), aligned on barcodes ----
    ts_series = {p: pd.DataFrame({"cons_alt": s["cons_alt"].values,
                                  "DP": s["DP"].values},
                                 index=s["cell_barcode"].values)
                 for p, s in true_long.groupby("pos")}
    site_ids = list(ts_series)
    adj = {i: set() for i in range(len(site_ids))}
    for i in range(len(site_ids)):
        si = ts_series[site_ids[i]]
        for j in range(i + 1, len(site_ids)):
            sj = ts_series[site_ids[j]]
            common = si.index.intersection(sj.index)
            if len(common) < 20:
                continue
            cs = PH.cosegregation(si.loc[common, "cons_alt"].values,
                                  si.loc[common, "DP"].values,
                                  sj.loc[common, "cons_alt"].values,
                                  sj.loc[common, "DP"].values)
            if cs["concordance"] >= 0.3:
                adj[i].add(j); adj[j].add(i)
    seen = set(); seeds = []
    for i in range(len(site_ids)):
        if i in seen:
            continue
        stack = [i]; comp = []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u); comp.append(u)
            stack.extend(adj[u] - seen)
        if len(comp) >= 2:
            seeds.append([int(site_ids[k]) for k in comp])

    # validate recovered seeds against the injected haplotype-group truth
    true_hap = {}
    for hg, s in true_long[true_long["hap_group"] >= 0].groupby("hap_group"):
        true_hap[int(hg)] = sorted(set(int(p) for p in s["pos"].unique()))
    n_true_hap = len(true_hap)
    # a seed "recovers" a true group if it contains >=2 of that group's sites
    recovered = 0
    for grp in true_hap.values():
        for seed in seeds:
            if len(set(grp) & set(seed)) >= 2:
                recovered += 1
                break
    phasing_seed_validation = {
        "n_true_haplotype_groups": n_true_hap,
        "true_haplotype_groups": {str(k): v for k, v in true_hap.items()},
        "n_seeds_recovered": len(seeds),
        "n_true_groups_recovered": recovered,
    }

    summary = {
        "eb_prior": prior,
        "n_true_sites": len(true_pos),
        "n_artifact_sites": int(art_long["pos"].nunique()),
        "feature_weights": weights, "feature_intercept": intercept,
        "baseline_specificity_pr_auc": float(base_sp),
        "all_features_specificity_pr_auc": float(
            abl.loc[abl["config"] == "all_features", "specificity_pr_auc"].iloc[0]),
        "all_features_detection_pr_auc": float(
            abl.loc[abl["config"] == "all_features", "detection_pr_auc"].iloc[0]),
        "selected_model_specificity_pr_auc": float(
            abl.loc[abl["config"] == "kept_features", "specificity_pr_auc"].iloc[0]),
        "selected_model_detection_pr_auc": float(
            abl.loc[abl["config"] == "kept_features", "detection_pr_auc"].iloc[0]),
        "features_kept": keep, "features_dropped": dropped,
        "calibration": {
            "pre_ece_all": pre, "recalibration_chosen": best,
            "platt_params": platt_ab,
            "pre_ece_test": pre_te, "post_ece_isotonic": post_iso,
            "post_ece_platt": post_platt,
            "reliability_pre": rd_pre, "reliability_post": rd_post,
        },
        "callability": csum,
        "n_haplotype_seeds": len(seeds),
        "haplotype_seeds": seeds[:20],
        "phasing_seed_validation": phasing_seed_validation,
    }
    with open(os.path.join(args.outdir, "phase3_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    sys.stderr.write(f"[p3] KEEP={keep} DROP={dropped}\n")
    sys.stderr.write(f"[p3] specificity baseline={base_sp:.4f} all={summary['all_features_specificity_pr_auc']:.4f}\n")
    sys.stderr.write(f"[p3] calibration pre-ECE(all)={pre['ece']:.4f} post-ECE({best})="
                     f"{(post_iso if best=='isotonic' else post_platt)['ece']:.4f}\n")
    print(json.dumps({"keep": keep, "drop": dropped,
                      "spec_base": float(base_sp),
                      "spec_all": summary["all_features_specificity_pr_auc"],
                      "det_all": summary["all_features_detection_pr_auc"],
                      "pre_ece": pre["ece"],
                      "post_ece": (post_iso if best == "isotonic" else post_platt)["ece"]}))


def false_col(df):
    return np.zeros(len(df), bool)


if __name__ == "__main__":
    main()


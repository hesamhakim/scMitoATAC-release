#!/usr/bin/env python
"""
scMitoHet Phase 3 — CALIBRATION assessment of the final fused posterior.

Science-team directive (IN SCOPE): demonstrate the headline claim
"confidence x -> measured precision x" semi-synthetically on the injection
ground truth, since real-data calibration (Phase 5 / J5) is out of scope.

Provides:
  * reliability_diagram   : predicted P(carrier) vs observed carrier frequency
                            in probability bins (equal-width or quantile).
  * expected_calibration_error (ECE) and max_calibration_error (MCE).
  * stratified ECE/MCE by coverage tier x VAF.
  * isotonic and Platt (logistic) recalibration maps fit on held-out cells,
    with pre/post ECE.

Pure numpy/scipy (no sklearn) to keep the stock bioinformatics env sufficient.
"""
import numpy as np


def reliability_diagram(y_true, p_pred, n_bins=10, strategy="quantile"):
    """Bin predictions and compute per-bin (mean_pred, obs_freq, count).
    strategy: 'quantile' (equal-count) or 'uniform' (equal-width)."""
    y_true = np.asarray(y_true, float); p_pred = np.asarray(p_pred, float)
    m = ~np.isnan(p_pred)
    y_true, p_pred = y_true[m], p_pred[m]
    if len(p_pred) == 0:
        return {"mean_pred": [], "obs_freq": [], "count": [], "bin_edges": []}
    if strategy == "quantile":
        edges = np.unique(np.quantile(p_pred, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([p_pred.min(), p_pred.max() + 1e-9])
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p_pred, edges[1:-1], right=False), 0, len(edges) - 2)
    mean_pred, obs_freq, count = [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() == 0:
            continue
        mean_pred.append(float(p_pred[sel].mean()))
        obs_freq.append(float(y_true[sel].mean()))
        count.append(int(sel.sum()))
    return {"mean_pred": mean_pred, "obs_freq": obs_freq, "count": count,
            "bin_edges": edges.tolist()}


def ece_mce(y_true, p_pred, n_bins=10, strategy="quantile"):
    """Expected & max calibration error (weighted by bin count)."""
    rd = reliability_diagram(y_true, p_pred, n_bins=n_bins, strategy=strategy)
    mp = np.array(rd["mean_pred"]); of = np.array(rd["obs_freq"])
    ct = np.array(rd["count"], float)
    if ct.sum() == 0:
        return {"ece": float("nan"), "mce": float("nan"), "n": 0}
    gaps = np.abs(mp - of)
    ece = float(np.sum(ct / ct.sum() * gaps))
    mce = float(gaps.max()) if len(gaps) else float("nan")
    return {"ece": ece, "mce": mce, "n": int(ct.sum())}


def stratified_ece(df, tier_col, vaf_col, y_col, p_col, n_bins=8):
    """ECE/MCE per (tier, vaf) cell. Returns list of dict rows."""
    rows = []
    for (tier, vaf), sub in df.groupby([tier_col, vaf_col]):
        y = sub[y_col].values.astype(float)
        p = sub[p_col].values.astype(float)
        if len(p) < 20 or y.sum() < 2 or (len(y) - y.sum()) < 2:
            em = {"ece": float("nan"), "mce": float("nan"), "n": int(len(p))}
        else:
            em = ece_mce(y, p, n_bins=min(n_bins, max(2, int(len(p) / 10))))
        rows.append({"tier": tier, "vaf": float(vaf), "n_cells": int(len(p)),
                     "n_pos": int(y.sum()), "ece": em["ece"], "mce": em["mce"]})
    return rows


# ---------------- recalibration maps ----------------------------------------
def fit_isotonic(y_true, p_pred):
    """Pool-adjacent-violators isotonic regression. Returns a callable map."""
    y_true = np.asarray(y_true, float); p_pred = np.asarray(p_pred, float)
    m = ~np.isnan(p_pred)
    y_true, p_pred = y_true[m], p_pred[m]
    order = np.argsort(p_pred)
    x = p_pred[order]; y = y_true[order]
    # PAVA
    w = np.ones_like(y)
    yhat = y.copy().astype(float)
    i = 0
    # standard PAVA with weighted means
    level_y = list(yhat); level_w = list(w); level_x = list(x)
    blocks_y = []; blocks_w = []; blocks_xmax = []
    for k in range(len(x)):
        cy = y[k]; cw = 1.0; cxmax = x[k]
        while blocks_y and blocks_y[-1] > cy:
            py = blocks_y.pop(); pw = blocks_w.pop(); pxm = blocks_xmax.pop()
            cy = (py * pw + cy * cw) / (pw + cw)
            cw = pw + cw
            cxmax = max(pxm, cxmax)
        blocks_y.append(cy); blocks_w.append(cw); blocks_xmax.append(cxmax)
    # expand block right-edges (thresholds) and fitted values
    thr = np.array(blocks_xmax); val = np.array(blocks_y)

    def _map(p):
        p = np.asarray(p, float)
        idx = np.searchsorted(thr, p, side="left")
        idx = np.clip(idx, 0, len(val) - 1)
        return val[idx]
    return _map


def fit_platt(y_true, p_pred, l2=1.0):
    """Platt scaling: logistic regression of y on logit(p). L2-regularized so it
    does not diverge on (near-)separable data. Returns (callable, {a,b})."""
    from scipy.optimize import minimize
    y_true = np.asarray(y_true, float); p_pred = np.asarray(p_pred, float)
    m = ~np.isnan(p_pred)
    y = y_true[m]; p = np.clip(p_pred[m], 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))

    def nll(ab):
        a, b = ab
        logit = np.clip(a * z + b, -30, 30)   # clip to avoid exp overflow
        # numerically-stable log-loss via logaddexp
        ll = np.logaddexp(0.0, logit) - y * logit
        # L2 shrink toward identity map (a=1,b=0)
        return float(np.sum(ll) + l2 * ((a - 1.0) ** 2 + b ** 2))
    res = minimize(nll, [1.0, 0.0], method="L-BFGS-B",
                   bounds=[(-10, 10), (-10, 10)])
    a, b = res.x

    def _map(pp):
        pp = np.clip(np.asarray(pp, float), 1e-6, 1 - 1e-6)
        zz = np.log(pp / (1 - pp))
        return 1.0 / (1.0 + np.exp(-(a * zz + b)))
    return _map, {"a": float(a), "b": float(b)}


# ============================================================================
# Phase 4 — PER-STRATUM calibration (science-team directive; supersedes the
# single global isotonic map that capped confidence at 0.66).
#
# Root cause of the 0.66 ceiling (verified Phase 3): a single global isotonic
# map is dominated by the abundant low-VAF / trough cells whose true carrier
# frequency is low, so it pulls DOWN the high-VAF bins where precision is
# already 1.0. The fix is to fit a SEPARATE monotone map per stratum
# (per-tier, ideally per-tier x VAF-band): each stratum is calibrated against
# its OWN reliability, so the licensed deep high-VAF stratum (Tier-B 25%) can
# reach ~1.0 while the trough strata stay low -- both honest.
# ============================================================================
def _vaf_band_fn(vaf_bands):
    def _band(v):
        for lo, hi, name in vaf_bands:
            if lo <= v < hi:
                return name
        return "hi"
    return _band


def fit_per_stratum_isotonic(df, y_col, p_col, tier_col, vaf_col=None,
                             min_n=40, min_pos=3, vaf_bands=None):
    """Fit one isotonic map per stratum with a TWO-LEVEL FALLBACK.

    Stratum = tier (if vaf_col is None) or (tier, vaf_band). A row is calibrated
    by the most specific map that was populated enough to fit:
        (tier, band) map  ->  tier map  ->  global map
    so an under-populated VAF band borrows its tier's robust map rather than the
    global map (which is what re-introduces the ceiling). Returns
    (predict_fn, info). Requires y in {0,1}, p in [0,1].
    """
    import numpy as np
    df = df.copy()
    tiers = df[tier_col].astype(str)
    if vaf_col is not None and vaf_bands is not None:
        bandfn = _vaf_band_fn(vaf_bands)
        bands = df[vaf_col].astype(float).map(bandfn)
        df["_stratum"] = list(zip(tiers, bands))
        use_bands = True
    elif vaf_col is not None:
        df["_stratum"] = list(zip(tiers, df[vaf_col].astype(float)))
        use_bands = False
    else:
        df["_stratum"] = tiers
        use_bands = None

    def _fit(y, p):
        return fit_isotonic(y, p)

    def _ok(y):
        npos = int(y.sum())
        return len(y) >= min_n and npos >= min_pos and (len(y) - npos) >= min_pos

    # level 0: global
    global_map = _fit(df[y_col].values, df[p_col].values)
    # level 1: per-tier
    tier_maps = {}
    for t, sub in df.groupby(tier_col):
        y = sub[y_col].values.astype(float)
        if _ok(y):
            tier_maps[str(t)] = _fit(y, sub[p_col].values.astype(float))
    # level 2: per-stratum (tier, band)
    strat_maps = {}
    info = {}
    for strat, sub in df.groupby("_stratum"):
        y = sub[y_col].values.astype(float)
        npos = int(y.sum())
        if _ok(y):
            strat_maps[strat] = _fit(y, sub[p_col].values.astype(float))
            info[str(strat)] = {"n": int(len(y)), "n_pos": npos, "level": "stratum"}
        else:
            tk = str(strat[0]) if isinstance(strat, tuple) else str(strat)
            lvl = "tier" if tk in tier_maps else "global"
            info[str(strat)] = {"n": int(len(y)), "n_pos": npos, "level": lvl}

    def predict(sub_df):
        sub = sub_df
        tk = sub[tier_col].astype(str)
        if use_bands is True:
            bandfn = _vaf_band_fn(vaf_bands)
            strata = list(zip(tk, sub[vaf_col].astype(float).map(bandfn)))
        elif use_bands is False:
            strata = list(zip(tk, sub[vaf_col].astype(float)))
        else:
            strata = list(tk)
        p = sub[p_col].values.astype(float)
        out = np.empty(len(p), float)
        for i, (st, ti, pi) in enumerate(zip(strata, tk, p)):
            m = strat_maps.get(st) or tier_maps.get(str(ti)) or global_map
            out[i] = float(m(np.array([pi]))[0])
        return out

    return predict, info

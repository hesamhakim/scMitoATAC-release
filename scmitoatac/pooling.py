#!/usr/bin/env python
"""
scMitoHet Phase 4.4 — METACELL / CLONE POOLING -> per-cluster heteroplasmy (Tier C).

Promotes the clone partition from a confidence feature to a COVERAGE-POOLING
UNIT. For trough positions (~80% of chrM) a per-cell low-VAF call is not
licensed (median-position depth ~4x -> a 2% variant is ~0.08 expected ALT
molecules per cell). Summing REAL consensus AD/DP across the cells of a
homogeneity-validated clone reaches callable pooled depth, and a hierarchical
beta-binomial returns a per-CLONE heteroplasmy estimate + credible interval.

This is aggregation of real observations + empirical-Bayes shrinkage across
clones -- NOT imputation. No unobserved count is ever synthesized (contrast
SAVER/scVI/ALRA, firmly excluded by the design note): a cell with DP=0 at the
position contributes (0,0) to the pool, nothing more.

Model
-----
Per clone c at a position, pool the member cells' consensus counts:
    K_c = sum_i alt_ci ,  N_c = sum_i dp_ci   (real observations only)

Population hyperprior (the hierarchical / strength-borrowing layer), EB-fit
across all clones' pooled fractions by method-of-moments on a Beta(a0,b0):
    theta_c ~ Beta(a0, b0)

Per-clone posterior (Beta-Binomial conjugacy, pooled counts as the likelihood):
    theta_c | K_c, N_c  ~  Beta(a0 + K_c, b0 + N_c - K_c)
    point   = (a0 + K_c) / (a0 + b0 + N_c)
    CI      = Beta quantiles (default 95%)

Detection / callability test (is the clone a real carrier vs error only):
    H0: theta_c = mu_e   (position/tier error null, over-dispersed beta-binom)
    one-sided pooled tail p = P(X >= K_c | N_c, mu_e, s_e)   [beta-binomial]
    callable if  p < alpha  AND  K_c >= min_alt_pool  AND  CI_low > mu_e
"""
import numpy as np
from scipy.stats import betabinom, beta as beta_dist


def _bb_sf(k, n, mu, s):
    """P(X >= k | n, beta-binom(mu, s)) inclusive, over-dispersed."""
    a = mu * s
    b = (1.0 - mu) * s
    if k <= 0:
        return 1.0
    # sf(k-1) = P(X >= k)
    return float(betabinom.sf(k - 1, n, a, b))


def fit_population_prior(pooled_fracs, min_var=1e-6, prior_strength_cap=1e4):
    """EB hyperprior Beta(a0,b0) by method-of-moments across clones' pooled
    fractions. Returns (a0, b0). Falls back to a weak prior when the spread is
    degenerate (all clones near one value)."""
    p = np.asarray(pooled_fracs, float)
    p = p[np.isfinite(p)]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if len(p) < 3:
        return 0.5, 0.5  # Jeffreys, near-uninformative
    m = p.mean()
    v = max(p.var(ddof=1), min_var)
    # method of moments for Beta
    common = m * (1 - m) / v - 1.0
    if common <= 0 or not np.isfinite(common):
        return 0.5, 0.5
    a0 = m * common
    b0 = (1 - m) * common
    s = a0 + b0
    if s > prior_strength_cap:  # cap so a huge clone-count doesn't dominate
        a0 *= prior_strength_cap / s
        b0 *= prior_strength_cap / s
    return float(max(a0, 1e-3)), float(max(b0, 1e-3))


def pooled_estimate(K, N, a0, b0, mu_e, s_e, alpha=0.01, min_alt_pool=3,
                    ci=0.95):
    """Per-clone pooled heteroplasmy estimate + callability.

    K, N   : pooled ALT and DP (sums of REAL per-cell consensus counts).
    a0, b0 : EB population hyperprior (fit_population_prior).
    mu_e, s_e : position/tier error null (beta-binomial).
    Returns dict: vaf_point, vaf_ci_low, vaf_ci_high, pooled_alt, pooled_dp,
                  tail_p, callable, ci_low_gt_null.
    """
    K = float(K); N = float(N)
    if N <= 0:
        return {"vaf_point": np.nan, "vaf_ci_low": np.nan, "vaf_ci_high": np.nan,
                "pooled_alt": K, "pooled_dp": N, "tail_p": 1.0,
                "callable": False, "ci_low_gt_null": False}
    post_a = a0 + K
    post_b = b0 + (N - K)
    point = post_a / (post_a + post_b)
    lo = float(beta_dist.ppf((1 - ci) / 2, post_a, post_b))
    hi = float(beta_dist.ppf(1 - (1 - ci) / 2, post_a, post_b))
    tail_p = _bb_sf(int(round(K)), int(round(N)), mu_e, s_e)
    ci_low_gt_null = lo > mu_e
    is_callable = bool((tail_p < alpha) and (K >= min_alt_pool) and ci_low_gt_null)
    return {"vaf_point": float(point), "vaf_ci_low": lo, "vaf_ci_high": hi,
            "pooled_alt": K, "pooled_dp": N, "tail_p": float(tail_p),
            "callable": is_callable, "ci_low_gt_null": bool(ci_low_gt_null)}


def pool_clone_cells(alt, dp):
    """Sum real per-cell consensus counts into (K, N). alt, dp are arrays over
    the clone's member cells. DP=0 cells contribute nothing (no imputation)."""
    alt = np.asarray(alt, float); dp = np.asarray(dp, float)
    m = dp > 0
    return float(alt[m].sum()), float(dp[m].sum()), int(m.sum())


def call_clones(clone_table, mu_e_by_pos, s_e, alpha=0.01, min_alt_pool=3,
                ci=0.95, group_cols=("pos", "clone_id")):
    """Call every clone in a long table of per-cell (pos, clone_id, alt, dp).

    clone_table : DataFrame with columns pos, clone_id, cons_alt, DP.
    mu_e_by_pos : dict pos -> error mean (fallback to global if missing).
    Returns a DataFrame of per-clone pooled estimates + callability, using an
    EB population prior fit across the clones present.
    """
    import pandas as pd
    g = clone_table.groupby(list(group_cols))
    recs = []
    fracs = []
    for keys, sub in g:
        K, N, ncov = pool_clone_cells(sub["cons_alt"].values, sub["DP"].values)
        fracs.append(K / N if N > 0 else np.nan)
        recs.append({**dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,))),
                     "K": K, "N": N, "n_cells": int(len(sub)), "n_cov_cells": ncov})
    a0, b0 = fit_population_prior(fracs)
    out = []
    for r in recs:
        pos = r.get("pos")
        mu_e = float(mu_e_by_pos.get(pos, mu_e_by_pos.get("__global__", 1e-4)))
        est = pooled_estimate(r["K"], r["N"], a0, b0, mu_e, s_e,
                              alpha=alpha, min_alt_pool=min_alt_pool, ci=ci)
        out.append({**r, **est, "mu_e": mu_e, "a0": a0, "b0": b0})
    return pd.DataFrame(out)

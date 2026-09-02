#!/usr/bin/env python
"""
scMitoHet Phase 4.4(b) — mtDNA-genotype HOMOGENEITY TEST (first-class).

Before a clone is pooled (pooling.py) its member cells must be certified to
share one mtDNA genotype at the pooled position. A chimeric cluster -- cells
from >1 lineage pooled together -- yields a high-coverage (therefore
high-confidence) VAF that belongs to NO real cell: it is CONFIDENTLY WRONG, the
worst failure mode. So the homogeneity test is not optional; it gates pooling.

No published mtDNA method certifies genotype homogeneity before pooling
(MitoDelta / SComatic pool by expression cell-type). This extends the mcRigor
metacell-homogeneity idea from expression to GENOTYPE.

Test
----
At a candidate position, the clone's per-cell consensus counts (alt_i, dp_i)
are the data. Two hypotheses:

  H0 (homogeneous): all covered cells share ONE beta-binomial carrier mean
     theta (plus over-dispersion s). One lineage -> one theta.
  H1 (chimeric):    a two-component beta-binomial mixture -- fraction f of
     cells at theta_hi, (1-f) at theta_lo -- i.e. two sub-lineages with
     different genotypes pooled together.

LR = 2 * (loglik_H1 - loglik_H0).  Under H0, LR ~ chi2 with ~2 extra df
(f, and the second mean). SPLIT the clone (declare chimeric) when

    LR > chi2_thresh (default chi2_2 @ 0.01 = 9.21)
    AND |theta_hi - theta_lo| >= min_sep      (means genuinely separated)
    AND min(f, 1-f) >= min_mass               (both sub-populations real)

The EM two-component fit reuses the same over-dispersed beta-binomial as the
caller, so the null it is tested against is the SAME error model used to pool.
"""
import numpy as np
from scipy.stats import betabinom
from scipy.special import logsumexp


def _bb_logpmf(k, n, mu, s):
    mu = min(max(mu, 1e-6), 1 - 1e-6)
    a = mu * s; b = (1 - mu) * s
    return betabinom.logpmf(k, n, a, b)


def _fit_single(k, n, s, mu_grid=None):
    """H0: one beta-binomial mean. MLE mean = pooled fraction (with the fixed
    over-dispersion s); returns (mu, loglik)."""
    tot = n.sum()
    mu = k.sum() / tot if tot > 0 else 1e-6
    mu = min(max(mu, 1e-6), 1 - 1e-6)
    ll = _bb_logpmf(k, n, mu, s).sum()
    return mu, float(ll)


def _fit_two(k, n, s, max_iter=200, tol=1e-6, seed=0):
    """H1: two-component beta-binomial mixture by EM. Returns
    (f, mu_lo, mu_hi, loglik)."""
    rng = np.random.default_rng(seed)
    frac = np.where(n > 0, k / n, 0.0)
    # init means at low/high quantiles of the per-cell fraction
    mu_lo = max(np.quantile(frac, 0.25), 1e-5)
    mu_hi = max(np.quantile(frac, 0.90), mu_lo * 3 + 1e-4)
    mu_lo = min(mu_lo, 1 - 1e-6); mu_hi = min(mu_hi, 1 - 1e-6)
    f = 0.3
    prev = -np.inf
    for _ in range(max_iter):
        ll_lo = np.log(max(1 - f, 1e-12)) + _bb_logpmf(k, n, mu_lo, s)
        ll_hi = np.log(max(f, 1e-12)) + _bb_logpmf(k, n, mu_hi, s)
        Z = np.logaddexp(ll_lo, ll_hi)
        r_hi = np.exp(ll_hi - Z)
        f = float(np.clip(r_hi.mean(), 1e-4, 1 - 1e-4))
        w_hi = r_hi; w_lo = 1 - r_hi
        denom_hi = (w_hi * n).sum(); denom_lo = (w_lo * n).sum()
        if denom_hi > 1e-9:
            mu_hi = float(np.clip((w_hi * k).sum() / denom_hi, 1e-6, 1 - 1e-6))
        if denom_lo > 1e-9:
            mu_lo = float(np.clip((w_lo * k).sum() / denom_lo, 1e-6, 1 - 1e-6))
        ll = float(Z.sum())
        if abs(ll - prev) < tol:
            break
        prev = ll
    # order so mu_lo <= mu_hi
    if mu_lo > mu_hi:
        mu_lo, mu_hi = mu_hi, mu_lo
        f = 1 - f
    return f, mu_lo, mu_hi, ll


def homogeneity_test(alt, dp, s=None, mu_e=1e-4, chi2_thresh=9.21,
                     min_sep=0.05, min_mass=0.10, min_cells=10):
    """Test whether a clone's cells share one mtDNA genotype at a position.

    Returns dict:
      chimeric   : bool (True => SPLIT the clone before pooling)
      LR, f, mu_lo, mu_hi, sep, n_cov
    A clone flagged chimeric must NOT be pooled as one unit.
    """
    alt = np.asarray(alt, float); dp = np.asarray(dp, float)
    m = dp > 0
    k, n = alt[m], dp[m]
    n_cov = int(m.sum())
    if n_cov < min_cells:
        return {"chimeric": False, "LR": 0.0, "f": 0.0, "mu_lo": mu_e,
                "mu_hi": mu_e, "sep": 0.0, "n_cov": n_cov,
                "reason": "too_few_cells"}
    if s is None:
        # weak default over-dispersion; caller normally passes the site s_e
        s = 20.0
    mu0, ll0 = _fit_single(k, n, s)
    f, mu_lo, mu_hi, ll1 = _fit_two(k, n, s)
    LR = float(2 * (ll1 - ll0))
    sep = float(mu_hi - mu_lo)
    mass = float(min(f, 1 - f))
    chimeric = bool((LR > chi2_thresh) and (sep >= min_sep) and (mass >= min_mass))
    return {"chimeric": chimeric, "LR": max(LR, 0.0), "f": float(f),
            "mu_lo": float(mu_lo), "mu_hi": float(mu_hi), "sep": sep,
            "mass": mass, "n_cov": n_cov, "mu0": float(mu0), "reason": "tested"}


def split_if_chimeric(alt, dp, cell_ids=None, s=None, mu_e=1e-4, **kw):
    """If chimeric, return the two sub-clone index sets by EM responsibility;
    else return a single group. Used to actually partition before pooling."""
    res = homogeneity_test(alt, dp, s=s, mu_e=mu_e, **kw)
    alt = np.asarray(alt, float); dp = np.asarray(dp, float)
    if not res["chimeric"]:
        return [np.arange(len(alt))], res
    if s is None:
        s = 20.0
    m = dp > 0
    k, n = alt[m], dp[m]
    idx = np.where(m)[0]
    ll_lo = _bb_logpmf(k, n, res["mu_lo"], s)
    ll_hi = _bb_logpmf(k, n, res["mu_hi"], s)
    assign_hi = ll_hi > ll_lo
    grp_hi = idx[assign_hi]
    grp_lo = idx[~assign_hi]
    return [grp_lo, grp_hi], res

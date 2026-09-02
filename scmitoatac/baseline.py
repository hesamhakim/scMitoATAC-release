#!/usr/bin/env python
"""
scMitoHet 1.3 baseline (the BASELINE TO BEAT) — scMitoMut-style per-site
beta-binomial q-value, reproduced so it can be run on the SAME spiked matrix as
the Phase 2 core for an apples-to-apples comparison.

Per site: fit a beta-binomial background null on ALT/DP over all cells with
iterative WT-trimming; per-cell upper-tail p = P(X >= AD | DP, null);
BH-adjust across covered cells within the site. The per-cell score = 1 - qvalue
(higher = more confident carrier). This mirrors the Phase 1 baseline summary
(global sequencing-error null pooled from low-AF background).

Critically, the baseline consumes the NAIVE molecule counts (one sampled read
per UMI family -> retains per-read error), whereas Phase 2 consumes the
CONSENSUS counts. That is the fair contrast: same statistics scaffold, the
Phase 2 gain is the consensus front end + empirical null + mixture test.
"""
import numpy as np
from scipy.stats import betabinom


def _bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def fit_null_trim(alt, dp, trim_iter=3, trim_q=0.99, global_mu=None,
                  global_s=None):
    alt = np.asarray(alt, float); dp = np.asarray(dp, float)
    m = dp > 0
    a, d = alt[m], dp[m]
    if len(d) < 5:
        return (global_mu or 3.37e-3), (global_s or 245.0)
    keep = np.ones(len(d), bool)
    for _ in range(trim_iter):
        frac = np.where(d > 0, a / d, 0)
        thr = np.quantile(frac[keep], trim_q)
        new = keep & (frac <= max(thr, 1e-9))
        if new.sum() < 5 or new.sum() == keep.sum():
            break
        keep = new
    mu = a[keep].sum() / max(d[keep].sum(), 1)
    p = a[keep] / d[keep]
    nbar = d[keep].mean()
    var_p = np.average((p - mu) ** 2)
    bvar = mu * (1 - mu) / max(nbar, 1)
    if var_p <= bvar or mu <= 0 or mu >= 1:
        rho = 1e-4
    else:
        rho = np.clip((var_p - bvar) / (mu * (1 - mu) - bvar), 1e-4, 0.99)
    s = np.clip((1 - rho) / rho, 1.0, 5e5)
    if mu <= 0:
        mu = global_mu or 3.37e-3
    return float(mu), float(s)


def baseline_site_scores(alt, dp, global_mu=3.37e-3, global_s=245.0):
    """Return per-cell (pvalue, qvalue, score=1-q) for one site."""
    alt = np.asarray(alt, float); dp = np.asarray(dp, float)
    mu, s = fit_null_trim(alt, dp, global_mu=global_mu, global_s=global_s)
    a_ = mu * s; b_ = (1 - mu) * s
    covered = dp > 0
    pvals = np.ones(len(dp))
    pvals[covered] = betabinom.sf(alt[covered] - 1, dp[covered].astype(int), a_, b_)
    q = np.ones(len(dp))
    q[covered] = _bh(pvals[covered])
    score = 1.0 - q
    return {"pvalue": pvals, "qvalue": q, "score": score, "mu": mu, "s": s,
            "covered": covered}

#!/usr/bin/env python
"""
scMitoHet 2.2 + 2.3 — position-specific empirical null with empirical-Bayes
shrinkage, plus the mtDNA-specific RNA error component.

2.2  For each site, fit a beta-binomial null on ALT/DP over the REFERENCE cells
     (cells not carrying the variant), giving a site-specific background error
     rate mu_s and overdispersion (concentration s_s). Shrink each site's
     (mu_s, s_s) toward a genome-wide prior estimated by pooling all reference
     cells across all clean sites (empirical Bayes). This extends the
     scMitoMut-style per-locus WT fit to a shared prior so low-depth sites
     borrow strength.

2.3  The genome-wide prior IS the mtDNA panel-of-normals: it is fit from the
     reference-cell background across all sites, so RT-error + RNA-editing
     inflation shows up as elevated site background beyond the sequencing-error
     floor. Sites whose fitted mu_s sits well above the PoN floor get a WIDER
     null (higher mu, lower concentration) rather than being called -- this is
     the RNA error model folded into the beta-binomial overdispersion. No
     nuclear REDIportal/gnomAD is used; the reference is mtDNA-only.

Beta-binomial: P(k | n) = C(n,k) B(k+a, n-k+b)/B(a,b), a=mu*s, b=(1-mu)*s.
mu = mean alt fraction, s = concentration (a+b); larger s = tighter (less
overdispersed).
"""
import numpy as np
from scipy.special import betaln, digamma
from scipy.optimize import minimize_scalar


def _bb_logpmf(k, n, a, b):
    from scipy.special import gammaln
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
            + betaln(k + a, n - k + b) - betaln(a, b))


def fit_betabinom_mom(k, n, wmin=1.0):
    """Method-of-moments beta-binomial fit of alt-fraction. Returns (mu, s).
    k, n arrays (alt, depth). Robust when most k==0."""
    k = np.asarray(k, float); n = np.asarray(n, float)
    m = n > 0
    k, n = k[m], n[m]
    if len(n) == 0:
        return 1e-3, 200.0
    p = k / n
    mu = k.sum() / n.sum()  # pooled mean fraction (weights by depth)
    if mu <= 0:
        mu = 0.5 / (n.sum() + 1)  # pseudo
    # overdispersion from variance of per-cell fractions vs binomial expectation
    nbar = n.mean()
    var_p = np.average((p - mu) ** 2)
    binom_var = mu * (1 - mu) / max(nbar, 1)
    if var_p <= binom_var or mu >= 1:
        rho = 1e-4
    else:
        # intra-class correlation rho; s = (1-rho)/rho
        rho = (var_p - binom_var) / (mu * (1 - mu) - binom_var)
        rho = min(max(rho, 1e-4), 0.99)
    s = (1 - rho) / rho
    s = min(max(s, wmin), 5e5)
    return float(mu), float(s)


def fit_global_prior(site_alt_dp, single_err_floor=4.6e-4):
    """Pool all reference (alt,dp) across sites -> genome-wide PoN prior.
    site_alt_dp: list of (alt_array, dp_array). Returns dict."""
    all_k = np.concatenate([a for a, d in site_alt_dp]) if site_alt_dp else np.array([0.])
    all_n = np.concatenate([d for a, d in site_alt_dp]) if site_alt_dp else np.array([1.])
    mu0, s0 = fit_betabinom_mom(all_k, all_n)
    mu0 = max(mu0, single_err_floor)
    return {"mu0": float(mu0), "s0": float(s0),
            "single_err_floor": float(single_err_floor),
            "n_obs": int(all_n.sum())}


def eb_shrink(mu_s, s_s, n_eff, prior, shrink_strength=50.0):
    """Empirical-Bayes shrink a site's (mu_s, s_s) toward genome-wide prior.
    Weight by evidence (n_eff reference cells): more cells -> trust site fit."""
    w = n_eff / (n_eff + shrink_strength)
    mu = w * mu_s + (1 - w) * prior["mu0"]
    mu = max(mu, prior["single_err_floor"])
    # for concentration, shrink in log space toward prior s0
    ls = w * np.log(max(s_s, 1.0)) + (1 - w) * np.log(max(prior["s0"], 1.0))
    s = float(np.exp(ls))
    return float(mu), s, float(w)


def fit_site_null(alt, dp, prior, ref_mask=None, trim_iter=3, trim_q=0.99,
                  shrink_strength=50.0):
    """Fit an EB-shrunk beta-binomial null for one site.

    alt, dp : per-cell alt & depth arrays (consensus molecules).
    ref_mask: optional bool mask of reference cells; if None, iteratively trim
              the high-alt-fraction tail (candidate carriers) so the null is fit
              on background only.
    Returns dict with mu, s (shrunk), raw mu/s, n_ref, pon_excess flag.
    """
    alt = np.asarray(alt, float); dp = np.asarray(dp, float)
    cov = dp > 0
    a, d = alt[cov], dp[cov]
    if ref_mask is not None:
        rm = np.asarray(ref_mask, bool)[cov]
    else:
        # Significance-based iterative WT-trim (scMitoMut-style): start from the
        # robust lower half (non-carriers dominate), then repeatedly drop any
        # cell whose alt count is a p<trim_p outlier vs the current fitted null,
        # until convergence. Removes carriers regardless of their VAF, so the
        # null is not contaminated even when the true carrier fraction is high.
        frac = np.where(d > 0, a / d, 0)
        med = np.median(frac)
        rm = frac <= max(med, 1e-9)  # majority-vote seed: >=50% are non-carriers
        if rm.sum() < 5:
            rm = np.ones(len(d), bool)
        trim_p = 0.01
        for _ in range(max(trim_iter, 8)):
            n_ref0 = rm.sum()
            if n_ref0 < 5:
                break
            mu_i, s_i = fit_betabinom_mom(a[rm], d[rm])
            a_i = max(mu_i, 1e-9) * s_i; b_i = (1 - mu_i) * s_i
            from scipy.stats import betabinom as _bb
            # upper-tail p for every covered cell under the current null
            pv = _bb.sf(a - 1, d.astype(int), a_i, b_i)
            new = pv > trim_p  # keep cells consistent with the null
            if new.sum() < 5:
                break
            if new.sum() == n_ref0 and np.array_equal(new, rm):
                rm = new
                break
            rm = new
    n_ref = int(rm.sum())
    if n_ref < 3:
        mu_s, s_s = prior["mu0"], prior["s0"]
    else:
        mu_s, s_s = fit_betabinom_mom(a[rm], d[rm])
    mu, s, w = eb_shrink(mu_s, s_s, n_ref, prior, shrink_strength)
    # PoN excess: site background well above genome-wide floor. In RNA this flagged
    # RNA-editing-prone sites; in ATAC (no RNA editing) it flags NUMT-mapping / low-complexity
    # context artifacts (the Phase-2.4 artifact class). Exposed as artifact_prone; editing_prone
    # kept as a backward-compat alias so downstream RNA code paths do not break.
    pon_excess = float(mu / max(prior["mu0"], 1e-9))
    return {"mu": mu, "s": s, "mu_raw": float(mu_s), "s_raw": float(s_s),
            "n_ref": n_ref, "eb_weight": w, "pon_excess": pon_excess,
            "artifact_prone": bool(pon_excess > 3.0),
            "editing_prone": bool(pon_excess > 3.0)}


def bb_upper_tail(k, n, mu, s):
    """P(X >= k | n, beta-binom(mu,s)). Vectorizable over k,n scalars."""
    from scipy.stats import betabinom
    a = mu * s; b = (1 - mu) * s
    if n <= 0:
        return 1.0
    return float(betabinom.sf(k - 1, int(n), a, b))

#!/usr/bin/env python
"""
scMitoHet 2.4 + 2.5 + 2.6 — cross-cell mixture test, per-cell depth
normalization, and multi-feature fusion into ONE calibrated score.

2.4  Cross-cell mixture LR. Per site, fit:
       H0: all cells' alt counts ~ one overdispersed beta-binomial error
           component (the site null mu_e, s_e).
       H1: a mixture pi*carrier + (1-pi)*error, carrier ~ beta-binomial with
           mean mu_c > mu_e. Fit pi, mu_c by EM; LR = 2*(llH1 - llH0).
     LR is the "structured recurrence" statistic: high LR => the site carries a
     real carrier population distinct from error (MQuad framework re-aimed at
     per-variant reality). This is a SITE-level evidence weight fed into the
     per-cell score, not a per-cell call.

2.5  Per-cell depth normalization. A cell's evidence is bounded by its depth:
     the posterior for a low-VAF carrier at depth n is only resolvable if n is
     large enough. We compute a per-cell DETECTION POWER = P(observe >= k_min
     alt molecules | carrier at the site's tested VAF) and expose an effective
     depth. Cells below a power floor are emitted as 'unknown' (never a silent
     negative): their carrier posterior is reported with an explicit
     low-power flag and a widened CI rather than a confident 0.

2.6  Fusion. The per-cell/per-variant confidence is a posterior probability of
     carrying the variant:

        P(carrier | k, n, site) =
            pi_site * BB(k | n, mu_c, s_c)
          -------------------------------------------------------
          pi_site*BB(k|n,mu_c,s_c) + (1-pi_site)*BB(k|n,mu_e,s_e)

     where (mu_e, s_e) is the EB-shrunk site null (2.2/2.3), (pi_site, mu_c) come
     from the mixture fit (2.4), and the whole thing is depth-aware (2.5) because
     BB(k|n,..) collapses toward the prior at low n -> low-depth cells get
     posteriors near pi_site (uninformative), i.e. 'unknown', not 0. The site
     mixture LR (2.4) modulates pi_site: a site with no H1 evidence has
     pi_site -> 0 so no cell there can score high. This is a principled
     combination -- one posterior -- NOT heuristic additive boosts.
"""
import numpy as np
from scipy.stats import betabinom
from scipy.special import logsumexp


def _bb_logpmf(k, n, mu, s):
    a = mu * s; b = (1 - mu) * s
    return betabinom.logpmf(k, n, a, b)


def mixture_lr(alt, dp, mu_e, s_e, min_carrier_mu=None, max_iter=100, tol=1e-5,
               s_c=None):
    """2.4 cross-cell mixture test. Returns dict with LR, pi, mu_c."""
    alt = np.asarray(alt, float); dp = np.asarray(dp, float)
    m = dp > 0
    k, n = alt[m], dp[m]
    if len(n) < 10 or k.sum() == 0:
        return {"LR": 0.0, "pi": 0.0, "mu_c": mu_e, "s_c": s_e, "n": int(len(n)),
                "converged": False}
    if s_c is None:
        s_c = max(s_e * 0.5, 2.0)  # carrier component a bit more dispersed
    # init
    frac = k / n
    mu_c = max(np.quantile(frac, 0.95), mu_e * 3, (min_carrier_mu or 0))
    mu_c = min(mu_c, 0.95)
    pi = 0.05
    ll_e = _bb_logpmf(k, n, mu_e, s_e)
    prev = -np.inf
    conv = False
    for _ in range(max_iter):
        ll_c = _bb_logpmf(k, n, mu_c, s_c)
        # responsibilities
        log_num_c = np.log(max(pi, 1e-12)) + ll_c
        log_num_e = np.log(max(1 - pi, 1e-12)) + ll_e
        Z = np.logaddexp(log_num_c, log_num_e)
        r_c = np.exp(log_num_c - Z)
        # M-step
        pi = np.clip(r_c.mean(), 1e-6, 0.999)
        wsum = r_c.sum()
        if wsum > 1e-6:
            mu_c = np.clip((r_c * k).sum() / (r_c * n).sum(), mu_e * 1.5, 0.99)
        ll = Z.sum()
        if abs(ll - prev) < tol:
            conv = True
            break
        prev = ll
    ll_h1 = np.logaddexp(np.log(max(pi, 1e-12)) + _bb_logpmf(k, n, mu_c, s_c),
                         np.log(max(1 - pi, 1e-12)) + ll_e).sum()
    ll_h0 = ll_e.sum()
    LR = float(2 * (ll_h1 - ll_h0))
    return {"LR": max(LR, 0.0), "pi": float(pi), "mu_c": float(mu_c),
            "s_c": float(s_c), "n": int(len(n)), "converged": conv}


def detection_power(dp, mu_c, k_min=2):
    """2.5 per-cell detection power = P(>= k_min alt | carrier at mu_c, binomial
    on depth dp). Low power -> cell cannot license a negative call."""
    dp = np.asarray(dp, float)
    from scipy.stats import binom
    pw = np.where(dp >= k_min, 1 - binom.cdf(k_min - 1, dp, mu_c), 0.0)
    pw = np.where(dp <= 0, 0.0, pw)
    return pw


def _logit(p):
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return float(np.log(p / (1 - p)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fuse_posterior(alt, dp, site_null, mix, k_min=2, power_floor=0.5,
                   site_feature_logit=0.0, contam_adjust=None):
    """2.6 per-cell carrier posterior. Returns per-cell arrays.

    site_null: dict from fit_site_null (mu, s) [error component].
    mix:       dict from mixture_lr (pi, mu_c, s_c, LR).

    Phase 3 fold-in (both default to no-op, so Phase 2 behavior is unchanged):
      site_feature_logit : scalar SITE-level logit adjustment to pi_site from the
                           Phase 3 artifact features (3.1/3.2/3.4). Added in
                           logit space -> a principled prior update, NOT a hard
                           cut and NOT a heuristic multiplicative boost. Positive
                           => site more variant-like; negative => artifact-like.
      contam_adjust      : per-cell logit adjustment (Phase 3.5 contamination
                           term; ContaminationModel returns 0 by default).
    """
    alt = np.asarray(alt, float); dp = np.asarray(dp, float)
    mu_e, s_e = site_null["mu"], site_null["s"]
    pi, mu_c, s_c = mix["pi"], mix["mu_c"], mix["s_c"]
    LR = mix["LR"]

    # site-level evidence gate: convert LR to a soft site prior multiplier.
    # LR ~ chi2_1 under H0; sites with LR<~3.84 (p>0.05) get suppressed pi.
    site_evidence = 1.0 / (1.0 + np.exp(-(LR - 3.84) / 2.0))  # logistic gate
    pi_site_2 = pi * site_evidence
    pi_site_2 = min(max(pi_site_2, 1e-6), 0.999)

    # ---- Phase 3: fold artifact features into pi_site in LOGIT space ----
    # logit(pi_site_3) = logit(pi_site_2) + site_feature_logit
    pi_site = _sigmoid(_logit(pi_site_2) + float(site_feature_logit))
    pi_site = min(max(pi_site, 1e-6), 0.999)

    n = dp.copy()
    covered = n > 0
    k = alt.copy()

    # per-cell prior: pi_site optionally shifted per-cell by contamination term
    if contam_adjust is None:
        pi_cell = np.full(len(n), pi_site)
    else:
        ca = np.asarray(contam_adjust, float)
        pi_cell = _sigmoid(_logit(pi_site) + ca)
        pi_cell = np.clip(pi_cell, 1e-6, 0.999)

    log_c = np.full(len(n), -np.inf)
    log_e = np.full(len(n), -np.inf)
    log_c[covered] = np.log(pi_cell[covered]) + _bb_logpmf(k[covered], n[covered], mu_c, s_c)
    log_e[covered] = np.log(1 - pi_cell[covered]) + _bb_logpmf(k[covered], n[covered], mu_e, s_e)
    Z = np.logaddexp(log_c, log_e)
    post = np.zeros(len(n))
    post[covered] = np.exp(log_c[covered] - Z[covered])

    power = detection_power(n, max(mu_c, mu_e * 3), k_min=k_min)
    low_power = (power < power_floor)
    # 'unknown' cells: covered but underpowered AND observed no alt -> cannot
    # confidently call negative. Flag them; posterior stays at prior-ish value.
    unknown = covered & low_power & (k < k_min)

    return {"posterior": post, "power": power, "low_power": low_power,
            "unknown": unknown, "covered": covered, "pi_site": pi_site,
            "pi_site_prefeature": pi_site_2, "site_feature_logit": float(site_feature_logit),
            "site_evidence": float(site_evidence)}

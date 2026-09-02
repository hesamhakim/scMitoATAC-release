"""Between-cluster differential mtDNA heteroplasmy (Phase 10).

Built ON TOP of the frozen caller (CALLER_FREEZE.md): this module consumes frozen outputs
(`site_null` (mu_e, s_e), `fusion` per-cell carrier posterior, `pooling.pooled_estimate`) and
NEVER alters scoring math. It answers, for a user-supplied partition, two questions per variant:

  Axis 1 (rate / exclusivity): does carrier PREVALENCE differ across clusters, or is the variant
      restricted to a subset of clusters?  -> Firth-penalized logistic, coverage-adjusted.
  Axis 2 (VAF level): for a variant carried in >=2 clusters, does the per-cell heteroplasmy
      INTENSITY differ?  -> beta-binomial GLM + per-cluster pooled_estimate.

A partition-free clonality pre-screen (frozen `homogeneity.homogeneity_test`) gates a primary panel.

Standing rule (Phase 9): every between-cluster comparison adjusts for per-cell chrM coverage.
Firth is used because a cluster-EXCLUSIVE variant is perfect separation, where the plain logistic
MLE diverges; Firth keeps the penalized LR finite and exact.
"""
import numpy as np
from scipy import stats
from scipy.special import expit, betaln

# frozen / non-frozen primitives consumed as-is
from ..homogeneity import homogeneity_test          # frozen-math clonality detector
from ..pooling import pooled_estimate, fit_population_prior, pool_clone_cells

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# coverage design (flexible, non-linear in log10 depth)
# --------------------------------------------------------------------------------------
def coverage_basis(dp, kind="decile", n_bins=10):
    """Design columns for the per-cell coverage nuisance. `dp` = per-cell depth at the position.

    'decile'  -> one-hot fixed effects over depth quantile bins (flexible, no linearity assumption).
    'linear'  -> a single log10(dp) column (the simpler comparator; escalate to decile if the
                 calibration null flags residual confound).
    Returns an (n, k) float array WITHOUT an intercept column (the caller adds one).
    """
    logdp = np.log10(np.clip(dp.astype(float), 0.5, None))
    if kind == "linear":
        z = (logdp - logdp.mean()) / (logdp.std() + _EPS)
        return z.reshape(-1, 1)
    # decile fixed effects: rank into bins, drop the first as reference
    q = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(logdp, q)
    b = np.searchsorted(edges, logdp, side="right")   # 0..n_bins-1
    cols = sorted(set(b.tolist()))[1:]                # drop reference bin
    if not cols:
        return np.zeros((len(dp), 0))
    return np.column_stack([(b == c).astype(float) for c in cols])


def _dummies(labels, drop_first=True):
    """One-hot cluster dummies aligned to sorted unique labels; returns (X, level_names)."""
    levels = sorted(set(labels.tolist()))
    use = levels[1:] if drop_first else levels
    X = np.column_stack([(labels == g).astype(float) for g in use]) if use else np.zeros((len(labels), 0))
    return X, use


# --------------------------------------------------------------------------------------
# Firth-penalized logistic regression
# --------------------------------------------------------------------------------------
def firth_logit(X, y, max_iter=200, tol=1e-7, ridge=1e-8):
    """Firth bias-reduced logistic IRLS. `X` must already include an intercept column.

    Penalized score adds the Jeffreys term via the hat-diagonal adjustment
    y* = y + h*(0.5 - pi). Returns beta (finite even under perfect separation).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        pi = expit(X @ beta)
        w = np.clip(pi * (1.0 - pi), 1e-10, None)
        XtWX = (X * w[:, None]).T @ X + ridge * np.eye(p)
        XtWX_inv = np.linalg.pinv(XtWX)
        # hat diagonal h_i = w_i * x_i' (X'WX)^-1 x_i
        h = w * np.einsum("ij,ij->i", X @ XtWX_inv, X)
        U = X.T @ (y - pi + h * (0.5 - pi))
        step = XtWX_inv @ U
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def firth_penloglik(X, y, beta):
    """Penalized log-likelihood l(beta) + 0.5 log|X'WX| (the quantity the Firth LR compares)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    pi = np.clip(expit(X @ beta), _EPS, 1 - _EPS)
    ll = float(np.sum(y * np.log(pi) + (1 - y) * np.log(1 - pi)))
    w = np.clip(pi * (1 - pi), 1e-10, None)
    _, logdet = np.linalg.slogdet((X * w[:, None]).T @ X + 1e-8 * np.eye(X.shape[1]))
    return ll + 0.5 * logdet


def _firth_lr(X_null, X_full, y):
    b0 = firth_logit(X_null, y)
    b1 = firth_logit(X_full, y)
    lr = 2.0 * (firth_penloglik(X_full, y, b1) - firth_penloglik(X_null, y, b0))
    return max(lr, 0.0), b1


# --------------------------------------------------------------------------------------
# Axis 1 -- carrier-rate differential + cluster-exclusivity
# --------------------------------------------------------------------------------------
def rate_test(carrier, cluster, dp, coverage="decile"):
    """Coverage-adjusted between-cluster carrier-RATE test (Firth logistic LRT).

    carrier: 0/1 per cell (drop U cells upstream; pass only C+/C- cells).
    cluster: per-cell label. dp: per-cell depth at the position (coverage nuisance).
    Returns dict: LR, df, p, enriched_in, adj_rate (per cluster at median coverage), n_carriers.
    """
    carrier = np.asarray(carrier, float)
    cluster = np.asarray(cluster)
    dp = np.asarray(dp, float)
    cov = coverage_basis(dp, kind=coverage)
    D, levels = _dummies(cluster, drop_first=True)
    n = len(carrier)
    ones = np.ones((n, 1))
    X_null = np.hstack([ones, cov])
    X_full = np.hstack([ones, cov, D])
    lr, b1 = _firth_lr(X_null, X_full, carrier)
    df = D.shape[1]
    p = float(stats.chi2.sf(lr, df)) if df > 0 else 1.0
    # adjusted per-cluster carrier probability at cohort-median coverage
    all_levels = sorted(set(cluster.tolist()))
    cov_med = np.median(cov, axis=0, keepdims=True) if cov.shape[1] else np.zeros((1, 0))
    # adjusted per-cluster carrier probability at cohort-median coverage; base level (dropped dummy) = all-zero row
    adj = {}
    for g in all_levels:
        drow = np.array([[1.0 if g == lv else 0.0 for lv in levels]]) if levels else np.zeros((1, 0))
        xrow = np.hstack([np.ones((1, 1)), cov_med, drow])
        adj[g] = float(expit((xrow @ b1).ravel()[0]))
    n_car = {g: int(carrier[cluster == g].sum()) for g in all_levels}
    enriched = max(adj, key=adj.get)
    return {"LR": lr, "df": df, "p": p, "enriched_in": enriched,
            "adj_rate": adj, "n_carriers": n_car, "n": n}


def exclusivity(carrier, cluster, dp, n_perm=1000, n_strata=10, seed=0):
    """Exclusivity statistic E (top-cluster carrier share) vs coverage-only expectation E_adj,
    with a coverage-stratified label-shuffle p-value, plus a Cochran-Mantel-Haenszel stratified test.
    """
    carrier = np.asarray(carrier, float)
    cluster = np.asarray(cluster)
    dp = np.asarray(dp, float)
    levels = sorted(set(cluster.tolist()))
    tot = carrier.sum()
    if tot < 1:
        return {"E": 0.0, "E_adj": np.nan, "fold": np.nan, "perm_p": 1.0, "cmh_p": 1.0, "top": None}
    share = {g: carrier[cluster == g].sum() / tot for g in levels}
    top = max(share, key=share.get)
    E = float(share[top])
    # coverage-only expectation of the top cluster's share: fit carrier ~ 1 + coverage, predict, sum per cluster
    cov = coverage_basis(dp, kind="decile", n_bins=n_strata)
    X0 = np.hstack([np.ones((len(dp), 1)), cov])
    b0 = firth_logit(X0, carrier)
    phat = expit(X0 @ b0)
    exp_share = {g: phat[cluster == g].sum() for g in levels}
    tot_exp = sum(exp_share.values()) + _EPS
    E_adj = float(exp_share[top] / tot_exp)
    # coverage-stratified permutation of labels (shuffle within depth deciles, mixed strata only)
    strata = _decile_strata(dp, n_strata)
    sfrac = _shuffleable_frac(cluster, strata)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        perm = _shuffle_within(cluster, strata, rng)
        sh = {g: carrier[perm == g].sum() / tot for g in levels}
        if max(sh.values()) >= E - 1e-12:
            ge += 1
    perm_p = (ge + 1) / (n_perm + 1)
    cmh_p = _cmh_top_vs_rest(carrier, cluster == top, strata)
    return {"E": E, "E_adj": E_adj, "fold": E / (E_adj + _EPS),
            "perm_p": float(perm_p), "cmh_p": float(cmh_p), "top": top,
            "shuffleable_frac": sfrac}


# --------------------------------------------------------------------------------------
# Axis 2 -- VAF-level differential
# --------------------------------------------------------------------------------------
def _bb_negloglik(params, alt, dp, D, s):
    """Beta-binomial NLL with logit(mu) = intercept + D@coef, fixed overdispersion s."""
    b = params
    eta = b[0] + (D @ b[1:] if D.shape[1] else 0.0)
    mu = np.clip(expit(eta), 1e-6, 1 - 1e-6)
    a = mu * s
    bb = (1 - mu) * s
    ll = (betaln(alt + a, dp - alt + bb) - betaln(a, bb))
    return -float(np.sum(ll))


def betabinom_glm(alt, dp, cluster, s):
    """Per-cell beta-binomial GLM LR test for a between-cluster VAF-level difference.

    logit(mu_c) = intercept + cluster dummies; H0 drops the dummies. Overdispersion s fixed
    (from the frozen site null). Returns LR, df, p.
    """
    from scipy.optimize import minimize
    alt = np.asarray(alt, float); dp = np.asarray(dp, float)
    m = dp > 0
    alt, dp, cluster = alt[m], dp[m], np.asarray(cluster)[m]
    D, _ = _dummies(cluster, drop_first=True)
    p0 = np.array([np.log((alt.sum() + 0.5) / (dp.sum() - alt.sum() + 0.5))])
    r0 = minimize(_bb_negloglik, p0, args=(alt, dp, np.zeros((len(alt), 0)), s), method="Nelder-Mead",
                  options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
    if D.shape[1] == 0:
        return {"LR": 0.0, "df": 0, "p": 1.0}
    p1 = np.concatenate([r0.x, np.zeros(D.shape[1])])
    r1 = minimize(_bb_negloglik, p1, args=(alt, dp, D, s), method="Nelder-Mead",
                  options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 8000})
    lr = max(2.0 * (r0.fun - r1.fun), 0.0)
    df = D.shape[1]
    return {"LR": float(lr), "df": df, "p": float(stats.chi2.sf(lr, df))}


def per_cluster_vaf(alt, dp, cluster, mu_e, s_e):
    """Per-cluster pooled EB VAF + 95% CI + callability via the frozen pooled_estimate."""
    alt = np.asarray(alt, float); dp = np.asarray(dp, float); cluster = np.asarray(cluster)
    levels = sorted(set(cluster.tolist()))
    fracs = []
    for g in levels:
        K, N, _ = pool_clone_cells(alt[cluster == g], dp[cluster == g])
        fracs.append(K / N if N > 0 else 0.0)
    a0, b0 = fit_population_prior(np.array(fracs)) if len(fracs) >= 3 else (0.5, 0.5)
    out = {}
    for g in levels:
        K, N, _ = pool_clone_cells(alt[cluster == g], dp[cluster == g])
        out[g] = pooled_estimate(K, N, a0, b0, mu_e, s_e)
    return out


# --------------------------------------------------------------------------------------
# Stage 0 -- partition-free clonality pre-screen (frozen homogeneity_test)
# --------------------------------------------------------------------------------------
def prescreen(alt, dp, s_e):
    """Label-free clonality flag over ALL covered cells (frozen homogeneity_test).

    Returns the full result; `chimeric` (LR>9.21 AND sep>=0.05 AND mass>=0.10) is the gate.
    Depth/within-carrier-VAF limited: absence of structure is NOT certified absence.
    """
    return homogeneity_test(np.asarray(alt, float), np.asarray(dp, float), s=s_e)


# --------------------------------------------------------------------------------------
# coverage-stratified permutation null (shared calibration primitive)
# --------------------------------------------------------------------------------------
def _decile_strata(dp, n_strata=10):
    logdp = np.log10(np.clip(np.asarray(dp, float), 0.5, None))
    q = np.linspace(0, 1, n_strata + 1)[1:-1]
    edges = np.quantile(logdp, q)
    return np.searchsorted(edges, logdp, side="right")


def _shuffle_within(labels, strata, rng):
    """Permute labels WITHIN each depth stratum, but only where a stratum holds >=2 distinct
    clusters. A single-cluster stratum has nothing to exchange (and shuffling it would fabricate
    a spurious no-op that hides non-identifiability), so it stays fixed."""
    out = labels.copy()
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        if len(set(labels[idx].tolist())) < 2:
            continue
        out[idx] = labels[rng.permutation(idx)]
    return out


def _shuffleable_frac(cluster, strata):
    """Fraction of cells sitting in depth strata that contain >=2 clusters = the coverage-overlap
    (positivity) mass. Low value => clusters occupy disjoint coverage support => the between-cluster
    comparison is not coverage-identifiable and must abstain rather than license."""
    cluster = np.asarray(cluster)
    mixed = np.zeros(len(cluster), bool)
    for s in np.unique(strata):
        m = strata == s
        if len(set(cluster[m].tolist())) >= 2:
            mixed[m] = True
    return float(mixed.mean())


def coverage_stratified_null(stat_fn, cluster, dp, n_perm=200, n_strata=10, seed=0):
    """Empirical null for a between-cluster statistic under coverage-stratified label shuffling.

    Returns (null_stats, shuffleable_frac). Labels are permuted only within depth strata holding
    >=2 clusters (preserving each cluster's depth profile, destroying only the cluster<->genotype
    association where the two overlap). `shuffleable_frac` is the positivity mass: a hit is licensed
    only if the observed statistic exceeds the null AND shuffleable_frac is adequate.
    """
    cluster = np.asarray(cluster)
    strata = _decile_strata(dp, n_strata)
    sfrac = _shuffleable_frac(cluster, strata)
    rng = np.random.default_rng(seed)
    null = np.array([stat_fn(_shuffle_within(cluster, strata, rng)) for _ in range(n_perm)])
    return null, sfrac


def _cmh_top_vs_rest(carrier, is_top, strata):
    """Cochran-Mantel-Haenszel test of carrier x (top-cluster vs rest) association across depth strata."""
    carrier = np.asarray(carrier).astype(bool)
    is_top = np.asarray(is_top).astype(bool)
    num = 0.0; den = 0.0
    for s in np.unique(strata):
        m = strata == s
        a = np.sum(carrier[m] & is_top[m]); b = np.sum(carrier[m] & ~is_top[m])
        c = np.sum(~carrier[m] & is_top[m]); d = np.sum(~carrier[m] & ~is_top[m])
        nS = a + b + c + d
        if nS < 2:
            continue
        num += a - (a + b) * (a + c) / nS
        den += (a + b) * (c + d) * (a + c) * (b + d) / (nS ** 2 * (nS - 1) + _EPS)
    if den <= 0:
        return 1.0
    chi = (abs(num) - 0.5) ** 2 / den
    return float(stats.chi2.sf(max(chi, 0.0), 1))


# --------------------------------------------------------------------------------------
# POOLED between-cluster test -- the depth-robust primary path
# --------------------------------------------------------------------------------------
def pooled_between_cluster(alt, dp, cluster, mu_e, s_e, min_pooled_depth=100,
                           n_null=200, n_strata=10, seed=0):
    """Depth-ROBUST between-cluster VAF test. Pool every cell within each cluster (compensating for low
    PER-CELL coverage) and compare pooled VAFs across clusters against a size- and coverage-matched
    random-repooling null. Requires adequate POOLED depth per cluster (n_cells x per-cell depth), NOT
    per-cell confident carriers -- this is the mechanism by which metacells/clusters make un-enriched
    scATAC usable, and it works at 2-3x per-cell depth as long as clusters have enough cells.

    The pooled fraction K/N is an unbiased estimate of a cluster's true VAF regardless of its depth, and the
    common-mode error floor cancels in the between-cluster contrast (CALLER_FREEZE 3), so this axis is both
    depth-robust and confound-robust. The null shuffles cluster labels within per-cell-depth deciles,
    matching group sizes and coverage while destroying the cluster<->genotype association.

    Returns per-cluster pooled VAF (EB point + CI + callability via the frozen pooled_estimate), the
    depth-weighted between-cluster VAF dispersion statistic, its coverage-matched null threshold, an
    empirical p, and a `licensed` flag (statistic beats the null AND positivity mass is adequate).
    """
    alt = np.asarray(alt, float); dp = np.asarray(dp, float); cluster = np.asarray(cluster)
    m = dp > 0
    alt, dp, cluster = alt[m], dp[m], cluster[m]
    levels = sorted(set(cluster.tolist()))

    def _wvar_pooled(lab):
        vs, ws = [], []
        for g in levels:
            gm = lab == g
            N = dp[gm].sum()
            if N <= 0:
                continue
            vs.append(alt[gm].sum() / N); ws.append(N)   # unbiased pooled fraction, depth-weighted
        if len(vs) < 2:
            return 0.0
        vs = np.array(vs); ws = np.array(ws)
        mbar = np.average(vs, weights=ws)
        return float(np.average((vs - mbar) ** 2, weights=ws))

    # per-cluster pooled EB estimate (frozen) + feasibility on POOLED depth
    fracs = [alt[cluster == g].sum() / max(dp[cluster == g].sum(), 1) for g in levels]
    a0, b0 = fit_population_prior(np.array(fracs)) if len(fracs) >= 3 else (0.5, 0.5)
    percl = {}
    for g in levels:
        gm = cluster == g
        K = alt[gm].sum(); N = dp[gm].sum()
        est = pooled_estimate(K, N, a0, b0, mu_e, s_e)
        est["pooled_depth"] = float(N); est["n_cells"] = int(gm.sum())
        percl[g] = est

    T_obs = _wvar_pooled(cluster)
    strata = _decile_strata(dp, n_strata)
    sfrac = _shuffleable_frac(cluster, strata)
    rng = np.random.default_rng(seed)
    null = np.array([_wvar_pooled(_shuffle_within(cluster, strata, rng)) for _ in range(n_null)])
    thr = float(np.quantile(null, 0.95))
    enough_pooled = all(percl[g]["pooled_depth"] >= min_pooled_depth for g in levels)
    licensed = bool(T_obs > thr and sfrac >= 0.5)
    top = max(percl, key=lambda g: percl[g]["vaf_point"])
    return {"T_obs": T_obs, "null_q95": thr, "empirical_p": float((null >= T_obs).mean() if len(null) else 1.0),
            "licensed": licensed, "shuffleable_frac": sfrac, "enough_pooled_depth": enough_pooled,
            "top": top, "per_cluster": {g: {"vaf": percl[g]["vaf_point"], "ci_low": percl[g]["vaf_ci_low"],
                                            "ci_high": percl[g]["vaf_ci_high"], "callable": bool(percl[g]["callable"]),
                                            "pooled_depth": percl[g]["pooled_depth"], "n_cells": percl[g]["n_cells"]}
                                        for g in levels}}

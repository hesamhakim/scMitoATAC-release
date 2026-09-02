"""Step 3 — between-compartment differences against the random-pool null (relative-preferred).

The specificity floor is a reproducible BIAS (report §13.4), hence common-mode across compartments
within one sample -> it CANCELS for between-compartment differences. So relative claims beat absolute
VAF, and the in-sample arbiter is the partition-vs-random null (validated in 6.3(a) and prerequisite #1):
a real between-compartment difference must exceed what random re-pooling of the same cells produces.
"""
import numpy as np

def random_pool_null(cell_alt, cell_dp, pos, group_sizes, n_draws=200, seed=2024):
    """Null distribution of pooled VAF at `pos` under RANDOM pooling (labels shuffled).

    Draw random cell groups of the given sizes and return their pooled VAFs -- the distribution a
    between-compartment difference must beat to be called real (not a pooling artifact).
    """
    rng = np.random.default_rng(seed)
    a = cell_alt[:, pos]; d = cell_dp[:, pos]
    n = len(a); out = []
    for _ in range(n_draws):
        for gs in group_sizes:
            idx = rng.choice(n, size=min(gs, n), replace=False)
            dd = d[idx].sum()
            out.append(a[idx].sum() / dd if dd > 0 else np.nan)
    return np.asarray(out)

def relative_difference(vaf_a, vaf_b, null_dist, q=0.95):
    """Is |vaf_a - vaf_b| beyond the random-pool null band? Returns (delta, null_hi, licensed)."""
    delta = abs(vaf_a - vaf_b)
    null_hi = np.nanquantile(null_dist, q)
    return dict(delta=delta, null_threshold=float(null_hi), licensed=bool(delta > null_hi))

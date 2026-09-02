"""Step 2 — metacell calling: pool cells by partition label, call via the FROZEN caller, abstain per depth.

A metacell has no intrinsic biology; it inherits meaning from the partition (master plan v3.5 §7-D).
Step 0 supplies the partition as a label column in .obs (genetic, scATAC-native) -- this layer consumes
it, never reimplements clustering. Pooling + calling uses the frozen scmitoatac primitives; nothing here
alters scoring math.

AnnData/MuData I/O is imported lazily (optional `deploy` extra). The numeric core
(`pool_partition_counts`) operates on plain arrays so it is testable without AnnData.
"""
import numpy as np
from .. import pooling, fusion, site_null  # frozen caller primitives

def pool_partition_counts(cell_alt, cell_dp, labels, partition):
    """Sum alt/depth over the cells in one partition -> pooled (alt, dp) per position.

    cell_alt, cell_dp : (n_cells, n_pos) arrays of per-cell alt and total depth at each position.
    labels            : (n_cells,) partition label per cell (from .obs, Step 0).
    partition         : the label to pool.
    """
    m = np.asarray(labels) == partition
    if m.sum() == 0:
        raise ValueError(f"partition {partition!r} has no cells")
    return cell_alt[m].sum(0), cell_dp[m].sum(0), int(m.sum())

def call_metacells(cell_alt, cell_dp, labels, per_cell_chrm_depth, target_vaf,
                   mu_e_by_pos=None, s_e=None, a0=None, b0=None):
    """Call heteroplasmy per (partition, position) with per-compartment abstention.

    Returns a list of dicts: one per partition, carrying the feasibility verdict (Step 1) and,
    for feasible compartments, the pooled estimate per position from the FROZEN
    `pooling.pooled_estimate` -- EB posterior point + credible interval + null-tail callability.
    Compartments failing the feasibility gate are marked abstained and NOT called.

    mu_e_by_pos, s_e : the position error-null (beta-binomial) parameters. When BOTH are given the
        calibrated estimator runs and each record carries `pooled_vaf` (EB posterior point),
        `vaf_ci_low`/`vaf_ci_high`, `tail_p`, and `callable` (the null-tail test), with
        `estimator="pooled_estimate"`.
        When they are absent the calibrated null-tail test is undefined, so the record falls back to
        the raw pooled fraction and is LABELLED `estimator="raw_uncalibrated"` with `callable=None` --
        the deficiency is explicit rather than silent. Deployment should supply the null.
    a0, b0 : EB population hyperprior; defaults to Jeffreys (0.5, 0.5), matching the other callers.
    """
    from .feasibility import compartment_feasible
    calibrated = (mu_e_by_pos is not None) and (s_e is not None)
    if a0 is None or b0 is None:
        a0, b0 = 0.5, 0.5                      # Jeffreys, as in phase4_benchmark / rescue_screens
    out = []
    for part in sorted(set(labels)):
        alt, dp, n = pool_partition_counts(cell_alt, cell_dp, labels, part)
        feas = compartment_feasible(n, per_cell_chrm_depth, target_vaf)
        rec = dict(partition=part, n_cells=n, **feas)
        if not feas["feasible"]:
            rec.update(pooled_vaf=None, estimator="abstained")   # abstained: NOT called
            out.append(rec)
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.where(dp > 0, alt / dp, np.nan)
        if not calibrated:
            rec.update(pooled_vaf=raw, pooled_vaf_raw=raw, estimator="raw_uncalibrated",
                       vaf_ci_low=None, vaf_ci_high=None, tail_p=None, callable=None)
            out.append(rec)
            continue
        # --- calibrated path: the frozen pooled_estimate per position ---
        mu_e = np.broadcast_to(np.asarray(mu_e_by_pos, dtype=float), raw.shape)
        P = len(raw)
        point = np.full(P, np.nan); lo = np.full(P, np.nan); hi = np.full(P, np.nan)
        tail = np.full(P, np.nan); call = np.zeros(P, dtype=bool)
        for j in range(P):
            if not (dp[j] > 0) or not np.isfinite(mu_e[j]):
                continue
            est = pooling.pooled_estimate(alt[j], dp[j], a0, b0, float(mu_e[j]), s_e)
            point[j] = est["vaf_point"]; lo[j] = est["vaf_ci_low"]; hi[j] = est["vaf_ci_high"]
            tail[j] = est["tail_p"]; call[j] = bool(est["callable"])
        rec.update(pooled_vaf=point, pooled_vaf_raw=raw, vaf_ci_low=lo, vaf_ci_high=hi,
                   tail_p=tail, callable=call, estimator="pooled_estimate")
        out.append(rec)
    return out

# ---- AnnData plug-in point (Step 0/2 boundary) ----
def call_from_anndata(adata, partition_key, layer_alt="mt_alt", layer_dp="mt_dp",
                      per_cell_chrm_depth=None, target_vaf=0.01, **kw):
    """AnnData-facing entry: read per-cell mt alt/depth layers + the partition label column, then call.

    Requires anndata (optional `deploy` extra). partition_key is the .obs column produced by Step 0.
    If per_cell_chrm_depth is None it is MEASURED from the data (median non-zero per-cell depth),
    honoring the 'use the sample's own depth' rule (never a constant).
    """
    try:
        import anndata  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("call_from_anndata needs the optional 'deploy' extra: pip install scmitoatac[deploy]") from e
    alt = np.asarray(adata.layers[layer_alt].todense() if hasattr(adata.layers[layer_alt], "todense")
                     else adata.layers[layer_alt])
    dp = np.asarray(adata.layers[layer_dp].todense() if hasattr(adata.layers[layer_dp], "todense")
                    else adata.layers[layer_dp])
    labels = adata.obs[partition_key].to_numpy()
    if per_cell_chrm_depth is None:
        pc = dp.sum(1)  # per-cell total across positions... use median positive per-cell mean depth
        per_cell_chrm_depth = float(np.median(dp[dp > 0])) if (dp > 0).any() else 0.0
    return call_metacells(alt, dp, labels, per_cell_chrm_depth, target_vaf, **kw)

#!/usr/bin/env python
"""
scMitoHet Phase 3.5 — contamination / index-hopping estimate. DEFERRED but
STUBBED (per science-team + the Phase 2 report).

Rationale for deferral (recorded, NOT dropped from the design):
  * haplocheck is not installed on RCC.
  * The Phase 1 contamination estimate on this library (5k_pbmc_v3) is 0.0.
  * A meaningful cross-sample index-hopping estimate needs multiple libraries
    from the same flowcell / a demultiplex table, which is a cross-sample
    (Phase 5 / J1) concern, not a single-library one.

The INTERFACE is implemented so the fused posterior already carries a
per-cell-prior contamination term. It defaults to 0 (no-op) and can be
re-enabled when haplocheck is available or at cross-sample scale, WITHOUT
changing the posterior's signature — the term folds into the same single score
as an additive logit adjustment on the per-cell carrier prior, exactly like the
other Phase 3 features.

Semantics when enabled: an ambient/index-hop contamination fraction `alpha`
(per library, or per cell if a per-cell estimate is available) inflates the
expected background ALT at a site by the population VAF of the contaminant
haplotype. A low-VAF apparent carrier whose ALT is fully explained by `alpha`
times the ambient VAF is down-weighted (its carrier posterior shrinks toward
the null). With alpha=0 the term is exactly 0 and the posterior is unchanged.
"""
import numpy as np


class ContaminationModel:
    """Per-cell-prior contamination term. Default is a no-op (alpha=0).

    enabled=False (default): contamination_logit_adjustment returns 0 for all
    cells -> the fused posterior is identical to the no-contamination case.

    To re-enable (future / cross-sample): construct with enabled=True and an
    `alpha` (scalar library contamination fraction or per-cell array) and an
    `ambient_vaf` (the population/ambient ALT fraction at the site). The
    adjustment down-weights cells whose observed ALT is consistent with
    contamination rather than a genuine carrier state.
    """

    def __init__(self, alpha=0.0, enabled=False, source="deferred:haplocheck_unavailable"):
        self.alpha = alpha
        self.enabled = bool(enabled)
        self.source = source

    def contamination_logit_adjustment(self, alt, dp, ambient_vaf=0.0,
                                       mu_e=None):
        """Additive logit adjustment to the per-cell carrier prior.
        Returns an array (len == len(alt)). alpha=0 or disabled -> all zeros."""
        alt = np.asarray(alt, float); dp = np.asarray(dp, float)
        if not self.enabled or np.all(np.asarray(self.alpha) == 0) or ambient_vaf <= 0:
            return np.zeros(len(alt))
        alpha = np.asarray(self.alpha, float)
        # expected ALT molecules attributable to contamination
        exp_contam = alpha * ambient_vaf * dp
        obs_frac = np.where(dp > 0, alt / dp, 0.0)
        contam_frac = alpha * ambient_vaf
        # if observed ALT fraction is within ~1 sd of the contamination-only
        # expectation, down-weight (negative logit); else no penalty.
        with np.errstate(divide="ignore", invalid="ignore"):
            excess = obs_frac - contam_frac
        sd = np.sqrt(np.maximum(contam_frac * (1 - contam_frac) / np.maximum(dp, 1), 1e-12))
        z = np.where(dp > 0, excess / sd, 0.0)
        # z<=1 => ALT explained by contamination => strong down-weight
        adj = -2.0 * (1.0 / (1.0 + np.exp((z - 1.0) * 2.0)))
        return adj

    def to_dict(self):
        return {"alpha": self.alpha, "enabled": self.enabled,
                "source": self.source}


# module-level default: the deferred no-op used everywhere in Phase 3.
DEFAULT = ContaminationModel(alpha=0.0, enabled=False)

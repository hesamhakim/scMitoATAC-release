"""Cluster-aware spike-in for validating the between-cluster differential-heteroplasmy method.

The existing spike-ins (`atac_spike_benchmark.build_spiked_matrix`, `phase3_benchmark.build_true_matrix`,
`phase6_partition_spikein`) cannot implant a variant restricted to a SUPPLIED cluster label vector at a chosen
within-carrier VAF, nor give different clusters different VAF. This module does, reusing the per-cell
Binomial(dp, vaf) injection pattern. Real per-cell depth `dp` is kept (it carries the coverage heterogeneity
that creates the confound the method must survive); only the alt counts at spike positions are synthesized, so
the truth is exactly what was injected (not circular).

Two spike kinds:
  exclusive     {kind, pos, carrier_clusters, within_carrier_vaf, carrier_frac}
      carriers only in `carrier_clusters` (each cell w.p. carrier_frac) at within_carrier_vaf; else error floor.
  differential  {kind, pos, vaf_by_cluster={g: vaf}, carrier_frac}
      each named cluster carries at its own vaf; equal vafs across clusters = the null (must NOT fire).
"""
import numpy as np


def spike_variant(dp, labels, spec, err_floor=3e-4, rng=None):
    """Per-cell alt vector for ONE implanted variant, given per-cell depth `dp` and `labels`."""
    if rng is None:
        rng = np.random.default_rng(0)
    dp = np.asarray(dp)
    labels = np.asarray(labels)
    n = len(dp)
    dpi = np.clip(dp.astype(int), 0, None)
    alt = rng.binomial(dpi, err_floor).astype(float)          # background error everywhere
    cf = spec.get("carrier_frac", 1.0)
    if spec["kind"] == "exclusive":
        for g in spec["carrier_clusters"]:
            car = (labels == g) & (rng.random(n) < cf)
            alt[car] = rng.binomial(dpi[car], spec["within_carrier_vaf"]).astype(float)
    elif spec["kind"] == "differential":
        for g, v in spec["vaf_by_cluster"].items():
            car = (labels == g) & (rng.random(n) < cf)
            alt[car] = rng.binomial(dpi[car], v).astype(float)
    else:
        raise ValueError("unknown spike kind: %r" % spec.get("kind"))
    alt = np.minimum(alt, dp.astype(float))                    # alt cannot exceed depth
    return alt


def build_cluster_spiked_matrix(dp, labels, spikes, err_floor=3e-4, rng=None):
    """Implant several spikes. Returns (alt_by_pos: {pos: alt_vector}, truth: {pos: spec})."""
    if rng is None:
        rng = np.random.default_rng(0)
    alt_by_pos, truth = {}, {}
    for spec in spikes:
        alt_by_pos[spec["pos"]] = spike_variant(dp, labels, spec, err_floor, rng)
        truth[spec["pos"]] = dict(spec)
    return alt_by_pos, truth

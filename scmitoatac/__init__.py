"""scmitoatac — calibrated per-cell mtDNA heteroplasmy confidence core for ATAC/Multiome.

Ported from scMitoHet (3' scRNA-seq predecessor) and reworked for ATAC through
Phase 3 (all three items below delivered and regression-locked in
tests/test_pipeline_regression.py):
  - consensus_pileup: UMI-consensus -> fragment-level dedup (ATAC has no UMIs) [3.1 done]
  - site_null: re-derived empirical null on ATAC error structure [3.2 done]
  - spike_benchmark: models systematic artifacts (NUMT, mapping bias), not
    only shot noise [3.3 done]

FROZEN 2026-07-22 (v0.1.0) — the calibrated caller is frozen at its characterized
operating envelope after Phases 0-6 + the two deployment prerequisites:
  - specificity floor ~0.03% (consistent across 19 ccRCC donors + REH + lung);
  - standard recovers enriched at r=0.88 on genuine low-het (2-10%) sites;
  - licensed between-source accuracy floor 0.5-2% VAF, depth-dependent
    (2% @100-250x, 1% @500x, 0.5% @>=2000x pooled);
  - operating-range sensitivity coverage ~97% within 3x (99.5% detected), with a
    small site-specific false-negative residual (e.g. chrM:4421, 13772).
Numeric behavior is pinned by the regression lock; do not alter scoring math
without re-freezing and re-versioning.

Provenance: scMitoHet repo-root scmitohet/ package (Phase 5 closed 2026-07-19),
all site_null.py copies byte-identical (md5 1ca61fe9) at port time.
"""
__all__ = ["consensus_pileup", "site_null", "fusion", "baseline", "spike_benchmark",
           "artifact_features", "callability", "phasing", "contamination",
           "calibration", "phase3_benchmark",
           "clone_inference", "homogeneity", "pooling", "phase4_benchmark",
           "deploy"]

__version__ = "0.1.0"  # FROZEN caller release (2026-07-22); see docs/deployment/CALLER_FREEZE.md


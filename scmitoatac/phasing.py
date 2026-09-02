#!/usr/bin/env python
"""
scMitoHet Phase 3.4 — read/UMI-level molecular phasing of co-occurring variants.

Two candidate variants that ride the SAME physical mtDNA molecules should
co-segregate across cells CONCORDANTLY (cis): a cell carrying one carries the
other, at a consistent fraction. This has two payoffs, both requested by the
build order:

  (1) ARTIFACT REJECTION (the feature). A recurrent artifact does NOT ride a
      real molecular lineage. So:
        * A candidate that co-segregates CONCORDANTLY with a pseudobulk-anchored
          REAL variant is guilt-by-association with a real haplotype -> its
          carrier prior is RAISED (this is the phasing feature fed to 2.4/2.6).
        * A candidate whose only "co-occurrence" is coverage-correlated (both
          apparent in the same high-read cells, i.e. shared artifact structure)
          shows NO cis molecular phasing -> no boost.
      Concordance is measured across cells by the phi coefficient of the
      carrier x carrier contingency table, gated by a co-segregation LR so
      chance overlaps in a few deep cells don't score.

  (2) HAPLOTYPE SEEDS for Phase 4. Concordantly-phased site groups are emitted
      as haplotype seeds (greedy transitive grouping over the concordance graph)
      — the anchor set Phase 4.1 clonal co-segregation builds on.

Unexploited on unenriched 10x: enrichment assays (MAESTER) phase within long
amplicons; here we phase via CROSS-CELL co-segregation of molecule counts,
which needs no enrichment. Full within-molecule (CB,UB) multi-position phasing
(seeing two ALT alleles on ONE UMI family) requires a molecule-level
multi-position readout and is the Phase-4 haplotype-refinement step; the
cross-cell statistic here is its testable, assay-agnostic proxy.
"""
import numpy as np
import pandas as pd


def _carrier_call(alt, dp, min_alt=2, min_frac=0.0):
    """Binary per-cell carrier indicator for a site (>= min_alt ALT molecules)."""
    alt = np.asarray(alt, float); dp = np.asarray(dp, float)
    return (alt >= min_alt) & (dp > 0)


def cosegregation(alt_i, dp_i, alt_j, dp_j, min_alt=2):
    """Cross-cell co-segregation of two candidate sites over the cells covered
    at BOTH. Returns dict: phi (contingency correlation), LR (2x2 independence
    LR), n_both, n_cocarrier, concordance (phi gated by significance)."""
    ci = _carrier_call(alt_i, dp_i, min_alt)
    cj = _carrier_call(alt_j, dp_j, min_alt)
    both = (np.asarray(dp_i) > 0) & (np.asarray(dp_j) > 0)
    if both.sum() < 20:
        return {"phi": 0.0, "LR": 0.0, "n_both": int(both.sum()),
                "n_cocarrier": 0, "concordance": 0.0}
    a = int((ci & cj & both).sum())   # both carrier
    b = int((ci & ~cj & both).sum())
    cc = int((~ci & cj & both).sum())
    d = int((~ci & ~cj & both).sum())
    n = a + b + cc + d
    # phi coefficient
    denom = np.sqrt(float((a+b)*(cc+d)*(a+cc)*(b+d)))
    phi = float((a*d - b*cc) / denom) if denom > 0 else 0.0
    # G-test / likelihood-ratio for independence of the 2x2 table
    obs = np.array([[a, b], [cc, d]], float) + 0.5
    rt = obs.sum(1, keepdims=True); ctot = obs.sum(0, keepdims=True); tot = obs.sum()
    exp = rt * ctot / tot
    LR = float(2 * np.sum(obs * np.log(obs / exp)))
    # concordance = positive phi gated by a significant, non-trivial overlap
    sig = 1.0 / (1.0 + np.exp(-(LR - 3.84) / 2.0))
    concordance = float(max(phi, 0.0) * sig) if a >= 3 else 0.0
    return {"phi": phi, "LR": LR, "n_both": n, "n_cocarrier": a,
            "concordance": concordance}


def phasing_feature(site_alt, site_dp, partner_sites, min_alt=2):
    """Phasing feature for ONE candidate site = the MAX cross-cell concordance
    with any partner (an anchor real variant, or another candidate).
    partner_sites: list of (alt_array, dp_array). Returns (max_concordance,
    best_partner_idx). Higher => the candidate rides a shared molecular lineage
    with a partner => artifact-unlikely => carrier prior RAISED."""
    best = 0.0; best_j = -1
    for j, (pa, pd_) in enumerate(partner_sites):
        cs = cosegregation(site_alt, site_dp, pa, pd_, min_alt=min_alt)
        if cs["concordance"] > best:
            best = cs["concordance"]; best_j = j
    return float(best), best_j


def seed_haplotypes(site_ids, alt_dp_by_site, concordance_thresh=0.3,
                    min_alt=2):
    """Greedy transitive grouping of sites into haplotype seeds over the
    concordance graph (edges where concordance >= thresh). Returns list of
    lists of site_ids. These seed Phase-4 clonal co-segregation (4.1)."""
    n = len(site_ids)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        ai, di = alt_dp_by_site[site_ids[i]]
        for j in range(i + 1, n):
            aj, dj = alt_dp_by_site[site_ids[j]]
            cs = cosegregation(ai, di, aj, dj, min_alt=min_alt)
            if cs["concordance"] >= concordance_thresh:
                adj[i].add(j); adj[j].add(i)
    seen = set(); groups = []
    for i in range(n):
        if i in seen:
            continue
        stack = [i]; comp = []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u); comp.append(u)
            stack.extend(adj[u] - seen)
        if len(comp) >= 2:
            groups.append([site_ids[k] for k in comp])
    return groups

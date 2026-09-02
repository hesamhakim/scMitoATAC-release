#!/usr/bin/env python
"""
scMitoHet Phase 3.3 — per-cell / per-position CALLABILITY MAP.

Formalizes the three-tier A/B/C licensing boundary the coverage model
prescribes, as an explicit per-library, per-position, per-cell map. This is the
"honest about the floor" deliverable: where a cell is not licensed to call at a
tested VAF, the report is 'unknown', NOT a false-negative.

Coverage physics (verified on 5k_pbmc_v3, deepest public 10x):
  * per-cell coverage is extreme (Gini 0.757; a few 3' peaks hold most reads).
  * A per-cell call needs >= k_min (default 2) ALT MOLECULES to be licensed,
    i.e. depth n must satisfy n * VAF >= k_min.

Three tiers (per position, from the per-cell consensus-depth distribution):
  * Tier A  high-VAF per-cell genome-wide. A cell is licensed at VAF_A (default
            25%) if its depth clears the k_min-molecule floor there. Since
            n*0.25>=2 needs n>=8, most covered cells reach Tier A only at
            positions with a healthy median depth.
  * Tier B  low-VAF per-cell ONLY at deep peaks. A POSITION is Tier-B-licensed
            if a sufficient fraction (TAU_COV, default 0.8) of its cells clear
            the k_min floor at VAF_B (the sample's low-VAF target, default 5%).
            The per-cell floor there is n >= k_min / VAF_B (=40 at 5%).
  * Tier C  low-VAF per-CLONE only (troughs). Positions where per-cell Tier-B
            is not licensed; calling is deferred to Phase-4 pooling. Reported
            per-cell as 'unknown' at low VAF, never FN.

Tier boundaries SHIFT WITH DEPTH -> they are set per sample by the Phase 0.4
depth routing (TAU_COV, VAF_B). This module computes:
  * per-(cell,pos) licensed tier at a tested VAF,
  * per-position tier label + the depth-dependent per-cell VAF floor,
  * a callability summary (licensed fractions per tier) for the report.
"""
import numpy as np
import pandas as pd


def percell_vaf_floor(dp, k_min=2):
    """Minimum VAF a cell of depth dp can license under a >=k_min-molecule bar.
    floor_vaf = k_min / dp (dp<=0 -> 1.0, i.e. nothing licensed)."""
    dp = np.asarray(dp, float)
    return np.where(dp > 0, k_min / np.maximum(dp, 1e-9), 1.0)


def cell_tier(dp, vaf_tested, vaf_A=0.25, vaf_B=0.05, k_min=2):
    """Per-cell licensed tier at a tested VAF.
      'A' if dp*vaf_tested >= k_min AND vaf_tested >= vaf_A
      'B' if dp*vaf_tested >= k_min AND vaf_A > vaf_tested >= vaf_B
      'C' otherwise (not per-cell licensed at this VAF -> pooling regime)
    Returns array of tier chars."""
    dp = np.asarray(dp, float)
    exp_alt = dp * vaf_tested
    licensed = exp_alt >= k_min
    tier = np.full(dp.shape, "C", dtype="<U1")
    tier[licensed & (vaf_tested >= vaf_B)] = "B"
    tier[licensed & (vaf_tested >= vaf_A)] = "A"
    return tier


def position_tier(dp_array, vaf_B=0.05, k_min=2, tau_cov=0.8):
    """Label a POSITION's per-cell low-VAF licensing tier from its per-cell
    depth distribution. A position is:
      Tier B  if frac(cells with dp >= k_min/vaf_B) >= tau_cov  (deep peak)
      Tier A' if it clears the floor only at high VAF (some but < tau_cov cells
              clear the low-VAF floor) -> licensed genome-wide at high VAF only
      Tier C  otherwise (trough; per-clone pooling only)
    Returns dict with tier, frac_cells_lowvaf_licensed, median_dp, floor_vaf.
    """
    dp = np.asarray(dp_array, float)
    dp = dp[dp > 0]
    if len(dp) == 0:
        return {"tier": "C", "frac_cells_lowvaf_licensed": 0.0,
                "median_dp": 0.0, "n_cells": 0, "floor_vaf_median": 1.0}
    floor_dp_B = k_min / vaf_B
    frac_B = float(np.mean(dp >= floor_dp_B))
    med = float(np.median(dp))
    floor_vaf_med = float(k_min / med) if med > 0 else 1.0
    if frac_B >= tau_cov:
        tier = "B"
    elif frac_B > 0 or med >= (k_min / 0.25):
        tier = "A"   # high-VAF-only per-cell licensing
    else:
        tier = "C"
    return {"tier": tier, "frac_cells_lowvaf_licensed": frac_B,
            "median_dp": med, "n_cells": int(len(dp)),
            "floor_vaf_median": floor_vaf_med}


def build_callability_map(pileup_df, routing=None, vaf_A=0.25, vaf_B=0.05,
                          k_min=2, tau_cov=0.8, depth_col="n_mol"):
    """Build the per-library callability map from a consensus pileup.
    routing: optional dict from Phase 0.4 (keys vaf_B, tau_cov override).
    Returns (position_map_df, summary_dict).
    """
    if routing:
        vaf_B = float(routing.get("TAU_VAF_B", routing.get("vaf_B", vaf_B)))
        tau_cov = float(routing.get("TAU_COV5", routing.get("tau_cov", tau_cov)))
    rows = []
    for pos, sub in pileup_df.groupby("pos"):
        pt = position_tier(sub[depth_col].values, vaf_B=vaf_B, k_min=k_min,
                            tau_cov=tau_cov)
        pt["pos"] = int(pos)
        rows.append(pt)
    pm = pd.DataFrame(rows)
    summary = {
        "n_positions": int(len(pm)),
        "n_tierB": int((pm["tier"] == "B").sum()),
        "n_tierA": int((pm["tier"] == "A").sum()),
        "n_tierC": int((pm["tier"] == "C").sum()),
        "vaf_B": vaf_B, "vaf_A": vaf_A, "k_min": k_min, "tau_cov": tau_cov,
        "median_floor_vaf_tierB": float(pm.loc[pm["tier"] == "B", "floor_vaf_median"].median())
            if (pm["tier"] == "B").any() else None,
    }
    return pm, summary

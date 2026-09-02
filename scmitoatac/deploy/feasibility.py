"""Step 1 — per-compartment feasibility gate (depth-floor calculator).

Keys off the sample's OWN measured per-cell chrM depth (NOT a constant; the census spans
~20x-2200x, a 100-fold range -- master plan v3.5 §7-D.1 Refinement 1). The licensed VAF floor
as a function of pooled depth is the FROZEN 6.2 titration result (report §14.1 / Fig 15):

    pooled depth   licensed floor (sens>=0.80, prec>=0.90 vs per-site null)
      100x            2%
      250x            2%
      500x            1%
      >=2000x         0.5%

cells_needed(target_vaf) answers the deployment planning question directly:
"how many cells must this compartment pool to license a call at target_vaf?"
"""
from bisect import bisect_left

# FROZEN operating points (pooled_depth -> licensed floor VAF); from phase6_spikein_summary.json.
# Monotone step function; between grid points we take the conservative (deeper-requiring) side.
_FLOOR_BY_DEPTH = [(100, 0.02), (250, 0.02), (500, 0.01), (2000, 0.005), (10000, 0.005)]

def licensed_floor(pooled_depth):
    """Licensed VAF floor at a given pooled depth (conservative step interpolation)."""
    if pooled_depth < _FLOOR_BY_DEPTH[0][0]:
        return None  # below 100x: no licensed call (compartment abstains)
    floor = _FLOOR_BY_DEPTH[0][1]
    for d, f in _FLOOR_BY_DEPTH:
        if pooled_depth >= d:
            floor = f
    return floor

def depth_needed(target_vaf):
    """Minimum pooled depth to license a call at target_vaf (inverse of licensed_floor)."""
    # smallest grid depth whose floor <= target_vaf
    for d, f in _FLOOR_BY_DEPTH:
        if f <= target_vaf:
            return d
    return None  # target below what any tested depth licenses

def cells_needed(target_vaf, per_cell_chrm_depth):
    """Cells this compartment must pool to license target_vaf, given its OWN measured per-cell depth.

    per_cell_chrm_depth: median chrM depth per cell MEASURED in this sample (Step 0 QC), never assumed.
    Returns None if target_vaf is below the frozen tested floor (0.5%).
    """
    if per_cell_chrm_depth <= 0:
        raise ValueError("per_cell_chrm_depth must be > 0 (measure it in Step 0)")
    d = depth_needed(target_vaf)
    if d is None:
        return None
    import math
    return int(math.ceil(d / per_cell_chrm_depth))

def compartment_feasible(n_cells, per_cell_chrm_depth, target_vaf):
    """Feasibility verdict for one compartment at a target VAF.

    Returns dict(feasible, pooled_depth, licensed_floor, cells_needed, verdict).
    A compartment that cannot reach its target's required depth ABSTAINS at the compartment level.
    """
    pooled = n_cells * per_cell_chrm_depth
    floor = licensed_floor(pooled)
    need = cells_needed(target_vaf, per_cell_chrm_depth)
    feasible = (need is not None) and (n_cells >= need)
    if floor is None:
        verdict = "ABSTAIN (pooled depth <100x: no licensed call)"
    elif need is None:
        verdict = f"ABSTAIN (target {target_vaf:.1%} below frozen tested floor 0.5%)"
    elif feasible:
        verdict = f"OK (licenses {floor:.1%}; needs {need} cells, has {n_cells})"
    else:
        verdict = f"ABSTAIN (needs {need} cells for {target_vaf:.1%}, has {n_cells}; licenses {floor:.1%} at current depth)"
    return dict(feasible=feasible, pooled_depth=pooled, licensed_floor=floor,
                cells_needed=need, verdict=verdict)

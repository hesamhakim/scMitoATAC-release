"""Step 4 — claim licensing + used_in_partition double-dipping guard.

Rules (master plan v3.5 §7-D.4):
  - ABSOLUTE VAF is licensed only ABOVE the per-sample depth floor AND at abstention-passing sites.
  - RELATIVE between-compartment differences are licensed more liberally (Step 3 null).
  - Variants USED IN THE PARTITION get the strictest CI treatment (the calibrated CI stays honest at
    clustering variants). In the nested CNV->mtDNA design the tag propagates through BOTH levels;
    mtDNA-partition variants are the sharpest double-dipping case (Refinement 3).
"""

def license_call(pos, vaf, pooled_depth, licensed_floor, abstained, used_in_partition,
                 kind="absolute"):
    """Return a licensing verdict dict for one (position, compartment) call."""
    if kind == "absolute":
        ok = (licensed_floor is not None) and (not abstained) and (vaf is not None) and (vaf >= licensed_floor)
        note = "absolute VAF licensed" if ok else "absolute VAF NOT licensed (below floor / abstained)"
    else:  # relative
        ok = not abstained
        note = "relative difference licensed (see Step 3 null)" if ok else "compartment abstained"
    flags = []
    if used_in_partition:
        flags.append("USED_IN_PARTITION: apply strictest CI (double-dipping guard); "
                     "if mtDNA-partition, this is the sharpest case")
    return dict(pos=pos, kind=kind, licensed=bool(ok), note=note, flags=flags)

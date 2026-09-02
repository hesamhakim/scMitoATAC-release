#!/usr/bin/env python
"""scmitoatac.deploy — Phase-7 deployment-layer tests.

Locks the feasibility (Step 1) arithmetic against the FROZEN 6.2 depth->floor curve, and exercises
the metacell numeric core + Step 3 null + Step 4 licensing on a fixed synthetic input. AnnData is NOT
required (the AnnData entry point is not exercised here). Run: python tests/test_deploy_layer.py
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from scmitoatac.deploy import feasibility as fz
from scmitoatac.deploy import metacell as mc
from scmitoatac.deploy import nullcompare as nc
from scmitoatac.deploy import licensing as lic

def test_licensed_floor_matches_frozen_62_curve():
    assert fz.licensed_floor(50) is None            # <100x: abstain
    assert fz.licensed_floor(100) == 0.02           # 2% @100x
    assert fz.licensed_floor(250) == 0.02
    assert fz.licensed_floor(500) == 0.01           # 1% @500x
    assert fz.licensed_floor(2000) == 0.005         # 0.5% @2000x
    assert fz.licensed_floor(50000) == 0.005

def test_cells_needed_inverts_the_curve():
    # ~40x per cell (PBMC placeholder): 2%->~3 cells(100x), 1%->~13(500x), 0.5%->50(2000x)
    assert fz.cells_needed(0.02, 40) == 3           # ceil(100/40)
    assert fz.cells_needed(0.01, 40) == 13          # ceil(500/40)
    assert fz.cells_needed(0.005, 40) == 50         # ceil(2000/40)
    assert fz.cells_needed(0.001, 40) is None       # below frozen tested floor
    # a deep sample needs far fewer cells (own-depth rule)
    assert fz.cells_needed(0.005, 2000) == 1        # ceil(2000/2000)

def test_compartment_feasible_abstains_when_shallow():
    # 5 cells at 40x -> pooled 200x, licenses 2% but NOT 1% (needs 13 cells)
    r1 = fz.compartment_feasible(5, 40, 0.02); assert r1["feasible"] and r1["licensed_floor"] == 0.02
    r2 = fz.compartment_feasible(5, 40, 0.01); assert not r2["feasible"]

def test_metacell_pool_and_call():
    rng = np.random.default_rng(7)
    n_cells, n_pos = 100, 3
    labels = np.array(["A"]*60 + ["B"]*40)
    dp = np.full((n_cells, n_pos), 45)                 # 45x/cell
    alt = np.zeros((n_cells, n_pos), dtype=int)
    # inject a 10% variant at pos 1 in partition A only
    aA = labels == "A"
    alt[aA, 1] = rng.binomial(45, 0.10, size=aA.sum())
    res = mc.call_metacells(alt, dp, labels, per_cell_chrm_depth=45, target_vaf=0.02)
    byp = {r["partition"]: r for r in res}
    assert byp["A"]["feasible"] and byp["B"]["feasible"]         # both >20 cells @45x -> pooled >900x
    assert abs(byp["A"]["pooled_vaf"][1] - 0.10) < 0.03         # A recovers ~10%
    assert byp["B"]["pooled_vaf"][1] < 0.01                     # B ~0

def test_null_and_licensing():
    rng = np.random.default_rng(3)
    n_cells, n_pos = 200, 2
    labels = np.array(["A"]*100 + ["B"]*100)
    dp = np.full((n_cells, n_pos), 50)
    alt = np.zeros((n_cells, n_pos), dtype=int)
    alt[labels=="A", 0] = rng.binomial(50, 0.08, size=100)     # 8% in A
    null = nc.random_pool_null(alt, dp, pos=0, group_sizes=[50], n_draws=100, seed=1)
    rel = nc.relative_difference(0.08, 0.0, null)
    assert rel["licensed"]                                      # real 8% diff beats random-pool null
    v = lic.license_call(0, 0.08, 5000, 0.005, abstained=False, used_in_partition=True)
    assert v["licensed"] and any("USED_IN_PARTITION" in f for f in v["flags"])

def test_call_metacells_calibrated_uses_pooled_estimate():
    """The fix: with an error-null supplied, call_metacells routes through the FROZEN
    pooling.pooled_estimate and returns CI + null-tail callability (not a raw fraction)."""
    import numpy as np
    from scmitoatac.deploy import metacell
    rng = np.random.default_rng(0)
    n_cells, n_pos = 400, 3
    dp = np.full((n_cells, n_pos), 40.0)
    alt = np.zeros((n_cells, n_pos))
    alt[:, 1] = rng.binomial(40, 0.05, size=n_cells)      # a real ~5% variant at pos 1
    labels = np.array([f"g{i % 4}" for i in range(n_cells)])
    recs = metacell.call_metacells(alt, dp, labels, per_cell_chrm_depth=40.0, target_vaf=0.02,
                                   mu_e_by_pos=np.full(n_pos, 3e-4), s_e=200.0)
    feas = [r for r in recs if r["feasible"]]
    assert feas, "expected feasible compartments at 100 cells x 40x"
    r0 = feas[0]
    assert r0["estimator"] == "pooled_estimate", r0["estimator"]
    for k in ("vaf_ci_low", "vaf_ci_high", "tail_p", "callable", "pooled_vaf_raw"):
        assert k in r0 and r0[k] is not None, k
    # CI must bracket the point estimate at the variant position
    assert r0["vaf_ci_low"][1] <= r0["pooled_vaf"][1] <= r0["vaf_ci_high"][1]
    # the real 5% variant is callable; the silent positions are not
    assert bool(r0["callable"][1]) is True
    assert bool(r0["callable"][0]) is False

def test_call_metacells_labels_uncalibrated_fallback():
    """Without an error-null the null-tail test is undefined -> raw fraction, EXPLICITLY labelled."""
    import numpy as np
    from scmitoatac.deploy import metacell
    dp = np.full((200, 2), 40.0); alt = np.zeros((200, 2)); alt[:, 0] = 2.0
    labels = np.array([f"g{i % 2}" for i in range(200)])
    recs = metacell.call_metacells(alt, dp, labels, per_cell_chrm_depth=40.0, target_vaf=0.02)
    feas = [r for r in recs if r["feasible"]]
    assert feas and feas[0]["estimator"] == "raw_uncalibrated"
    assert feas[0]["callable"] is None          # deficiency explicit, not silent
    assert np.isclose(feas[0]["pooled_vaf"][0], 0.05)

def test_call_metacells_abstained_marked():
    import numpy as np
    from scmitoatac.deploy import metacell
    dp = np.full((6, 2), 3.0); alt = np.zeros((6, 2))
    labels = np.array(["g0"] * 6)               # 6 cells at 3x cannot license 1%
    recs = metacell.call_metacells(alt, dp, labels, per_cell_chrm_depth=3.0, target_vaf=0.01)
    assert recs[0]["feasible"] is False
    assert recs[0]["pooled_vaf"] is None and recs[0]["estimator"] == "abstained"

if __name__ == "__main__":
    fns = [f for f in dict(globals()) if f.startswith("test_")]
    p = 0
    for fn in fns:
        globals()[fn](); print("PASS", fn); p += 1
    print(f"\n{p}/{len(fns)} passed")

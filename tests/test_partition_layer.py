#!/usr/bin/env python
"""scmitoatac.deploy.partition -- format/registry/synthetic/comparison tests.

Exercises the whole Step-0 boundary WITHOUT any external partitioner: the standard PartitionResult
format, its metacell-granularity contract, the registry, the synthetic generators, and the
cross-tool disagreement metric. This is the decoupled build/test the user asked for.
"""
import numpy as np
from scmitoatac.deploy import partition as P

N = 2000  # a dataset large enough that metacell granularity => hundreds of groups

def test_random_metacell_granularity():
    r = P.run("random_metacell", N, target_size=50, seed=1)
    ok, issues = r.validate(granularity="metacell")
    assert ok, issues
    q = r.qc()
    assert q["n_groups"] >= 20, q                 # hundreds, not coarse
    assert 10 <= q["median_group_size"] <= 200, q  # metacell scale
    assert q["n_assigned"] == N

def test_structured_metacell_valid():
    r = P.run("structured_metacell", N, target_size=40, n_blocks=8, purity=0.85, seed=2)
    ok, issues = r.validate(granularity="metacell")
    assert ok, issues
    assert r.level == "metacell"

def test_coarse_clone_fails_metacell_contract():
    r = P.run("coarse_clone", N, n_clones=3)
    ok, issues = r.validate(granularity="metacell")
    assert not ok and any("coarse" in i for i in issues), issues   # negative control
    # but legal at clone level
    ok2, _ = r.validate(granularity="clone")
    assert ok2

def test_registry_lists_synthetic_and_kinds():
    reg = P.list_partitioners()
    for n in ("random_metacell", "structured_metacell", "coarse_clone"):
        assert n in reg and reg[n]["kind"] == "synthetic", reg
    assert reg["coarse_clone"]["level"] == "clone"

def test_external_adapter_run_refused():
    class Dummy(P.ExternalAdapter):
        name = "dummy_ext"; axis = "CNV"; level = "clone"
    P.register(Dummy())
    try:
        P.run("dummy_ext", 100); assert False, "external run() should refuse"
    except TypeError:
        pass

def test_comparison_diversity():
    # two different tools (different seeds/structure) on the SAME cells -> should disagree (ARI < 1)
    ids = np.array([f"cell_{i}" for i in range(N)])
    class _AD:  # minimal adata-like with obs_names
        obs_names = ids
    ad = _AD()
    r1 = P.run("structured_metacell", ad, n_blocks=8, purity=0.9, seed=1)
    r2 = P.run("structured_metacell", ad, n_blocks=8, purity=0.9, seed=99)
    r3 = P.run("random_metacell", ad, seed=7)
    cmp = P.compare_partitions([r1, r2, r3])
    # diversity scale (matches eval diversity.py): HIGH vi_dissimilarity = tools DO disagree
    assert cmp["mean_vi_dissimilarity"] > 0.1, cmp["mean_vi_dissimilarity"]
    assert "pairwise" in cmp and len(cmp["pairwise"]) == 3
    for p in cmp["pairwise"]:
        assert 0.0 <= p["vi_dissimilarity"] <= 1.0 and 0.0 <= p["ami_agreement"] <= 1.0
    # identical partitions are maximally SIMILAR: VI ~ 0
    assert P._norm_vi(list(r1.labels), list(r1.labels)) < 1e-9

def test_comparison_requires_same_universe():
    r1 = P.run("random_metacell", 100, seed=1)
    r2 = P.run("random_metacell", 200, seed=1)   # different universe
    try:
        P.compare_partitions([r1, r2]); assert False, "should reject mismatched universes"
    except ValueError:
        pass

def test_to_obs_roundtrip():
    ids = np.array([f"cell_{i}" for i in range(300)])
    class _AD:
        def __init__(s): s.obs_names = ids; s.obs = {}; s.uns = {}
    ad = _AD()
    r = P.run("random_metacell", ad, target_size=30, seed=3)
    key = r.to_obs(ad)
    assert key in ad.obs and key in ad.uns
    assert ad.uns[key]["method"] == "random_metacell"
    assert ad.uns[key]["qc"]["n_groups"] == r.n_groups

if __name__ == "__main__":
    fns = [f for f in dir() if f.startswith("test_")]
    p = 0
    for fn in fns:
        globals()[fn](); print("PASS", fn); p += 1
    print(f"\n{p}/{len(fns)} passed")

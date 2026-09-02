"""Unit tests for the Phase-10 between-cluster differential-heteroplasmy method.

Locks the statistical behavior that the validation depends on. Does NOT touch the frozen caller
(tests/test_pipeline_regression.py stays 3/3); this only exercises the non-frozen deploy layer.
"""
import numpy as np
import pytest

from scmitoatac.deploy import between_cluster as bc

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster_spikein as cs


def test_firth_finite_under_separation():
    X = np.column_stack([np.ones(6), np.array([-3., -2, -1, 1, 2, 3])])
    y = np.array([0., 0, 0, 1, 1, 1])
    b = bc.firth_logit(X, y)
    assert np.all(np.isfinite(b))          # plain logistic MLE would diverge here


def test_rate_detects_true_exclusive():
    rng = np.random.default_rng(1)
    n = 2000
    cl = np.array(["A", "B"])[rng.integers(0, 2, n)]
    dp = np.full(n, 50.)
    carr = np.where(cl == "A", rng.random(n) < 0.20, rng.random(n) < 0.01).astype(float)
    r = bc.rate_test(carr, cl, dp)
    assert r["p"] < 1e-4 and r["enriched_in"] == "A"


def test_overlap_coverage_confound_not_licensed():
    # true VAF uniform, detection depth-driven, clusters differ in mean depth but OVERLAP.
    # The parametric p can leave a small residual, so the ARBITER is the coverage-stratified
    # permutation null: within a depth decile there is no cluster effect, so a pure coverage
    # confound must NOT be licensed.
    rng = np.random.default_rng(2)
    n = 6000
    cl = np.array(["A", "B"])[rng.integers(0, 2, n)]
    dp = np.clip(np.where(cl == "A", rng.normal(38, 12, n), rng.normal(24, 12, n)), 5, 90).astype(float)
    carr = (rng.binomial(dp.astype(int), 0.02) >= 1).astype(float)
    statfn = lambda c: bc.rate_test(carr, c, dp, coverage="decile")["LR"]
    obs = statfn(cl)
    null, sfrac = bc.coverage_stratified_null(statfn, cl, dp, n_perm=200)
    licensed = (obs > np.quantile(null, 0.95)) and (sfrac >= 0.5)
    assert not licensed                     # pure coverage confound is not licensed by the permutation null


def test_disjoint_support_is_not_licensed():
    # non-identifiable: clusters occupy disjoint depth ranges -> positivity guard must withhold license
    rng = np.random.default_rng(3)
    n = 4000
    cl = np.array(["A", "B"])[rng.integers(0, 2, n)]
    dp = np.where(cl == "A", rng.integers(40, 80, n), rng.integers(3, 12, n)).astype(float)
    carr = (rng.binomial(dp.astype(int), 0.02) >= 1).astype(float)
    statfn = lambda c: bc.rate_test(carr, c, dp, coverage="decile")["LR"]
    obs = statfn(cl)
    null, sfrac = bc.coverage_stratified_null(statfn, cl, dp, n_perm=100)
    licensed = (obs > np.quantile(null, 0.95)) and (sfrac >= 0.5)
    assert not licensed and sfrac < 0.5


def test_betabinom_glm_vaf_gradient():
    rng = np.random.default_rng(4)
    n = 3000
    cl = np.array(["A", "B"])[rng.integers(0, 2, n)]
    dp = rng.integers(20, 60, n).astype(float)
    alt = rng.binomial(dp.astype(int), np.where(cl == "A", 0.10, 0.02)).astype(float)
    assert bc.betabinom_glm(alt, dp, cl, s=50.)["p"] < 1e-3
    alt0 = rng.binomial(dp.astype(int), 0.05).astype(float)
    assert bc.betabinom_glm(alt0, dp, cl, s=50.)["p"] > 0.05


def test_exclusivity_statistic():
    rng = np.random.default_rng(5)
    n = 3000
    cl = np.array(["A", "B", "C"])[rng.integers(0, 3, n)]
    dp = rng.integers(20, 60, n).astype(float)
    e = bc.exclusivity(np.where(cl == "A", rng.random(n) < 0.3, 0.).astype(float), cl, dp, n_perm=300)
    assert e["E"] > 0.9 and e["perm_p"] < 0.05 and e["top"] == "A"
    e2 = bc.exclusivity((rng.random(n) < 0.1).astype(float), cl, dp, n_perm=300)
    assert e2["perm_p"] > 0.05


def test_spikein_recovers_exclusive_and_differential():
    rng = np.random.default_rng(6)
    n = 4000
    cl = np.array(["A", "B", "C"])[rng.integers(0, 3, n)]
    dp = rng.integers(30, 60, n).astype(float)
    spikes = [
        {"kind": "exclusive", "pos": 1000, "carrier_clusters": ["B"], "within_carrier_vaf": 0.10, "carrier_frac": 0.6},
        {"kind": "differential", "pos": 2000, "vaf_by_cluster": {"A": 0.10, "B": 0.02, "C": 0.02}, "carrier_frac": 1.0},
        {"kind": "differential", "pos": 3000, "vaf_by_cluster": {"A": 0.05, "B": 0.05, "C": 0.05}, "carrier_frac": 1.0},  # null
    ]
    alt_by_pos, truth = cs.build_cluster_spiked_matrix(dp, cl, spikes, rng=rng)
    # exclusive: rate axis licenses and points at B
    carr = (alt_by_pos[1000] >= 1).astype(float)
    r = bc.rate_test(carr, cl, dp)
    assert r["p"] < 1e-4 and r["enriched_in"] == "B"
    # differential: VAF axis fires for the gradient, not for the equal-VAF null
    assert bc.betabinom_glm(alt_by_pos[2000], dp, cl, s=50.)["p"] < 1e-2
    assert bc.betabinom_glm(alt_by_pos[3000], dp, cl, s=50.)["p"] > 0.05


def test_shallow_cluster_abstains():
    # a tiny/shallow cluster should be flagged infeasible by the frozen feasibility gate
    from scmitoatac.deploy.feasibility import compartment_feasible
    feasible_big = compartment_feasible(n_cells=1000, per_cell_chrm_depth=20, target_vaf=0.02)
    feasible_tiny = compartment_feasible(n_cells=5, per_cell_chrm_depth=3, target_vaf=0.02)
    assert feasible_big["feasible"] and not feasible_tiny["feasible"]


# ---- pooled between-partition test (the depth-robust primary path) ----

def test_pooled_recovers_at_low_per_cell_depth():
    # THE mechanism claim: many cells at ~1-3x per cell pool to clear the floor and recover a between-partition
    # VAF difference -- i.e. un-enriched per-cell depth is fine given enough cells per partition.
    rng = np.random.default_rng(20)
    n = 6000
    cl = np.array(["A", "B"])[rng.integers(0, 2, n)]
    dp = rng.integers(1, 4, n).astype(float)                       # 1-3x per cell (un-enriched)
    alt_by_pos, _ = cs.build_cluster_spiked_matrix(
        dp, cl, [{"kind": "differential", "pos": 1, "vaf_by_cluster": {"A": 0.05, "B": 0.01}, "carrier_frac": 1.0}], rng=rng)
    res = bc.pooled_between_cluster(alt_by_pos[1], dp, cl, mu_e=3e-4, s_e=100.0, n_null=200)
    assert res["licensed"] and res["enough_pooled_depth"] and res["top"] == "A"


def test_pooled_null_equal_vaf_not_licensed():
    rng = np.random.default_rng(21)
    n = 6000
    cl = np.array(["A", "B"])[rng.integers(0, 2, n)]
    dp = rng.integers(1, 4, n).astype(float)
    alt_by_pos, _ = cs.build_cluster_spiked_matrix(
        dp, cl, [{"kind": "differential", "pos": 1, "vaf_by_cluster": {"A": 0.03, "B": 0.03}, "carrier_frac": 1.0}], rng=rng)
    res = bc.pooled_between_cluster(alt_by_pos[1], dp, cl, mu_e=3e-4, s_e=100.0, n_null=200)
    assert not res["licensed"]


def test_pooled_abstains_below_pooled_floor():
    rng = np.random.default_rng(22)
    n = 40
    cl = np.array(["A", "B"])[rng.integers(0, 2, n)]
    dp = rng.integers(1, 3, n).astype(float)                       # ~40 cells x ~2x -> pooled < 100x
    alt_by_pos, _ = cs.build_cluster_spiked_matrix(
        dp, cl, [{"kind": "differential", "pos": 1, "vaf_by_cluster": {"A": 0.05, "B": 0.01}, "carrier_frac": 1.0}], rng=rng)
    res = bc.pooled_between_cluster(alt_by_pos[1], dp, cl, mu_e=3e-4, s_e=100.0, n_null=100, min_pooled_depth=100)
    assert not res["enough_pooled_depth"]


def test_pooled_disjoint_coverage_not_licensed():
    # disjoint per-cell depth support between partitions -> not coverage-identifiable -> abstain
    rng = np.random.default_rng(23)
    n = 4000
    cl = np.array(["A", "B"])[rng.integers(0, 2, n)]
    dp = np.where(cl == "A", rng.integers(40, 80, n), rng.integers(1, 4, n)).astype(float)
    alt_by_pos, _ = cs.build_cluster_spiked_matrix(
        dp, cl, [{"kind": "differential", "pos": 1, "vaf_by_cluster": {"A": 0.05, "B": 0.01}, "carrier_frac": 1.0}], rng=rng)
    res = bc.pooled_between_cluster(alt_by_pos[1], dp, cl, mu_e=3e-4, s_e=100.0, n_null=200)
    assert res["shuffleable_frac"] < 0.5 and not res["licensed"]

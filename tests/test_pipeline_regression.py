#!/usr/bin/env python
"""
scMitoATAC Phase 3.6 — self-containment REGRESSION LOCK.

Pins the numeric behavior of the ported confidence-layer modules
(site_null / fusion / baseline / calibration / callability) on a FIXED synthetic
input with a fixed seed. If any refactor silently changes the scoring math, this
test fails. The point is self-containment: the project must reproduce its own
Phase-3 results without reference to scMitoHet. Run: `python -m pytest -q
tests/test_pipeline_regression.py` (or plain `python tests/test_pipeline_regression.py`).

Reference values were computed 2026-07-20 from the committed modules.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "scmitoatac")
sys.path.insert(0, PKG)

import site_null as sn          # noqa
import fusion as fu             # noqa
import baseline as bl           # noqa
import calibration as cal       # noqa
import callability as cab       # noqa

# --- fixed reference (frozen 2026-07-20, full precision) ---
REF = {
    "prior_mu0": 0.024417169946100395, "prior_s0": 4.154558427255345,
    "site_mu": 0.0025246829789115173, "site_s": 4517.649336658138,
    "pon_excess": 0.10339785423472994,
    "mix_pi": 0.11336528335556315, "mix_LR": 2124.0236010813787,
    "post_mean_carrier": 0.943269500702221, "post_mean_noncarrier": 0.00019662313714406782,
    "n_unknown": 0, "baseline_max": 0.6414056289608296,
    "ece": 0.14588472005513473, "mce": 0.2500257479023491,
    "pos_tier": "A", "pos_median_dp": 31.0,
}
RTOL = 1e-6


def _fixture():
    rng = np.random.default_rng(20240720)
    n = 500
    dp = rng.poisson(30, n) + 1
    alt = rng.binomial(dp, 2.5e-4)
    car = np.zeros(n, bool); car[:60] = True
    alt[car] = rng.binomial(dp[car], 0.20)
    return dp, alt, car, rng


def test_scoring_chain_regression():
    dp, alt, car, rng = _fixture()
    prior = sn.fit_global_prior([(alt.astype(float), dp.astype(float))], single_err_floor=2.32e-4)
    snf = sn.fit_site_null(alt, dp, prior)
    mix = fu.mixture_lr(alt, dp, snf["mu"], snf["s"])
    fz = fu.fuse_posterior(alt, dp, snf, mix, k_min=2)
    bs = bl.baseline_site_scores(alt, dp)

    got = {
        "prior_mu0": prior["mu0"], "prior_s0": prior["s0"],
        "site_mu": float(snf["mu"]), "site_s": float(snf["s"]),
        "pon_excess": float(snf["pon_excess"]),
        "mix_pi": float(mix["pi"]), "mix_LR": float(mix["LR"]),
        "post_mean_carrier": float(fz["posterior"][car].mean()),
        "post_mean_noncarrier": float(fz["posterior"][~car].mean()),
        "baseline_max": float(bs["score"].max()),
    }
    for k, v in got.items():
        assert np.isclose(v, REF[k], rtol=RTOL, atol=1e-8), f"{k}: got {v}, ref {REF[k]}"
    assert int(fz["unknown"].sum()) == REF["n_unknown"]
    # separation invariant (the science, not just the number): carrier >> non-carrier
    assert fz["posterior"][car].mean() > 0.9
    assert fz["posterior"][~car].mean() < 0.01


def test_calibration_primitives():
    # Reproduce the exact reference RNG stream: the fixture draws, THEN the calib draw.
    rng = np.random.default_rng(20240720)
    n = 500
    dp = rng.poisson(30, n) + 1
    alt = rng.binomial(dp, 2.5e-4)
    car = np.zeros(n, bool); car[:60] = True
    alt[car] = rng.binomial(dp[car], 0.20)
    yt = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1] * 20)
    pp = np.clip(yt * 0.7 + rng.random(len(yt)) * 0.3, 0, 1)
    em = cal.ece_mce(yt, pp, n_bins=5)
    assert np.isclose(em["ece"], REF["ece"], rtol=RTOL), f"ece {em['ece']} vs {REF['ece']}"
    assert np.isclose(em["mce"], REF["mce"], rtol=RTOL), f"mce {em['mce']} vs {REF['mce']}"


def test_callability():
    dp, alt, car, rng = _fixture()
    pt = cab.position_tier(dp, vaf_B=0.05, k_min=2)
    assert pt["tier"] == REF["pos_tier"]
    assert np.isclose(pt["median_dp"], REF["pos_median_dp"], rtol=RTOL)


if __name__ == "__main__":
    test_scoring_chain_regression()
    test_calibration_primitives()
    test_callability()
    print("ALL REGRESSION TESTS PASS")
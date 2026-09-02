"""Minimal reproducible example for scMitoATAC.

Runs in seconds on synthetic data with no downloads, and exercises the two layers
described in the paper:

  1. Dynamic metacell sizing -- how many cells a compartment must pool to license a
     target heteroplasmy, computed from the sample's OWN measured per-cell chrM
     depth rather than a fixed constant.
  2. The coverage-stratified between-population test -- comparing pooled
     heteroplasmy between two cell populations against a null that matches group
     size and sequencing depth, returning an explicit licensed / abstain verdict.

Run:  python examples/minimal_example.py
"""
import numpy as np

from scmitoatac.deploy import feasibility as fz
from scmitoatac.deploy import between_cluster as bc

# ------------------------------------------------------------------ 1. envelope
print("Depth-to-detection envelope (the planning table)")
for d in (100, 250, 500, 1000, 2000, 5000):
    print("   pooled depth %5dx  ->  licensed floor %.1f%%" % (d, 100 * fz.licensed_floor(d)))

print("\nDynamic metacell sizing, from a sample's own measured per-cell depth")
for per_cell in (5.0, 30.0, 150.0):        # shallow PBMC ... mitochondria-rich tissue
    for target in (0.02, 0.01):
        print("   per-cell chrM depth %5.0fx, target %4.0f%%  ->  pool %s cells"
              % (per_cell, target * 100, fz.cells_needed(target, per_cell)))

# -------------------------------------------- 2. between-population level test
rng = np.random.default_rng(0)
n = 1200
cluster = np.where(np.arange(n) < 600, "malignant", "normal")
mu_e, s_e = 4.6e-4, 200.0                  # per-position background error null


def depths(kind):
    """Overlapping depth (comparable populations) vs disjoint depth (the hard case)."""
    if kind == "overlapping":
        dp = np.where(cluster == "malignant", rng.poisson(40, n), rng.poisson(28, n))
    else:                                   # malignant sequenced ~5x deeper
        dp = np.where(cluster == "malignant", rng.poisson(60, n), rng.poisson(12, n))
    dp = dp.astype(float); dp[dp < 1] = 1.0
    return dp


def draw(dp, vaf_malignant, vaf_normal):
    v = np.where(cluster == "malignant", vaf_malignant, vaf_normal)
    return rng.binomial(dp.astype(int), np.clip(v + mu_e, 0, 1)).astype(float)


def report(title, dp, alt):
    r = bc.pooled_between_cluster(alt, dp, cluster, mu_e, s_e, seed=0)
    print("\n%s" % title)
    for g, st in r["per_cluster"].items():
        print("   %-10s pooled VAF %6.3f%%  (%d cells, pooled depth %.0fx)"
              % (g, 100 * st["vaf"], st["n_cells"], st["pooled_depth"]))
    print("   p=%.3f  positivity=%.2f  ->  %s"
          % (r["empirical_p"], r["shuffleable_frac"],
             "LICENSED" if r["licensed"] else "ABSTAIN"))
    return r


dp_ov = depths("overlapping")
report("(a) Real shift, comparable depth -- malignant 6%, normal 0%:",
       dp_ov, draw(dp_ov, 0.06, 0.0))

report("(b) No biological difference, same depths -- both 2%:",
       dp_ov, draw(dp_ov, 0.02, 0.02))

dp_dj = depths("disjoint")
report("(c) Real shift, but the populations share no coverage range:",
       dp_dj, draw(dp_dj, 0.06, 0.0))

print("""
(a) is licensed: the shift beats a null that matched size and depth.
(b) is not: there is nothing to find, and the test says so.
(c) is the honest refusal. Pooled VAF is read-weighted, so a deeper cluster reads
    higher on its own. When two populations share no coverage range, depth and
    biology are not separable -- the positivity gate abstains instead of reporting
    a difference that depth alone could explain.""")

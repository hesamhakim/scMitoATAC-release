"""Step 0 boundary — the standard PartitionResult format, adapter registry, and
built-in synthetic metacell generators.

WHY THIS EXISTS (master plan v3.5 §7-D.0; user direction 2026-07-22):
scMitoATAC does not wait for, vendor, or re-implement external partitioners. Instead it fixes
ONE standard format -- `PartitionResult` -- that every partitioner (external tool OR built-in
synthetic generator) is normalized into, and feeds that to the frozen caller's metacell layer.
Because the format is fixed here, the whole tool is built and tested NOW against arbitrary
(random / structured) metacell assignments, decoupled from the external-tool evaluation.

DESIGN GOALS (all enforced by `PartitionResult.validate`):
  1. METACELL GRANULARITY, not coarse clones. Target 10-200 cells per group => HUNDREDS of
     partitions on a real dataset, not 2-4 malignant/normal blocks. A 2-4 group partition is a
     legal `level="clone"` result but is flagged when metacell granularity was requested.
  2. COMPARABILITY. Every PartitionResult carries the cell-id universe it was computed on; two
     results are comparable only if they cover the SAME cells. This is what lets diverse tools
     be compared on identical data.
  3. DIVERSITY IS THE POINT. Different partitioners SHOULD disagree; `compare_partitions`
     quantifies that disagreement (adjusted Rand index) so each tool's partition can be scored
     by how much independent structure it contributes when fed to the caller.

INTEGRATION MODEL (unchanged): for an EXTERNAL tool, scMitoATAC EMITS an optimized CLI recipe
(`Adapter.recipe`) that the user runs against their own install, then INGESTS the tool's output
(`Adapter.parse`) into a PartitionResult. Built-in SYNTHETIC generators implement `.run` directly
(no external step) and exist to build/test the tool today.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
import datetime as _dt

# Recognized partition axes (from docs/deployment/partition_tool_shortlist.md) + synthetic.
AXES = {"CNV", "ecDNA", "nuclear_SNP_donor", "nuclear_SNP_somatic", "mtDNA", "synthetic"}
LEVELS = {"metacell", "clone", "donor"}

# Metacell-granularity contract (user direction 2026-07-22).
METACELL_MIN_CELLS = 10      # smallest admissible group at metacell granularity
METACELL_MAX_MEDIAN = 200    # median group size above this is "coarse", not metacell
METACELL_MIN_GROUPS = 20     # fewer groups than this on a large dataset => coarse, flagged


@dataclass
class PartitionResult:
    """The one format every partitioner is normalized into before it reaches the caller.

    labels    : (n_cells,) group id per cell (int/str). -1 or "" == unassigned (dropped from calling).
    cell_ids  : (n_cells,) cell barcodes -- the cell UNIVERSE, required for cross-tool comparability.
    method    : partitioner name (e.g. "Copy-scAT", "random_metacell").
    version   : partitioner version string.
    axis      : one of AXES.
    level     : "metacell" | "clone" | "donor".
    params    : the exact parameters used (for the run manifest).
    used_in_partition : variant positions the partition CONSUMED (mtDNA axis); [] otherwise.
                        Read by licensing.py's double-dipping guard.
    provenance: {inputs, command/recipe, timestamp, tool_output_path?}.
    """
    labels: np.ndarray
    cell_ids: np.ndarray
    method: str
    version: str = "unknown"
    axis: str = "synthetic"
    level: str = "metacell"
    params: dict = field(default_factory=dict)
    used_in_partition: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.labels = np.asarray(self.labels)
        self.cell_ids = np.asarray(self.cell_ids)
        if "timestamp" not in self.provenance:
            self.provenance["timestamp"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    # ---- assigned-cell mask (unassigned = -1 or "") ----
    def _assigned_mask(self):
        lab = self.labels
        if lab.dtype.kind in "iu":
            return lab >= 0
        return np.array([str(x) not in ("", "-1", "nan", "NaN", "None") for x in lab])

    def group_sizes(self):
        """dict group_id -> n_cells, over ASSIGNED cells only."""
        m = self._assigned_mask()
        vals, counts = np.unique(self.labels[m], return_counts=True)
        return dict(zip(vals.tolist(), counts.tolist()))

    @property
    def n_groups(self):
        return len(self.group_sizes())

    @property
    def n_cells(self):
        return len(self.cell_ids)

    @property
    def median_group_size(self):
        gs = list(self.group_sizes().values())
        return float(np.median(gs)) if gs else 0.0

    def qc(self):
        gs = self.group_sizes()
        sizes = np.array(list(gs.values())) if gs else np.array([0])
        m = self._assigned_mask()
        return dict(n_cells=self.n_cells, n_assigned=int(m.sum()),
                    n_unassigned=int((~m).sum()), n_groups=self.n_groups,
                    median_group_size=float(np.median(sizes)),
                    min_group_size=int(sizes.min()), max_group_size=int(sizes.max()))

    def validate(self, granularity="metacell", strict=False):
        """Check the result against the format contract. Returns (ok, issues[]).

        granularity="metacell" enforces the 10-200 cells/group, hundreds-of-groups contract.
        granularity="clone"/"donor"/None relaxes it (coarse partitions are legal at those levels).
        strict=True raises on any issue instead of returning them.
        """
        issues = []
        if len(self.labels) != len(self.cell_ids):
            issues.append(f"labels ({len(self.labels)}) and cell_ids ({len(self.cell_ids)}) length mismatch")
        if self.axis not in AXES:
            issues.append(f"axis {self.axis!r} not in {sorted(AXES)}")
        if self.level not in LEVELS:
            issues.append(f"level {self.level!r} not in {sorted(LEVELS)}")
        if len(set(self.cell_ids)) != len(self.cell_ids):
            issues.append("cell_ids are not unique (breaks comparability)")
        gs = self.group_sizes()
        if gs:
            small = {g: n for g, n in gs.items() if n < METACELL_MIN_CELLS}
            if granularity == "metacell":
                if small:
                    issues.append(f"{len(small)} group(s) below {METACELL_MIN_CELLS} cells "
                                  f"(metacell floor); e.g. {dict(list(small.items())[:3])}")
                if self.median_group_size > METACELL_MAX_MEDIAN:
                    issues.append(f"median group size {self.median_group_size:.0f} > "
                                  f"{METACELL_MAX_MEDIAN} (coarse, not metacell granularity)")
                if self.n_groups < METACELL_MIN_GROUPS and self.n_cells >= 400:
                    issues.append(f"only {self.n_groups} groups on {self.n_cells} cells "
                                  f"(looks like coarse clones, not metacells)")
        else:
            issues.append("no assigned cells")
        ok = len(issues) == 0
        if strict and not ok:
            raise ValueError("PartitionResult.validate failed:\n  - " + "\n  - ".join(issues))
        return ok, issues

    # ---- AnnData plug points ----
    def to_obs(self, adata, key=None):
        """Write labels into adata.obs[key] and metadata into adata.uns[key]. Requires cell-id match."""
        key = key or f"partition_{self.method}"
        obs_ids = np.asarray(adata.obs_names)
        if not np.array_equal(obs_ids, self.cell_ids):
            # align by barcode
            pos = {c: i for i, c in enumerate(self.cell_ids)}
            if not set(obs_ids).issubset(pos):
                raise ValueError("adata contains cells not in this PartitionResult; cannot align")
            lab = np.array([self.labels[pos[c]] for c in obs_ids])
        else:
            lab = self.labels
        adata.obs[key] = lab
        adata.uns[key] = self.manifest()
        return key

    def manifest(self):
        """The run-manifest dict written to .uns (metadata only, no per-cell arrays)."""
        d = dict(method=self.method, version=self.version, axis=self.axis, level=self.level,
                 params=self.params, used_in_partition=list(self.used_in_partition),
                 provenance=self.provenance, qc=self.qc())
        return d


# ============================ Registry ============================
_REGISTRY = {}

def register(adapter):
    """Register an adapter under adapter.name. Accepts a class (instantiated here) or an instance.
    Returns the original object so it still works as a class decorator."""
    inst = adapter() if isinstance(adapter, type) else adapter
    _REGISTRY[inst.name] = inst
    return adapter

def list_partitioners():
    """Return {name: {axis, level, kind}} for every registered partitioner."""
    return {n: dict(axis=a.axis, level=a.level,
                    kind=("synthetic" if getattr(a, "synthetic", False) else "external"))
            for n, a in _REGISTRY.items()}

def get(name):
    if name not in _REGISTRY:
        raise KeyError(f"partitioner {name!r} not registered; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]

def run(name, adata_or_n, **params):
    """Run a registered partitioner -> PartitionResult.

    Synthetic generators run directly. External adapters raise here (they are two-step:
    emit `recipe(...)` for the user to run, then `parse(...)` the output); use those methods.
    """
    a = get(name)
    if not getattr(a, "synthetic", False):
        raise TypeError(f"{name!r} is an external adapter: call .recipe(params) then .parse(output). "
                        "Only synthetic generators support run().")
    return a.run(adata_or_n, **params)


# ============================ External-adapter contract ============================
class ExternalAdapter:
    """Base for an external partitioner. scMitoATAC EMITS recipe(), the user runs it, we parse().

    Subclasses set name/axis/level and implement defaults(), recipe(), check_inputs(), parse().
    The evaluation team's per-tool card supplies the validated recipe + parse logic.
    """
    name = "external"; axis = "CNV"; level = "clone"; synthetic = False
    def defaults(self): return {}
    def check_inputs(self, adata):
        """Return list of blocker/warning strings (empty = ready). Enforces operational gates."""
        raise NotImplementedError
    def recipe(self, params=None, inputs=None):
        """Return the optimized CLI command string for the user to run against their own install."""
        raise NotImplementedError
    def parse(self, tool_output_path, cell_ids):
        """Ingest the tool's native output -> PartitionResult on the given cell universe."""
        raise NotImplementedError


# ============================ Built-in synthetic generators (testable NOW) ============================
class _SyntheticBase:
    axis = "synthetic"; level = "metacell"; synthetic = True
    def _cellids(self, adata_or_n):
        if isinstance(adata_or_n, (int, np.integer)):
            n = int(adata_or_n)
            return np.array([f"cell_{i}" for i in range(n)]), n
        ids = np.asarray(adata_or_n.obs_names)
        return ids, len(ids)


@register
class RandomMetacells(_SyntheticBase):
    """Assign cells to random metacells of ~target_size. The NULL partition: no real structure,
    hundreds of groups at metacell granularity. Used as the random-pool baseline and to exercise
    the format/caller path without any external tool."""
    name = "random_metacell"
    def run(self, adata_or_n, target_size=50, seed=2024, **kw):
        ids, n = self._cellids(adata_or_n)
        rng = np.random.default_rng(seed)
        k = max(1, round(n / target_size))
        labels = rng.integers(0, k, size=n)
        return PartitionResult(labels=labels, cell_ids=ids, method=self.name, version="1.0",
                               axis="synthetic", level="metacell",
                               params=dict(target_size=target_size, seed=seed, n_groups=k),
                               provenance=dict(inputs="synthetic", command=f"random_metacell(target_size={target_size})"))


@register
class StructuredMetacells(_SyntheticBase):
    """Metacells with LATENT structure: cells carry a hidden block label, and metacells are drawn
    mostly within-block (purity `p`). Two StructuredMetacells runs with different seeds/purity
    disagree on exact membership but share block structure -- a stand-in for two real tools that
    both see biology but partition it differently. Lets us test cross-tool comparison + diversity."""
    name = "structured_metacell"
    def run(self, adata_or_n, target_size=50, n_blocks=8, purity=0.85, seed=2024, **kw):
        ids, n = self._cellids(adata_or_n)
        rng = np.random.default_rng(seed)
        block = rng.integers(0, n_blocks, size=n)                 # latent structure
        k = max(1, round(n / target_size))
        # assign each metacell to a home block; cells join their block's metacells w.p. purity
        mc_block = rng.integers(0, n_blocks, size=k)
        labels = np.empty(n, dtype=int)
        by_block = {b: np.where(mc_block == b)[0] for b in range(n_blocks)}
        for i in range(n):
            if rng.random() < purity and len(by_block[block[i]]):
                labels[i] = rng.choice(by_block[block[i]])
            else:
                labels[i] = rng.integers(0, k)
        return PartitionResult(labels=labels, cell_ids=ids, method=self.name, version="1.0",
                               axis="synthetic", level="metacell",
                               params=dict(target_size=target_size, n_blocks=n_blocks, purity=purity, seed=seed),
                               provenance=dict(inputs="synthetic",
                                               command=f"structured_metacell(n_blocks={n_blocks},purity={purity})"))


@register
class CoarseClones(_SyntheticBase):
    """2-4 coarse clones (malignant/normal-style). Legal at level='clone' but intentionally FAILS
    metacell-granularity validation -- the negative control for the granularity contract."""
    name = "coarse_clone"; level = "clone"
    def run(self, adata_or_n, n_clones=3, seed=2024, **kw):
        ids, n = self._cellids(adata_or_n)
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, n_clones, size=n)
        return PartitionResult(labels=labels, cell_ids=ids, method=self.name, version="1.0",
                               axis="synthetic", level="clone",
                               params=dict(n_clones=n_clones, seed=seed),
                               provenance=dict(inputs="synthetic", command=f"coarse_clone(n_clones={n_clones})"))


# ============================ Cross-tool comparison (diversity metric) ============================
# DIVERSITY-SCALE CONVENTION (unified with scripts/partition_eval/diversity.py, 2026-07-24):
# report DISSIMILARITY where HIGH = orthogonal carving = valuable, plus AMI agreement where
# HIGH = redundant. AMI >= REDUNDANT_AMI flags a pair that carves too alike to both add value.
# Same estimator as the eval workspace (sklearn adjusted_mutual_info_score / mutual_info_score /
# entropy) so the two report IDENTICAL numbers on the same partitions.
REDUNDANT_AMI = 0.5

def _entropy(labels):
    """Shannon entropy (nats) of a label vector, matching sklearn.metrics.cluster.entropy.
    Local implementation: sklearn's `entropy` is deprecated (removed in 1.10), so compute it
    directly to stay identical to diversity.py's numbers without depending on the deprecated symbol."""
    import math
    vals, counts = np.unique(np.asarray(labels), return_counts=True)
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    return float(-np.sum(p * (np.log(counts) - math.log(n))))

def _norm_vi(a, b):
    """Normalized Variation of Information in [0,1]; 0 = identical, 1 = maximally different.
    Matches scripts/partition_eval/diversity.py::_norm_vi (H(a)+H(b)-2*MI)/log(n)."""
    import math
    from sklearn.metrics import mutual_info_score
    mi = mutual_info_score(a, b)
    vi = _entropy(a) + _entropy(b) - 2 * mi
    n = len(a)
    return float(vi / math.log(n)) if n > 1 else 0.0

def compare_partitions(results):
    """Given a list of PartitionResults on the SAME cell universe, return pairwise DIVERSITY.

    Reports the same scale as the eval workspace's diversity.py:
      - vi_dissimilarity, one_minus_ami : HIGH = orthogonal carving = valuable diversity
      - ami_agreement                   : HIGH = redundant
      - redundant                       : ami_agreement >= REDUNDANT_AMI (0.5)
    plus per-result granularity QC. Raises if the cell universes differ (comparability requirement).
    sklearn is imported lazily so the package stays numpy-only on import.
    """
    if len(results) < 2:
        raise ValueError("need >=2 PartitionResults to compare")
    from itertools import combinations
    from sklearn.metrics import adjusted_mutual_info_score
    ref = set(results[0].cell_ids)
    for r in results[1:]:
        if set(r.cell_ids) != ref:
            raise ValueError(f"{r.method!r} covers a different cell universe; not comparable")
    # align every partition to results[0]'s cell order (label values namespaced by method so
    # identical integer ids from different tools are not conflated)
    order = {c: i for i, c in enumerate(results[0].cell_ids)}
    aligned = {}
    for r in results:
        idx = np.array([order[c] for c in r.cell_ids])
        lab = np.empty(len(r.labels), dtype=object); lab[idx] = r.labels
        aligned[r.method] = np.array([f"{r.method}:{x}" for x in lab])
    names = [r.method for r in results]
    pairs = []
    for n1, n2 in combinations(names, 2):
        a, b = aligned[n1], aligned[n2]
        ami = float(adjusted_mutual_info_score(a, b))
        vi = _norm_vi(a, b)
        pairs.append(dict(tool_a=n1, tool_b=n2,
                          vi_dissimilarity=round(vi, 3),      # high = good (diverse)
                          one_minus_ami=round(1 - ami, 3),    # high = good (diverse)
                          ami_agreement=round(ami, 3),        # high = redundant
                          redundant=bool(ami >= REDUNDANT_AMI)))
    return dict(methods=names,
                granularity={r.method: r.qc() for r in results},
                pairwise=pairs,
                redundant_pairs=[f"{p['tool_a']} ~ {p['tool_b']} (AMI {p['ami_agreement']})"
                                 for p in pairs if p["redundant"]],
                mean_vi_dissimilarity=round(float(np.mean([p["vi_dissimilarity"] for p in pairs])), 3),
                interpretation="HIGH vi_dissimilarity / one_minus_ami = orthogonal carving = valuable "
                               "diversity; AMI >= 0.5 = redundant tools that add no diversity.")

#!/usr/bin/env python
"""
Metacell layer for scMitoATAC — the metacell is the reporting unit (never the cell).

A metacell is a group of cells within a cluster, sized so its pooled chrM depth can DETECT a variant down to a
minimum VAF (LOD). Rationale: un-enriched scATAC gives ~0.03-0.13x chrM/cell, so a per-cell VAF is ~single-read
noise; a metacell pools enough reads to estimate/detect VAF reliably. Pipeline unit going forward.

Sizing (per sample/cluster):
  D_min = a_min / LOD            required pooled depth to expect >= a_min alt reads at VAF=LOD  (a_min=3, LOD=0.05 -> 60)
  k     = ceil(D_min / d-bar)    cells per metacell; d-bar = mean per-cell read depth at a position
  auto-relax: if a cluster of n cells yields < M_min metacells at k, enlarge k = floor(n / M_min) and report the
              ACHIEVED LOD = a_min / (d-bar * k). Never silently fail; report LOD_ach + k + n_metacells.
d-bar is computed per position from the pileup with the UNION WHITELIST cardinality as denominator (the pileup is
sparse — only covered (cell,pos) rows exist). A representative d-bar = median per-cell depth over covered positions.

Formation within a cluster: kNN (default; greedy neighbourhoods on the LSI embedding -> cell-state-homogeneous) or
random (shuffle-partition). Aggregation: sum the per-cell cons_A/C/G/T (and depth) into metacell x position counts.

This module is imported by metacell_score.py / metacell_verify.py and is CLI-testable:
  metacell.py <tok> <partition_csv> [--lod 0.05 --a-min 3 --m-min 5 --method knn|random --track human]
"""
import os, sys, argparse, math
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sample_chrm as SC  # _load_cells_fragpileup(path, positions) -> (barcodes, base{A..T:nCxnP}, depth, positions)

ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASES = ["A", "C", "G", "T"]
CAND_LO, CAND_HI, CAND_MINDEPTH = 0.005, 0.40, 100     # candidate gate (matches sweep_score.assemble)


_REGISTRY_PATH = f"{ROOT}/work/autoscan/pileup_registry.tsv"   # optional tok->(pileup,wl) override for non-phase16
_REGISTRY = None


def _registry():
    """Lazy-load the optional pileup registry (tok<TAB>pileup<TAB>wl). Lets non-phase16 samples (e.g. phase15
    BMMC/MPAL) be scored without symlinks or touching the phase16 default. Absent file -> empty (no override)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = {}
        if os.path.exists(_REGISTRY_PATH):
            for i, line in enumerate(open(_REGISTRY_PATH)):
                p = line.rstrip("\n").split("\t")
                if i == 0 and p[0] == "tok":
                    continue
                if len(p) >= 2:
                    _REGISTRY[p[0]] = (p[1], p[2] if len(p) >= 3 else None)
    return _REGISTRY


def pileup_path(tok, track="human"):
    ov = _registry().get(tok)
    if ov:
        return ov[0]
    return f"{ROOT}/work/phase16_cancer_scan/{track}/mtdna/{tok}/{tok}_decoy_fragpileup.tsv.gz"


def whitelist_n(tok, track="human"):
    ov = _registry().get(tok)
    wl = ov[1] if (ov and ov[1]) else f"{ROOT}/work/phase16_cancer_scan/{track}/mtdna/{tok}/{tok}_wl.txt"
    if wl and os.path.exists(wl):
        return sum(1 for _ in open(wl))
    return None


def candidate_positions(tok, track="human"):
    """Positions passing the candidate gate (pooled 2nd-allele frac in [0.5%,40%], pooled depth>=100), from bulk."""
    bulk = SC._load_bulk_fragpileup(pileup_path(tok, track))       # DataFrame idx=pos, cols A,C,G,T,ref
    tot = bulk[BASES].sum(axis=1)
    ref = bulk["ref"]
    alt2 = bulk[BASES].apply(lambda r: sorted(r.values)[-2], axis=1)   # 2nd-highest base count
    frac = np.where(tot > 0, alt2 / tot, 0.0)
    keep = bulk.index[(frac >= CAND_LO) & (frac <= CAND_HI) & (tot >= CAND_MINDEPTH)]
    return sorted(int(p) for p in keep), bulk


def dbar_representative(base, depth, wl_n):
    """Representative mean per-cell depth (median over positions of total_depth/whitelist_n)."""
    if wl_n is None or wl_n == 0:
        wl_n = base["A"].shape[0]                      # fall back to covered-cell count
    perpos_total = depth.sum(axis=0)                   # nP: total reads per position (over covered cells)
    dbar_pos = perpos_total / float(wl_n)              # mean per-cell depth per position
    dbar_pos = dbar_pos[dbar_pos > 0]
    return float(np.median(dbar_pos)) if dbar_pos.size else 0.0, dbar_pos


def size_metacell(n_cells, dbar, lod=0.05, a_min=3, m_min=5):
    """Return (k, lod_achieved, n_metacells, relaxed). k cells/metacell to detect VAF=lod; auto-relax if too few."""
    if dbar <= 0 or n_cells < 1:
        return None, float("nan"), 0, True
    d_min = a_min / lod
    k = max(1, math.ceil(d_min / dbar))
    relaxed = False
    if n_cells // k < m_min:                           # too few metacells -> relax to hit M_min
        k = max(1, n_cells // m_min)
        relaxed = True
    lod_ach = a_min / (dbar * k) if k > 0 else float("nan")
    n_mc = n_cells // k
    return int(k), float(lod_ach), int(n_mc), relaxed


def form_metacells(idx, emb, k, method="knn", seed=0):
    """Partition cell indices `idx` (within one cluster) into metacells of ~k cells. Returns list of index arrays."""
    idx = np.asarray(idx)
    if len(idx) < k or k <= 1:
        return [idx] if len(idx) else []
    rng = np.random.default_rng(seed)
    if method == "random" or emb is None:
        perm = rng.permutation(idx)
        return [perm[i:i + k] for i in range(0, len(perm) - len(perm) % k, k)] or [perm]
    # kNN greedy: seed with a random unassigned cell, grab its k-1 nearest unassigned neighbours (euclidean in LSI)
    E = emb[idx]                                        # local coords
    unassigned = set(range(len(idx)))
    mcs = []
    order = list(rng.permutation(len(idx)))
    while len(unassigned) >= k:
        seed_i = next(i for i in order if i in unassigned)
        pool = np.fromiter(unassigned, int)
        d = np.linalg.norm(E[pool] - E[seed_i], axis=1)
        take = pool[np.argsort(d)[:k]]
        mcs.append(idx[take]); unassigned -= set(take.tolist())
    if unassigned:                                      # fold leftovers into nearest existing metacell centroid
        cent = np.array([E[[np.where(idx == m)[0][0] for m in mc]].mean(0) for mc in mcs])
        for i in unassigned:
            j = int(np.argmin(np.linalg.norm(cent - E[i], axis=1)))
            mcs[j] = np.append(mcs[j], idx[i])
    return mcs


def build(tok, partition_csv, lod=0.05, a_min=3, m_min=5, method="knn", track="human", emb_map=None, positions=None):
    """Core entry: size + form metacells per cluster, aggregate base/depth. Returns a dict with the metacell
    long table (metacell, group, pos, alt-not-yet; base counts) and per-cluster sizing metadata."""
    part = pd.read_csv(partition_csv)
    part["barcode"] = part["barcode"].astype(str)
    if positions is None:
        positions, _ = candidate_positions(tok, track)
    barcodes, base, depth, positions = SC._load_cells_fragpileup(pileup_path(tok, track), positions)
    bidx = {b: i for i, b in enumerate(barcodes)}
    wl_n = whitelist_n(tok, track)
    dbar, _ = dbar_representative(base, depth, wl_n)

    # embedding aligned to `barcodes` (for kNN); emb_map: {barcode: vector}
    emb = None
    if method == "knn" and emb_map is not None:
        dim = len(next(iter(emb_map.values())))
        emb = np.full((len(barcodes), dim), np.nan)
        for b, i in bidx.items():
            if b in emb_map:
                emb[i] = emb_map[b]
        if np.isnan(emb).all():
            emb = None

    rows_meta, mc_records, sizing = [], [], []
    mc_counter = 0
    for g, sub in part.groupby("group"):
        cell_ix = np.array([bidx[b] for b in sub.barcode if b in bidx])
        n = len(cell_ix)
        if n == 0:
            continue
        k, lod_ach, n_mc, relaxed = size_metacell(n, dbar, lod, a_min, m_min)
        sizing.append(dict(group=str(g), n_cells=int(n), dbar=round(dbar, 4), k=k, lod_target=lod,
                           lod_achieved=round(lod_ach, 4) if lod_ach == lod_ach else None,
                           n_metacells=n_mc, relaxed=relaxed,
                           under_resolved=bool(n_mc < m_min)))
        if k is None:
            continue
        # embedding subset for this cluster (kNN); if any nan, fall back to random for the cluster
        emb_g = emb if (emb is not None and not np.isnan(emb[cell_ix]).any()) else None
        meth = method if emb_g is not None else ("random" if method == "knn" else method)
        mcs = form_metacells(cell_ix, emb_g, k, meth)
        for mc in mcs:
            mc_id = f"mc{mc_counter}"; mc_counter += 1
            agg = {b: base[b][mc, :].sum(axis=0) for b in BASES}     # per-pos summed counts over metacell members
            dep = sum(agg[b] for b in BASES)
            for j, pos in enumerate(positions):
                if dep[j] > 0:
                    rows_meta.append((mc_id, str(g), int(pos), int(agg["A"][j]), int(agg["C"][j]),
                                      int(agg["G"][j]), int(agg["T"][j]), int(dep[j])))
            mc_records.append((mc_id, str(g), len(mc)))
    long = pd.DataFrame(rows_meta, columns=["metacell", "group", "pos", "A", "C", "G", "T", "depth"])
    manifest = pd.DataFrame(mc_records, columns=["metacell", "group", "n_cells"])
    return dict(long=long, manifest=manifest, sizing=pd.DataFrame(sizing), positions=positions, dbar=dbar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tok"); ap.add_argument("partition_csv")
    ap.add_argument("--lod", type=float, default=0.05); ap.add_argument("--a-min", type=int, default=3)
    ap.add_argument("--m-min", type=int, default=5); ap.add_argument("--method", default="knn")
    ap.add_argument("--track", default="human"); ap.add_argument("--lsi-npy"); ap.add_argument("--lsi-cells")
    a = ap.parse_args()
    emb_map = None
    if a.lsi_npy and a.lsi_cells:                      # optional cached LSI (barcode order in --lsi-cells)
        E = np.load(a.lsi_npy); cells = [l.strip() for l in open(a.lsi_cells)]
        emb_map = {c: E[i] for i, c in enumerate(cells)}
    r = build(a.tok, a.partition_csv, a.lod, a.a_min, a.m_min, a.method, a.track, emb_map)
    print(f"[{a.tok}] d-bar={r['dbar']:.4f} reads/cell; {len(r['positions'])} candidate positions; "
          f"{r['manifest'].metacell.nunique()} metacells total")
    print(r["sizing"].to_string(index=False))


if __name__ == "__main__":
    main()

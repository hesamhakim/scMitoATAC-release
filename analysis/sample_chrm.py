#!/usr/bin/env python
"""
Per-cell chrM count loader for the partition scoring system.

Two on-disk formats feed the same interface:
  - fragpileup (ccRCC): work/phase5/ccrcc/{donor}/results/{donor}_atac_fragpileup.tsv.gz
    columns cell_barcode, pos, cons_A/C/G/T (per-cell consensus base counts).
  - mgatk (De Rop):     work/phase5/mgatk/mgatk_{cond}/final/mgatk_{cond}.{A,C,G,T}.txt.gz
    each `pos,barcode,fwd,rev`; depth from mgatk_{cond}.coverage.txt.gz (`pos,barcode,cov`);
    ref base from chrM_refAllele.txt.

Interface (both formats):
  load_bulk(sample_id)  -> DataFrame indexed by pos, columns A,C,G,T (summed over cells) + 'ref'
                           (ref known for mgatk; inferred as the modal base for fragpileup).
  load_cells(sample_id, positions)
                        -> (barcodes: list[str],
                            base: {b: (n_cells x n_pos) int array} for b in ACGT,
                            depth: (n_cells x n_pos) int array,
                            positions: list[int])

Barcodes are returned verbatim; the scorer reconciles them to a PartitionResult's labels
(stripping any 'cell-' prefix, as subpool.py:40 does).
"""
import os
import numpy as np
import pandas as pd

ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASES = ["A", "C", "G", "T"]

# sample_id -> source descriptor. chrM_slice / WGS truth are recorded where present so the
# scorer's precision side can find them; loader itself only needs type + path.
SAMPLES = {
    "ccRCC_RCC112": {"type": "fragpileup", "path": f"{ROOT}/work/phase5/ccrcc/RCC112/results/RCC112_atac_fragpileup.tsv.gz",
                     "wgs": f"{ROOT}/work/phase5/ccrcc/RCC112/results/RCC112_wgs.pileup"},
    "ccRCC_RCC106": {"type": "fragpileup", "path": f"{ROOT}/work/phase5/ccrcc/RCC106/results/RCC106_atac_fragpileup.tsv.gz"},
    "ccRCC_RCC86":  {"type": "fragpileup", "path": f"{ROOT}/work/phase5/ccrcc/RCC86/results/RCC86_atac_fragpileup.tsv.gz"},
    "derop_standard": {"type": "mgatk", "dir": f"{ROOT}/work/phase5/mgatk/mgatk_standard/final", "prefix": "mgatk_standard"},
    "derop_enriched": {"type": "mgatk", "dir": f"{ROOT}/work/phase5/mgatk/mgatk_enriched/final", "prefix": "mgatk_enriched"},
    # mtDOGMA: deep multiome (per-cell chrM ~2194) but NO WGS truth -> site-null nomination (deep
    # enough to have power). The decisive replication test: real complementarity on a deep sample?
    "mtdogma_PBMC": {"type": "fragpileup", "path": f"{ROOT}/work/phase5/mtdogma/mtdogma_arc_fragpileup.tsv.gz"},
    # --- Phase 8 cell-level-ground-truth panel (survey high-mtDNA-coverage slices; full-depth
    # pileups at work/phase8/celltruth/pileup/). Tiers per the approved plan. ---
    "glioma_4021":  {"type": "fragpileup", "path": f"{ROOT}/work/phase8/celltruth/pileup/glioma_4021_fragpileup.tsv.gz"},   # T1 human glioma (primary)
    "glioma_2937":  {"type": "fragpileup", "path": f"{ROOT}/work/phase8/celltruth/pileup/glioma_2937_fragpileup.tsv.gz"},   # T1
    "glioma_2932":  {"type": "fragpileup", "path": f"{ROOT}/work/phase8/celltruth/pileup/glioma_2932_fragpileup.tsv.gz"},   # T1 (control cohort)
    "mlung_kp_3019": {"type": "fragpileup", "path": f"{ROOT}/work/phase8/celltruth/pileup/mlung_kp_3019_fragpileup.tsv.gz"}, # T2 mouse lung tumor
    "mlung_kp_3017": {"type": "fragpileup", "path": f"{ROOT}/work/phase8/celltruth/pileup/mlung_kp_3017_fragpileup.tsv.gz"}, # T2
    "bmmc_s4d8":    {"type": "fragpileup", "path": f"{ROOT}/work/phase8/celltruth/pileup/bmmc_s4d8_fragpileup.tsv.gz"},     # T3 human BMMC (donor pos.control)
    "bmmc_s1d3":    {"type": "fragpileup", "path": f"{ROOT}/work/phase8/celltruth/pileup/bmmc_s1d3_fragpileup.tsv.gz"},     # T3
    "tcell_cpep1":  {"type": "fragpileup", "path": f"{ROOT}/work/phase8/celltruth/pileup/tcell_cpep1_fragpileup.tsv.gz"},   # T3 human CD8 T (knee-cut)
    "mhsc_lthsc":   {"type": "fragpileup", "path": f"{ROOT}/work/phase8/celltruth/pileup/mhsc_lthsc_fragpileup.tsv.gz"},    # T4 mouse HSC (homogeneous ctrl)
    "cochlea_p1wt": {"type": "fragpileup", "path": f"{ROOT}/work/phase8/celltruth/pileup/cochlea_p1wt_fragpileup.tsv.gz"},  # T4 mouse cochlea
}

# truth sources for truth-anchored candidate nomination: the KNOWN heteroplasmic positions
# (WGS/bulk). Recovery is then measured over real variants, not site-null-nominated positions.
TRUTH = {
    "ccRCC_RCC112": {"type": "reh_sites", "path": f"{ROOT}/work/phase5/ccrcc/RCC112/results/reh_sites.csv"},
    "ccRCC_RCC106": {"type": "reh_sites", "path": f"{ROOT}/work/phase5/ccrcc/RCC106/results/reh_sites.csv"},
    "ccRCC_RCC86":  {"type": "reh_sites", "path": f"{ROOT}/work/phase5/ccrcc/RCC86/results/reh_sites.csv"},
    "derop_standard": {"type": "derop_truth", "path": f"{ROOT}/work/phase4/derop/derop_enr_truth.csv"},
    "derop_enriched": {"type": "derop_truth", "path": f"{ROOT}/work/phase4/derop/derop_enr_truth.csv"},
}


def load_truth_positions(sample_id, vaf_lo=0.01, vaf_hi=0.95):
    """Known heteroplasmic chrM positions for a sample (or None if no truth on disk).

    reh_sites: per-donor WGS validation table, keep wgs_het==True.
    derop_truth: bulk WGS VAF table, keep intermediate VAF (real heteroplasmy).
    """
    t = TRUTH.get(sample_id)
    if t is None:
        return None
    df = pd.read_csv(t["path"])
    if t["type"] == "reh_sites":
        het = df[df["wgs_het"].astype(bool)]
        return sorted(int(p) for p in het["pos"])
    if t["type"] == "derop_truth":
        sel = df[(df["wgs_vaf"] > vaf_lo) & (df["wgs_vaf"] < vaf_hi)]
        return sorted(int(p) for p in sel["pos"])
    return None


_FRAG_COLS = ["cell_barcode", "pos", "cons_A", "cons_C", "cons_G", "cons_T"]


def _frag_iter(path, positions=None):
    """Yield fragpileup chunks (optionally filtered to `positions`)."""
    pset = set(int(p) for p in positions) if positions is not None else None
    for ch in pd.read_csv(path, sep="\t", usecols=_FRAG_COLS, chunksize=2_000_000):
        yield ch[ch.pos.isin(pset)] if pset is not None else ch


def _load_bulk_fragpileup(path):
    acc = None
    for ch in _frag_iter(path):
        g = ch.groupby("pos")[["cons_A", "cons_C", "cons_G", "cons_T"]].sum()
        acc = g if acc is None else acc.add(g, fill_value=0)
    acc.columns = BASES
    acc["ref"] = acc[BASES].idxmax(axis=1)  # fragpileup has no ref file: modal base = ref
    return acc.astype({b: "int64" for b in BASES})


def _load_cells_fragpileup(path, positions):
    positions = [int(p) for p in positions]
    rows = [ch for ch in _frag_iter(path, positions) if len(ch)]
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=_FRAG_COLS)
    barcodes = sorted(df.cell_barcode.unique().tolist())
    bidx = {b: i for i, b in enumerate(barcodes)}
    pidx = {p: j for j, p in enumerate(positions)}
    nC, nP = len(barcodes), len(positions)
    # one row per (cell, pos) -> (bi, pj) pairs unique -> direct vectorized assignment
    bi = df.cell_barcode.map(bidx).to_numpy()
    pj = df.pos.astype(int).map(pidx).to_numpy()
    base = {b: np.zeros((nC, nP), dtype="int64") for b in BASES}
    for b in BASES:
        base[b][bi, pj] = df[f"cons_{b}"].to_numpy()
    depth = sum(base[b] for b in BASES)
    return barcodes, base, depth, positions


def _mgatk_paths(desc):
    d, pre = desc["dir"], desc["prefix"]
    return ({b: f"{d}/{pre}.{b}.txt.gz" for b in BASES},
            f"{d}/{pre}.coverage.txt.gz", f"{d}/chrM_refAllele.txt")


def _load_bulk_mgatk(desc):
    base_paths, cov_path, ref_path = _mgatk_paths(desc)
    out = pd.DataFrame()
    for b in BASES:
        df = pd.read_csv(base_paths[b], header=None, names=["pos", "bc", "fwd", "rev"])
        out[b] = df.assign(c=df.fwd + df.rev).groupby("pos")["c"].sum()
    out = out.fillna(0).astype("int64")
    ref = pd.read_csv(ref_path, sep="\t", header=None, names=["pos", "ref"]).set_index("pos")["ref"]
    out["ref"] = ref.reindex(out.index)
    return out


def _load_cells_mgatk(desc, positions):
    positions = [int(p) for p in positions]
    pset = set(positions)
    base_paths, cov_path, _ = _mgatk_paths(desc)
    # collect barcodes appearing at the candidate positions (union across coverage)
    cov = pd.read_csv(cov_path, header=None, names=["pos", "bc", "depth"])
    cov = cov[cov.pos.isin(pset)]
    barcodes = sorted(cov.bc.unique().tolist())
    bidx = {b: i for i, b in enumerate(barcodes)}
    pidx = {p: j for j, p in enumerate(positions)}
    nC, nP = len(barcodes), len(positions)
    base = {b: np.zeros((nC, nP), dtype="int64") for b in BASES}
    for b in BASES:
        df = pd.read_csv(base_paths[b], header=None, names=["pos", "bc", "fwd", "rev"])
        df = df[df.pos.isin(pset) & df.bc.isin(bidx)]
        base[b][df.bc.map(bidx).to_numpy(), df.pos.astype(int).map(pidx).to_numpy()] = (df.fwd + df.rev).to_numpy()
    depth = np.zeros((nC, nP), dtype="int64")
    depth[cov.bc.map(bidx).to_numpy(), cov.pos.astype(int).map(pidx).to_numpy()] = cov["depth"].to_numpy()
    return barcodes, base, depth, positions


def load_bulk(sample_id):
    d = SAMPLES[sample_id]
    return _load_bulk_fragpileup(d["path"]) if d["type"] == "fragpileup" else _load_bulk_mgatk(d)


def load_cells(sample_id, positions):
    d = SAMPLES[sample_id]
    return (_load_cells_fragpileup(d["path"], positions) if d["type"] == "fragpileup"
            else _load_cells_mgatk(d, positions))


if __name__ == "__main__":
    import sys
    sid = sys.argv[1] if len(sys.argv) > 1 else "derop_standard"
    bulk = load_bulk(sid)
    # nominate a few positions with the most non-ref support, just to exercise load_cells
    nonref = bulk.apply(lambda r: r[[b for b in BASES if b != r["ref"]]].max(), axis=1)
    top = nonref.sort_values(ascending=False).head(5).index.tolist()
    bc, base, depth, pos = load_cells(sid, top)
    print(f"[{sid}] bulk positions={len(bulk)}; top nonref pos={top}")
    print(f"  cells loaded={len(bc)}; depth matrix={depth.shape}; "
          f"median depth at cand={np.median(depth[depth > 0]) if (depth > 0).any() else 0:.0f}")

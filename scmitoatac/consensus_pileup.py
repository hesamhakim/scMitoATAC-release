#!/usr/bin/env python
"""
scMitoHet 2.1 — UMI-family consensus base collapse.

For each candidate chrM position, pile up reads, group by (CB, UB) family, and
emit TWO per-(cell, position) molecule-level base-count matrices from the SAME
substrate so the consensus effect is isolated cleanly:

  * NAIVE molecule counts  : one base per family, taken from a single sampled
                             read in the family -> retains per-read RT/seq error.
  * CONSENSUS molecule cnts : family majority vote over all reads -> corrects
                             per-read error when family size >= 2.

Both matrices share the same DP (number of molecules covering the site); the
ONLY difference is within-family error correction. Per-molecule consensus
quality is assigned from the learned residual-by-family-size model
(k>=4 effectively error-free).

MAPQ handling: 10x/STARsolo assign MAPQ 255 to unique mappers; we ACCEPT them
(min-MAPQ default 0). Duplicates are KEPT (they are the family members consensus
needs) -- we do not skip is_duplicate.

Output: long-format parquet, one row per (cell, pos) that has >=1 molecule:
  cell_barcode, pos, ref_base, n_reads, n_mol,
  cons_A, cons_C, cons_G, cons_T, cons_N,
  naive_A, naive_C, naive_G, naive_T, naive_N,
  mean_family_size, mean_cons_qual
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np
import pysam

BASES = "ACGT"
BIDX = {b: i for i, b in enumerate(BASES)}


def load_family_qual_model(params_path):
    """Return dict family_size -> phred consensus quality from learned residuals."""
    with open(params_path) as fh:
        p = json.load(fh)
    resid = p.get("consensus_residual_by_family_size", {})
    single_err = float(p.get("implied_single_read_error_rate", 4.6e-4))
    qual = {}
    for k, v in resid.items():
        r = float(v.get("modeled_majority_residual", 0.0))
        r = max(r, 1e-12)
        qual[int(k)] = min(60.0, -10.0 * np.log10(r))
    # family size 1 -> single-read error floor
    qual[1] = min(60.0, -10.0 * np.log10(max(single_err, 1e-12)))
    return qual, single_err


def consensus_of_family(base_quals):
    """base_quals: list of (base_char, qual). Return (cons_base, naive_base, n_reads).
    Consensus = majority vote (ties broken by summed quality). Naive = a single
    deterministically-sampled read's base (first, order preserved from BAM)."""
    counts = np.zeros(4)
    qsum = np.zeros(4)
    naive_base = None
    for b, q in base_quals:
        i = BIDX.get(b)
        if i is None:
            continue
        counts[i] += 1
        qsum[i] += q
        if naive_base is None:
            naive_base = b
    if counts.sum() == 0:
        return None, None, 0
    top = np.max(counts)
    winners = np.where(counts == top)[0]
    if len(winners) == 1:
        cons_i = winners[0]
    else:  # tie -> highest summed quality
        cons_i = winners[np.argmax(qsum[winners])]
    return BASES[cons_i], naive_base, int(counts.sum())


def pileup_positions(bam_path, positions, contig, barcodes_set,
                     qual_model, min_bq=13, min_mapq=0, cb_tag="CB", ub_tag="UB",
                     max_depth=8000000):
    bam = pysam.AlignmentFile(bam_path, "rb")
    rows = []
    trunc_hits = []
    spos = sorted(positions)
    for pi, pos1 in enumerate(spos):  # 1-based
        pos0 = pos1 - 1
        n_read_visits = 0
        # family[(cb,ub)] -> list of (base, qual)
        fam = defaultdict(list)
        for col in bam.pileup(contig, pos0, pos0 + 1, truncate=True,
                              min_base_quality=0, min_mapping_quality=min_mapq,
                              stepper="nofilter", max_depth=max_depth,
                              ignore_overlaps=False, flag_filter=0):
            if col.reference_pos != pos0:
                continue
            n_read_visits = col.nsegments
            for pr in col.pileups:
                if pr.is_del or pr.is_refskip or pr.query_position is None:
                    continue
                r = pr.alignment
                if r.is_unmapped or r.is_secondary or r.is_supplementary:
                    continue
                if r.mapping_quality < min_mapq:
                    continue
                q = r.query_qualities[pr.query_position]
                if q < min_bq:
                    continue
                if not (r.has_tag(cb_tag) and r.has_tag(ub_tag)):
                    continue
                cb = r.get_tag(cb_tag)
                if barcodes_set is not None and cb not in barcodes_set:
                    continue
                ub = r.get_tag(ub_tag)
                b = r.query_sequence[pr.query_position]
                fam[(cb, ub)].append((b, q))
        # collapse families per cell
        cell = defaultdict(lambda: {"cons": np.zeros(5), "naive": np.zeros(5),
                                     "n_reads": 0, "n_mol": 0, "fsz": 0, "cq": 0.0})
        for (cb, ub), bq in fam.items():
            cons_b, naive_b, nr = consensus_of_family(bq)
            if cons_b is None:
                continue
            d = cell[cb]
            ci = BIDX.get(cons_b, 4)
            ni = BIDX.get(naive_b, 4)
            d["cons"][ci] += 1
            d["naive"][ni] += 1
            d["n_reads"] += nr
            d["n_mol"] += 1
            d["fsz"] += nr
            d["cq"] += qual_model.get(min(nr, max(qual_model)), qual_model.get(1))
        if n_read_visits >= max_depth:
            trunc_hits.append((pos1, n_read_visits))
        if (pi + 1) % 50 == 0 or pi == len(spos) - 1:
            sys.stderr.write(f"[consensus_pileup] {pi+1}/{len(spos)} positions done "
                             f"(pos {pos1}, {n_read_visits} read-visits)\n")
            sys.stderr.flush()
        for cb, d in cell.items():
            nm = d["n_mol"]
            rows.append((cb, pos1, d["n_reads"], nm,
                         int(d["cons"][0]), int(d["cons"][1]), int(d["cons"][2]),
                         int(d["cons"][3]), int(d["cons"][4]),
                         int(d["naive"][0]), int(d["naive"][1]), int(d["naive"][2]),
                         int(d["naive"][3]), int(d["naive"][4]),
                         d["fsz"] / nm if nm else 0.0,
                         d["cq"] / nm if nm else 0.0))
    bam.close()
    if trunc_hits:
        sys.stderr.write(f"[consensus_pileup] WARNING: {len(trunc_hits)} positions hit "
                         f"max_depth cap {max_depth}: {trunc_hits[:10]}\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bam", required=True)
    ap.add_argument("--positions", required=True, help="CSV with a 'pos' column (1-based)")
    ap.add_argument("--barcodes", required=True, help="one CB per line")
    ap.add_argument("--consensus-params", required=True)
    ap.add_argument("--contig", default="MT")
    ap.add_argument("--min-bq", type=int, default=13)
    ap.add_argument("--min-mapq", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=8000000,
                    help="pysam pileup max_depth; must exceed the deepest position's read count")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-chunks", type=int, default=1,
                    help="split positions into this many interleaved chunks (job-array parallelism)")
    ap.add_argument("--chunk", type=int, default=0, help="0-based chunk index to process")
    args = ap.parse_args()

    import pandas as pd
    positions = sorted(pd.read_csv(args.positions)["pos"].astype(int).tolist())
    if args.n_chunks > 1:
        # interleave so each chunk gets a mix of peak/mid/trough (balanced load)
        positions = positions[args.chunk::args.n_chunks]
    with open(args.barcodes) as fh:
        bcs = set(l.strip() for l in fh if l.strip())
    qual_model, single_err = load_family_qual_model(args.consensus_params)
    sys.stderr.write(f"[consensus_pileup] {len(positions)} positions, {len(bcs)} barcodes, "
                     f"single_err={single_err:.2e}, qual_model_sizes={sorted(qual_model)[:8]}...\n")

    rows = pileup_positions(args.bam, positions, args.contig, bcs, qual_model,
                            min_bq=args.min_bq, min_mapq=args.min_mapq,
                            max_depth=args.max_depth)
    cols = ["cell_barcode", "pos", "n_reads", "n_mol",
            "cons_A", "cons_C", "cons_G", "cons_T", "cons_N",
            "naive_A", "naive_C", "naive_G", "naive_T", "naive_N",
            "mean_family_size", "mean_cons_qual"]
    df = pd.DataFrame(rows, columns=cols)
    if args.out.endswith(".parquet"):
        df.to_parquet(args.out, index=False)
    else:
        df.to_csv(args.out, index=False, compression="gzip" if args.out.endswith(".gz") else None)
    sys.stderr.write(f"[consensus_pileup] wrote {len(df)} (cell,pos) rows -> {args.out}\n")
    # quick summary
    agg = df.groupby("pos").agg(n_cells=("cell_barcode", "nunique"),
                                tot_mol=("n_mol", "sum"),
                                tot_reads=("n_reads", "sum")).describe()
    sys.stderr.write(str(agg) + "\n")


if __name__ == "__main__":
    main()

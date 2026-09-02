#!/usr/bin/env python
"""
scMitoATAC Phase 4.1/4.2/4.3 — REH real-data validation against WGS truth.

REH is a clonal B-ALL cell line: the SAME biosample (SAMN13831871) has 10x scATAC
and Illumina PCR-free WGS. The WGS mitogenome is the ground truth. On a clonal line
the truth is (near-)homoplasmic at nearly all sites, so this is primarily a
SPECIFICITY test:

  4.1 specificity — where WGS says homoplasmic-reference (truth VAF ~ 0), the ATAC
      caller must NOT license a heteroplasmy call (after calibration + abstention).
  4.2 VAF concordance — at sites where WGS shows a real minor allele, the ATAC pooled
      VAF should track the WGS VAF.
  4.3 artifact-site abstention on real truth — at the Phase-2.4 panel positions, the
      calibrated abstention policy must hold: retained calls meet the precision target
      or abstain; WGS-confirmed variants outside the panel must survive.

Inputs:
  --atac-pileup   fragment-level consensus pileup over chrM (from fragment_pileup.py on
                  the REH ATAC BAM), cols include pos, cell_barcode, n_mol, cons_alt...
  --wgs-pileup    samtools mpileup of REH WGS over chrM (pos, ref, dp, bases, quals)
  --panel         work/phase1/results/phase2_atac_artifact_panel.csv (chrM_pos ...)
  --floor         Phase-3.2 error_floor.json (measured consensus/naive floor)
Outputs: reh_validation_summary.json, reh_sites.csv
"""
import argparse, json, os, sys, re
import numpy as np
import pandas as pd

BASES = "ACGTN"
CONTROL = set(range(1, 577)) | set(range(16024, 16570))  # exclude control region


def parse_wgs_pileup(path):
    """samtools mpileup -> per-pos {ref, dp, alt_counts, vaf(minor-allele frac)}."""
    rows = []
    with open(path) as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            pos = int(f[1]); ref = f[2].upper(); dp = int(f[3]); reads = f[4]
            # strip mapping start/end markers and indels
            reads = re.sub(r"\^.", "", reads).replace("$", "")
            reads = re.sub(r"[+-]\d+[ACGTNacgtn]+", "", reads)
            counts = {b: 0 for b in "ACGT"}
            i = 0
            while i < len(reads):
                ch = reads[i]
                if ch in ".,":
                    if ref in counts:
                        counts[ref] += 1
                elif ch.upper() in counts:
                    counts[ch.upper()] += 1
                i += 1
            tot = sum(counts.values())
            alt = tot - counts.get(ref, 0)
            vaf = alt / tot if tot else 0.0
            rows.append({"pos": pos, "ref": ref, "wgs_dp": tot,
                         "wgs_alt": alt, "wgs_vaf": vaf})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atac-pileup", required=True)
    ap.add_argument("--wgs-pileup", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--floor", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--het-truth-thresh", type=float, default=0.02,
                    help="WGS VAF above which a site is a TRUE heteroplasmy")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    wgs = parse_wgs_pileup(a.wgs_pileup)
    panel = pd.read_csv(a.panel)
    panel_pos = set(int(p) for p in panel["chrM_pos"].values)
    floor = json.load(open(a.floor))
    cons_floor = floor.get("cons", {}).get("measured_floor", 2.32e-4)

    # ATAC fragment pileup -> pooled per-position naive & consensus VAF
    # fragment_pileup writes TAB-delimited (.tsv.gz); fall back to comma for plain .csv
    sep = "\t" if ".tsv" in a.atac_pileup else ","
    ap_df = pd.read_csv(a.atac_pileup, sep=sep,
                        compression="gzip" if a.atac_pileup.endswith(".gz") else None)
    # detect column style (cons_A.. naive_A.. or aggregated)
    cons_cols = [f"cons_{b}" for b in BASES if f"cons_{b}" in ap_df.columns]
    naive_cols = [f"naive_{b}" for b in BASES if f"naive_{b}" in ap_df.columns]
    # need the reference base per position from WGS
    refmap = dict(zip(wgs["pos"], wgs["ref"]))
    grp = ap_df.groupby("pos")
    rows = []
    for pos, sub in grp:
        ref = refmap.get(int(pos))
        if ref is None:
            continue
        cons_tot = sub[cons_cols].sum().sum()
        naive_tot = sub[naive_cols].sum().sum()
        cons_ref = sub[f"cons_{ref}"].sum() if f"cons_{ref}" in sub else 0
        naive_ref = sub[f"naive_{ref}"].sum() if f"naive_{ref}" in sub else 0
        cons_vaf = (cons_tot - cons_ref) / cons_tot if cons_tot else 0.0
        naive_vaf = (naive_tot - naive_ref) / naive_tot if naive_tot else 0.0
        rows.append({"pos": int(pos), "ref": ref, "atac_cons_dp": int(cons_tot),
                     "atac_cons_vaf": float(cons_vaf), "atac_naive_vaf": float(naive_vaf),
                     "n_cells": int(sub["cell_barcode"].nunique())})
    atac = pd.DataFrame(rows)

    m = atac.merge(wgs[["pos", "wgs_dp", "wgs_alt", "wgs_vaf"]], on="pos", how="inner")
    m["is_panel"] = m["pos"].isin(panel_pos)
    m["is_control"] = m["pos"].isin(CONTROL)
    m["wgs_het"] = m["wgs_vaf"] >= a.het_truth_thresh
    core = m[~m["is_control"]].copy()

    # 4.1 specificity: where WGS is homoplasmic-ref (not het), does ATAC consensus stay near floor?
    homo = core[~core["wgs_het"]]
    het = core[core["wgs_het"]]
    spec = {
        "n_homoplasmic_sites": int(len(homo)),
        "atac_cons_vaf_median_at_homoplasmic": float(homo["atac_cons_vaf"].median()),
        "atac_cons_vaf_p95_at_homoplasmic": float(homo["atac_cons_vaf"].quantile(0.95)),
        "cons_floor_measured": cons_floor,
        # false-positive sites: ATAC consensus VAF > 1% where WGS homoplasmic
        "n_atac_fp_gt1pct": int((homo["atac_cons_vaf"] > 0.01).sum()),
        "fp_sites_are_panel_frac": float(homo.loc[homo["atac_cons_vaf"] > 0.01, "is_panel"].mean())
            if (homo["atac_cons_vaf"] > 0.01).any() else 0.0,
    }
    # 4.2 concordance at real het sites
    conc = {"n_het_sites": int(len(het))}
    if len(het) >= 2:
        conc["pearson_atac_wgs_vaf"] = float(np.corrcoef(het["atac_cons_vaf"], het["wgs_vaf"])[0, 1])
    conc["het_sites"] = het[["pos", "ref", "wgs_vaf", "atac_cons_vaf", "is_panel"]].to_dict("records")
    # 4.3 artifact-site behavior: panel sites vs clean sites, ATAC consensus VAF where WGS homoplasmic
    panel_homo = homo[homo["is_panel"]]; clean_homo = homo[~homo["is_panel"]]
    art = {
        "n_panel_homoplasmic": int(len(panel_homo)),
        "atac_cons_vaf_median_panel": float(panel_homo["atac_cons_vaf"].median()) if len(panel_homo) else None,
        "atac_cons_vaf_median_clean": float(clean_homo["atac_cons_vaf"].median()) if len(clean_homo) else None,
        "panel_over_clean_ratio": (float(panel_homo["atac_cons_vaf"].median() /
                                         max(clean_homo["atac_cons_vaf"].median(), 1e-9))
                                   if len(panel_homo) and len(clean_homo) else None),
    }
    out = {"specificity_4_1": spec, "concordance_4_2": conc, "artifact_4_3": art,
           "n_sites_compared": int(len(core)), "het_truth_thresh": a.het_truth_thresh}
    json.dump(out, open(os.path.join(a.outdir, "reh_validation_summary.json"), "w"),
              indent=2, default=float)
    m.to_csv(os.path.join(a.outdir, "reh_sites.csv"), index=False)
    print(json.dumps(out, indent=1, default=float)[:2000])


if __name__ == "__main__":
    main()
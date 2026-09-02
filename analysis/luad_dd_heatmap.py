#!/usr/bin/env python
"""
Variant x cluster mean-METACELL-VAF heatmap per sample, one panel per lens that carries candidates. Built directly
from the already-computed metacell summaries (work/luad_dd/<tok>/mc_summary_<tok>_<axis>.tsv) -- no recompute, and
strictly metacell resolution. Columns annotated with cell type + CNV/tumor burden (single-cell annotations from
clusters_annot). Shows the multi-variant enrichment pattern in the malignant subclone / epithelial cells.
Usage: luad_dd_heatmap.py [tok ...]   (default: all 5 LUAD samples that have candidates)
Out: docs/reports/figures/luad_meeting/heatmap_<tok>.png
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pubstyle as PS; PS.apply()
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DD = f"{ROOT}/work/luad_dd"; FIG = f"{ROOT}/figures"
VAFCMAP = LinearSegmentedColormap.from_list("vaf", ["#f7f7f7", "#f4a582", "#b2182b"])
LENSES = ["cnvSubclone", "cellType", "tumorNormal"]           # panel order (malignant-subclone first)
ALLSAMP = ["LUAD_AL05", "LUAD_AL08", "LUAD_AL10", "LUAD_AL11", "LUAD_AL12"]


def region_map():
    r = {}
    for tok in ALLSAMP:
        f = f"{DD}/{tok}/mc_calib_{tok}.tsv"
        if os.path.exists(f):
            d = pd.read_csv(f, sep="\t")
            for t in d.itertuples():
                r[int(t.pos)] = str(t.region).split("/")[0]
    return r


def main():
    os.makedirs(FIG, exist_ok=True)
    toks = sys.argv[1:] or ALLSAMP
    cand = pd.read_csv(f"{DD}/luad_dd_candidates.tsv", sep="\t")
    cand = cand[cand.call == "candidate"]
    reg = region_map()
    for tok in toks:
        cc = cand[cand['sample'] == tok]
        cpos = sorted(cc.pos.unique(), key=lambda p: -cc[cc.pos == p].T_conc.max())   # candidate variants, by effect
        if not cpos:
            print(f"[{tok}] no candidates; skip heatmap"); continue
        ann = pd.read_csv(f"{DD}/{tok}/clusters_annot_{tok}.tsv", sep="\t") if os.path.exists(f"{DD}/{tok}/clusters_annot_{tok}.tsv") else pd.DataFrame()
        panels = []
        for lens in LENSES:
            f = f"{DD}/{tok}/mc_summary_{tok}_{lens}.tsv"
            if not os.path.exists(f):
                continue
            s = pd.read_csv(f, sep="\t")
            s = s[s.pos.isin(cpos) & s.cluster.astype(str).ne("")]
            if not len(s):
                continue
            M = s.pivot_table(index="pos", columns="cluster", values="mean_mcvaf", aggfunc="first").reindex(cpos)
            panels.append((lens, M))
        if not panels:
            print(f"[{tok}] no metacell summaries; skip"); continue
        fw = min(PS.COL - 0.22, 2.3 + 1.55 * len(panels))   # headroom for the larger tick labels under tight bbox
        wr = [max(4.0, M.shape[1]) for _, M in panels]     # give each lens width ∝ its cluster count (floor keeps 2-col panels legible)
        fig, axs = plt.subplots(1, len(panels), figsize=(fw, 0.36 * len(cpos) + 1.7), squeeze=False,
                                gridspec_kw={"width_ratios": wr})
        for j, (lens, M) in enumerate(panels):
            ax = axs[0][j]
            im = ax.imshow(M.values, aspect="auto", cmap=VAFCMAP, vmin=0, vmax=max(0.4, np.nanmax(M.values)))
            ax.set_xticks(range(M.shape[1]))
            # column labels: for cellType the cluster IS the cell type (label alone, rotated); for CNV/tumorNormal
            # show cluster + cell type + aneuploidy burden (marks the malignant column).
            labs = []
            for c in M.columns:
                a = ann[(ann.axis == lens) & (ann.cluster.astype(str) == str(c))] if len(ann) else pd.DataFrame()
                ct = a.top_celltype.iloc[0] if len(a) else ""
                tf = a.tumor_frac.iloc[0] if len(a) and pd.notna(a.tumor_frac.iloc[0]) else None
                if lens == "cellType":
                    labs.append(str(c))                                            # cluster == cell type
                elif lens == "tumorNormal":
                    labs.append({"normal_diploid": "normal", "tumor_aneuploid": "tumor"}.get(str(c), str(c)))
                else:
                    # short code c<k>·b<tumor-fraction> (cell type dropped; key given in the caption)
                    labs.append(f"{str(c).replace('cnvcnv', 'c')}·b{tf:.2f}" if tf is not None
                                else str(c).replace('cnvcnv', 'c'))
            rot = 0 if lens == "tumorNormal" else 90
            ax.set_xticklabels(labs, fontsize=7.5, rotation=rot, ha="center")
            ax.set_yticks(range(len(cpos)))
            if j == 0:
                ax.set_yticklabels([f"m.{p} {reg.get(p,'')}" for p in cpos], fontsize=7.5)
            else:
                ax.set_yticklabels([])
            # heatmaps communicate localization by color alone (no printed cell values)
            ax.set_title(lens)     # lens name is the panel identifier (cnvSubclone / cellType / tumorNormal)
        fig.colorbar(im, ax=axs[0].tolist(), fraction=0.03, pad=0.02, label="mean metacell VAF")
        fig.suptitle(f"{tok.replace('LUAD_','')} — candidate mtDNA variant × cluster (metacell VAF)")
        PS.save(fig, f"{FIG}/heatmap_{tok}.png"); plt.close(fig)
        print(f"wrote heatmap_{tok}.png ({len(cpos)} variants × {len(panels)} lenses)")


if __name__ == "__main__":
    main()

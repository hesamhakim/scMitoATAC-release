#!/usr/bin/env python
"""
Dimension-reduction (LSI -> t-SNE) of a LUAD sample's cells, with the mtDNA shift overlaid at METACELL resolution.
Cells are embedded from a genome-wide 10 kb ATAC tile matrix (built from the fragments file; umap-learn absent so we
use sklearn t-SNE, the ped_hgg recipe). Cell-level colorings are legitimate single-cell annotations from OTHER
algorithms: CNV compartment, aneuploidy burden, gene-activity cell type. Heteroplasmy is shown ONLY as metacell
points (LOD-sized pools within each compartment), coloured by pooled metacell VAF -- never a per-cell VAF.

Usage: luad_dd_embed.py <tok> [--track human --max-cells 15000 --bin 10000 --pos <variant> --seed 0]
  --pos defaults to the sample's top malignant tumor-gain candidate.
Out: work/luad_dd/<tok>/embed/{lsi,tsne}_<tok>.npy + cells.txt ; docs/reports/figures/luad_meeting/embed_<tok>.png
"""
import os, sys, argparse, gzip
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pubstyle as PS; PS.apply()
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import metacell as MC
import sample_chrm as SC

ROOT = MC.ROOT; BASES = MC.BASES
FIG = f"{ROOT}/figures"
VAFCMAP = LinearSegmentedColormap.from_list("vaf", ["#f7f7f7", "#f4a582", "#b2182b"])
COMPc = {"normal_diploid": PS.BLUE, "tumor_aneuploid": PS.RED}
MIN_MC_DEPTH = 10           # minimum pooled depth for a metacell to be shown (VAF resolution ~0.1)


def cnv_burden(tok, track):
    for base in (f"{ROOT}/work/phase16_cancer_scan/{track}/partitions/{tok}",):
        rt = f"{base}/epiAneufinder_results/results_table.tsv"
        if os.path.exists(rt) and os.path.getsize(rt) > 0:
            df = pd.read_csv(rt, sep=r"\s+"); cells = [c for c in df.columns if c not in ("seq", "start", "end")]
            b = (df[cells] != 1).mean(axis=0); b.index = [c.replace("cell-", "") for c in b.index]
            return b.astype(float)
    return pd.Series(dtype=float)


def build_tiles(frag, cellset, cell_i, BIN):
    tile_i = {}; rows = []; cols = []
    for ch in pd.read_csv(frag, sep="\t", header=None, usecols=[0, 1, 3],
                          names=["chrom", "start", "bc"], chunksize=5_000_000, comment="#"):
        ch = ch[ch.bc.isin(cellset)]
        if not len(ch):
            continue
        ch = ch[ch.chrom.astype(str).str.match(r"^(chr)?([0-9]{1,2}|X)$")]
        tid = ch.chrom.astype(str) + ":" + (ch.start.to_numpy() // BIN).astype(str)
        for bc, t in zip(ch.bc.to_numpy(), tid.to_numpy()):
            j = tile_i.get(t)
            if j is None:
                j = len(tile_i); tile_i[t] = j
            rows.append(cell_i[bc]); cols.append(j)
    X = sp.coo_matrix((np.ones(len(rows), "float32"), (rows, cols)),
                      shape=(len(cell_i), len(tile_i))).tocsr()
    X.data[:] = 1.0
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tok"); ap.add_argument("--track", default="human"); ap.add_argument("--max-cells", type=int, default=15000)
    ap.add_argument("--bin", type=int, default=10000); ap.add_argument("--pos", type=int, default=None); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok = a.tok; rng = np.random.default_rng(a.seed)
    out = f"{ROOT}/work/luad_dd/{tok}/embed"; os.makedirs(out, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    PART = f"{ROOT}/work/phase16_cancer_scan/{a.track}/partitions/{tok}"
    tn = pd.read_csv(f"{PART}/part_{tok}_tumorNormal_k2.csv"); tn["barcode"] = tn.barcode.astype(str)

    # pick the target variant = top malignant tumor-gain candidate (or --pos)
    pos = a.pos
    if pos is None:
        c = pd.read_csv(f"{ROOT}/work/luad_dd/luad_dd_candidates.tsv", sep="\t")
        c = c[(c['sample'] == tok) & (c.call == "candidate") & (c.direction == "tumor-gain")
              & (c.is_malignant_cluster == True) & (~c.high_baseline)]
        pos = int(c.sort_values("T_conc", ascending=False).pos.iloc[0]) if len(c) else int(
            pd.read_csv(f"{ROOT}/work/luad_dd/{tok}/finalists_{tok}.tsv", sep="\t").pos.iloc[0])

    comp = tn.set_index("barcode").group.to_dict()          # full compartment map (before subsample)
    lsf, tsf, clf = f"{out}/lsi_{tok}.npy", f"{out}/tsne_{tok}.npy", f"{out}/cells.txt"
    if os.path.exists(lsf) and os.path.exists(tsf) and os.path.exists(clf):
        cells = [b for b in open(clf).read().split() if b]      # reuse cached embedding (skip heavy LSI+t-SNE)
        emb = np.load(lsf); xy = np.load(tsf)
        print(f"[{tok}] reuse cached embedding: {len(cells)} cells (target m.{pos})")
    else:
        if len(tn) > a.max_cells:                                # keep ALL tumor cells (the compartment of interest),
            tum = tn[tn.group == "tumor_aneuploid"]              # cap normal to fill the budget -> tumor is not diluted
            nor = tn[tn.group == "normal_diploid"]
            if len(tum) > a.max_cells:
                tum = tum.sample(n=a.max_cells, random_state=a.seed); nor = nor.iloc[:0]
            else:
                keep_nor = max(1, a.max_cells - len(tum))
                if len(nor) > keep_nor:
                    nor = nor.sample(n=keep_nor, random_state=a.seed)
            tn = pd.concat([tum, nor], ignore_index=True)
        cells = tn.barcode.tolist(); cell_i = {b: i for i, b in enumerate(cells)}; cellset = set(cells)
        print(f"[{tok}] embedding {len(cells)} cells (target m.{pos})")
        X = build_tiles(f"{ROOT}/work/phase16_cancer_scan/{a.track}/sources/{tok}.fragments.tsv.gz", cellset, cell_i, a.bin)
        cs = np.asarray((X > 0).sum(0)).ravel(); keep = (cs >= 5) & (cs <= 0.95 * len(cells))
        X = X[:, keep]
        print(f"[{tok}] tile matrix {X.shape} (>=5-cell tiles)")
        emb = TruncatedSVD(n_components=min(30, X.shape[1] - 1), random_state=0).fit_transform(
            TfidfTransformer().fit_transform(X))[:, 1:]
        xy = TSNE(n_components=2, init="pca", random_state=0, perplexity=30).fit_transform(emb)
        np.save(lsf, emb); np.save(tsf, xy); open(clf, "w").write("\n".join(cells) + "\n")

    # cell annotations (single-cell, from other algorithms -- legitimate)
    burden = cnv_burden(tok, a.track)
    ctf = f"{ROOT}/work/luad_dd/{tok}/part_{tok}_celltype.csv"
    ctmap = pd.read_csv(ctf).assign(barcode=lambda d: d.barcode.astype(str)).set_index("barcode").group.to_dict() if os.path.exists(ctf) else {}
    bvec = np.array([burden.get(b, np.nan) for b in cells]); cvec = [comp[b] for b in cells]
    ctypes = [ctmap.get(b, "n/a") for b in cells]

    # METACELL overlay: pool member cells' alt/depth at the variant (never per-cell VAF); centroid = mean t-SNE
    bc_p, base, depth, _ = SC._load_cells_fragpileup(MC.pileup_path(tok, a.track), [pos])
    pmap = {str(b): k for k, b in enumerate(bc_p)}
    tot = {b: int(base[b][:, 0].sum()) for b in BASES}; modal = max(tot, key=tot.get)
    def cell_ad(bc):                                    # (alt, depth) for a cell at the variant, 0 if absent
        k = pmap.get(bc)
        if k is None:
            return 0, 0
        d = int(depth[k, 0]); return d - int(base[modal][k, 0]), d
    dbar = np.mean([cell_ad(b)[1] for b in cells]) or 0.5
    mrows = []
    for g in ("normal_diploid", "tumor_aneuploid"):
        idx = np.array([i for i, b in enumerate(cells) if comp[b] == g])
        if len(idx) < 10:
            continue
        k, *_ = MC.size_metacell(len(idx), dbar); k = max(k, 10)
        # SPATIALLY-coherent metacells: KMeans in LSI space within the compartment (so centroids spread across the
        # embedding and show the VAF gradient), instead of random pools whose centroids collapse to the middle.
        ng = max(2, len(idx) // k)
        lab = KMeans(n_clusters=ng, n_init=3, random_state=0).fit_predict(emb[idx])
        for gi in range(ng):
            grp = idx[lab == gi]
            if len(grp) < max(5, k // 2):
                continue
            ad = [cell_ad(cells[i]) for i in grp]; alt = sum(x for x, _ in ad); dep = sum(y for _, y in ad)
            if dep < MIN_MC_DEPTH:                      # a metacell VAF needs enough reads (LOD spirit); drop noisy low-depth pools
                continue
            mrows.append(dict(x=xy[grp, 0].mean(), y=xy[grp, 1].mean(), vaf=alt / dep, n=len(grp), comp=g))
    mc = pd.DataFrame(mrows)

    # ---- 2x2 figure ----
    fig, axs = plt.subplots(2, 2, figsize=(PS.COL, 6.6)); (a1, a2), (a3, a4) = axs
    for g in ("normal_diploid", "tumor_aneuploid"):
        m = [i for i in range(len(cells)) if cvec[i] == g]
        a1.scatter(xy[m, 0], xy[m, 1], s=2, alpha=0.4, c=COMPc[g], edgecolors="none", label=g)
    a1.legend(frameon=False, markerscale=4, loc="upper right"); a1.set_title("CNV compartment (single-cell)")
    sc = a2.scatter(xy[:, 0], xy[:, 1], s=2, alpha=0.5, c=bvec, cmap="viridis"); a2.set_title("aneuploidy burden (single-cell)")
    fig.colorbar(sc, ax=a2, fraction=0.045, pad=0.02, label="CNV burden")
    cats = [c for c in dict.fromkeys(ctypes) if c != "n/a"]     # order-preserving unique (pandas3-safe)
    cmap = plt.get_cmap("tab20", max(len(cats), 1))
    for k, ct in enumerate(cats):
        m = [i for i in range(len(cells)) if ctypes[i] == ct]
        a3.scatter(xy[m, 0], xy[m, 1], s=2, alpha=0.5, color=cmap(k), edgecolors="none", label=ct)
    a3.legend(fontsize=7.5, frameon=False, markerscale=5, ncol=2, loc="upper right"); a3.set_title("ATAC gene-activity cell type (single-cell)")
    # METACELL-ONLY heteroplasmy view: one point per metacell (no cell background, so cells and metacell VAF are
    # never mixed in a single panel). Same coordinate frame as the single-cell annotation panels above.
    if len(mc):
        s4 = a4.scatter(mc.x, mc.y, s=24 + 70 * mc.n / mc.n.max(), c=mc.vaf, cmap=VAFCMAP, vmin=0,
                        vmax=0.5, edgecolors="k", linewidths=0.4)
        fig.colorbar(s4, ax=a4, fraction=0.045, pad=0.02, label="metacell VAF")
    a4.set_xlim(xy[:, 0].min(), xy[:, 0].max()); a4.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    reg = ""
    try:
        rr = pd.read_csv(f"{ROOT}/work/luad_dd/{tok}/mc_calib_{tok}.tsv", sep="\t"); reg = str(rr[rr.pos == pos].region.iloc[0]).split("/")[0]
    except Exception:
        pass
    a4.set_title(f"m.{pos} {reg} — metacell VAF (size∝n)")
    for lbl, ax in zip("abcd", (a1, a2, a3, a4)):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        PS.panel(ax, lbl, x=-0.04, y=1.01)
    fig.suptitle(f"{tok.replace('LUAD_','')} — ATAC embedding ({len(cells)} cells); heteroplasmy at metacell resolution")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); PS.save(fig, f"{FIG}/embed_{tok}.png"); plt.close(fig)
    print(f"[{tok}] wrote embed_{tok}.png  ({len(mc)} metacells overlaid; dbar={dbar:.2f})")


if __name__ == "__main__":
    main()

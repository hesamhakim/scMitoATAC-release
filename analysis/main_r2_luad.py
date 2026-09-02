#!/usr/bin/env python
"""Main Figure 3 for the scMitoATAC manuscript: LUAD -- heteroplasmy shifts localise to the malignant compartment.

Composite, publication quality, authored at final print size (<=6.9in wide, 300 DPI, tight bbox). Six panels port
data/logic (numbers unchanged) from the existing LUAD generators; only the visual design is re-worked to fit one page:
  (a) cohort effect-vs-significance volcano across 5 tumors x all lenses   [port: luad_dd_volcano]
  (b) AL10 variant x cluster metacell-VAF heatmap, CNV-subclone + cell type [port: luad_dd_heatmap, drop tumorNormal]
  (c) AL10 ATAC embedding: CNV compartment + clone-variant metacell VAF      [port: luad_dd_embed, 2 sub-views]
  (d) per-metacell VAF for m.9973 by cell type (malignant/epithelial ~2x)    [port: luad_dd_figs.strip, one variant]
  (e) surviving malignant gains are orthogonal to ALK status                 [port: luad_dd_figs.cohort_alk]
  (f) the abstention that makes it credible: downsampling leaves the confound + scMitoMut returns 0 carriers
                                                                             [port: luad_covctrl_rebuttal_fig(b) + scmitomut_compare(b)]
Reads only precomputed work/ products; recomputes nothing but the (cached) embedding overlay.
Out: docs/manuscript/figures/main/main_r2_luad.png
general env: numpy, pandas, scipy, sklearn, matplotlib.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pubstyle as PS; PS.apply()
import metacell as MC
import sample_chrm as SC

ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DD = f"{ROOT}/work/luad_dd"; SM = f"{ROOT}/work/scmitomut_hth"
OUT = f"{ROOT}/figures/main_r2_luad.png"
BLUE, RED, GREEN, GREY, INK, FILL = PS.BLUE, PS.RED, PS.GREEN, PS.GREY, PS.INK, PS.FILL
BASES = MC.BASES
VAFCMAP = LinearSegmentedColormap.from_list("vaf", ["#f7f7f7", "#f4a582", "#b2182b"])
TOK = "LUAD_AL10"; PFLOOR = 1.0 / 1001.0
HB = "#6a3d9a"                       # high-baseline (germline-seg) candidate colour (from luad_dd_volcano)
OXPHOS_TRIO = {9973, 6510, 14607}    # the AL10 localizing trio; only these heatmap rows get in-cell numbers

# five-char cell-type abbreviations so nothing clips in the dense panels
CT_ABBR = {"Epithelial": "Epi", "AT2": "AT2", "Ciliated": "Cil", "Club": "Club", "Endothelial": "Endo",
           "Fibroblast": "Fibr", "Myeloid": "Mye", "NK": "NK", "Plasma": "Plas", "Tcell": "Tc", "Bcell": "Bc"}
# the surviving calls to annotate on the volcano (task-specified): AL10 OXPHOS trio + AL11/AL08 MT-ND5/MT-CYB
SURV = [("LUAD_AL10", 9973, "AL10 m.9973 CO3"), ("LUAD_AL10", 6510, "AL10 m.6510 CO1"),
        ("LUAD_AL10", 14607, "AL10 m.14607 ND6"), ("LUAD_AL11", 13145, "AL11 m.13145 ND5"),
        ("LUAD_AL08", 15059, "AL08 m.15059 CYB")]


# ============================================================ data loaders (numbers unchanged) ==
def load_volcano():
    s = pd.read_csv(f"{DD}/stats_fdr.tsv", sep="\t")
    c = pd.read_csv(f"{DD}/luad_dd_candidates.tsv", sep="\t")
    c = c[c.call == "candidate"][["sample", "axis", "pos", "direction", "is_malignant_cluster", "high_baseline"]]
    c = c.rename(columns={"axis": "method"})
    m = s.merge(c, on=["sample", "method", "pos"], how="left")
    m["is_cand"] = m.direction.notna()
    return m


def cand_colour(r):
    if not r.is_cand:
        return GREY
    if bool(r.high_baseline):
        return HB
    if r.direction == "tumor-gain" and bool(r.is_malignant_cluster):
        return RED
    return BLUE


def region_map():
    r = {}
    for tok in ["LUAD_AL05", "LUAD_AL08", "LUAD_AL10", "LUAD_AL11", "LUAD_AL12"]:
        f = f"{DD}/{tok}/mc_calib_{tok}.tsv"
        if os.path.exists(f):
            d = pd.read_csv(f, sep="\t")
            for t in d.itertuples():
                r[int(t.pos)] = str(t.region).split("/")[0]
    return r


def embed_overlay(pos=9973):
    """Reuse the cached LSI+t-SNE embedding; recompute the metacell VAF overlay at m.9973 (luad_dd_embed logic)."""
    emb = f"{DD}/{TOK}/embed"
    cells = [b for b in open(f"{emb}/cells.txt").read().split() if b]
    lsi = np.load(f"{emb}/lsi_{TOK}.npy"); xy = np.load(f"{emb}/tsne_{TOK}.npy")
    tn = pd.read_csv(f"{ROOT}/work/phase16_cancer_scan/human/partitions/{TOK}/part_{TOK}_tumorNormal_k2.csv")
    comp = {str(b): g for b, g in zip(tn.barcode.astype(str), tn.group)}
    cvec = [comp.get(b, "normal_diploid") for b in cells]

    bc_p, base, depth, _ = SC._load_cells_fragpileup(MC.pileup_path(TOK, "human"), [pos])
    pmap = {str(b): k for k, b in enumerate(bc_p)}
    tot = {b: int(base[b][:, 0].sum()) for b in BASES}; modal = max(tot, key=tot.get)

    def cell_ad(bc):
        k = pmap.get(bc)
        if k is None:
            return 0, 0
        d = int(depth[k, 0]); return d - int(base[modal][k, 0]), d

    dbar = np.mean([cell_ad(b)[1] for b in cells]) or 0.5
    mrows = []
    for g in ("normal_diploid", "tumor_aneuploid"):
        idx = np.array([i for i, b in enumerate(cells) if comp.get(b) == g])
        if len(idx) < 10:
            continue
        k, *_ = MC.size_metacell(len(idx), dbar); k = max(k, 10)
        ng = max(2, len(idx) // k)
        lab = KMeans(n_clusters=ng, n_init=3, random_state=0).fit_predict(lsi[idx])
        for gi in range(ng):
            grp = idx[lab == gi]
            if len(grp) < max(5, k // 2):
                continue
            ad = [cell_ad(cells[i]) for i in grp]; alt = sum(a for a, _ in ad); dep = sum(d for _, d in ad)
            if dep < 10:
                continue
            mrows.append(dict(x=xy[grp, 0].mean(), y=xy[grp, 1].mean(), vaf=alt / dep, n=len(grp), comp=g))
    return xy, cvec, pd.DataFrame(mrows)


def rarefaction_bars():
    """luad_covctrl_rebuttal_fig panel (b): original vs downsampled rho( VAF , depth ) on the same malignant metacells."""
    CONS = ["cons_A", "cons_C", "cons_G", "cons_T"]; MALIG = "tumor_aneuploid"; POS = 9973
    rng = np.random.default_rng(0)
    sc = pd.read_csv(f"{DD}/{TOK}/{TOK}_cellscores.tsv.gz", sep="\t")
    sc["compartment"] = sc["compartment"].fillna("unknown").astype(str)
    pcs = [c for c in sc.columns if c.startswith("PC")]
    sc["mc"] = KMeans(n_clusters=max(2, round(len(sc) / 40)), n_init=4, random_state=0).fit_predict(sc[pcs].to_numpy())
    bc2mc = dict(zip(sc.barcode, sc.mc))
    mfrac = sc.groupby("mc").compartment.agg(lambda s: (s == MALIG).mean())
    malig = set(mfrac[mfrac >= 0.5].index)
    acc = {}
    for chk in pd.read_csv(f"{ROOT}/work/phase16_cancer_scan/human/mtdna/{TOK}/{TOK}_decoy_fragpileup.tsv.gz",
                           sep="\t", usecols=["cell_barcode", "pos"] + CONS, chunksize=2_000_000):
        chk = chk[(chk.pos == POS) & chk.cell_barcode.isin(bc2mc)]
        if not len(chk):
            continue
        chk["mc"] = chk.cell_barcode.map(bc2mc)
        for mc, sub in chk.groupby("mc"):
            acc[mc] = acc.get(mc, np.zeros(4)) + sub[CONS].to_numpy().sum(axis=0)
    rows = [(m, c.sum(), 1 - c.max() / c.sum(), c) for m, c in acc.items() if m in malig and c.sum() > 0]
    depth = np.array([r[1] for r in rows]); vaf = np.array([r[2] for r in rows]); counts = [r[3] for r in rows]
    Ds = [30, 40]; orig_r, rare_r, ns = [], [], []
    for D in Ds:
        idx = np.where(depth >= D)[0]
        vr = np.array([np.mean(1 - rng.multinomial(D, counts[i] / counts[i].sum(), size=40).max(1) / D) for i in idx])
        orig_r.append(spearmanr(vaf[idx], depth[idx])[0]); rare_r.append(spearmanr(vr, depth[idx])[0]); ns.append(len(idx))
    return Ds, orig_r, rare_r, ns


def scmitomut_funnel():
    inp = pd.read_csv(f"{SM}/input_{TOK}.tsv", sep="\t").merge(pd.read_csv(f"{SM}/comp_{TOK}.tsv", sep="\t"),
                                                              on="cell_barcode", how="left")
    s = inp[inp["loc"] == 9973]; mal = s[s.compartment == "tumor_aneuploid"]
    sm = pd.read_csv(f"{SM}/scmitomut_summary_{TOK}.tsv", sep="\t")
    sm["pos"] = sm["loc"].astype(str).str.replace("chrM.", "", regex=False).astype(int)
    n_cov = len(mal); n_alt2 = int((mal.alt >= 2).sum()); n_carr = int(sm[sm.pos == 9973].n_carrier_q05.iloc[0])
    return n_cov, n_alt2, n_carr


# ============================================================ panels ==
def panel_a(ax, m):
    rng = np.random.default_rng(0)
    y = -np.log10(np.clip(m.p_clone.values, PFLOOR, 1)) + rng.normal(0, 0.03, len(m))
    nc = ~m.is_cand.values
    # non-candidate cloud -> light hexbin density (task: drop the grey cloud to a low-alpha density)
    ax.hexbin(m.T_conc.values[nc], y[nc], gridsize=(34, 13), cmap="Greys", mincnt=1,
              alpha=0.55, linewidths=0, zorder=1)
    # candidate positions on top, coloured by class
    cols = np.array([cand_colour(r) for r in m.itertuples()], dtype=object)
    cm = m.is_cand.values
    ax.scatter(m.T_conc.values[cm], y[cm], s=26, c=list(cols[cm]), edgecolors="k",
               linewidths=0.4, alpha=0.95, zorder=3)
    ax.axhline(-np.log10(PFLOOR), ls=":", color="#999", lw=0.8, zorder=2)
    ax.text(ax.get_xlim()[1], -np.log10(PFLOOR), " p-floor", va="bottom", ha="right", fontsize=6, color="#888")
    ax.axvline(0.03, ls=":", color="#ccc", lw=0.8, zorder=2)
    # annotate ONLY the surviving calls, fanned below the p-floor with connectors
    for i, (samp, pos, lab) in enumerate(SURV):
        d = m[(m['sample'] == samp) & (m.pos == pos) & m.is_cand & (m.direction == "tumor-gain")
              & (m.is_malignant_cluster == True)]
        if not len(d):
            d = m[(m['sample'] == samp) & (m.pos == pos) & m.is_cand]
        if not len(d):
            continue
        r = d.sort_values("T_conc", ascending=False).iloc[0]
        ax.annotate(lab, (r.T_conc, -np.log10(max(r.p_clone, PFLOOR))),
                    fontsize=6.0, xytext=(10, -12 - 11 * i), textcoords="offset points", ha="left",
                    arrowprops=dict(arrowstyle="-", color=RED, lw=0.5), zorder=4)
    ax.set_xlabel("effect size  T$_{conc}$  (hot − pooled VAF)"); ax.set_ylabel("−log$_{10}$ raw p")
    ax.set_title("Effect size versus significance (5 tumors × all lenses)", fontsize=8)
    h = [Line2D([0], [0], marker='o', ls='', mfc=RED, mec='k', mew=0.3, label='malignant gain'),
         Line2D([0], [0], marker='o', ls='', mfc=BLUE, mec='k', mew=0.3, label='other cand.'),
         Line2D([0], [0], marker='o', ls='', mfc=HB, mec='k', mew=0.3, label='germline-seg')]
    ax.legend(handles=h, fontsize=5.6, loc="lower right", handletextpad=0.2, borderpad=0.2)


def panel_b(axcnv, axct, cax, reg):
    cand = pd.read_csv(f"{DD}/luad_dd_candidates.tsv", sep="\t"); cand = cand[cand.call == "candidate"]
    cc = cand[cand['sample'] == TOK]
    cpos = sorted(cc.pos.unique(), key=lambda p: -cc[cc.pos == p].T_conc.max())
    ann = pd.read_csv(f"{DD}/{TOK}/clusters_annot_{TOK}.tsv", sep="\t")
    im = None
    for ax, lens, title in [(axcnv, "cnvSubclone", "CNV subclone"), (axct, "cellType", "cell type")]:
        s = pd.read_csv(f"{DD}/{TOK}/mc_summary_{TOK}_{lens}.tsv", sep="\t")
        s = s[s.pos.isin(cpos) & s.cluster.astype(str).ne("")]
        M = s.pivot_table(index="pos", columns="cluster", values="mean_mcvaf", aggfunc="first").reindex(cpos)
        im = ax.imshow(M.values, aspect="auto", cmap=VAFCMAP, vmin=0, vmax=0.42)
        ax.set_xticks(range(M.shape[1]))
        labs = []
        for c in M.columns:
            if lens == "cellType":
                labs.append(CT_ABBR.get(str(c), str(c)[:4]))
            else:                                     # cnvSubclone: cluster id + malignant-burden flag
                a = ann[(ann.axis == lens) & (ann.cluster.astype(str) == str(c))]
                tf = a.tumor_frac.iloc[0] if len(a) and pd.notna(a.tumor_frac.iloc[0]) else None
                labs.append(f"{str(c).replace('cnvcnv','c')}·b{tf:.2f}" if tf is not None else str(c))
        ax.set_xticklabels(labs, fontsize=6.4, rotation=90)
        ax.set_yticks(range(len(cpos)))
        if lens == "cnvSubclone":
            ax.set_yticklabels([f"m.{p} {reg.get(p, '').replace('MT-', '')}" for p in cpos], fontsize=6.5)
            for lab, p in zip(ax.get_yticklabels(), cpos):     # bold the key localizing OXPHOS rows
                if p in OXPHOS_TRIO:
                    lab.set_fontweight("bold")
        else:
            ax.set_yticklabels([])
        # heatmap communicates localization by color; the key OXPHOS rows carry a thin outline
        # instead of printed cell values
        for yi, p in enumerate(cpos):
            if p in OXPHOS_TRIO:
                ax.add_patch(Rectangle((-0.5, yi - 0.5), M.shape[1], 1, fill=False,
                                       edgecolor=INK, lw=1.1, zorder=5))
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.tick_params(length=2)
    cb = plt.colorbar(im, cax=cax); cb.set_label("mean VAF", fontsize=6.5); cb.ax.tick_params(labelsize=6)


def panel_c(axvaf, cax, xy, mc):
    """Single embedding: metacell m.9973 VAF over a faint tissue backdrop. The CNV-compartment
    view (normal vs tumor) is shown in the fuller embedding panel of Supplementary Figure S14."""
    axvaf.scatter(xy[:, 0], xy[:, 1], s=1.2, c="#dcdcdc", edgecolors="none", zorder=1)  # tissue backdrop
    if len(mc):
        s4 = axvaf.scatter(mc.x, mc.y, s=10 + 34 * mc.n / mc.n.max(), c=mc.vaf, cmap=VAFCMAP,
                           vmin=0, vmax=0.5, edgecolors="k", linewidths=0.3, zorder=3)
        cb = plt.colorbar(s4, cax=cax); cb.set_label("VAF", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    axvaf.set_xlim(xy[:, 0].min(), xy[:, 0].max()); axvaf.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    axvaf.set_title("m.9973 VAF on the ATAC embedding (size ∝ n)", fontsize=6.8, pad=3)
    axvaf.set_xticks([]); axvaf.set_yticks([])
    for sp in axvaf.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6)


def panel_d(ax, pos=9973, reg=""):
    vec = pd.read_csv(f"{DD}/{TOK}/mc_vec_{TOK}_{pos}_cellType.tsv", sep="\t")
    stats = [(str(g), s.vaf.to_numpy()) for g, s in vec.groupby("group") if len(s)]
    stats.sort(key=lambda kv: -kv[1].mean()); top = stats[0][0]
    rng = np.random.default_rng(0)
    for i, (name, v) in enumerate(stats):
        col = RED if name == top else "#4d4d4d"
        ax.scatter(rng.normal(i, 0.08, len(v)), v, s=8, alpha=0.5, color=col, edgecolors="none")
        ax.plot([i - 0.34, i + 0.34], [v.mean()] * 2, color=col, lw=2.0)
        ax.text(i, 0.505, f"{v.mean():.2f}", ha="center", va="bottom", fontsize=5.6, color=col)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels([CT_ABBR.get(n, n[:5]) for n, _ in stats], rotation=40, ha="right", fontsize=7.0)
    ax.set_ylim(0, 0.55); ax.axhline(0.05, ls=":", color="#aaa", lw=0.8)
    ax.set_ylabel("metacell VAF", fontsize=7.5)
    ax.set_title(f"m.9973 {reg} metacell VAF by cell type", fontsize=7.5, pad=12)
    # descriptor sits just above the axis (below the title), clear of the in-axis mean-number row
    ax.text(0.5, 1.015, "dot = metacell · bar = cluster mean · dotted = 5% LOD", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=5.6, color="#555")


def panel_e(ax, reg):
    cand = pd.read_csv(f"{DD}/luad_dd_candidates.tsv", sep="\t"); cand = cand[cand.call == "candidate"]
    cov = pd.read_csv(f"{DD}/luad_dd_covnull.tsv", sep="\t")
    # surviving = licensed by the coverage-stratified null on any lens (non germline-seg); malignancy is
    # judged on the candidate side (covnull's is_malignant keys off CNV clusters, missing cell-type-licensed AL08)
    surv = set(map(tuple, cov[(cov.licensed) & (~cov.high_baseline)][["sample", "pos"]].values))
    g = cand[(cand.direction == "tumor-gain") & (cand.is_malignant_cluster == True) & (~cand.high_baseline)]
    g = g[[(s, p) in surv for s, p in zip(g['sample'], g.pos)]]
    g = g.sort_values("T_conc", ascending=False).drop_duplicates(["sample", "pos"])
    g = g.sort_values("T_conc", ascending=True)
    ALKc = {"ALK+": RED, "WT": BLUE}
    y = np.arange(len(g))
    labels = [f"{r.sample.replace('LUAD_','')} m.{r.pos} {CT_ABBR.get(str(r.region).split('/')[0], str(r.region).split('/')[0].replace('MT-',''))}"
              for r in g.itertuples()]
    ax.barh(y, g.T_conc, color=[ALKc.get(a, GREY) for a in g.alk], edgecolor="white", height=0.68)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.0)
    ax.set_xlim(0, g.T_conc.max() * 1.18)
    ax.set_xlabel("cluster-pooled Δ VAF (T$_{conc}$)", fontsize=7.5)
    ax.set_title("Retained gains versus ALK status", fontsize=8)
    ax.legend(handles=[Patch(fc=ALKc["ALK+"], label="ALK+"), Patch(fc=ALKc["WT"], label="WT")],
              fontsize=6.2, loc="lower right", handlelength=1.0, borderpad=0.25)
    ax.tick_params(length=2)


def panel_f(axr, axfun, Ds, orig_r, rare_r, ns, funnel):
    # f-i: rarefaction leaves the confound intact
    x = np.arange(len(Ds)); w = 0.36
    axr.bar(x - w / 2, orig_r, w, color=GREY, label="original", edgecolor="white")
    axr.bar(x + w / 2, rare_r, w, color=RED, label="downsampled", edgecolor="white")
    axr.axhline(0, color=INK, lw=0.8)
    for xi, (o, r, n) in enumerate(zip(orig_r, rare_r, ns)):
        axr.text(xi, max(o, r) + 0.03, f"n={n}", ha="center", va="bottom", fontsize=5.4, color=INK)
    axr.set_xticks(x); axr.set_xticklabels([f"D={d}" for d in Ds]); axr.set_ylim(0, 0.72)
    axr.set_ylabel("ρ( VAF , depth )", fontsize=7.0)
    axr.set_title("Downsampling\nleaves it intact", fontsize=6.8, pad=3)
    axr.legend(fontsize=5.4, loc="upper right", handlelength=1.0, handletextpad=0.3, borderpad=0.2)
    # f-ii: per-cell presence caller funnel -> 0 carriers
    n_cov, n_alt2, n_carr = funnel
    bars = [n_cov, n_alt2, n_carr]
    axfun.bar([0, 1, 2], bars, color=[GREY, GREY, RED], edgecolor="white", width=0.62)
    for i, v in enumerate(bars):
        axfun.text(i, v + max(bars) * 0.02, f"{v}", ha="center", va="bottom", fontsize=6.0)
    axfun.set_xticks([0, 1, 2])
    axfun.set_xticklabels(["malig.", "≥2 alt", "carrier"], fontsize=6.0)
    axfun.set_ylim(0, max(bars) * 1.22); axfun.set_ylabel("cells", fontsize=7.0)
    axfun.set_title("scMitoMut per-cell:\n0 carriers", fontsize=6.8, pad=3)
    axfun.tick_params(length=2)


# ============================================================ compose ==
def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    reg = region_map()
    mv = load_volcano()
    xy, cvec, mc = embed_overlay(9973)
    # four-panel figure: a volcano | b heatmap (top), c single embedding | d metacell VAF (bottom).
    # ALK panel removed (its story is Supplementary Figure S15); one embedding kept (the other is in S14).
    fig = plt.figure(figsize=(6.9, 5.45))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.9, 1.55], hspace=0.52, wspace=0.42,
                          left=0.10, right=0.90, top=0.885, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    # panel b: two heatmaps + shared colorbar
    gb = gs[0, 1].subgridspec(1, 3, width_ratios=[4, 11, 0.55], wspace=0.12)
    ax_bcnv = fig.add_subplot(gb[0, 0]); ax_bct = fig.add_subplot(gb[0, 1]); cax_b = fig.add_subplot(gb[0, 2])
    # panel c: single VAF embedding + colorbar
    gc = gs[1, 0].subgridspec(1, 2, width_ratios=[1, 0.06], wspace=0.1)
    ax_cvaf = fig.add_subplot(gc[0, 0]); cax_c = fig.add_subplot(gc[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a(ax_a, mv)
    panel_b(ax_bcnv, ax_bct, cax_b, reg)
    panel_c(ax_cvaf, cax_c, xy, mc)
    panel_d(ax_d, 9973, reg.get(9973, ""))

    PS.panel(ax_a, "a", x=-0.15, y=1.10, fontsize=10)
    PS.panel(ax_bcnv, "b", x=-0.45, y=1.10, fontsize=10)
    PS.panel(ax_cvaf, "c", x=-0.08, y=1.06, fontsize=10)
    PS.panel(ax_d, "d", x=-0.15, y=1.14, fontsize=10)

    if not PS.GB_SUB:   # GB: the overall title lives in the figure legend, not the graphic
        fig.suptitle("LUAD: heteroplasmy shifts localise to the malignant compartment (5 tumors)",
                     fontsize=9, fontweight="bold", y=0.985)
    PS.save(fig, OUT); plt.close(fig)
    print("wrote", OUT)
    print(f"c: {len(mc)} metacells overlaid; 4-panel LUAD (ALK -> S15, one embedding -> S14)")


if __name__ == "__main__":
    main()

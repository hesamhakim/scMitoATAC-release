#!/usr/bin/env python
"""Publication figures + summary tables for the ped-HGG m.9438 deep-dive (2932, 2937)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle as PS
PS.apply()

ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DD = f"{ROOT}/work/phase18/pedhgg_deepdive"
FIG = f"{DD}/figures"; os.makedirs(FIG, exist_ok=True)
P16 = f"{ROOT}/work/phase16_cancer_scan/human"
TOKS = {"2932": "primary pGBM", "2937": "recurrent pGBM"}
CARRIER = 0.60
STATE_ORDER = ["OPC-like", "NPC-like", "AC-like", "MES-like", "Cycling", "Oligodendrocyte",
               "Microglia/Macro", "T-cell", "Endothelial", "Pericyte", "Neuron"]
STATE_COL = dict(zip(STATE_ORDER, plt.cm.tab20(np.linspace(0, 1, len(STATE_ORDER)))))


def cnv_burden(tok):
    rt = f"{P16}/partitions/{tok}/epiAneufinder_results/results_table.tsv"
    df = pd.read_csv(rt, sep=r"\s+")
    cc = [c for c in df.columns if c.startswith("cell-")]
    b = (df[cc].to_numpy() != 1).mean(axis=0)
    return pd.DataFrame({"barcode": [c.replace("cell-", "") for c in cc], "burden": b})


def load(tok):
    d = {}
    d["ts"] = np.load(f"{DD}/{tok}/tsne.npy")
    d["cl"] = pd.read_csv(f"{DD}/{tok}/clusters_vaf.tsv", sep="\t")
    d["ps"] = pd.read_csv(f"{DD}/{tok}/percell_state.tsv", sep="\t")
    d["en"] = pd.read_csv(f"{DD}/{tok}/enrichment.tsv", sep="\t")
    d["cc"] = pd.read_csv(f"{DD}/{tok}/concordance.tsv", sep="\t", index_col=0)
    d["burden"] = cnv_burden(tok)
    d["cnv4"] = pd.read_csv(f"{P16}/partitions/{tok}/part_{tok}_cnv_k4.csv")
    return d


def fig_overview(tok, d):
    ts, cl, ps = d["ts"], d["cl"], d["ps"]
    ok = cl.depth.to_numpy() >= 3
    vaf = cl.vaf.to_numpy()
    fig, ax = plt.subplots(2, 3, figsize=(6.5, 4.2))
    fig.suptitle(f"ped-HGG {tok} ({TOKS[tok]}) — m.9438 MT-CO3 A>G carrier subpopulation", y=1.0)
    # (a) TSNE by per-cell VAF
    a = ax[0, 0]
    a.scatter(ts[~ok, 0], ts[~ok, 1], s=6, c="#dddddd", label="depth<3")
    sc = a.scatter(ts[ok, 0], ts[ok, 1], s=10, c=vaf[ok], cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(sc, ax=a, label="m.9438 VAF"); a.set_title("per-cell m.9438 VAF"); a.set_xticks([]); a.set_yticks([])
    # (b) TSNE by clustering (KMeans_k8)
    a = ax[0, 1]
    lab = cl["KMeans_k8"].astype(str).to_numpy()
    for i, g in enumerate(sorted(set(lab))):
        m = lab == g
        a.scatter(ts[m, 0], ts[m, 1], s=8, color=plt.cm.tab10(i % 10), label=g)
    a.set_title("clusters (KMeans k=8)"); a.set_xticks([]); a.set_yticks([]); a.legend(fontsize=6, markerscale=1.5, ncol=2)
    # (c) TSNE by dominant cell-state
    a = ax[0, 2]
    st = cl[["barcode"]].merge(ps[["barcode", "dominant_state"]], on="barcode", how="left")["dominant_state"].to_numpy()
    for s in STATE_ORDER:
        m = st == s
        if m.any():
            a.scatter(ts[m, 0], ts[m, 1], s=8, color=STATE_COL[s], label=s)
    a.scatter(ts[pd.isna(st), 0], ts[pd.isna(st), 1], s=6, c="#eeeeee")
    a.set_title("gene-activity cell state"); a.set_xticks([]); a.set_yticks([]); a.legend(fontsize=5.5, markerscale=1.5, ncol=2)
    # (d) per-cell VAF histogram (carrier cluster vs rest, by CNV carrier)
    a = ax[1, 0]
    car = d["carrier_atac_mask"]
    a.hist(vaf[ok & car], bins=25, range=(0, 1), alpha=0.7, label=f"carrier cluster (n={int((ok&car).sum())})", color="#c0392b")
    a.hist(vaf[ok & ~car], bins=25, range=(0, 1), alpha=0.6, label=f"rest (n={int((ok&~car).sum())})", color="#2980b9")
    a.axvline(CARRIER, ls="--", c="k", lw=1); a.set_yscale("log"); a.set_xlabel("m.9438 VAF"); a.set_ylabel("cells")
    a.set_title("carrier cluster vs rest"); a.legend(fontsize=6)
    # (e) per-cluster mean VAF barplot (KMeans_k8)
    a = ax[1, 1]
    en = d["en"]; sub = en[en.method == "KMeans_k8"].copy().sort_values("mean_vaf", ascending=False, na_position="last")
    colors = ["#c0392b" if x else "#95a5a6" for x in sub.is_carrier_cluster]
    a.bar(sub.cluster.astype(str), sub.mean_vaf.fillna(0), color=colors)
    a.set_ylabel("mean m.9438 VAF"); a.set_xlabel("KMeans k=8 cluster"); a.set_title("per-cluster VAF")
    # (f) aneuploidy burden per CNV(k4) cluster + carrier VAF
    a = ax[1, 2]
    m = d["burden"].merge(d["cnv4"], on="barcode")
    bg = m.groupby("group").burden.mean()
    envaf = d["en"][d["en"].method == "epiAneuCNV_k4"].set_index("cluster")["mean_vaf"]
    carrier_clu = d["en"][(d["en"].method == "epiAneuCNV_k4") & d["en"].is_carrier_cluster].cluster.iloc[0]
    order = bg.index.tolist()
    colors = ["#c0392b" if g == carrier_clu else "#95a5a6" for g in order]
    a.bar([str(g) for g in order], [bg[g] for g in order], color=colors)
    a.set_ylabel("aneuploidy burden (frac bins != normal)"); a.set_xlabel("epiAneufinder CNV cluster")
    a.set_title("CNV burden per cluster")
    for i, g in enumerate(order):
        a.text(i, bg[g] + 0.005, f"VAF\n{envaf.get(g, np.nan):.2f}", ha="center", fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    PS.save(fig, f"{FIG}/overview_{tok}.png"); plt.close(fig)


def fig_crossmethod(tok, d):
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.3),
                           gridspec_kw={"width_ratios": [1.25, 1]})
    J = d["cc"].astype(float)
    im = ax[0].imshow(J.to_numpy(), cmap="magma", vmin=0, vmax=1)
    ax[0].set_xticks(range(len(J)))
    ax[0].set_xticklabels(J.columns, rotation=45, ha="right", rotation_mode="anchor")
    ax[0].set_yticks(range(len(J))); ax[0].set_yticklabels(J.index)
    ax[0].tick_params(labelsize=6, length=2)
    cb = plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03)
    cb.set_label("Jaccard (carrier-cell sets)"); cb.ax.tick_params(labelsize=6)
    ax[0].set_title("cross-method carrier-set concordance")
    PS.panel(ax[0], "a", x=-0.28)
    cap = d["capture"]
    ax[1].barh(cap.method, cap["capture"], color=PS.BLUE, label="capture (recall)")
    ax[1].barh(cap.method, cap["purity"], left=None, color="none",
               edgecolor=PS.INK, linewidth=0.6)
    ax[1].plot(cap["purity"], cap.method, "o", color=PS.RED, ms=4, label="purity (precision)")
    ax[1].set_xlabel("fraction"); ax[1].set_xlim(0, 1)
    ax[1].tick_params(labelsize=6, length=2)
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False,
                 handlelength=1.2, columnspacing=1.4)
    ax[1].set_title("capture / purity per method")
    PS.panel(ax[1], "b", x=-0.30)
    fig.tight_layout()
    PS.save(fig, f"{FIG}/crossmethod_{tok}.png"); plt.close(fig)


def capture_table(tok, d):
    cl = d["cl"]; ok = cl.depth >= 3; truecar = ok & (cl.vaf >= CARRIER)
    methods = [c for c in cl.columns if c not in ("barcode", "vaf", "depth")]
    rows = []; carrier_atac_mask = None
    for m in methods:
        lab = cl[m].astype(str)
        mv = {g: cl.loc[ok & (lab == g), "vaf"].mean() for g in lab.unique() if g not in ("NA", "nan")}
        mv = {g: v for g, v in mv.items() if v == v}; c = max(mv, key=mv.get)
        inclu = (lab == c) & ok
        rows.append(dict(method=m, carrier_cluster=c, n=int(inclu.sum()),
                         capture=round((inclu & truecar).sum() / max(1, truecar.sum()), 2),
                         purity=round((inclu & truecar).sum() / max(1, inclu.sum()), 2)))
        if m == "KMeans_k8":
            carrier_atac_mask = ((cl[m].astype(str) == c)).to_numpy()
    d["carrier_atac_mask"] = carrier_atac_mask
    d["capture"] = pd.DataFrame(rows)
    return d["capture"]


def fig_compare(D):
    fig, ax = plt.subplots(1, 3, figsize=(PS.COL, 2.5))
    # (a) VAF distributions (carrier cluster) both samples
    for tok, c in [("2932", PS.BLUE), ("2937", PS.RED)]:
        d = D[tok]; cl = d["cl"]; ok = cl.depth >= 3; car = d["carrier_atac_mask"]
        v = cl.vaf.to_numpy()[ok & car]
        ax[0].hist(v, bins=25, range=(0, 1), alpha=0.6, label=f"{tok} ({TOKS[tok]})", color=c)
    ax[0].set_xlabel("m.9438 VAF (carrier cluster)"); ax[0].set_ylabel("cells")
    ax[0].legend(); ax[0].set_title("carrier VAF distribution"); PS.panel(ax[0], "a", y=1.14)
    # (b) CNV burden: carrier vs non-carrier CNV cluster
    labels, cb, ncb = [], [], []
    for tok in TOKS:
        d = D[tok]; m = d["burden"].merge(d["cnv4"], on="barcode")
        cc = d["en"][(d["en"].method == "epiAneuCNV_k4") & d["en"].is_carrier_cluster].cluster.iloc[0]
        labels.append(tok); cb.append(m[m.group == cc].burden.mean()); ncb.append(m[m.group != cc].burden.mean())
    x = np.arange(len(labels)); w = 0.35
    ax[1].bar(x - w / 2, cb, w, label="carrier CNV subclone", color=PS.RED)
    ax[1].bar(x + w / 2, ncb, w, label="other CNV clusters", color=PS.GREY)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].set_ylabel("aneuploidy burden")
    ax[1].set_ylim(0, max(max(cb), max(ncb)) * 1.42)
    ax[1].legend(fontsize=6, loc="upper center", ncol=2, columnspacing=1.0, handletextpad=0.4)
    ax[1].set_title("carrier subclone is CNV-quiet"); PS.panel(ax[1], "b", y=1.14)
    # (c) malignant_frac carrier vs rest
    cm, nm = [], []
    for tok in TOKS:
        d = D[tok]; cs = pd.read_csv(f"{DD}/{tok}/cluster_state.tsv", sep="\t")
        sub = cs[cs.method == "epiAneuCNV_k4"]
        cc = d["en"][(d["en"].method == "epiAneuCNV_k4") & d["en"].is_carrier_cluster].cluster.iloc[0]
        cm.append(sub[sub.cluster == cc].malignant_frac.iloc[0]); nm.append(sub[sub.cluster != cc].malignant_frac.mean())
    ax[2].bar(x - w / 2, cm, w, label="carrier subclone", color=PS.RED)
    ax[2].bar(x + w / 2, nm, w, label="other clusters", color=PS.GREY)
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels); ax[2].set_ylabel("malignant gene-activity fraction")
    ax[2].set_ylim(0, max(max(cm), max(nm)) * 1.42)
    ax[2].legend(fontsize=6, loc="upper center", ncol=2, columnspacing=1.0, handletextpad=0.4)
    ax[2].set_title("lower malignant activity"); PS.panel(ax[2], "c", y=1.14)
    fig.tight_layout(); PS.save(fig, f"{FIG}/compare_2932_2937.png"); plt.close(fig)


def main():
    D = {}
    summ = []
    for tok in TOKS:
        d = load(tok)
        capture_table(tok, d)
        fig_overview(tok, d)
        fig_crossmethod(tok, d)
        D[tok] = d
        cl = d["cl"]; ok = cl.depth >= 3
        cc = d["en"][(d["en"].method == "epiAneuCNV_k4") & d["en"].is_carrier_cluster].cluster.iloc[0]
        m = d["burden"].merge(d["cnv4"], on="barcode")
        summ.append(dict(sample=tok, type=TOKS[tok], atac_cells=len(cl), covered=int(ok.sum()),
                         carriers=int((ok & (cl.vaf >= CARRIER)).sum()),
                         carrier_CNV_subclone=cc, carrier_burden=round(m[m.group == cc].burden.mean(), 3),
                         other_burden=round(m[m.group != cc].burden.mean(), 3),
                         mean_capture=round(d["capture"].capture.mean(), 2),
                         n_methods_significant=int(len(d["capture"]))))
    fig_compare(D)
    pd.DataFrame(summ).to_csv(f"{DD}/SUMMARY.tsv", sep="\t", index=False)
    print(pd.DataFrame(summ).to_string(index=False))
    print(f"\nfigures -> {FIG}/  (overview_{{2932,2937}}, crossmethod_{{2932,2937}}, compare_2932_2937)")


if __name__ == "__main__":
    main()

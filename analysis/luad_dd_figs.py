#!/usr/bin/env python
"""
LUAD deep-dive figures. (1) Per candidate sample: METACELL VAF distribution by cluster (strip + mean bar) for each
candidate position, on the tumor/normal and cell-type axes -> shows WHERE the shift concentrates and its within-
cluster spread. Metacell (not per-cell) is the resolution floor: single cells lack the chrM coverage for low
heteroplasmy. (2) Cohort: candidate effect (Delta VAF) per sample, colored by ALK status.
Reads work/luad_dd/luad_dd_candidates.tsv + mc_vec_*.tsv. Skips missing inputs gracefully.
Usage: luad_dd_figs.py
Out: docs/reports/figures/luad_dd/metacell_<tok>.png (per candidate sample) + cohort_alk.png
"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pubstyle as PS; PS.apply()
import matplotlib.pyplot as plt

ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DD = f"{ROOT}/work/luad_dd"; FIG = f"{ROOT}/figures"
HOT, COLD = PS.RED, "#4d4d4d"; ALKc = {"ALK+": PS.RED, "WT": PS.BLUE}


def strip(ax, vec, title=None):
    # one point per METACELL; highlight the highest-mean cluster (localization). Bar = mean metacell-VAF; annotate
    # mean + fraction of metacells with VAF>=0.2 (c20) + n metacells. title is set only on the top row (column header).
    stats = [(str(g), s.vaf.to_numpy()) for g, s in vec.groupby("group") if len(s)]
    if not stats:
        ax.set_axis_off(); ax.text(0.5, 0.5, "no metacells", ha="center", fontsize=8); return
    stats.sort(key=lambda kv: -kv[1].mean()); top = stats[0][0]
    rng = np.random.default_rng(0)
    for i, (name, v) in enumerate(stats):
        col = HOT if name == top else COLD
        ax.scatter(rng.normal(i, 0.07, len(v)), v, s=9, alpha=0.45, color=col, edgecolors="none")
        ax.plot([i - 0.32, i + 0.32], [v.mean()] * 2, color=col, lw=2.2)
        # few clusters (tumor/normal): full μ / c20 / n; many clusters (cell type): just the mean, to avoid overlap
        lab = (f"μ={v.mean():.2f} c20={(v >= 0.2).mean():.0%} n={len(v)}" if len(stats) <= 4 else f"{v.mean():.2f}")
        ax.text(i, 0.505, lab, ha="center", va="bottom", fontsize=6.6 if len(stats) <= 4 else 7.0, color=col)
    ax.set_xticks(range(len(stats))); ax.set_xticklabels([n[:12] for n, _ in stats], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylim(0, 0.5); ax.axhline(0.05, ls=":", color="#aaa", lw=0.8)          # VAF 0-50% (data max ~0.53)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)


def metacell_panels(tok, cand):
    # lead with clean low-baseline gains; high-baseline (germline-seg) rows sort last
    srt = (["high_baseline", "is_malignant_cluster", "T_conc"], [True, False, False]) if "high_baseline" in cand.columns else (["T_conc"], [False])
    sub = cand[cand['sample'] == tok].sort_values(srt[0], ascending=srt[1])
    poss = list(dict.fromkeys(sub.pos.tolist()))[:3]
    if not poss:
        return None
    axes_lbl = ["tumorNormal", "cellType"]
    colhdr = {"tumorNormal": "tumor vs normal", "cellType": "cell type"}   # one header per column (top row)
    fig, axs = plt.subplots(len(poss), 2, figsize=(PS.COL, 2.2 * len(poss)), squeeze=False)
    for i, pos in enumerate(poss):
        row = sub[sub.pos == pos].iloc[0]
        reg = str(row.region).split("/")[0] if pd.notna(row.region) else ""
        for jc, lbl in enumerate(axes_lbl):
            f = f"{DD}/{tok}/mc_vec_{tok}_{pos}_{lbl}.tsv"
            axc = axs[i][jc]
            if not os.path.exists(f):
                axc.set_axis_off(); axc.text(0.5, 0.5, f"{lbl}: n/a", ha="center", fontsize=8); continue
            strip(axc, pd.read_csv(f, sep="\t"), title=(colhdr[lbl] if i == 0 else None))  # column header on top row only
            if jc == 0:                                                   # variant row-label once, on the left
                axc.set_ylabel(f"m.{pos} {reg}\nmetacell VAF", fontsize=8.5)
    fig.suptitle(f"{tok.replace('LUAD_','')} ({cand[cand['sample']==tok].alk.iloc[0]}) — metacell mtDNA VAF by cluster "
                 f"(dot = metacell; red = top-mean cluster; bar = mean; dotted = 5% LOD)", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); p = f"{FIG}/metacell_{tok}.png"
    PS.save(fig, p); plt.close(fig)
    return p


def cohort_alk(cand):
    """Faceted lollipop: ALK+ and WT candidates in their own columns, effect (T_conc) on a shared x-axis,
    dot colored by shift direction. The exhaustive per-candidate detail lives in Supplementary Table S7."""
    from matplotlib.lines import Line2D
    if not len(cand):
        return None

    def dcol(d):
        d = str(d)
        if d.startswith("tumor-gain"):
            return PS.RED
        if d.startswith("enriched"):
            return "#e08214"          # orange: enriched-cluster shift
        return PS.GREY                # normal / loss

    def dlabel(r):
        return (f"{r.sample.replace('LUAD_','')} m.{r.pos} "
                f"{str(r.region).split('/')[0].replace('MT-','') if pd.notna(r.region) else ''}")

    groups = [("ALK+", cand[cand.alk == "ALK+"]), ("WT", cand[cand.alk == "WT"])]
    nmax = max(len(g) for _, g in groups) or 1
    xmax = cand.T_conc.max() * 1.12
    fig, axes = plt.subplots(1, 2, figsize=(PS.COL, 0.24 * nmax + 1.5), sharex=True)
    for ax, (name, g) in zip(axes, groups):
        g = g.sort_values("T_conc", ascending=True).reset_index(drop=True)
        y = np.arange(len(g))
        for yi, r in zip(y, g.itertuples()):
            ax.hlines(yi, 0, r.T_conc, color="#d0d0d0", lw=1.0, zorder=1)
            ax.scatter(r.T_conc, yi, s=44, color=dcol(r.direction), edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_yticks(y); ax.set_yticklabels([dlabel(r) for r in g.itertuples()], fontsize=6.8)
        ax.set_ylim(-0.6, nmax - 0.4)
        ax.set_xlim(0, xmax)
        ax.set_title(f"{name}  (n = {len(g)})", fontsize=9, fontweight="bold")
        ax.set_xlabel("cluster-pooled Δ VAF (T$_{conc}$)")
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([0], [0], marker="o", ls="", mfc=PS.RED, mec="white", label="tumor-gain"),
               Line2D([0], [0], marker="o", ls="", mfc="#e08214", mec="white", label="enriched-cluster"),
               Line2D([0], [0], marker="o", ls="", mfc=PS.GREY, mec="white", label="normal / loss")]
    axes[1].legend(handles=handles, fontsize=6.2, loc="lower right", frameon=False,
                   title="shift direction", title_fontsize=6.5, handletextpad=0.3, labelspacing=0.3)
    fig.suptitle("LUAD candidate mtDNA shifts by ALK status (unenriched scATAC)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); p = f"{FIG}/cohort_alk.png"; PS.save(fig, p); plt.close(fig)
    return p


def main():
    os.makedirs(FIG, exist_ok=True)
    cf = f"{DD}/luad_dd_candidates.tsv"
    if not os.path.exists(cf):
        print("no candidates file"); return
    c = pd.read_csv(cf, sep="\t")
    cand = c[c.call == "candidate"].copy()
    made = []
    for tok in sorted(cand['sample'].unique()):
        p = metacell_panels(tok, cand)
        if p:
            made.append(p)
    p = cohort_alk(cand)
    if p:
        made.append(p)
    print(f"wrote {len(made)} figures -> {FIG}")
    for m in made:
        print("  " + m)


if __name__ == "__main__":
    main()

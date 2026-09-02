#!/usr/bin/env python
"""
Effect-vs-significance ("volcano") plots for the LUAD meeting report. x = T_conc (between-cluster effect size,
the real discriminator); y = -log10(raw p_clone). NOTE: p_clone is a permutation p floored at 1/(n_null+1) ~= 1e-3,
so significance saturates near y~3 and MANY positions are "significant" at raw p -- the separation is on EFFECT and
localization, not p. Candidate positions (localized, non-mixing, non-artifact) are colored by direction; q_clone<0.10
(BH context) is drawn filled vs open. Per strong sample + a cohort panel.
Usage: luad_dd_volcano.py
Out: docs/reports/figures/luad_meeting/volcano_<tok>.png + volcano_cohort.png
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pubstyle as PS; PS.apply()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DD = f"{ROOT}/work/luad_dd"; FIG = f"{ROOT}/figures"
STRONG = ["LUAD_AL10", "LUAD_AL11", "LUAD_AL08"]
GAIN, OTHER, HB, BG = PS.RED, PS.BLUE, "#6a3d9a", "#d9d9d9"
PFLOOR = 1.0 / 1001.0


def load():
    s = pd.read_csv(f"{DD}/stats_fdr.tsv", sep="\t")
    c = pd.read_csv(f"{DD}/luad_dd_candidates.tsv", sep="\t")
    c = c[c.call == "candidate"][["sample", "axis", "pos", "direction", "is_malignant_cluster", "high_baseline"]]
    c = c.rename(columns={"axis": "method"})
    m = s.merge(c, on=["sample", "method", "pos"], how="left")
    m["is_cand"] = m.direction.notna()
    return m


def cls(r):
    if not r.is_cand:
        return BG
    if bool(r.high_baseline):
        return HB
    if r.direction == "tumor-gain" and bool(r.is_malignant_cluster):
        return GAIN
    return OTHER


def volcano(ax, d, title, annotate=True):
    rng = np.random.default_rng(0)
    y = -np.log10(np.clip(d.p_clone.values, PFLOOR, 1)) + rng.normal(0, 0.03, len(d))  # tiny jitter (p-floor saturates)
    cols = [cls(r) for r in d.itertuples()]
    filled = (d.q_clone.values < 0.10)
    # background (non-candidate) first, then candidates on top. Split each into open (q>=0.10) vs
    # bordered (q<0.10) with SCALAR edgecolors -- a per-point edgecolors LIST makes matplotlib draw a
    # real edge on every point (it ignores "none" in list form), which falsely rings all points.
    cols = np.array(cols, dtype=object)
    for mask, z in [(~d.is_cand.values, 1), (d.is_cand.values, 3)]:
        for passq, ec, sz, lw in [(False, "none", 22, 0.0), (True, "k", 52, 1.1)]:
            m2 = mask & (filled == passq)
            if not m2.any():
                continue
            ax.scatter(d.T_conc.values[m2], y[m2], s=sz, c=list(cols[m2]),
                       edgecolors=ec, linewidths=lw, alpha=0.9, zorder=z + (0 if not passq else 1))
    ax.axhline(-np.log10(PFLOOR), ls=":", color="#888", lw=0.8)
    ax.text(ax.get_xlim()[1], -np.log10(PFLOOR), " p-floor (1/1001)", va="center", ha="left", fontsize=6, color="#888")
    ax.axvline(0.03, ls=":", color="#bbb", lw=0.8)
    if annotate:
        cand = d[d.is_cand & (d.direction == "tumor-gain") & (d.is_malignant_cluster == True)]
        cand = cand.sort_values("T_conc", ascending=False).drop_duplicates("pos").head(5)
        for i, r in enumerate(cand.itertuples()):     # fan labels out below the p-floor with connector lines
            ax.annotate(f"m.{int(r.pos)} {str(r.region).split('/')[0]}",
                        (r.T_conc, -np.log10(max(r.p_clone, PFLOOR))),
                        fontsize=6.5, xytext=(12, -14 - 12 * i), textcoords="offset points",
                        ha="left", arrowprops=dict(arrowstyle="-", color="#b2182b", lw=0.5))
    ax.set_xlabel("effect size  T_conc  (hot-cluster − pooled metacell VAF)"); ax.set_ylabel("−log10 raw p_clone")
    ax.set_title(title); ax.spines[["top", "right"]].set_visible(False)


def legend(ax):
    h = [Line2D([0], [0], marker='o', ls='', mfc=GAIN, mec='none', label='malignant tumor-gain'),
         Line2D([0], [0], marker='o', ls='', mfc=OTHER, mec='none', label='other candidate (loss / cell-type)'),
         Line2D([0], [0], marker='o', ls='', mfc=HB, mec='none', label='high-baseline (germline-seg)'),
         Line2D([0], [0], marker='o', ls='', mfc=BG, mec='none', label='non-candidate position'),
         Line2D([0], [0], marker='o', ls='', mfc='w', mec='k', label='q_clone < 0.10 (BH, context)')]
    ax.legend(handles=h, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=2)          # below the plot, never over the data


def main():
    os.makedirs(FIG, exist_ok=True)
    m = load()
    for tok in STRONG:
        d = m[m['sample'] == tok]
        if not len(d):
            continue
        fig, ax = plt.subplots(figsize=(5.4, 4.9))
        volcano(ax, d, f"{tok.replace('LUAD_','')} — effect size versus significance")
        legend(ax)
        fig.tight_layout(); PS.save(fig, f"{FIG}/volcano_{tok}.png"); plt.close(fig)
        print(f"wrote volcano_{tok}.png")
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    volcano(ax, m, "LUAD cohort — mtDNA effect size versus significance")
    legend(ax)
    fig.tight_layout(); PS.save(fig, f"{FIG}/volcano_cohort.png"); plt.close(fig)
    print("wrote volcano_cohort.png")


if __name__ == "__main__":
    main()

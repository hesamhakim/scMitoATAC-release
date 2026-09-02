#!/usr/bin/env python
"""Composite MAIN FIGURES 2 and 5 (v2) for the scMitoATAC manuscript.

Figure 2  main_v2_fig2_specificity.png  "Specificity and accuracy against external DNA reference data"
  (a) REH false-VAF: clean-vs-panel homoplasmic-site specificity strip (log y).
      Ported from scripts/reh_validation.py make_figure, panel (a).
  (b) ccRCC 19-donor per-donor lollipop of ATAC consensus VAF at homoplasmic sites.
      Ported from manuscript_data_figs.fig7_ccrcc_crossdonor, panel (a).
  (c) mgatk standard-vs-enriched scatter (r=0.974 all het; r=0.88 genuine 2-10%).
      Ported from phase5_mgatk_3arm_cmp.make_figure, panel (a).
  (d) three arms share a sub-0.1% specificity floor.
      Ported from phase5_mgatk_3arm_cmp.make_figure, panel (b).
  (e) per-stratum ECE lollipop (raw -> recalibrated).
      Ported from manuscript_data_figs.fig4_calibration, panel (b) only.

Figure 5  main_v2_fig5_band_boundary.png  "Operating inside the licensed band, and the boundary"
  (a) licensed-band clonal-hematopoiesis panel (marrow clones, clean blood).
      Ported from main_fig_composites_c.main_fig5, panel (d).
  (b) NEW: k-stability as the discriminator (values from FINDINGS_phase15, Arm A).
  (c) coverage-stratified null is calibrated (stratified vs plain-shuffle FPR).
      Ported from fig_phase10.py FPR-bar panel.
  (d) NEW: honest low-band boundary (ATAC-vs-WGS concordance r by band).

Interpreter: python >= 3.12 (see environment.yml)
Reads committed data files read-only; writes only the two PNGs.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pubstyle as PS
PS.apply()

ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{ROOT}/data"
OUTDIR = f"{ROOT}/figures"
LIGHTBLUE = "#a6c8e0"
ANN = 6

ARTIFACT_POS = 3106  # rCRS N-spacer (reh)


# =====================================================================
# MAIN FIGURE 2 — specificity & accuracy vs external DNA truth
# =====================================================================
def _f2_a_reh(ax):
    """REH homoplasmic-site specificity: clean vs panel VAF strip (log y)."""
    resdir = f"{ROOT}/data/reh"
    sites = pd.read_csv(f"{resdir}/reh_sites.csv")
    summ = json.load(open(f"{resdir}/reh_validation_summary.json"))
    core = sites[~sites["is_control"]].copy()
    homo = core[~core["wgs_het"]].copy()
    clean = homo[~homo["is_panel"]]
    panel = homo[homo["is_panel"]]
    ratio = summ["artifact_4_3"]["panel_over_clean_ratio"]
    floor_pct = summ["specificity_4_1"]["cons_floor_measured"] * 100

    rng = np.random.default_rng(0)
    disp_floor = 3e-3
    for xc, grp, col in [(0, clean, PS.GREY), (1, panel, PS.RED)]:
        y = np.clip(grp["atac_cons_vaf"].values * 100, disp_floor, None)
        x = xc + rng.uniform(-0.16, 0.16, size=len(y))
        ax.scatter(x, y, s=3, c=col, alpha=0.35, linewidths=0, rasterized=True)
        med = grp["atac_cons_vaf"].median() * 100
        ax.plot([xc - 0.28, xc + 0.28], [med, med], color=col, lw=2.4,
                solid_capstyle="round")
    ax.axhline(floor_pct, ls="--", lw=0.8, color=PS.INK)
    ax.text(1.55, floor_pct, f"floor\n{floor_pct:.3f}%", va="center", ha="right",
            fontsize=ANN, color=PS.GREY)
    ax.set_yscale("log")
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"clean\n(n={len(clean):,})", f"panel\n(n={len(panel)})"])
    ax.set_ylabel("ATAC consensus VAF (%), log")
    ax.set_title(f"REH: homoplasmic-site\nspecificity ({ratio:.1f}x panel/clean)")
    PS.panel(ax, "a", x=-0.40, y=1.24)


def _f2_b_ccrcc(ax):
    """ccRCC per-donor lollipop of ATAC consensus VAF at WGS-homoplasmic sites."""
    d = pd.read_csv(f"{DATA}/ccRCC_53_perdonor.csv")
    summ = json.load(open(f"{DATA}/ccRCC_53_summary.json"))
    d = d.copy()
    d["floor_pct"] = d["atac_median_vaf_at_homoplasmic"] * 100
    d = d.sort_values("floor_pct", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    pooled_med = summ["cross_donor_artifact_floor_median"] * 100
    ax.hlines(y, 0, d["floor_pct"].values, color=PS.GREY, lw=0.7, zorder=1)
    ax.scatter(d["floor_pct"].values, y, s=18, color=PS.BLUE, zorder=3)
    ax.axvline(pooled_med, ls="--", lw=1.2, c=PS.RED)
    ax.text(pooled_med + 0.0015, 3.0, f"pooled median\n{pooled_med:.3f}%",
            fontsize=ANN, color=PS.RED, va="center", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels(d["donor"].values, fontsize=6.5)
    ax.set_xlabel("ATAC consensus VAF at\nWGS-homoplasmic sites (%)")
    ax.set_title(f"ccRCC: specificity floor\nconsistent across {summ['n_donors_ok']} donors")
    ax.text(0.97, 0.03, f"n={summ['n_homoplasmic_sites_pooled']:,} site-obs",
            transform=ax.transAxes, fontsize=ANN, color=PS.GREY, ha="right", va="bottom")
    PS.panel(ax, "b", x=-0.52, y=1.24)


def _f2_c_mgatk_scatter(ax):
    """mgatk standard vs enriched scatter, het sites split at 10%."""
    m = pd.read_csv(f"{DATA}/mgatk_3arm_perpos.csv")
    s = json.load(open(f"{DATA}/mgatk_3arm_summary.json"))
    HET = 0.02
    enr_het = m[(m.mgatk_enr_vaf >= HET) & (m.mgatk_enr_vaf <= 0.95)].dropna(
        subset=["mgatk_std_vaf"])
    low = enr_het[enr_het.mgatk_enr_vaf <= 0.10]
    high = enr_het[enr_het.mgatk_enr_vaf > 0.10]
    r_all = s["concordance_mgatk_std_vs_enr"]
    r_low = float(np.corrcoef(low.mgatk_enr_vaf, low.mgatk_std_vaf)[0, 1])

    ax.plot([0, 100], [0, 100], ls="--", lw=0.8, color=PS.GREY, zorder=0)
    ax.scatter(high.mgatk_enr_vaf * 100, high.mgatk_std_vaf * 100, s=14, c=LIGHTBLUE,
               linewidths=0, label="germline / markers")
    ax.scatter(low.mgatk_enr_vaf * 100, low.mgatk_std_vaf * 100, s=14, c=PS.RED,
               linewidths=0, label="genuine low-het (2-10%)")
    ax.text(0.03, 0.97, f"all het: r = {r_all:.3f} (n={len(enr_het)})\n"
            f"genuine 2-10%: r = {r_low:.2f} (n={len(low)})",
            transform=ax.transAxes, va="top", ha="left", fontsize=6)
    ax.set_xlim(-3, 105)
    ax.set_ylim(-3, 105)
    ax.set_xlabel("mgatk on enriched\n(mtscATAC) VAF (%)")
    ax.set_ylabel("mgatk on standard\n(10x scATAC) VAF (%)")
    ax.set_title("mgatk: standard recovers\nenriched het")
    ax.legend(loc="lower right", fontsize=5.6, handletextpad=0.3)
    PS.panel(ax, "c", x=-0.42, y=1.24)


def _f2_d_mgatk_floor(ax):
    """Specificity floor across three arms."""
    m = pd.read_csv(f"{DATA}/mgatk_3arm_perpos.csv")
    s = json.load(open(f"{DATA}/mgatk_3arm_summary.json"))
    HET = 0.02
    enr_floor = m.loc[m.mgatk_enr_vaf < HET, "mgatk_enr_vaf"].median() * 100
    std_floor = s["mgatk_std_median_vaf_at_enr_homo"] * 100
    ours_floor = s["ours_std_median_vaf_at_enr_homo"] * 100
    vals = [enr_floor, std_floor, ours_floor]
    cols = [LIGHTBLUE, PS.BLUE, PS.RED]
    ax.bar([0, 1, 2], vals, color=cols, width=0.66)
    for xi, v in zip([0, 1, 2], vals):
        ax.text(xi, v + max(vals) * 0.02, f"{v:.3f}%", ha="center", va="bottom",
                fontsize=6.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["mgatk\nenriched", "mgatk\nstandard", "ours\nstandard"])
    ax.set_ylabel("median VAF at\nhomoplasmic sites (%)")
    ax.set_ylim(0, max(vals) * 1.45)
    ed = s["median_enr_dp"]
    sd = s["median_std_dp"]
    ax.text(0.5, 0.99, f"enriched {ed:.0f}x, standard {sd:.0f}x\n({ed/sd:.1f}x deeper)",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.6, color=PS.GREY)
    ax.set_title("All three arms share a\nsub-0.1% floor")
    PS.panel(ax, "d", x=-0.24, y=1.24)


def _f2_e_ece(ax):
    """Reliability diagram: clean-site posterior tracks the diagonal; artifact sites are over-confident.
    Recalibration collapses the artifact ECE from 0.37 to ~0.001 (annotated)."""
    rel = pd.read_csv(f"{DATA}/atac_reliability.csv")

    def ece(sub):
        w = sub["count"] / sub["count"].sum()
        return float((w * (sub["mean_pred"] - sub["obs_freq"]).abs()).sum())

    ece_vals = {k: ece(rel[rel["site_class"] == k]) for k in ["clean", "artifact"]}
    ax.plot([0, 1], [0, 1], ls=":", color=PS.INK, lw=1.1, zorder=2)   # perfect calibration
    ax.text(0.62, 0.70, "perfect\ncalibration", rotation=40, fontsize=5.6, color=PS.GREY,
            ha="center", va="center", rotation_mode="anchor")
    sa = rel[rel["site_class"] == "artifact"].sort_values("mean_pred")
    # shade the over-confidence wedge: gap between the artifact curve and the diagonal
    ax.fill_between(sa["mean_pred"], sa["obs_freq"], sa["mean_pred"], color=PS.RED, alpha=0.12, zorder=1)
    for k, col, mk in [("clean", PS.BLUE, "o"), ("artifact", PS.RED, "s")]:
        s = rel[rel["site_class"] == k].sort_values("mean_pred")
        ax.plot(s["mean_pred"], s["obs_freq"], marker=mk, color=col, lw=1.8, ms=3.5, zorder=3,
                label=f"{k} (raw ECE {ece_vals[k]:.2f})")
    ax.annotate("predicts 0.84,\nonly 0.07 real", xy=(0.84, 0.07), xytext=(0.36, 0.42),
                fontsize=5.8, color=PS.RED, arrowprops=dict(arrowstyle="->", color=PS.RED, lw=0.7))
    ax.text(0.03, 0.965, "recalibration -> artifact ECE ~0.001", transform=ax.transAxes, ha="left",
            fontsize=5.6, color=PS.INK, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("predicted P(carrier)"); ax.set_ylabel("observed carrier\nfrequency")
    ax.set_title("Calibration: artifact sites\nare over-confident")
    ax.legend(loc="upper center", fontsize=5.6, bbox_to_anchor=(0.5, 0.87), handletextpad=0.4)
    PS.panel(ax, "e", x=-0.30, y=1.16)


def main_fig2():
    os.makedirs(OUTDIR, exist_ok=True)
    out = f"{OUTDIR}/main_v2_fig2_specificity.png"
    # canvas trimmed so the tight-bbox output (which adds ~0.25in of left-margin
    # y-labels + panel letters) lands at <= PS.COL (6.9in) wide.
    fig, axes = plt.subplots(2, 3, figsize=(6.62, 5.0))
    _f2_a_reh(axes[0, 0])
    _f2_b_ccrcc(axes[0, 1])
    _f2_c_mgatk_scatter(axes[0, 2])
    _f2_d_mgatk_floor(axes[1, 0])
    _f2_e_ece(axes[1, 1])
    axes[1, 2].axis("off")
    fig.suptitle("Specificity and accuracy against external DNA reference data", y=1.01,
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout(h_pad=3.0, w_pad=2.6, rect=(0, 0, 1, 0.98))
    PS.save(fig, out)
    plt.close(fig)
    print(f"wrote {out}")


# =====================================================================
# MAIN FIGURE 5 — operating inside the licensed band, and the boundary
# =====================================================================
# Fig 5a: licensed-band clonal-hematopoiesis (verified; from main_fig_composites_c Fig-5d)
F5A = [
    dict(sample="BMMC_D6T1", tissue="marrow", vaf=1.23, variant="m.3943\nMT-ND1",
         kstab="3/4", clone=True),
    dict(sample="BMMC_D5T1", tissue="marrow", vaf=1.99, variant="m.11056\nMT-ND4",
         kstab="2/4", clone=True),
    dict(sample="PBMC_D12T3", tissue="blood", vaf=0.0, variant="clean",
         kstab="0/4", clone=False),
]

# Fig 5b: NEW k-stability discriminator (FINDINGS_phase15, Arm A)
# (k-cuts confident out of 4, between-subclone effect % VAF, retained?)
F5B = [
    dict(label="BMMC_D6T1 raw-max", k=1, vaf=6.32, retained=False, variant=None),
    dict(label="BMMC_D6T1 m.3943", k=3, vaf=1.23, retained=True, variant="m.3943 MT-ND1"),
    dict(label="BMMC_D5T1 raw-max", k=1, vaf=3.72, retained=False, variant=None),
    dict(label="BMMC_D5T1 m.11056", k=2, vaf=1.99, retained=True, variant="m.11056 MT-ND4"),
    dict(label="PBMC_D12T3", k=1, vaf=0.55, retained=False, variant=None),
]

# Fig 5d: NEW honest low-band boundary (aggregate ccRCC ATAC-vs-WGS concordance by band)
F5D = [
    ("overall", 0.73, PS.GREY),
    ("germline\n(>=10% VAF)", 0.89, PS.BLUE),
    ("2-10% band", 0.007, PS.RED),
]


def _f5_a_band(ax):
    x = np.arange(len(F5A))
    ymax = 5.8
    ax.axhspan(1.0, 5.0, color=PS.FILL, zorder=0)
    ax.text(len(F5A) - 0.5, 5.0, "licensed 1-5% band", ha="right", va="top",
            fontsize=ANN, color=PS.BLUE, style="italic")
    ax.axhline(1.0, color=PS.GREY, lw=0.6, ls="--", zorder=1)
    ax.axhline(5.0, color=PS.GREY, lw=0.6, ls="--", zorder=1)
    colors = [PS.BLUE if r["clone"] else PS.GREY for r in F5A]
    vafs = [r["vaf"] for r in F5A]
    ax.bar(x, vafs, width=0.55, color=colors, zorder=3)
    for i, r in enumerate(F5A):
        if r["clone"]:
            ax.annotate(f"{r['variant']} {r['kstab']}\n{r['vaf']:.2f}%",
                        xy=(i, r["vaf"]), xytext=(i, r["vaf"] + 0.30),
                        ha="center", va="bottom", fontsize=ANN, color=PS.INK, zorder=4)
        else:
            ax.annotate("clean 0/4\n0%", xy=(i, 0), xytext=(i, 0.30),
                        ha="center", va="bottom", fontsize=ANN, color=PS.INK, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['sample']}\n({r['tissue']})" for r in F5A], fontsize=6)
    ax.set_ylabel("k-stable clone VAF (%)")
    ax.set_ylim(0, ymax)
    ax.set_title("Licensed-band clone in\nmarrow, clean in blood")
    PS.panel(ax, "a", x=-0.20, y=1.14)


def _f5_b_kstability(ax):
    ymax = 7.2
    # retained region: k>=2 AND effect >=1%
    ax.axvspan(1.5, 4.5, color=PS.FILL, zorder=0)
    ax.axvline(1.5, color=PS.GREY, lw=0.6, ls="--", zorder=1)
    ax.axhline(1.0, color=PS.GREY, lw=0.6, ls=":", zorder=1)
    ax.text(4.42, 0.35, "retained region\n(k>=2, effect>=1%)", ha="right", va="bottom",
            fontsize=ANN, color=PS.BLUE, style="italic")

    for r in F5B:
        if r["retained"]:
            ax.scatter(r["k"], r["vaf"], s=70, color=PS.BLUE, zorder=4,
                       edgecolor="white", linewidth=0.6)
        else:
            ax.scatter(r["k"], r["vaf"], s=64, facecolors="none", edgecolors=PS.GREY,
                       linewidths=1.4, zorder=4)

    # label the two retained points with their variant
    ax.annotate(F5B[1]["variant"], xy=(3, 1.23), xytext=(3.0, 2.15),
                ha="center", va="bottom", fontsize=ANN, color=PS.BLUE,
                arrowprops=dict(arrowstyle="-|>", color=PS.BLUE, lw=0.9))
    ax.annotate(F5B[3]["variant"], xy=(2, 1.99), xytext=(1.72, 3.15),
                ha="left", va="bottom", fontsize=ANN, color=PS.BLUE,
                arrowprops=dict(arrowstyle="-|>", color=PS.BLUE, lw=0.9))
    # annotate the demoted largest raw effect
    ax.annotate("largest raw effect\n6.32% at k=1\ndemoted (healthy ctrl)",
                xy=(1, 6.32), xytext=(1.35, 6.35),
                ha="left", va="top", fontsize=ANN, color=PS.GREY,
                arrowprops=dict(arrowstyle="-|>", color=PS.GREY, lw=0.9))

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.5, 4.6)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("k-cuts (of 4) with a confident effect")
    ax.set_ylabel("between-subclone effect (% VAF)")
    ax.set_title("k-stability demotes the\nlargest raw false positive")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", ls="", mfc=PS.BLUE, mec="white",
               label="retained (k-stable)"),
        Line2D([0], [0], marker="o", ls="", mfc="none", mec=PS.GREY,
               label="demoted (k=1)"),
    ], loc="upper right", fontsize=6, handletextpad=0.3, borderpad=0.3)
    PS.panel(ax, "b", x=-0.20, y=1.14)


def _f5_c_null(ax):
    c = pd.read_csv(f"{DATA}/phase10_s3_calibration.tsv", sep="\t").iloc[0]
    vals = [c.fpr_stratified, c.fpr_plain]
    ax.bar(["coverage-\nstratified", "plain\nshuffle"], vals,
           color=[PS.BLUE, PS.GREY], width=0.6)
    ax.axhline(c.alpha, color=PS.RED, ls="--", lw=1.3, label=f"alpha={c.alpha:g}")
    for i, v in enumerate(vals):
        ax.text(i, v + 3e-5, f"{v:.4f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_ylabel("false-positive rate\n(permutation null)")
    ax.set_ylim(0, max(c.alpha * 1.3, 0.002))
    ax.set_title("Null is calibrated: no\nresidual coverage confound")
    ax.legend(loc="upper center", fontsize=6.5)
    PS.panel(ax, "c", x=-0.24, y=1.14)


def _f5_d_boundary(ax):
    labels = [r[0] for r in F5D]
    rvals = [r[1] for r in F5D]
    cols = [r[2] for r in F5D]
    x = np.arange(len(F5D))
    ax.bar(x, rvals, width=0.6, color=cols, edgecolor="white", linewidth=0.5)
    for xi, r in zip(x, rvals):
        ax.text(xi, r + 0.02, f"r = {r:.2f}" if r >= 0.1 else f"r = {r:.3f}",
                ha="center", va="bottom", fontsize=6.5, color=PS.INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ATAC-vs-WGS concordance (r)")
    ax.text(0.97, 0.52, "median WGS reference\ndepth 45× → a 4% VAF\nis ~2 alt reads\n(shot noise)",
            transform=ax.transAxes, fontsize=ANN, color=PS.GREY, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc=PS.FILL, ec=PS.GREY, lw=0.6))
    ax.set_title("Low-band failure is a reference-arm\ndepth limit, not the caller")
    PS.panel(ax, "d", x=-0.24, y=1.14)


def main_fig5():
    os.makedirs(OUTDIR, exist_ok=True)
    out = f"{OUTDIR}/main_v2_fig5_band_boundary.png"
    fig, axes = plt.subplots(2, 2, figsize=(PS.COL, 5.0))
    _f5_a_band(axes[0, 0])
    _f5_b_kstability(axes[0, 1])
    _f5_c_null(axes[1, 0])
    _f5_d_boundary(axes[1, 1])
    fig.suptitle("Operating inside the licensed band, and the boundary", y=1.01,
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout(h_pad=3.0, w_pad=2.6, rect=(0, 0, 1, 0.98))
    PS.save(fig, out)
    plt.close(fig)
    print(f"wrote {out}")
    print("Fig-5a plotted values:")
    for r in F5A:
        print(f"  {r['sample']} ({r['tissue']}): VAF={r['vaf']:.2f}%  "
              f"{r['variant'].replace(chr(10), ' ')} k-stab {r['kstab']}")
    print("Fig-5b plotted values (k, %VAF, retained):")
    for r in F5B:
        print(f"  {r['label']}: k={r['k']}, {r['vaf']:.2f}%, "
              f"retained={r['retained']}"
              + (f", {r['variant']}" if r['variant'] else ""))
    print("Fig-5d plotted values (band, r):")
    for lab, r, _ in F5D:
        print(f"  {lab.replace(chr(10), ' ')}: r={r}")


if __name__ == "__main__":
    main_fig2()
    main_fig5()

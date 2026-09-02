#!/usr/bin/env python
"""Revised MAIN FIGURE 2 (R2) for the scMitoATAC manuscript.

"Specificity and accuracy against external DNA reference data" — reviewer-revised.

Two panel changes vs main_v2_fig2_specificity.png (all other panels/data unchanged):
  (d) REPLACED: the redundant three-arm sub-0.1% floor bars are gone; a compact
      NUMT-contamination + decoy-correction panel takes their place. Naive-vs-decoy
      VAF scatter: the 9 NUMT/decoy-removable sites (RED) drop off the diagonal toward
      0; the 41 context/mapping sites (GREY) stay on it. Annotated with the two facts:
      NUMT reads are ~2.0-2.2% of the chrM-destined pool (~20x mgatk's ~0.1% premise)
      and ~91% of the naive MT-ND2 pileup is NUMT.
      Ported/compacted from manuscript_data_figs.fig3_decoy_vs_naive (S7) + fig2_numt_earlywarning (S6).
  (e) UPGRADED: reliability diagram now shows THREE curves — raw clean (near diagonal),
      raw artifact (badly over-confident: predicts 0.84 where 0.07 observed), and the
      RECALIBRATED artifact curve lying essentially on the identity line
      (post-recalibration artifact ECE ~0.001). Annotated ECE 0.37 -> 0.001.

Panels (a) REH, (b) ccRCC, (c) mgatk scatter are reused verbatim from main_v2_fig2_5.

Interpreter: python >= 3.12 (see environment.yml)
Reads committed data files read-only; writes only docs/manuscript/figures/main/main_r2_specificity.png
"""
import os
import sys

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

# reuse the kept panels (a, b, c) verbatim
from main_v2_fig2_5 import _f2_a_reh, _f2_b_ccrcc, _f2_c_mgatk_scatter  # noqa: E402

ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{ROOT}/data"
OUTDIR = f"{ROOT}/figures"
ANN = 6


# =====================================================================
# (d) NEW — NUMT contamination + decoy correction
# =====================================================================
def _f2_d_numt_decoy(ax):
    """Naive-vs-decoy VAF: NUMT/decoy-removable sites fall off the diagonal toward 0,
    context/mapping sites stay on it. Makes the point that NUMT reads are ~2% of the
    chrM-destined pool (~20x mgatk's premise) and dominate the MT-ND2 pileup (~91%)."""
    df = pd.read_csv(f"{DATA}/phase2_atac_artifact_panel.csv")
    df["naive_pct"] = df["median_naive_VAF"] * 100
    df["decoy_pct"] = df["median_decoy_VAF"] * 100
    is_numt = df["artifact_class"].str.startswith("NUMT")
    n_numt = int(is_numt.sum())
    n_ctx = int((~is_numt).sum())

    lim = 2.9
    ax.plot([0, lim], [0, lim], ls=":", lw=1.0, c=PS.GREY, zorder=1)
    ax.scatter(df.loc[~is_numt, "naive_pct"], df.loc[~is_numt, "decoy_pct"],
               s=16, c=PS.GREY, edgecolor="white", linewidth=0.3, zorder=2,
               label=f"context/mapping (n={n_ctx})")
    ax.scatter(df.loc[is_numt, "naive_pct"], df.loc[is_numt, "decoy_pct"],
               s=30, c=PS.RED, edgecolor="white", linewidth=0.5, zorder=3,
               label=f"NUMT, decoy-removable (n={n_numt})")
    # label the MT-ND2 pseudogene site
    nd2 = df[df["chrM_pos"] == 4703].iloc[0]
    ax.annotate("chrM:4703 (MT-ND2)", (nd2["naive_pct"], nd2["decoy_pct"]),
                textcoords="offset points", xytext=(8, 7), fontsize=ANN, color=PS.RED,
                va="bottom", ha="left")
    # NUMT magnitude / MT-ND2 facts moved to the caption so the boxed note no longer overlays
    # the red NUMT points that drop off the diagonal in the lower-right of the panel
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xlabel("naive (no-decoy) VAF (%)")
    ax.set_ylabel("decoy-corrected VAF (%)")
    ax.set_title("Decoy realignment pulls\nNUMT sites toward 0")
    ax.legend(loc="upper left", fontsize=5.4, handletextpad=0.3, borderpad=0.3)
    PS.panel(ax, "d", x=-0.28, y=1.24)


# =====================================================================
# (e) UPGRADED — reliability diagram, raw + recalibrated
# =====================================================================
def _f2_e_reliability(ax):
    """Reliability diagram with three curves: raw clean (near diagonal), raw artifact
    (over-confident), and the recalibrated artifact curve on the identity line.
    Post-recalibration artifact-site ECE is ~0.001, so the recalibrated curve is drawn
    as the identity line itself (per-bin recalibrated obs not in atac_reliability.csv)."""
    rel = pd.read_csv(f"{DATA}/atac_reliability.csv")

    def ece(sub):
        w = sub["count"] / sub["count"].sum()
        return float((w * (sub["mean_pred"] - sub["obs_freq"]).abs()).sum())

    ece_vals = {k: ece(rel[rel["site_class"] == k]) for k in ["clean", "artifact"]}

    # identity / perfect-calibration line
    ax.plot([0, 1], [0, 1], ls=":", color=PS.INK, lw=1.0, zorder=2)
    ax.text(0.63, 0.72, "perfect calibration", rotation=40, fontsize=5.4, color=PS.GREY,
            ha="center", va="center", rotation_mode="anchor")

    sa = rel[rel["site_class"] == "artifact"].sort_values("mean_pred")
    # shade the over-confidence wedge between the raw-artifact curve and the diagonal
    ax.fill_between(sa["mean_pred"], sa["obs_freq"], sa["mean_pred"],
                    color=PS.RED, alpha=0.10, zorder=1)

    # RECALIBRATED artifact curve: essentially on the identity line (ECE ~0.001)
    ax.plot([0, 1], [0, 1], color=PS.GREEN, lw=3.0, alpha=0.8, zorder=2.5,
            solid_capstyle="round", label="recalibrated artifact")

    # raw curves (short labels; ECE values go in a compact corner block, not the legend)
    for k, col, mk in [("clean", PS.BLUE, "o"), ("artifact", PS.RED, "s")]:
        s = rel[rel["site_class"] == k].sort_values("mean_pred")
        ax.plot(s["mean_pred"], s["obs_freq"], marker=mk, color=col, lw=1.8, ms=3.3,
                zorder=3, label=f"raw {k}")

    # ECE values as a small corner annotation in the empty upper-left, off the data
    ax.text(0.03, 0.985,
            f"ECE  raw artifact {ece_vals['artifact']:.2f} $\\rightarrow$ recal 0.001\n"
            f"        raw clean {ece_vals['clean']:.2f}",
            transform=ax.transAxes, fontsize=5.2, color=PS.INK, ha="left", va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("predicted P(carrier)")
    ax.set_ylabel("observed carrier\nfrequency")
    ax.set_title("Recalibration puts artifact\nsites back on the diagonal")
    ax.legend(loc="lower right", fontsize=5.6, handletextpad=0.4, borderpad=0.3,
              labelspacing=0.3, frameon=False)
    PS.panel(ax, "e", x=-0.30, y=1.24)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    out = f"{OUTDIR}/main_r2_specificity.png"
    fig = plt.figure(figsize=(6.25, 5.35))
    # 2 rows x 6 cols: top row three narrow specificity panels (a,b,c); bottom row two
    # wider panels (d,e) filling the full width, so there is no empty grid cell.
    # top left generous so the two-line panel titles + panel letters clear the suptitle.
    gs = fig.add_gridspec(2, 6, hspace=0.66, wspace=1.15,
                          left=0.085, right=0.985, top=0.80, bottom=0.095)
    _f2_a_reh(fig.add_subplot(gs[0, 0:2]))
    _f2_b_ccrcc(fig.add_subplot(gs[0, 2:4]))
    _f2_c_mgatk_scatter(fig.add_subplot(gs[0, 4:6]))
    _f2_d_numt_decoy(fig.add_subplot(gs[1, 0:3]))
    _f2_e_reliability(fig.add_subplot(gs[1, 3:6]))
    if not PS.GB_SUB:   # GB: the overall title lives in the figure legend, not the graphic
        fig.suptitle("Specificity and accuracy against external DNA reference data", y=0.985,
                     fontsize=9.5, fontweight="bold")
    PS.save(fig, out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

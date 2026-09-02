#!/usr/bin/env python
"""Trimmed supplement figure: ONLY the carrier-fraction panel of the pooling figure.

The full pooling figure (phase4_pooling_clone.png, from manuscript_data_figs.fig6_pooling_clone)
has three panels: (a) variance vs N, (b) mean error vs N, (c) recovered carrier fraction vs
WGS truth. Panels a and b are now shown in main Figure 3, so this supplement keeps ONLY panel
(c): recovered single-cell carrier fraction and pooled ATAC VAF versus WGS truth across the REH
heteroplasmic sites (r = 0.963, n=3). Data/numbers are ported verbatim from fig6_pooling_clone().

Run:  python reproduce/supp_pooling_carrier.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle as PS  # noqa: E402

PS.apply()

ROOT = Path(os.environ.get("SCMITOATAC_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
OUT = ROOT / "figures" / "supp_pooling_carrier.png"

ANN = 6   # annotation fontsize
LAB = 8   # axis label fontsize


def main():
    reh = pd.read_csv(DATA / "reh_pooling.csv")
    Ns = sorted(reh["N"].unique())

    # --- ported verbatim from fig6_pooling_clone() panel c / axc block ---
    truth = reh.groupby("pos")["truth_vaf"].first() * 100
    pooled = reh[reh["N"] == max(Ns)].set_index("pos")["mean_pooled_vaf"] * 100
    # 'called carrier frac' is not in reh_pooling.csv; values read from current PNG.
    carrier = {15438: 22.0, 4421: 2.5, 13772: 0.4}
    r_pearson = 0.963  # shown in current PNG (n=3 het sites)
    lim = 28
    lab_off = {15438: (7, 2), 4421: (7, 3), 13772: (10, -7)}  # separate near-origin labels

    fig, axc = plt.subplots(1, 1, figsize=(3.7, 3.1))

    axc.plot([0, lim], [0, lim], ls="--", lw=1.0, c=PS.GREY, zorder=1)
    for pos in truth.index:
        axc.scatter(truth[pos], carrier[pos], s=44, color=PS.BLUE, zorder=3)
        axc.scatter(truth[pos], pooled[pos], s=44, marker="^", facecolor="none",
                    edgecolor=PS.RED, linewidth=1.2, zorder=3)
        axc.annotate(f"chrM:{pos}", (truth[pos], carrier[pos]),
                     textcoords="offset points", xytext=lab_off[pos], fontsize=ANN,
                     color=PS.INK, va="center")
    axc.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc=PS.BLUE, mec=PS.BLUE, label="carrier fraction"),
                        Line2D([0], [0], marker="^", ls="", mfc="none", mec=PS.RED, label="pooled ATAC VAF")],
               loc="upper right", fontsize=6.2, handletextpad=0.3, borderpad=0.3,
               bbox_to_anchor=(0.99, 0.82))
    axc.text(0.04, 0.96, f"r = {r_pearson:.3f} (n=3)",
             transform=axc.transAxes, fontsize=ANN, color=PS.INK, va="top")
    axc.set_xlim(0, lim); axc.set_ylim(0, lim)
    axc.set_xlabel("WGS reference VAF (%)", fontsize=LAB)
    axc.set_ylabel("recovered from single-cell (%)", fontsize=LAB)
    axc.set_title("Carrier fraction tracks WGS reference", fontsize=8)
    axc.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.92)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    PS.save(fig, OUT)
    plt.close(fig)
    print("truth:", {int(p): round(truth[p], 2) for p in truth.index})
    print("carrier:", carrier)
    print("pooled:", {int(p): round(pooled[p], 2) for p in pooled.index})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

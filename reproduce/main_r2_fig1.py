#!/usr/bin/env python
"""Main Figure 1 for the scMitoATAC manuscript: the method and its licensed operating envelope.

Composite, publication quality, authored at final print size:
  (a) full-width compressed pipeline schematic (single row of boxes)
  (b) compressed coverage-stratified null schematic
  (c) titration: called VAF tracks known VAF over three orders of magnitude   [port: phase6_spikein.make_figure panel a]
  (d) licensed floor: sensitivity/precision vs known VAF per pooled depth      [port: phase6_spikein.make_figure panel b]
  (e) planning: cells needed to license a target VAF vs per-cell chrM depth     [new]

Reuses box()/arr() drawing helpers from manuscript_schematics.py (read-only).
Out: docs/manuscript/figures/main/main_r2_method_envelope.png
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pubstyle as PS
PS.apply()
# reuse the schematic drawing primitives (module import does not run its main())
from manuscript_schematics import box, arr, R, L, T, B  # noqa: E402

ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = f"{ROOT}/data"
FD = f"{ROOT}/assets"
OUT = f"{ROOT}/figures/main_r2_method_envelope.png"
BLUE, RED, GREEN, GREY, INK, FILL = PS.BLUE, PS.RED, PS.GREEN, PS.GREY, PS.INK, PS.FILL


# ----------------------------------------------------------------------------- panel a
def _abox(ax, x, y, w, h, label, ec, fc, n=None, ts=8.0):
    """One rounded workflow box with a vertically-centered label and an optional step number."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                                ec=ec, fc=fc, lw=1.3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=ts,
            fontweight="bold", color=INK, linespacing=1.05)
    if n is not None:
        ax.text(x + 0.09, y + h - 0.07, str(n), ha="left", va="top", fontsize=6.4, color=ec)
    return (x, y, w, h)


def panel_a(ax):
    """Two-row workflow, both rows left-to-right: row 1 = the frozen per-cell caller (steps 1-5),
    row 2 = the deployment / between-population layer (steps 6-9), joined by a wrap connector 5->6."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.85); ax.axis("off")
    bw, bh, gap, x0 = 1.70, 0.85, 0.30, 0.10
    y1, y2 = 2.78, 0.36                                       # row-1 / row-2 box bottoms
    xs1 = [x0 + i * (bw + gap) for i in range(5)]
    xs2 = [x0 + i * (bw + gap) for i in range(4)]

    r1 = ["Standard\nBAM", "Decoy realign.\n+ consensus\npileup", "Per-site\nnull",
          "Mixture\nfusion", "Recalibrate\n/ abstain"]
    r2 = ["Metacell\npooling", "Cluster\ndifferential", "Coverage-\nstratified null", "Calibrated\ncall"]
    ec2 = [BLUE, BLUE, RED, GREEN]
    fc2 = [FILL, FILL, FILL, "#eef7ee"]

    b1 = [_abox(ax, xs1[i], y1, bw, bh, r1[i], BLUE, FILL, n=i + 1) for i in range(5)]
    b2 = [_abox(ax, xs2[i], y2, bw, bh, r2[i], ec2[i], fc2[i], n=i + 6) for i in range(4)]
    for p, q in zip(b1, b1[1:]):
        arr(ax, R(p), L(q), lw=1.3)
    for p, q in zip(b2, b2[1:]):
        arr(ax, R(p), L(q), lw=1.3)

    # wrap connector 5 -> 6: down from box 5, left across the inter-row gap, down into box 6
    yc = 1.95
    x5c, x6c = xs1[4] + bw / 2, xs2[0] + bw / 2
    ax.plot([x5c, x5c], [y1, yc], color=INK, lw=1.3)
    ax.plot([x5c, x6c], [yc, yc], color=INK, lw=1.3)
    arr(ax, (x6c, yc), (x6c, y2 + bh), lw=1.3)

    # stage labels above each row
    ax.text((xs1[0] + xs1[4] + bw) / 2, y1 + bh + 0.14, "frozen per-cell caller  (v0.1.0)",
            ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=BLUE)
    ax.text((xs2[0] + xs2[3] + bw) / 2, yc + 0.10, "deployment / between-population inference",
            ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=BLUE)


# ----------------------------------------------------------------------------- panel b
def panel_b(ax):
    """Coverage-stratified null flow diagram (journal-style, supplied). Panel letter 'b' baked in."""
    ax.imshow(plt.imread(f"{FD}/covnull_clean.png")); ax.axis("off")


# ----------------------------------------------------------------------------- data for c/d
def _curve():
    C = pd.read_csv(f"{DATADIR}/phase6_spikein_curve.csv")
    s = json.load(open(f"{DATADIR}/phase6_spikein_summary.json"))
    depths = [int(d) for d in s["depths"]]
    cmap = plt.cm.viridis(np.linspace(0.08, 0.9, len(depths)))
    dcol = {d: cmap[i] for i, d in enumerate(depths)}
    return C, depths, dcol


# ----------------------------------------------------------------------------- panel c
def panel_c(ax, C, depths, dcol):
    ax.plot([0.07, 80], [0.07, 80], ls="--", lw=0.8, color=GREY, zorder=0, label="identity")
    for d in depths:
        sub = C[C.depth == d].sort_values("known_vaf")
        sub = sub[sub.median_called > 0]
        ax.plot(sub.known_vaf * 100, sub.median_called * 100, "-o", color=dcol[d], ms=2.6, lw=1.1, label=f"{d}×")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.07, 80); ax.set_ylim(0.08, 140)
    ax.set_xlabel("Known VAF (mixing fraction, %)")
    ax.set_ylabel("Median called VAF (%)")
    ax.set_title("Called VAF tracks known VAF\nover three orders of magnitude", fontsize=8)
    ax.legend(title="pooled depth", ncol=2, fontsize=5.5, title_fontsize=5.8, loc="upper left",
              handletextpad=0.3, columnspacing=0.8, borderpad=0.3, labelspacing=0.25)


# ----------------------------------------------------------------------------- panel d
def panel_d(ax, C, depths, dcol):
    for d in depths:
        sub = C[C.depth == d].sort_values("known_vaf")
        ax.plot(sub.known_vaf * 100, sub.precision, "-", color=dcol[d], lw=1.0, alpha=0.30, zorder=1)
        ax.plot(sub.known_vaf * 100, sub.sensitivity, "-o", color=dcol[d], ms=2.6, lw=1.1, label=f"{d}×", zorder=3)
    ax.axhline(0.80, ls=":", lw=0.9, color=GREY)
    ax.text(0.075, 0.815, "sens 0.80", va="bottom", ha="left", fontsize=5.6, color=GREY)
    ax.set_xscale("log"); ax.set_xlim(0.07, 80); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Known VAF (%)")
    ax.set_ylabel("Sensitivity (○) · precision (faint)")
    ax.set_title("Licensed floor:  2% @100–250× ·\n1% @500× · 0.5% @≥2000×", fontsize=8)
    ax.legend(title="pooled depth", ncol=2, fontsize=5.5, title_fontsize=5.8, loc="lower right",
              handletextpad=0.3, columnspacing=0.8, borderpad=0.3, labelspacing=0.25)


# ----------------------------------------------------------------------------- panel e
def panel_e(ax):
    """Planning: cells needed = ceil(k=3 molecule floor / VAF / per-cell chrM depth).
    Depth-to-license = 3/VAF (the k>=3 pooled-call floor). Targets span the operating range used for the
    proof-of-principle demonstrations (5-10% LOD) down to the aggressive 1%. The x-axis spans the per-cell
    chrM depths actually observed (~8x ped-HGG to ~950x lymph node), extended a decade each way."""
    targets = [(10.0, 30, "#2e7d32"), (5.0, 60, "#2166ac"), (2.0, 150, "#762a83"), (1.0, 300, "#b2182b")]
    x = np.logspace(np.log10(2), np.log10(10000), 300)
    ax.axvspan(7.8, 952, color="#eaf0f5", zorder=0)          # per-cell chrM depths observed in this study
    ax.text(86, 160, "per-cell reads\nobserved here", ha="center", va="top", fontsize=5.2, color="#5a6b7a")
    for vaf, dn, col in targets:
        ax.plot(x, np.ceil(dn / x), "-", color=col, lw=1.7, label=f"{vaf:g}% VAF")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(2, 10000); ax.set_ylim(0.8, 200)
    ax.set_xlabel("Per-cell chrM reads (unique / cell)")
    ax.set_ylabel("Cells needed (metacell size)")
    ax.set_title("Planning: cells to license\na target VAF", fontsize=8)
    ax.legend(title="target VAF", fontsize=5.6, title_fontsize=5.8, loc="upper right",
              handletextpad=0.4, borderpad=0.3, labelspacing=0.25)


# ----------------------------------------------------------------------------- assemble
def main():
    C, depths, dcol = _curve()
    fig = plt.figure(figsize=(6.6, 7.35))
    # row 0 holds the two-row workflow (~2.6:1); width_ratios give panel b (the coverage-null
    # diagram) more room so it is not the smallest panel
    gs = fig.add_gridspec(3, 2, height_ratios=[1.72, 1.05, 1.05],
                          width_ratios=[1.24, 1.0],
                          hspace=0.46, wspace=0.36,
                          left=0.085, right=0.978, top=0.995, bottom=0.085)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    panel_a(ax_a)          # two-row workflow (matplotlib)
    panel_b(ax_b)          # coverage-stratified null flow diagram (panel letter 'b' baked into the image)
    panel_c(ax_c, C, depths, dcol)
    panel_d(ax_d, C, depths, dcol)
    panel_e(ax_e)

    PS.panel(ax_a, "a", x=-0.02, y=1.0, fontsize=10)
    PS.panel(ax_c, "c", x=-0.24, y=1.18, fontsize=10)
    PS.panel(ax_d, "d", x=-0.24, y=1.18, fontsize=10)
    PS.panel(ax_e, "e", x=-0.24, y=1.18, fontsize=10)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # tight bbox but with minimal pad so the final file stays within the 6.9in column
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02)
    PS.save_submission(fig, OUT)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

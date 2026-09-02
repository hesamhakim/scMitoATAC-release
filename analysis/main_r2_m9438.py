#!/usr/bin/env python
"""Composite MAIN FIGURE (r2) for the scMitoATAC manuscript -- m.9438 as a pure ROBUSTNESS figure.

  "m.9438: why believe this is a real partition-specific signal, not an algorithmic or technical artifact?"

Identical to main_v2_fig4 EXCEPT panel (b): the "carrier subclone: CNV-quiet, less malignant"
biology panel is REPLACED by the per-cell pan-site co-elevation verifier (ported from
ped_hgg_specificity_fig panel d). Genuine clones are site-specific (co-elevation ~0) while
technical artifacts / donor-mixing are pan-site (co-elevation ~1).

Five panels (ped-HGG m.9438 MT-CO3 A>G flagship, samples 2932 + 2937):
  (a) carrier-VAF distribution, bimodal in primary (2932) and recurrent (2937) pGBM.
  (b) per-cell verifier: pan-site co-elevation for the two positives (2932, 2937), a normal-lung
      negative (FN4) and a 2-donor mixing negative (mult_T255), with the 0.30 artifact threshold.
  (c) cross-method carrier-set concordance: 13x13 Jaccard, mean of the two samples.
  (d) per-method capture (recall) + purity (precision): every method isolates the same carriers.
  (e) negative control: carriers concentrate in the two positives, not in normal lung (FN4).

Does NOT modify the source generators -- reuses main_v2_fig4's panel functions read-only and
writes a NEW png (main_r2_m9438.png); the v2 figure is untouched.
Interpreter: python >= 3.12 (see environment.yml)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pubstyle as PS
PS.apply()

# read-only reuse of the v2 figure's panel builders and data loaders
import main_v2_fig4 as M4
import ped_hgg_figures as PH

ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = f"{ROOT}/figures"
OUT = f"{OUTDIR}/main_r2_m9438.png"

PS.COL = 6.9

# Approved per-cell-verifier co-elevation values (FINDINGS_pedhgg_m9438_deepdive,
# ped_hgg_specificity_fig.coelev()). Genuine clones are site-specific (~0); artifacts / donor
# mixing are pan-site (~1). Hardcoded so the panel is deterministic regardless of VERIFY.tsv state.
COELEV = {"2932": 0.00, "2937": 0.00, "FN4": 0.96, "mult_T255": 0.98}
COELEV_THR = 0.30


# --------------------------------------------------------------------------- panel b (NEW: co-elevation verifier)
def _panel_b_coelev(ax):
    order = [("2932", "2932\n(pGBM)", PS.RED),
             ("2937", "2937\n(pGBM)", PS.RED),
             ("FN4", "FN4\n(normal\nlung)", PS.GREY),
             ("mult_T255", "mult_T255\n(2-donor\npool)", PS.INK)]
    labs = [l for _, l, _ in order]
    vals = [COELEV[k] for k, _, _ in order]
    cols = [c for _, _, c in order]
    x = np.arange(len(order))
    ax.bar(x, vals, color=cols, width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=5.5)
    ax.tick_params(axis="x", length=2)
    ax.set_ylabel("pan-site co-elevation")
    ax.set_ylim(0, 1.14)
    ax.axhline(COELEV_THR, ls="--", c=PS.INK, lw=1)
    ax.text(0.5, COELEV_THR + 0.03, "artifact threshold",
            ha="center", va="bottom", fontsize=6, color=PS.INK)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.set_title("per-cell verifier:\nreal: site-specific\nartifact: pan-site",
                 fontsize=8, pad=6)
    PS.panel(ax, "b", x=-0.24, y=1.14)
    return {k: v for k, _, _ in order for v in [COELEV[k]]}


# --------------------------------------------------------------------------- assemble
def build():
    os.makedirs(OUTDIR, exist_ok=True)
    D = {}
    for tok in PH.TOKS:
        d = PH.load(tok)
        PH.capture_table(tok, d)
        D[tok] = d

    # reading order a-b-c across the top, d-e across the bottom. Two gridspecs so the
    # bottom row can be inset from the left, leaving room for panel d's long method labels.
    fig = plt.figure(figsize=(6.45, 5.9))
    gtop = fig.add_gridspec(1, 3, left=0.085, right=0.965, top=0.93, bottom=0.585, wspace=0.62)
    gbot = fig.add_gridspec(1, 12, left=0.20, right=0.975, top=0.45, bottom=0.10, wspace=1.0)
    ax_a = fig.add_subplot(gtop[0, 0])
    ax_b = fig.add_subplot(gtop[0, 1])
    ax_c = fig.add_subplot(gtop[0, 2])
    ax_d = fig.add_subplot(gbot[0, 0:8])
    ax_e = fig.add_subplot(gbot[0, 8:12])

    M4._panel_a(ax_a, D)
    coelev = _panel_b_coelev(ax_b)
    M4._panel_c(ax_c, D)
    M4._panel_d(ax_d, D)
    caps = M4._panel_e(ax_e)

    PS.save(fig, OUT)
    plt.close(fig)
    print(f"wrote {OUT}")
    print("panel b pan-site co-elevation:", coelev)
    print("panel e captures:", caps)


if __name__ == "__main__":
    build()

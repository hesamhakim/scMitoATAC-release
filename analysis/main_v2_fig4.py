#!/usr/bin/env python
"""Composite MAIN FIGURE 4 (v2) for the scMitoATAC manuscript.

  "m.9438: a discrete clone that replicates and is method-invariant"

Five panels (ped-HGG m.9438 MT-CO3 A>G flagship, samples 2932 + 2937):
  (a) carrier-VAF distribution, bimodal in primary (2932) and recurrent (2937) pGBM.
      PORT of ped_hgg_figures.fig_compare panel (a).
  (b) MERGED grouped panel: the carrier subclone has lower aneuploidy burden AND lower
      malignant gene-activity than other clusters, in both patients (fig_compare b+c merged).
  (c) cross-method carrier-set concordance: 13x13 Jaccard, mean of the two samples.
      PORT of ped_hgg_figures.fig_crossmethod panel (a), REDESIGNED so labels do not clip
      (axes numbered 1..13; the method names are read off panel (d), which lists them in the
      same order).
  (d) per-method capture (recall) + purity (precision), mean of the two samples: every method
      isolates the same carrier population. PORT of fig_crossmethod panel (b).
  (e) negative control: carriers concentrate in the two positives (capture ~0.94-0.96) and not
      in normal lung (FN4, ~0.47). PORT of ped_hgg_specificity_fig panel (b).

Does NOT modify the source generators -- imports their data-loading helpers read-only.
Interpreter: python >= 3.12 (see environment.yml)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pubstyle as PS
PS.apply()

# read-only imports of the source generators (module-level code only builds constants / applies style)
import ped_hgg_figures as PH
import ped_hgg_specificity_fig as SPEC

ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = f"{ROOT}/figures"
OUT = f"{OUTDIR}/main_v2_fig4_m9438.png"

PS.COL = 6.9


# --------------------------------------------------------------------------- panel a
def _panel_a(ax, D):
    for tok, c in [("2932", PS.BLUE), ("2937", PS.RED)]:
        d = D[tok]; cl = d["cl"]; ok = cl.depth >= 3; car = d["carrier_atac_mask"]
        v = cl.vaf.to_numpy()[ok & car]
        ax.hist(v, bins=25, range=(0, 1), alpha=0.6,
                label=f"{tok} ({PH.TOKS[tok]})", color=c)
    ax.set_xlabel("m.9438 VAF (carrier cluster)")
    ax.set_ylabel("cells")
    ax.legend(fontsize=6, loc="upper center")
    ax.set_title("bimodal carrier VAF")
    PS.panel(ax, "a", x=-0.24, y=1.14)


# --------------------------------------------------------------------------- panel b (merged)
def _panel_b(ax, D):
    toks = list(PH.TOKS)
    burden_car, burden_oth, malig_car, malig_oth = [], [], [], []
    for tok in toks:
        d = D[tok]
        m = d["burden"].merge(d["cnv4"], on="barcode")
        cc = d["en"][(d["en"].method == "epiAneuCNV_k4") & d["en"].is_carrier_cluster].cluster.iloc[0]
        burden_car.append(m[m.group == cc].burden.mean())
        burden_oth.append(m[m.group != cc].burden.mean())
        cs = pd.read_csv(f"{PH.DD}/{tok}/cluster_state.tsv", sep="\t")
        sub = cs[cs.method == "epiAneuCNV_k4"]
        malig_car.append(sub[sub.cluster == cc].malignant_frac.iloc[0])
        malig_oth.append(sub[sub.cluster != cc].malignant_frac.mean())

    # two metric-blocks on one shared "fraction" axis (both quantities are fractions in [0,1])
    xb = np.array([0.0, 0.85])          # aneuploidy-burden block: 2932, 2937
    xm = np.array([2.25, 3.10])         # malignant-activity block: 2932, 2937
    w = 0.36
    ax.bar(xb - w / 2, burden_car, w, color=PS.RED)
    ax.bar(xb + w / 2, burden_oth, w, color=PS.GREY)
    ax.bar(xm - w / 2, malig_car, w, color=PS.RED)
    ax.bar(xm + w / 2, malig_oth, w, color=PS.GREY)

    ax.set_ylabel("fraction")
    ax.set_ylim(0, 0.72)
    ax.set_xticks(np.concatenate([xb, xm]))
    ax.set_xticklabels(toks + toks, fontsize=6)
    ax.tick_params(axis="x", length=2)
    # metric-block headers below the sample labels
    for xc, txt in [(xb.mean(), "aneuploidy\nburden"), (xm.mean(), "malignant\nactivity")]:
        ax.annotate(txt, xy=(xc, 0), xytext=(xc, -0.235), textcoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=6.5, color=PS.INK, annotation_clip=False)
    # light divider between the two blocks
    ax.axvline((xb[-1] + xm[0]) / 2, color=PS.GREY, lw=0.5, ls=":", ymax=0.82)

    leg = [Patch(facecolor=PS.RED, label="carrier subclone"),
           Patch(facecolor=PS.GREY, label="other clusters")]
    ax.legend(handles=leg, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, columnspacing=1.0, handlelength=1.1, handletextpad=0.4, frameon=False)
    ax.set_title("carrier subclone: CNV-quiet,\nless malignant", pad=18)
    PS.panel(ax, "b", x=-0.24, y=1.30)


# --------------------------------------------------------------------------- panel e (negative control)
def _panel_e(ax):
    toks = ["2932", "2937", "FN4"]
    labs = ["2932\n(pGBM)", "2937\n(pGBM)", "FN4\n(normal\nlung)"]
    caps = [SPEC.capture(t) for t in toks]
    cols = [PS.RED, PS.RED, PS.GREY]
    ax.bar(labs, caps, color=cols)
    ax.set_ylabel("best carrier-cell capture")
    ax.set_ylim(0, 1.14)
    ax.axhline(0.5, ls="--", c=PS.GREY, lw=1)
    ax.tick_params(axis="x", labelsize=6, length=2)
    ax.set_title("negative control:\ncaptured in positives, not lung")
    PS.panel(ax, "e", x=-0.20, y=1.14)
    for i, c in enumerate(caps):
        ax.text(i, c + 0.02, f"{c:.2f}", ha="center", fontsize=6.5)
    return dict(zip(toks, [round(c, 2) for c in caps]))


# --------------------------------------------------------------------------- panel c (13x13 Jaccard)
def _panel_c(ax, D):
    J = ((D["2932"]["cc"].astype(float) + D["2937"]["cc"].astype(float)) / 2.0)
    n = len(J)
    im = ax.imshow(J.to_numpy(), cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(range(1, n + 1), fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(range(1, n + 1), fontsize=6)
    ax.tick_params(length=2)
    ax.set_xlabel("method index (see d)", fontsize=7)
    ax.set_ylabel("method index", fontsize=7)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Jaccard (carrier sets)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("cross-method concordance", pad=8)
    PS.panel(ax, "c", x=-0.34, y=1.22)


# --------------------------------------------------------------------------- panel d (capture / purity)
def _panel_d(ax, D):
    order = list(D["2932"]["cc"].index)                 # canonical 13-method order (matches c)
    cap = {}
    for tok in PH.TOKS:
        cap[tok] = D[tok]["capture"].set_index("method")
    capture = np.array([np.mean([cap[t].loc[m, "capture"] for t in PH.TOKS]) for m in order])
    purity = np.array([np.mean([cap[t].loc[m, "purity"] for t in PH.TOKS]) for m in order])
    labels = [f"{i + 1}  {m}" for i, m in enumerate(order)]

    y = np.arange(len(order))[::-1]                      # method 1 at top
    ax.barh(y, capture, color=PS.BLUE, height=0.62, label="capture (recall)")
    ax.plot(purity, y, "o", color=PS.RED, ms=4, label="purity (precision)")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, 1.0); ax.set_xlabel("fraction")
    ax.tick_params(length=2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              fontsize=6.5, handlelength=1.2, columnspacing=1.4)
    ax.set_title("every method isolates the same carriers", pad=16)
    PS.panel(ax, "d", x=-0.14, y=1.14)


# --------------------------------------------------------------------------- assemble
def build():
    os.makedirs(OUTDIR, exist_ok=True)
    D = {}
    for tok in PH.TOKS:
        d = PH.load(tok)
        PH.capture_table(tok, d)
        D[tok] = d

    fig = plt.figure(figsize=(6.65, 5.4))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.28],
                          hspace=0.52, wspace=1.1,
                          left=0.085, right=0.985, top=0.90, bottom=0.115)
    ax_a = fig.add_subplot(gs[0, 0:4])
    ax_b = fig.add_subplot(gs[0, 4:8])
    ax_e = fig.add_subplot(gs[0, 8:12])
    ax_c = fig.add_subplot(gs[1, 0:3])
    ax_d = fig.add_subplot(gs[1, 5:12])

    _panel_a(ax_a, D)
    _panel_b(ax_b, D)
    caps = _panel_e(ax_e)
    _panel_c(ax_c, D)
    _panel_d(ax_d, D)

    PS.save(fig, OUT)
    plt.close(fig)
    print(f"wrote {OUT}")
    print("panel e captures:", caps)


if __name__ == "__main__":
    build()

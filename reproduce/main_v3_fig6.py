#!/usr/bin/env python
"""Composite MAIN FIGURE (v3) — "Low-VAF in real tissue, and what pooling does and does not buy".

Single 2x2 composite that pairs the licensed-band clonal-hematopoiesis demonstration
with the honest pooling contrast (variance defeats shot noise; mean error does not
defeat the systematic artifact floor).

  (a) Licensed-band clonal hematopoiesis: two healthy marrow samples carry a k-stable
      clone inside the licensed 1-5% band; a healthy blood control is clean.
      PORTED from main_v2_fig2_5._f5_a_band (F5A constants), verbatim.
  (b) k-stability as the discriminator: the largest raw effect (6.32%) is confident at
      only 1/4 resolutions and is demoted; the 1.23% (3/4) and 1.99% (2/4) calls are kept.
      PORTED from main_v2_fig2_5._f5_b_kstability (F5B constants), verbatim.
  (c) Pooling reduces variance at nearly the ideal 1/N rate (median log-log slope, all
      sites climbing, none plateauing).
      PORTED from manuscript_data_figs.fig6_pooling_clone panel (a).
  (d) But against external WGS truth the mean error plateaus at a systematic bias floor
      (De Rop within-assay keeps falling as 1/sqrt(N); REH-vs-WGS flattens near ~2%).
      PORTED from manuscript_data_figs.fig6_pooling_clone panel (b).

Interpreter: python >= 3.12 (see environment.yml)
Reads committed data files read-only; writes only the one PNG.
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
import pubstyle as PS  # noqa: E402
PS.apply()

ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{ROOT}/data"
OUTDIR = f"{ROOT}/figures"
ANN = 6
LAB = 8

# ---------------------------------------------------------------------------
# Panel (a) data — licensed-band clonal-hematopoiesis (verified; from Fig-5a)
# ---------------------------------------------------------------------------
F5A = [
    dict(sample="BMMC_D6T1", tissue="marrow", vaf=1.23, variant="m.3943\nMT-ND1",
         kstab="3/4", clone=True),
    dict(sample="BMMC_D5T1", tissue="marrow", vaf=1.99, variant="m.11056\nMT-ND4",
         kstab="2/4", clone=True),
    dict(sample="PBMC_D12T3", tissue="blood", vaf=0.0, variant="clean",
         kstab="0/4", clone=False),
]

# ---------------------------------------------------------------------------
# Panel (b) data — k-stability discriminator (FINDINGS_phase15, Arm A)
# (k-cuts confident out of 4, between-subclone effect % VAF, retained?)
# ---------------------------------------------------------------------------
F5B = [
    dict(label="BMMC_D6T1 raw-max", k=1, vaf=6.32, retained=False, variant=None),
    dict(label="BMMC_D6T1 m.3943", k=3, vaf=1.23, retained=True, variant="m.3943 MT-ND1"),
    dict(label="BMMC_D5T1 raw-max", k=1, vaf=3.72, retained=False, variant=None),
    dict(label="BMMC_D5T1 m.11056", k=2, vaf=1.99, retained=True, variant="m.11056 MT-ND4"),
    dict(label="PBMC_D12T3", k=1, vaf=0.55, retained=False, variant=None),
]


def _a_band(ax):
    """PORT of main_v2_fig2_5._f5_a_band."""
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


def _b_kstability(ax):
    """PORT of main_v2_fig2_5._f5_b_kstability."""
    ymax = 7.2
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

    ax.annotate(F5B[1]["variant"], xy=(3, 1.23), xytext=(3.0, 2.15),
                ha="center", va="bottom", fontsize=ANN, color=PS.BLUE,
                arrowprops=dict(arrowstyle="-|>", color=PS.BLUE, lw=0.9))
    ax.annotate(F5B[3]["variant"], xy=(2, 1.99), xytext=(1.72, 3.15),
                ha="left", va="bottom", fontsize=ANN, color=PS.BLUE,
                arrowprops=dict(arrowstyle="-|>", color=PS.BLUE, lw=0.9))
    ax.annotate("largest raw effect\n6.32% at k=1\ndemoted (healthy ctrl)",
                xy=(1, 6.32), xytext=(1.35, 6.35),
                ha="left", va="top", fontsize=ANN, color=PS.INK,
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=0.9))

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


def _c_variance(ax):
    """PORT of manuscript_data_figs.fig6_pooling_clone panel (a): variance vs N (log-log)."""
    dr = pd.read_csv(f"{DATA}/derop_pooling.csv")
    Ns = sorted(dr["N"].unique())
    vtab = dr.pivot_table(index="pos", columns="N", values="var_pooled_vaf")
    med_var = vtab.median(axis=0).values
    lx = np.log(np.array(Ns, float))
    slope = float(np.polyfit(lx, np.log(med_var), 1)[0])
    climb = int((vtab[Ns].diff(axis=1).iloc[:, 1:] < 0).all(axis=1).sum())
    n_sites = vtab.shape[0]

    for _, row in vtab.iterrows():
        ax.plot(Ns, row[Ns].values, color=PS.GREY, lw=0.9, alpha=0.35, zorder=1)
    ax.plot(Ns, med_var, "-o", color=PS.RED, ms=3.5, lw=1.8, zorder=3,
            label=f"median over {n_sites} sites")
    ideal = med_var[0] * (Ns[0] / np.array(Ns, float))
    ax.plot(Ns, ideal, ls="--", lw=1.1, c=PS.INK, zorder=2, label="ideal 1/N (slope -1)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pool size N (cells)")
    ax.set_ylabel("variance of pooled VAF estimate")
    ax.set_title("Pooling defeats shot noise:\nvariance falls as 1/N")
    ax.text(0.03, 0.05, f"median slope = {slope:.2f}\n{climb}/{n_sites} climb, 0 plateau",
            transform=ax.transAxes, fontsize=ANN, color=PS.INK, va="bottom")
    ax.legend(loc="upper right", fontsize=6.2)
    ax.tick_params(labelsize=7)
    PS.panel(ax, "c", x=-0.22, y=1.14)
    return slope, climb, n_sites


def _d_error(ax):
    """PORT of manuscript_data_figs.fig6_pooling_clone panel (b): mean |pooled - truth| vs N."""
    dr = pd.read_csv(f"{DATA}/derop_pooling.csv")
    reh = pd.read_csv(f"{DATA}/reh_pooling.csv")
    Ns = sorted(dr["N"].unique())
    vtab = dr.pivot_table(index="pos", columns="N", values="var_pooled_vaf")
    n_sites = vtab.shape[0]

    dr_err = dr.groupby("N")["mean_abs_err"].median().loc[Ns].values * 100
    reh_err = reh.groupby("N")["mean_abs_err"].median().reindex(Ns).values * 100
    ax.plot(Ns, dr_err, "-o", color=PS.BLUE, ms=4, lw=1.5,
            label=f"De Rop ({n_sites} sites, within-assay)")
    ax.plot(Ns, reh_err, "-s", color=PS.GREEN, ms=4, lw=1.5,
            label="REH (3 sites, WGS reference)")
    ideal_b = dr_err[0] / np.sqrt(np.array(Ns, float) / Ns[0])
    ax.plot(Ns, ideal_b, ls="--", lw=1.1, c=PS.INK, label=r"ideal 1/$\sqrt{N}$")
    # mark the ~2% systematic floor the WGS-truth arm plateaus at
    floor = float(reh_err[-1])
    ax.axhline(floor, color=PS.RED, ls=":", lw=1.0, zorder=1)
    ax.text(Ns[-1], floor * 1.16, f"~{floor:.1f}% systematic\nbias floor (vs WGS)",
            fontsize=ANN, color=PS.RED, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pool size N (cells)")
    ax.set_ylabel("mean |pooled VAF - reference| (%)")
    ax.set_title("But not the artifact:\nWGS error plateaus")
    ax.legend(loc="upper right", fontsize=6.2)
    ax.tick_params(labelsize=7)
    PS.panel(ax, "d", x=-0.22, y=1.14)
    return floor


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    out = f"{OUTDIR}/main_v3_fig6_lowvaf_pooling.png"
    fig, axes = plt.subplots(2, 2, figsize=(PS.COL, 5.0))
    _a_band(axes[0, 0])
    _b_kstability(axes[0, 1])
    slope, climb, n_sites = _c_variance(axes[1, 0])
    floor = _d_error(axes[1, 1])
    if not PS.GB_SUB:   # GB: the overall title lives in the figure legend, not the graphic
        fig.suptitle("Low-VAF in real tissue, and what pooling does and does not buy",
                     y=1.01, fontsize=9.5, fontweight="bold")
    fig.tight_layout(h_pad=3.4, w_pad=2.8, rect=(0, 0, 1, 0.98 if not PS.GB_SUB else 1.0))
    PS.save(fig, out)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"panel c: median slope={slope:.3f}, climb={climb}/{n_sites}")
    print(f"panel d: WGS-truth plateau floor = {floor:.2f}%")


if __name__ == "__main__":
    main()

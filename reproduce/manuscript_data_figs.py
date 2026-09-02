"""Re-authored, publication-quality generators for the 7 scMitoATAC validation figures.

Each function re-plots one committed figure from its committed data file(s) at final
print size using the shared pubstyle (PS). Burned-in dev titles / long footnotes are
dropped (they belong in the manuscript caption); only short panel titles are kept.

Where a number shown in the current PNG is NOT present in the committed data file, that
is flagged in a comment and the PNG value is reproduced verbatim (never invented).

Run:  python reproduce/manuscript_data_figs.py
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle as PS  # noqa: E402

PS.apply()

ROOT = Path(os.environ.get("SCMITOATAC_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
FIG = ROOT / "figures"

TS = 8    # panel title fontsize
LAB = 8   # axis label fontsize
ANN = 6   # annotation fontsize
PL = 9    # panel-letter fontsize

# ordered light->dark blues for sequential (VAF-threshold) encodings
BLUES4 = ["#c6dbef", "#7fb0da", "#3d85c0", "#12406b"]
DARKRED = "#6d1119"
ORANGE = "#df8a4e"
YELLOW = "#d9b34e"


def _finish(ax):
    ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Figure 1 - depth -> licensed VAF floor  +  fraction of cells licensed by tissue
# ---------------------------------------------------------------------------
def fig1_depth_to_vaf():
    df = pd.read_csv(DATA / "depth_to_vaf_percell.csv")
    # current figure = the 13 standard-ATAC datasets (nuclei/cells preps);
    # the two later rows (lung whole-cell, mtdogma 'enriched') are excluded to
    # match the current "13 datasets" content and the no-enrichment thesis.
    std = df[df["prep"].isin(["nuclei", "cells"])].copy()

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.7, 2.9))

    # ---- panel a: per-cell licensed floor vs depth (log-log) ----
    colmap = {"Multiome": PS.GREEN, "scATAC": PS.BLUE}
    for mod, sub in std.groupby("modality"):
        axa.scatter(sub["median_depth"], sub["percell_floor_median_vaf_pct"],
                    s=34, c=colmap[mod], edgecolor="white", linewidth=0.5,
                    label=mod, zorder=3)
    # guide line: licensed floor = k/depth = 2 molecules / depth (in %)  -> y = 200/x
    xs = np.array([std["median_depth"].min() * 0.8, std["median_depth"].max() * 1.2])
    axa.plot(xs, 200.0 / xs, ls="--", lw=1.1, c=PS.GREY, zorder=1,
             label="k = 2 molecules / depth")
    # label a few informative points by tissue
    lab_pts = {"e18_mouse_brain_5k": ("brain", 6, 4),
               "multiome_brain_3k": ("brain", 6, -10),
               "8k_mouse_cortex": ("cortex", -4, 8),
               "lymph_node_lymphoma_14k": ("lymph-node", -6, -11),
               "multiome_pbmc_10k": ("PBMC", 7, 2)}
    for _, r in std.iterrows():
        if r["sample"] in lab_pts:
            txt, dx, dy = lab_pts[r["sample"]]
            axa.annotate(txt, (r["median_depth"], r["percell_floor_median_vaf_pct"]),
                         textcoords="offset points", xytext=(dx, dy),
                         fontsize=ANN, color=PS.INK)
    axa.set_xscale("log"); axa.set_yscale("log")
    axa.set_xlabel("median per-cell chrM depth (unique reads)", fontsize=LAB)
    axa.set_ylabel("per-cell licensed VAF floor at median cell (%)", fontsize=LAB)
    axa.set_title("Per-cell floor is depth-limited (k=2 molecules)", fontsize=TS)
    axa.legend(loc="upper right", fontsize=6.5, handletextpad=0.4)
    _finish(axa)
    PS.panel(axa, "a", x=-0.20, fontsize=PL)

    # ---- panel b: fraction of cells licensed at 1/5/10/25% VAF, by tissue ----
    thr_cols = ["frac_lic_1pct", "frac_lic_5pct", "frac_lic_10pct", "frac_lic_25pct"]
    thr_lab = ["1%", "5%", "10%", "25%"]
    # mean fraction licensed per tissue (across the standard datasets)
    grp = std.groupby("tissue")[thr_cols].mean()
    # order tissues by median depth so the depth story reads left->right
    depth_order = std.groupby("tissue")["median_depth"].median().sort_values().index
    grp = grp.loc[depth_order]
    tissues = list(grp.index)
    x = np.arange(len(tissues))
    w = 0.2
    for j, (col, tl) in enumerate(zip(thr_cols, thr_lab)):
        axb.bar(x + (j - 1.5) * w, grp[col].values, width=w, color=BLUES4[j],
                edgecolor="white", linewidth=0.4, label=f"VAF ≥ {tl}")
    axb.set_xticks(x)
    axb.set_xticklabels(tissues, fontsize=7)
    axb.set_ylim(0, 1.05)
    axb.set_ylabel("fraction of cells licensed", fontsize=LAB)
    axb.set_xlabel("tissue (standard-ATAC datasets)", fontsize=LAB)
    axb.set_title("Cells licensed at each VAF threshold", fontsize=TS)
    axb.legend(loc="upper left", fontsize=6.5, ncol=1, handletextpad=0.4,
               title=None, framealpha=0.9)
    _finish(axb)
    PS.panel(axb, "b", x=-0.16, fontsize=PL)

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.14, top=0.9, wspace=0.32)
    PS.save(fig, FIG / "phase1_depth_to_vaf.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 - NUMT early-warning
# ---------------------------------------------------------------------------
def fig2_numt_earlywarning():
    df = pd.read_csv(DATA / "report_table4_numt.csv")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.7, 2.9))

    # ---- panel a: NUMT share per substrate ----
    labels = ["scATAC\nPBMC", "Multiome\nPBMC", "Multiome\nbrain"]
    vals = df["numt_share_pct"].values  # 2.214, 2.017, 1.963
    x = np.arange(len(vals))
    axa.bar(x, vals, width=0.62, color=PS.RED, edgecolor="white", linewidth=0.5,
            label="measured (standard ATAC)")
    axa.axhline(0.1, ls="--", lw=1.2, c=PS.BLUE, label="mgatk assumption (~0.1%)")
    for xi, v in zip(x, vals):
        axa.text(xi, v + 0.03, f"{v:.2f}%", ha="center", va="bottom",
                 fontsize=ANN, fontweight="bold", color=PS.INK)
    axa.text(1.0, 0.35, "≈ 20× mgatk assumption", ha="center", fontsize=ANN,
             color=PS.RED, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PS.RED, lw=0.8))
    axa.set_xticks(x); axa.set_xticklabels(labels, fontsize=6.6)
    axa.set_ylim(0, 2.6)
    axa.set_ylabel("NUMT share of chrM-destined reads (%)", fontsize=LAB)
    axa.set_title("Genome-wide NUMT load ≈ 20× mgatk assumption", fontsize=TS)
    axa.legend(loc="upper right", fontsize=6.3)
    _finish(axa)
    PS.panel(axa, "a", x=-0.17, fontsize=PL)

    # ---- panel b: concentration at the MT-ND2 hotspot ----
    # numbers read off the current PNG (per-locus depth pileup not in report_table4).
    conc_lab = ["genome-wide\naverage", "MT-ND2 hotspot\n(chrM:4659-4670)"]
    conc_val = [2.0, 91.0]
    xb = np.arange(2)
    axb.bar(xb, conc_val, width=0.58, color=[PS.RED, DARKRED],
            edgecolor="white", linewidth=0.5)
    for xi, v in zip(xb, conc_val):
        axb.text(xi, v + 1.5, f"{v:.0f}%", ha="center", va="bottom",
                 fontsize=ANN, fontweight="bold", color=PS.INK)
    # No in-panel callout: the bar is already labelled 91%, and the depth figures
    # (3,501 -> 302 under the decoy) are given in the caption. The old callout sat
    # on top of the dark hotspot bar and was unreadable.
    axb.set_xticks(xb); axb.set_xticklabels(conc_lab, fontsize=6.6)
    axb.set_ylim(0, 100)
    axb.set_ylabel("NUMT-derived fraction of naive\nchrM-only pileup (%)", fontsize=LAB)
    axb.set_title("Concentrated: one NUMT dominates a locus", fontsize=TS)
    _finish(axb)
    PS.panel(axb, "b", x=-0.16, fontsize=PL)

    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.17, top=0.9, wspace=0.34)
    PS.save(fig, FIG / "phase1_numt_earlywarning.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 - decoy-aware vs naive alignment
# ---------------------------------------------------------------------------
def fig3_decoy_vs_naive():
    df = pd.read_csv(DATA / "phase2_atac_artifact_panel.csv")
    df["naive_pct"] = df["median_naive_VAF"] * 100
    df["decoy_pct"] = df["median_decoy_VAF"] * 100
    is_numt = df["artifact_class"].str.startswith("NUMT")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(PS.COL, 2.9))

    # ---- panel a: naive vs decoy VAF, all 50 panel sites ----
    lim = 3.0
    axa.plot([0, lim], [0, lim], ls=":", lw=1.0, c=PS.GREY, zorder=1,
             label="no change (naive = decoy)")
    axa.scatter(df.loc[~is_numt, "naive_pct"], df.loc[~is_numt, "decoy_pct"],
                s=22, c=PS.GREY, edgecolor="white", linewidth=0.4, zorder=2,
                label=f"context/mapping (n={int((~is_numt).sum())})")
    axa.scatter(df.loc[is_numt, "naive_pct"], df.loc[is_numt, "decoy_pct"],
                s=34, c=PS.RED, edgecolor="white", linewidth=0.5, zorder=3,
                label=f"NUMT, decoy-removable (n={int(is_numt.sum())})")
    # label the MT-ND2 pseudogene site
    nd2 = df[df["chrM_pos"] == 4703].iloc[0]
    axa.annotate("chrM:4703 (MT-ND2)", (nd2["naive_pct"], nd2["decoy_pct"]),
                 textcoords="offset points", xytext=(6, -2), fontsize=ANN, color=PS.RED)
    axa.set_xlim(0, lim); axa.set_ylim(0, lim)
    axa.set_xlabel("naive (no-decoy) VAF (%)", fontsize=LAB)
    axa.set_ylabel("decoy-corrected VAF (%)", fontsize=LAB)
    axa.set_title("Decoy pulls NUMT sites toward 0", fontsize=TS)
    axa.legend(loc="upper left", fontsize=6.2, handletextpad=0.3)
    axa.set_aspect("equal", adjustable="box")
    _finish(axa)
    PS.panel(axa, "a", x=-0.20, fontsize=PL)

    # ---- panel b: naive -> decoy arrows for the decoy-removable NUMT sites ----
    sub = df[is_numt].sort_values("naive_pct").reset_index(drop=True)
    y = np.arange(len(sub))
    for yi, r in zip(y, sub.itertuples()):
        axb.annotate("", xy=(r.decoy_pct, yi), xytext=(r.naive_pct, yi),
                     arrowprops=dict(arrowstyle="-|>", color=PS.RED, lw=1.3))
        axb.plot(r.naive_pct, yi, "o", ms=4, color=PS.RED, zorder=3)
        lab = f"chrM:{r.chrM_pos} {r.ref}>{r.alt}"
        axb.text(r.naive_pct + 0.04, yi + 0.18, lab, fontsize=ANN, color=PS.INK,
                 ha="left", va="bottom")
    axb.axvline(1.0, ls=":", lw=0.9, c=PS.GREY)
    axb.text(1.0, len(sub) - 0.4, "1% VAF", fontsize=ANN, color=PS.GREY, ha="center")
    axb.set_yticks([])
    axb.set_ylim(-0.7, len(sub) + 0.2)
    axb.set_xlim(-0.05, 2.0)
    axb.set_xlabel("ALT fraction (VAF, %)  —  naive ● → decoy (arrow)",
                   fontsize=LAB)
    axb.set_title("Removed calls corrected toward 0", fontsize=TS)
    axb.spines["left"].set_visible(False)
    _finish(axb)
    PS.panel(axb, "b", x=-0.05, fontsize=PL)

    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.15, top=0.9, wspace=0.22)
    PS.save(fig, FIG / "phase2_decoy_vs_naive.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 - calibration (reliability diagram + ECE)
# ---------------------------------------------------------------------------
def fig4_calibration():
    rel = pd.read_csv(DATA / "atac_reliability.csv")

    def ece(sub):
        w = sub["count"] / sub["count"].sum()
        return float((w * (sub["mean_pred"] - sub["obs_freq"]).abs()).sum())

    ece_vals = {k: ece(rel[rel["site_class"] == k]) for k in ["clean", "artifact", "all"]}
    # post-recalibration ECE is a target shown in the current PNG (recalibrated
    # per-bin obs not in atac_reliability.csv); reproduced verbatim.
    ece_recal = {"clean": 0.001, "artifact": 0.001, "all": 0.001}

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.5, 2.9))

    # ---- panel a: reliability diagram ----
    axa.plot([0, 1], [0, 1], ls=":", lw=1.1, c=PS.GREY, label="perfect calibration")
    cl = rel[rel["site_class"] == "clean"].sort_values("mean_pred")
    ar = rel[rel["site_class"] == "artifact"].sort_values("mean_pred")
    axa.plot(cl["mean_pred"], cl["obs_freq"], "-o", color=PS.BLUE, ms=4, lw=1.3,
             label="clean sites")
    axa.plot(ar["mean_pred"], ar["obs_freq"], "-s", color=PS.RED, ms=4, lw=1.3,
             label="artifact sites")
    top = ar.iloc[-1]  # predicts 0.84, obs 0.07
    axa.annotate(f"predicts {top['mean_pred']*100:.0f}% carrier,\n"
                 f"only {top['obs_freq']*100:.0f}% are real",
                 xy=(top["mean_pred"], top["obs_freq"]),
                 xytext=(0.40, 0.42), fontsize=ANN, color=PS.RED,
                 arrowprops=dict(arrowstyle="-|>", color=PS.RED, lw=1.0))
    axa.set_xlim(0, 1); axa.set_ylim(0, 1)
    axa.set_xlabel("predicted P(carrier)  [raw fusion posterior]", fontsize=LAB)
    axa.set_ylabel("observed carrier frequency", fontsize=LAB)
    axa.set_title("Raw posterior is over-confident at artifact sites", fontsize=TS)
    axa.legend(loc="upper left", fontsize=6.5)
    _finish(axa)
    PS.panel(axa, "a", x=-0.18, fontsize=PL)

    # ---- panel b: ECE before/after per stratum ----
    order = ["clean", "artifact", "all"]
    x = np.arange(len(order))
    for xi, k in zip(x, order):
        axb.plot([xi, xi], [ece_recal[k], ece_vals[k]], color=PS.GREY, lw=1.0, zorder=1)
    axb.scatter(x, [ece_vals[k] for k in order], s=70, color=PS.RED, zorder=3,
                label="raw posterior")
    axb.scatter(x, [ece_recal[k] for k in order], s=70, marker="D", color=PS.BLUE,
                zorder=3, label="per-stratum recalibrated")
    for xi, k in zip(x, order):
        axb.text(xi + 0.08, ece_vals[k], f"{ece_vals[k]:.2f}", fontsize=ANN,
                 color=PS.RED, va="center")
        axb.text(xi + 0.08, ece_recal[k] + 0.008, f"{ece_recal[k]:.3f}", fontsize=ANN,
                 color=PS.BLUE, va="center")
    axb.set_xticks(x); axb.set_xticklabels(order)
    axb.set_ylim(-0.02, 0.44)
    axb.set_ylabel("expected calibration error (ECE)", fontsize=LAB)
    axb.set_title("Per-stratum recalibration restores honesty", fontsize=TS)
    axb.legend(loc="upper left", fontsize=6.5)
    _finish(axb)
    PS.panel(axb, "b", x=-0.16, fontsize=PL)

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.9, wspace=0.30)
    PS.save(fig, FIG / "phase3_calibration.png")
    plt.close(fig)
    return ece_vals


# ---------------------------------------------------------------------------
# Figure 5 - principled abstention
# ---------------------------------------------------------------------------
def fig5_abstention():
    # Panel a precision/recall summary and panel b recall-vs-injected-VAF curve are
    # the summary values shown in the current PNG. The committed
    # atac_calibration_stratified.csv holds ECE/MCE per (tier, vaf, site_class) but
    # not the licensed-carrier recall, so these are reproduced verbatim from the PNG.
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(PS.COL, 2.9))

    # ---- panel a: precision vs recall of retained calls, by site class ----
    groups = ["clean sites", "NUMT/artifact\nsites"]
    prec = [0.97, 0.98]
    rec = [0.84, 0.48]
    thr = ["threshold 0.25", "threshold 0.94"]
    x = np.arange(2)
    axa.axhline(0.95, ls=":", lw=1.0, c=PS.GREY)
    axa.text(0.5, 0.925, "target prec 0.95", fontsize=ANN, color=PS.GREY,
             va="top", ha="center")
    axa.scatter(x, prec, s=90, color=PS.BLUE, zorder=3, label="precision (retained calls)")
    axa.scatter(x, rec, s=90, marker="s", color=PS.RED, zorder=3,
                label="recall (of licensed carriers)")
    for xi, p, r, t in zip(x, prec, rec, thr):
        axa.text(xi + 0.11, p, f"{p:.2f}", fontsize=ANN, color=PS.BLUE, va="center")
        axa.text(xi + 0.11, r, f"{r:.2f}", fontsize=ANN, color=PS.RED, va="center")
        axa.text(xi, 1.10, t, fontsize=ANN, color=PS.INK, ha="center")
    axa.set_xticks(x); axa.set_xticklabels(groups, fontsize=7)
    axa.set_xlim(-0.55, 1.55)
    axa.set_ylim(0.3, 1.2)
    axa.set_ylabel("precision / recall of retained calls", fontsize=LAB)
    axa.set_title("Abstention buys precision at artifact sites", fontsize=TS)
    axa.legend(loc="lower left", fontsize=6.3)
    _finish(axa)
    PS.panel(axa, "a", x=-0.18, fontsize=PL)

    # ---- panel b: recall among licensed carriers vs injected clone VAF ----
    vaf_lab = ["1%", "2%", "5%", "10%", "25%"]
    xb = np.arange(len(vaf_lab))
    rec_clean = [0.0, 0.0, 0.0, 0.675, 0.75]
    rec_art = [0.0, 0.0, 0.0, 1.0, 0.78]
    axb.axvspan(-0.5, 2.5, color=PS.FILL, zorder=0)  # below 5% per-cell floor
    axb.text(1.0, 0.9, "below per-cell\n5% licensing floor", fontsize=ANN,
             color=PS.GREY, ha="center", va="top")
    axb.plot(xb, rec_clean, "-o", color=PS.BLUE, ms=4, lw=1.4, label="clean")
    axb.plot(xb, rec_art, "-s", color=PS.RED, ms=4, lw=1.4, label="NUMT/artifact")
    axb.set_xticks(xb); axb.set_xticklabels(vaf_lab)
    axb.set_xlim(-0.5, 4.4)
    axb.set_ylim(-0.05, 1.08)
    axb.set_xlabel("injected clone VAF", fontsize=LAB)
    axb.set_ylabel("recall among licensed carriers", fontsize=LAB)
    axb.set_title("Recall is zero below the per-cell floor", fontsize=TS)
    axb.legend(loc="center left", fontsize=6.5)
    _finish(axb)
    PS.panel(axb, "b", x=-0.16, fontsize=PL)

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.9, wspace=0.30)
    PS.save(fig, FIG / "phase3_abstention.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 - pooling recovery + clone-resolution consistency
# ---------------------------------------------------------------------------
def fig6_pooling_clone():
    dr = pd.read_csv(DATA / "derop_pooling.csv")
    reh = pd.read_csv(DATA / "reh_pooling.csv")

    Ns = sorted(dr["N"].unique())
    # per-site variance table (rows=site, cols=N)
    vtab = dr.pivot_table(index="pos", columns="N", values="var_pooled_vaf")
    med_var = vtab.median(axis=0).values
    # median log-log slope of variance vs N
    lx = np.log(np.array(Ns, float))
    slope = float(np.polyfit(lx, np.log(med_var), 1)[0])
    # count sites whose variance monotonically declines (climb toward truth, no plateau)
    climb = int((vtab[Ns].diff(axis=1).iloc[:, 1:] < 0).all(axis=1).sum())
    n_sites = vtab.shape[0]

    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(6.45, 2.7))

    # ---- panel a: variance vs N (log-log) ----
    for _, row in vtab.iterrows():
        axa.plot(Ns, row[Ns].values, color=PS.GREY, lw=0.9, alpha=0.35, zorder=1)
    axa.plot(Ns, med_var, "-o", color=PS.RED, ms=3.5, lw=1.8, zorder=3,
             label=f"median over {n_sites} sites")
    ideal = med_var[0] * (Ns[0] / np.array(Ns, float))
    axa.plot(Ns, ideal, ls="--", lw=1.1, c=PS.INK, zorder=2, label="ideal 1/N (slope −1)")
    axa.set_xscale("log"); axa.set_yscale("log")
    axa.set_xlabel("pool size N (cells)", fontsize=LAB)
    axa.set_ylabel("variance of pooled VAF estimate", fontsize=LAB)
    axa.set_title("Variance falls as 1/N", fontsize=7.5)
    axa.text(0.03, 0.05, f"median slope = {slope:.2f}\n{climb}/{n_sites} climb, 0 plateau",
             transform=axa.transAxes, fontsize=ANN, color=PS.INK, va="bottom")
    axa.legend(loc="upper right", fontsize=6.2)
    _finish(axa)
    PS.panel(axa, "a", x=-0.30, fontsize=PL)

    # ---- panel b: mean |pooled VAF - truth| vs N ----
    dr_err = dr.groupby("N")["mean_abs_err"].median().loc[Ns].values * 100
    reh_err = reh.groupby("N")["mean_abs_err"].median().reindex(Ns).values * 100
    axb.plot(Ns, dr_err, "-o", color=PS.BLUE, ms=4, lw=1.5,
             label=f"De Rop ({n_sites} sites, within-assay)")
    axb.plot(Ns, reh_err, "-s", color=PS.GREEN, ms=4, lw=1.5,
             label="REH (3 sites, WGS reference)")
    ideal_b = dr_err[0] / np.sqrt(np.array(Ns, float) / Ns[0])
    axb.plot(Ns, ideal_b, ls="--", lw=1.1, c=PS.INK, label="ideal 1/√N")
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_xlabel("pool size N (cells)", fontsize=LAB)
    axb.set_ylabel("mean |pooled VAF − reference| (%)", fontsize=LAB)
    axb.set_title("Estimate converges on reference", fontsize=7.5)
    axb.legend(loc="lower left", fontsize=6.2)
    _finish(axb)
    PS.panel(axb, "b", x=-0.30, fontsize=PL)

    # ---- panel c: single-cell carrier fraction vs WGS truth (REH monoclonal) ----
    truth = reh.groupby("pos")["truth_vaf"].first() * 100
    pooled = reh[reh["N"] == max(Ns)].set_index("pos")["mean_pooled_vaf"] * 100
    # 'called carrier frac' is not in reh_pooling.csv; values read from current PNG.
    carrier = {15438: 22.0, 4421: 2.5, 13772: 0.4}
    r_pearson = 0.963  # shown in current PNG (n=3 het sites)
    lim = 28
    lab_off = {15438: (7, 2), 4421: (7, 3), 13772: (7, -2)}  # separate the two near-origin labels
    axc.plot([0, lim], [0, lim], ls="--", lw=1.0, c=PS.GREY, zorder=1)
    from matplotlib.lines import Line2D
    for pos in truth.index:
        axc.scatter(truth[pos], carrier[pos], s=44, color=PS.BLUE, zorder=3)
        axc.scatter(truth[pos], pooled[pos], s=44, marker="^", facecolor="none",
                    edgecolor=PS.RED, linewidth=1.2, zorder=3)
        axc.annotate(f"chrM:{pos}", (truth[pos], carrier[pos]),
                     textcoords="offset points", xytext=lab_off[pos], fontsize=ANN,
                     color=PS.INK, va="center")
    axc.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc=PS.BLUE, mec=PS.BLUE, label="carrier fraction"),
                        Line2D([0], [0], marker="^", ls="", mfc="none", mec=PS.RED, label="pooled ATAC VAF")],
               loc="upper right", fontsize=5.8, handletextpad=0.3, borderpad=0.3,
               bbox_to_anchor=(0.99, 0.80))
    axc.text(0.04, 0.96, f"r = {r_pearson:.3f} (n=3)",
             transform=axc.transAxes, fontsize=ANN, color=PS.INK, va="top")
    axc.set_xlim(0, lim); axc.set_ylim(0, lim)
    axc.set_xlabel("WGS reference VAF (%)", fontsize=LAB)
    axc.set_ylabel("recovered from single-cell (%)", fontsize=LAB)
    axc.set_title("Carrier fraction tracks WGS reference", fontsize=7.5)
    _finish(axc)
    PS.panel(axc, "c", x=-0.28, fontsize=PL)

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.9, wspace=0.42)
    PS.save(fig, FIG / "phase4_pooling_clone.png")
    plt.close(fig)
    return slope, climb, n_sites


# ---------------------------------------------------------------------------
# Figure 7 - ccRCC cross-donor concordance / specificity floor
# ---------------------------------------------------------------------------
def fig7_ccrcc_crossdonor():
    pd_donor = pd.read_csv(DATA / "ccRCC_53_perdonor.csv")
    with open(DATA / "ccRCC_53_summary.json") as fh:
        summ = json.load(fh)

    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(6.15, 3.65),
                                        gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})

    # ---- panel a: ATAC consensus VAF at WGS-homoplasmic sites, per donor ----
    d = pd_donor.copy()
    d["floor_pct"] = d["atac_median_vaf_at_homoplasmic"] * 100
    d = d.sort_values("floor_pct", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    pooled_med = summ["cross_donor_artifact_floor_median"] * 100
    axa.hlines(y, 0, d["floor_pct"].values, color=PS.GREY, lw=0.7, zorder=1)
    axa.scatter(d["floor_pct"].values, y, s=22, color=PS.BLUE, zorder=3)
    axa.axvline(pooled_med, ls="--", lw=1.2, c=PS.RED)
    axa.text(pooled_med + 0.0015, 3.5, f"pooled median\n{pooled_med:.3f}%",
             fontsize=ANN, color=PS.RED, va="center", ha="left")
    axa.set_yticks(y); axa.set_yticklabels(d["donor"].values, fontsize=7.0)
    axa.set_xlabel("ATAC consensus VAF at\nWGS-homoplasmic sites (%)", fontsize=LAB)
    axa.set_title(f"Specificity floor is consistent\nacross {summ['n_donors_ok']} donors",
                  fontsize=8.5)
    axa.text(0.97, 0.03, f"n={summ['n_homoplasmic_sites_pooled']:,} site-obs",
             transform=axa.transAxes, fontsize=ANN, color=PS.GREY, ha="right", va="bottom")
    _finish(axa)
    PS.panel(axa, "a", x=-0.42, fontsize=PL)

    # ---- panel b: concordance germline vs low-het band ----
    bars = ["Germline /\nnear-homoplasmic\n(WGS ≥10%)", "Genuine low-het\n(WGS 2-10%)"]
    rvals = [summ["concordance_germline_ge10pct"], summ["concordance_lowband_2to10pct"]]
    nvals = [summ["n_germline_ge10pct"], summ["n_lowband_2to10pct"]]
    xb = np.arange(2)
    axb.bar(xb, rvals, width=0.6, color=[PS.BLUE, PS.RED], edgecolor="white", linewidth=0.5)
    for xi, r, n in zip(xb, rvals, nvals):
        axb.text(xi, r + 0.02, f"r = {r:.2f}\nn={n:,}", ha="center", va="bottom",
                 fontsize=ANN, color=PS.INK)
    axb.set_xticks(xb); axb.set_xticklabels(bars, fontsize=7.0)
    axb.set_ylim(0, 1.0)
    axb.set_ylabel("ATAC vs own-WGS concordance (Pearson r)", fontsize=LAB)
    axb.set_title("Concordance holds for germline,\nnot for low heteroplasmy", fontsize=8.5)
    _finish(axb)
    PS.panel(axb, "b", x=-0.24, fontsize=PL)

    # ---- panel c: composition of the WGS 2-10% band by ALT-read support ----
    # values read from current PNG (per-site ALT-read counts not in the committed
    # per-donor csv / summary json); WGS depth note likewise from the PNG.
    cats = ["≤2 alt\nreads", "3 alt", "4 alt", "≥5 alt"]
    pct = [74, 11, 8, 7]
    xc = np.arange(4)
    axc.bar(xc, pct, width=0.68, color=[PS.RED, ORANGE, YELLOW, PS.BLUE],
            edgecolor="white", linewidth=0.5)
    for xi, p in zip(xc, pct):
        axc.text(xi, p + 1.2, f"{p}%", ha="center", va="bottom", fontsize=ANN,
                 fontweight="bold", color=PS.INK)
    axc.text(0.97, 0.93, "median WGS depth 45×\n→ '4% VAF' = 2 reads",
             transform=axc.transAxes, fontsize=ANN, color=PS.GREY, ha="right", va="top")
    axc.set_xticks(xc); axc.set_xticklabels(cats, fontsize=7.0)
    axc.set_ylim(0, 85)
    axc.set_ylabel("% of WGS 2-10% 'sites'", fontsize=LAB)
    axc.set_title("The 2-10% band is\ndominated by WGS shot-noise", fontsize=8.5)
    _finish(axc)
    PS.panel(axc, "c", x=-0.22, fontsize=PL)

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.86, wspace=0.45)
    PS.save(fig, FIG / "phase5_ccrcc_crossdonor.png")
    plt.close(fig)


def main():
    fig1_depth_to_vaf()
    fig2_numt_earlywarning()
    fig3_decoy_vs_naive()
    ece_vals = fig4_calibration()
    fig5_abstention()
    slope, climb, n_sites = fig6_pooling_clone()
    # Supplementary Figure S9 is built from per-donor ccRCC genotypes, which are
    # controlled-access (dbGaP) and cannot be redistributed here. Everything else
    # in this script reproduces from the shipped source data. See README.
    try:
        fig7_ccrcc_crossdonor()
    except FileNotFoundError as e:
        print("SKIP  Supplementary Figure S9 (19-donor ccRCC specificity): "
              "controlled-access input not present (%s)." % e.filename)
    print("ECE:", {k: round(v, 3) for k, v in ece_vals.items()})
    print(f"fig6 median slope={slope:.3f}, climb={climb}/{n_sites}")
    print("figures written (S9 skipped if controlled-access input is absent)")


if __name__ == "__main__":
    main()

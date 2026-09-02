#!/usr/bin/env python
"""Manuscript rebuttal figure vs scmtMPM's coverage control (BAM downsampling).
Three panels, AL10 clone position m.9973, within the malignant compartment:
  A. The confound: pooled metacell VAF rises with coverage (read-weighted pooling).
  B. Downsampling to a common depth (scmtMPM's normalization) does NOT remove it -- on the SAME metacells, the
     VAF~depth correlation is essentially unchanged after rarefaction; it only "drops" where n collapses.
  C. Our fix: the coverage-stratified null's positivity mass (sfrac) abstains where clusters are coverage-disjoint
     (tumor-vs-normal) and licenses coverage-overlapping partitions (CNV-subclone, cell-type).
Palette: ColorBrewer RdBu endpoints, validated CVD-safe (blue #2166ac = licensed, red #b2182b = abstain/highlight);
grey #8c8c8c is neutral ink. Secondary encoding (marker shape) in C so identity is never colour-alone.
Usage: luad_covctrl_rebuttal_fig.py
Out: docs/reports/figures/manuscript/coverage_control_rebuttal.png
general env: numpy, pandas, scipy, sklearn, matplotlib.
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle as PS
PS.apply()

ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONS = ["cons_A", "cons_C", "cons_G", "cons_T"]; MALIG = "tumor_aneuploid"; POS = 9973; TOK = "LUAD_AL10"
BLUE, RED, GREY, INK = PS.BLUE, PS.RED, PS.GREY, PS.INK
rng = np.random.default_rng(0)


def malignant_metacells():
    sc = pd.read_csv(f"{ROOT}/work/luad_dd/{TOK}/{TOK}_cellscores.tsv.gz", sep="\t")
    sc["compartment"] = sc["compartment"].fillna("unknown").astype(str)
    pcs = [c for c in sc.columns if c.startswith("PC")]
    sc["mc"] = KMeans(n_clusters=max(2, round(len(sc) / 40)), n_init=4, random_state=0).fit_predict(sc[pcs].to_numpy())
    bc2mc = dict(zip(sc.barcode, sc.mc))
    mfrac = sc.groupby("mc").compartment.agg(lambda s: (s == MALIG).mean())   # metacell malignant-cell fraction
    malig = set(mfrac[mfrac >= 0.5].index)                                    # dominant-malignant (matches covnull)
    acc = {}
    for chk in pd.read_csv(f"{ROOT}/work/phase16_cancer_scan/human/mtdna/{TOK}/{TOK}_decoy_fragpileup.tsv.gz",
                           sep="\t", usecols=["cell_barcode", "pos"] + CONS, chunksize=2_000_000):
        chk = chk[(chk.pos == POS) & chk.cell_barcode.isin(bc2mc)]
        if not len(chk):
            continue
        chk["mc"] = chk.cell_barcode.map(bc2mc)
        for mc, sub in chk.groupby("mc"):
            acc[mc] = acc.get(mc, np.zeros(4)) + sub[CONS].to_numpy().sum(axis=0)
    rows = [(m, c.sum(), 1 - c.max() / c.sum(), c) for m, c in acc.items() if m in malig and c.sum() > 0]
    depth = np.array([r[1] for r in rows]); vaf = np.array([r[2] for r in rows]); counts = [r[3] for r in rows]
    return depth.astype(int), vaf, counts


def rarefy(counts, depth, D, ndraws=40):
    idx = np.where(depth >= D)[0]
    vr = np.array([np.mean(1 - rng.multinomial(D, counts[i] / counts[i].sum(), size=ndraws).max(1) / D) for i in idx])
    return idx, vr


def main():
    depth, vaf, counts = malignant_metacells()
    keepA = depth >= 10
    dA, vA = depth[keepA], vaf[keepA]
    rA, pA = spearmanr(vA, dA)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(PS.COL, 2.55))

    # ---- Panel A: the confound ----
    axA.scatter(dA, vA, s=13, c=GREY, alpha=0.7, edgecolors="white", linewidths=0.3)
    z = np.polyfit(dA, vA, 1); xs = np.linspace(dA.min(), dA.max(), 20)
    axA.plot(xs, np.polyval(z, xs), color=INK, lw=1.6)
    axA.text(0.04, 0.96, f"Spearman $\\rho$ = {rA:.2f}\np = {pA:.0e},  n = {len(dA)}", transform=axA.transAxes,
             va="top", ha="left", fontsize=6.2, bbox=dict(boxstyle="round", fc="white", ec="#ddd"))
    axA.set_xlabel("metacell chrM depth at m.9973"); axA.set_ylabel("metacell VAF")
    axA.set_title("VAF rises with coverage", fontsize=8)

    # ---- Panel B: downsampling does not remove it (grouped orig vs rarefied, same metacells) ----
    Ds = [30, 40, 50]; x = np.arange(len(Ds)); w = 0.38
    orig_r, rare_r, ns = [], [], []
    for D in Ds:
        idx, vr = rarefy(counts, depth, D)
        orig_r.append(spearmanr(vaf[idx], depth[idx])[0]); rare_r.append(spearmanr(vr, depth[idx])[0]); ns.append(len(idx))
    axB.bar(x - w / 2, orig_r, w, color=GREY, label="original VAF", edgecolor="white")
    axB.bar(x + w / 2, rare_r, w, color=RED, label="downsampled to D", edgecolor="white",
            hatch=["", "", "///"])                       # D=50 hatched = underpowered
    axB.axhline(0, color=INK, lw=0.9)
    axB.text(0.5, 0.02, r"$\rho$ = 0 means confound removed", transform=axB.transAxes,
             ha="center", va="bottom", fontsize=5.8, color=INK, style="italic")
    for xi, (o, r, n) in enumerate(zip(orig_r, rare_r, ns)):
        axB.text(xi, max(o, r) + 0.03, f"n={n}" + ("\n(underpow.)" if Ds[xi] >= 50 else ""),
                 ha="center", va="bottom", fontsize=5.6, color=INK)
    axB.set_xticks(x); axB.set_xticklabels([f"D={d}" for d in Ds]); axB.set_ylim(-0.05, 0.78)
    axB.set_ylabel(r"$\rho$( VAF , original depth )"); axB.legend(fontsize=6, loc="upper right")
    axB.set_title("Downsampling leaves it intact", fontsize=8)

    # ---- Panel C: positivity gate abstains on coverage-disjoint clusters ----
    cov = pd.read_csv(f"{ROOT}/work/luad_dd/luad_dd_covnull.tsv", sep="\t")
    order = ["tumorNormal", "cnvSubclone", "cellType"]
    lab = {"tumorNormal": "tumor-vs-\nnormal", "cnvSubclone": "CNV\nsubclone", "cellType": "cell\ntype"}
    cov = cov[cov.axis.isin(order)].copy()
    xmap = {a: i for i, a in enumerate(order)}
    for _, r in cov.iterrows():
        xi = xmap[r.axis] + rng.uniform(-0.16, 0.16)
        idn = r.sfrac >= 0.5                                  # positivity gate: >=0.5 identifiable, else abstain
        axC.scatter(xi, r.sfrac, s=30, c=(BLUE if idn else RED), marker=("o" if idn else "X"),
                    edgecolors="white", linewidths=0.4, alpha=0.9, zorder=3)
    axC.axhline(0.5, ls="--", color=INK, lw=1.0)
    axC.text(len(order) - 0.5, 0.52, "positivity threshold", ha="right", va="bottom", fontsize=6, color=INK)
    axC.annotate("coverage-disjoint\n→ abstain", xy=(0, cov[cov.axis == "tumorNormal"].sfrac.median()),
                 xytext=(0.2, 0.66), fontsize=6, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.7))
    axC.set_xticks(range(len(order))); axC.set_xticklabels([lab[a] for a in order], fontsize=6.5)
    axC.set_ylim(0, 1.05); axC.set_ylabel("coverage-overlap mass (sfrac)")
    from matplotlib.lines import Line2D
    axC.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc=BLUE, mec="white", label="identifiable"),
                        Line2D([0], [0], marker="X", ls="", mfc=RED, mec="white", label="abstain")],
               fontsize=6, loc="lower right")
    axC.set_title("Positivity gate abstains", fontsize=8)

    for ax, L in ((axA, "a"), (axB, "b"), (axC, "c")):
        PS.panel(ax, L)
    fig.tight_layout()
    out = f"{ROOT}/figures"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    PS.save(fig, out); plt.close(fig)
    print("wrote", out)
    print(f"A: rho={rA:.2f} n={len(dA)} | B: Ds={Ds} orig={[round(o,2) for o in orig_r]} "
          f"rare={[round(r,2) for r in rare_r]} n={ns} | C: {len(cov)} candidates")


if __name__ == "__main__":
    main()

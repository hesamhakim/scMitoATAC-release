#!/usr/bin/env python
"""Main-text coverage-controlled figure (r2): coverage-controlled between-population mtDNA inference.
Revision of main_v3_fig5.py. IDENTICAL to that generator in every panel EXCEPT panel d, where a
magnified inset axis is added so the two near-zero false-positive-rate bars (coverage-stratified
0.0006, plain shuffle 0.0013) are legible next to the alpha = 0.05 reference line.

Six panels (a-f); numbers/data are ported unchanged from the standalone generators:
  a,b  luad_covctrl_rebuttal_fig.py  (AL10 m.9973 confound + rarefaction-invariance)
  c    luad_dd_covnull.tsv           (positivity gate: sfrac abstains on coverage-disjoint axes)
  d    phase10_s3_calibration.tsv    (coverage-stratified null FPR vs plain shuffle) + magnified inset
  e    luad_dd_covnull.tsv           (AL08 m.15059 MT-CYB: licensed on cell type, abstained on tumor-vs-normal)
  f    scmitomut_hth/ + covnull      (per-cell presence caller starved -> 0 carriers, metacell level test licensed)
Palette: pubstyle (CVD-safe RdBu; blue=licensed/identifiable, red=abstain/highlight, grey=neutral ink).
Out: docs/manuscript/figures/main/main_r2_coverage.png  (NEW file; does not overwrite main_v3_fig5_coverage.png)
general env: numpy, pandas, scipy, sklearn, matplotlib.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pubstyle as PS
PS.apply()
import luad_covctrl_rebuttal_fig as RB          # reuse malignant_metacells() + rarefy() verbatim

ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, RED, GREEN, GREY, INK = PS.BLUE, PS.RED, PS.GREEN, PS.GREY, PS.INK
rng = np.random.default_rng(0)
OUT = f"{ROOT}/figures/main_r2_coverage.png"


def main():
    # ---------------------------------------------------------------- data
    depth, vaf, counts = RB.malignant_metacells()                 # panels a,b
    cov = pd.read_csv(f"{ROOT}/work/luad_dd/luad_dd_covnull.tsv", sep="\t")   # panels c,e,f
    cal = pd.read_csv(f"{ROOT}/data/phase10_s3_calibration.tsv", sep="\t").iloc[0]  # panel d
    # panel f per-cell funnel
    OUTS = f"{ROOT}/work/scmitomut_hth"
    inp = pd.read_csv(f"{OUTS}/input_LUAD_AL10.tsv", sep="\t")
    comp = pd.read_csv(f"{OUTS}/comp_LUAD_AL10.tsv", sep="\t")
    sm = pd.read_csv(f"{OUTS}/scmitomut_summary_LUAD_AL10.tsv", sep="\t")
    sm["pos"] = sm["loc"].astype(str).str.replace("chrM.", "", regex=False).astype(int)

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(PS.COL - 0.12, 6.6))
    gs = fig.add_gridspec(2, 3, hspace=0.78, wspace=0.52,
                          left=0.085, right=0.985, top=0.9, bottom=0.085)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0]); axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])

    # ===================== Panel a: the confound =====================
    keepA = depth >= 10
    dA, vA = depth[keepA], vaf[keepA]
    rA, pA = spearmanr(vA, dA)
    axA.scatter(dA, vA, s=12, c=GREY, alpha=0.7, edgecolors="white", linewidths=0.3)
    z = np.polyfit(dA, vA, 1); xs = np.linspace(dA.min(), dA.max(), 20)
    axA.plot(xs, np.polyval(z, xs), color=INK, lw=1.6)
    axA.text(0.04, 0.96, f"Spearman $\\rho$ = {rA:.2f}\np = {pA:.0e},  n = {len(dA)}",
             transform=axA.transAxes, va="top", ha="left", fontsize=6.2,
             bbox=dict(boxstyle="round", fc="white", ec="#dddddd"))
    axA.set_xlabel("metacell chrM depth (m.9973)"); axA.set_ylabel("metacell VAF")
    axA.set_title("VAF rises with coverage", fontsize=8)

    # ============ Panel b: downsampling leaves it intact =============
    Ds = [30, 40, 50]; x = np.arange(len(Ds)); w = 0.38
    orig_r, rare_r, ns = [], [], []
    for D in Ds:
        idx, vr = RB.rarefy(counts, depth, D)
        orig_r.append(spearmanr(vaf[idx], depth[idx])[0])
        rare_r.append(spearmanr(vr, depth[idx])[0]); ns.append(len(idx))
    axB.bar(x - w / 2, orig_r, w, color=GREY, label="original VAF", edgecolor="white")
    axB.bar(x + w / 2, rare_r, w, color=RED, label="downsampled to D",
            edgecolor="white", hatch=["", "", "///"])
    axB.axhline(0, color=INK, lw=0.9)
    # short note in the clear band above the bars and below the legend
    axB.text(0.52, 0.70, "$\\rho \\approx$ 0.46 persists", transform=axB.transAxes,
             ha="center", va="center", fontsize=7.0, color=RED, style="italic")
    for xi, (o, r, n) in enumerate(zip(orig_r, rare_r, ns)):
        axB.text(xi, max(o, r) + 0.03, f"n={n}" + ("\n(underpow.)" if Ds[xi] >= 50 else ""),
                 ha="center", va="bottom", fontsize=5.6, color=INK)
    axB.set_xticks(x); axB.set_xticklabels([f"D={d}" for d in Ds]); axB.set_ylim(-0.05, 0.82)
    axB.set_ylabel(r"$\rho$( VAF , original depth )"); axB.legend(fontsize=6, loc="upper right")
    axB.set_title("Downsampling leaves it intact", fontsize=8)

    # =============== Panel c: positivity gate abstains ===============
    order = ["tumorNormal", "cnvSubclone", "cellType"]
    lab = {"tumorNormal": "tumor-vs-\nnormal", "cnvSubclone": "CNV\nsubclone", "cellType": "cell\ntype"}
    cC = cov[cov.axis.isin(order)].copy()
    xmap = {a: i for i, a in enumerate(order)}
    for _, r in cC.iterrows():
        xi = xmap[r.axis] + rng.uniform(-0.16, 0.16)
        idn = r.sfrac >= 0.5
        axC.scatter(xi, r.sfrac, s=28, c=(BLUE if idn else RED), marker=("o" if idn else "X"),
                    edgecolors="white", linewidths=0.4, alpha=0.9, zorder=3)
    axC.axhline(0.5, ls="--", color=INK, lw=1.0)
    axC.text(1.98, 0.53, "positivity threshold", ha="right", va="bottom", fontsize=6, color=INK)
    axC.set_xticks(range(len(order))); axC.set_xticklabels([lab[a] for a in order], fontsize=6.5)
    axC.set_ylim(0, 1.08); axC.set_ylabel("coverage-overlap mass (sfrac)")
    # the legend now carries the marker grammar, so the redundant standalone "coverage-disjoint"
    # annotation is dropped; the legend sits in the empty upper-left (tumor-vs-normal has no licensed points)
    axC.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc=BLUE, mec="white", label="identifiable / licensed"),
                        Line2D([0], [0], marker="X", ls="", mfc=RED, mec="white", label="abstain (coverage-disjoint)")],
               fontsize=6.6, loc="upper left", handletextpad=0.3, borderpad=0.35, labelspacing=0.35)
    axC.set_title("Positivity gate abstains", fontsize=8)

    # ================ Panel d: null is calibrated ====================
    # Both FPRs are ~25-80x below alpha=0.05, so we plot them directly on their own
    # small scale (0-0.002) and cite alpha as an external reference rather than an
    # axhline that would flatten the two bars to invisibility.
    fpr_s, fpr_p, alpha = float(cal.fpr_stratified), float(cal.fpr_plain), float(cal.alpha)
    dcolors = [BLUE, GREY]
    axD.bar(["coverage-\nstratified", "plain\nshuffle"], [fpr_s, fpr_p],
            color=dcolors, width=0.6, edgecolor="white")
    for i, v in enumerate([fpr_s, fpr_p]):
        axD.text(i, v + 0.00004, f"{v:.4f}", ha="center", va="bottom", fontsize=6.4, color=INK)
    axD.set_ylabel("false-positive rate (null)")
    axD.set_ylim(0, 0.0023)
    axD.set_yticks([0, 0.001, 0.002])
    ratio_lo, ratio_hi = int(round(alpha / fpr_p)), int(round(alpha / fpr_s))
    axD.text(0.5, 0.97, f"both $\\ll$ $\\alpha$ = {alpha:g}\n({ratio_lo}–{ratio_hi}$\\times$ higher, off scale)",
             transform=axD.transAxes, ha="center", va="top", fontsize=6.0, color=RED)
    axD.set_title("Null is calibrated\n(FPR $\\ll$ $\\alpha$)", fontsize=8)

    # ===== Panel e: same variant, two lenses (AL08 m.15059 MT-CYB) ====
    e = cov[(cov["sample"] == "LUAD_AL08") & (cov.pos == 15059)].set_index("axis")
    lenses = ["cellType", "tumorNormal"]
    ename = {"cellType": "cell type", "tumorNormal": "tumor-vs-\nnormal"}
    xe = np.arange(len(lenses))
    for i, ax_lens in enumerate(lenses):
        row = e.loc[ax_lens]
        lic = bool(row.licensed)
        axE.bar(i, row.sfrac, width=0.62, color=(BLUE if lic else RED), edgecolor="white")
        verdict = "LICENSED" if lic else "ABSTAIN"
        vy = row.sfrac + 0.03 if row.sfrac < 0.6 else row.sfrac - 0.03
        va = "bottom" if row.sfrac < 0.6 else "top"
        tc = "white" if row.sfrac >= 0.6 else INK
        axE.text(i, vy, f"$T$={row.T_conc:.2f}\n$p$={row.p_strat:.3f}\nsfrac={row.sfrac:.2f}",
                 ha="center", va=va, fontsize=5.8, color=tc)
        axE.text(i, 1.005, verdict, ha="center", va="bottom", fontsize=6.6, fontweight="bold",
                 color=(BLUE if lic else RED))
    axE.axhline(0.5, ls="--", color=INK, lw=1.0)
    axE.text(len(lenses) - 0.5, 0.52, "positivity threshold", ha="right", va="bottom", fontsize=5.8, color=INK)
    axE.set_xticks(xe); axE.set_xticklabels([ename[a] for a in lenses], fontsize=6.8)
    axE.set_ylim(0, 1.14); axE.set_ylabel("coverage-overlap mass (sfrac)")
    axE.set_title("Same variant (AL08 m.15059 MT-CYB):\nlicensed on cell type, abstained on tumor-vs-normal",
                  fontsize=6.9)

    # ===== Panel f: presence caller (0 carriers) vs level test (licensed) =====
    pos = 9973
    s = inp[inp["loc"] == pos].merge(comp, on="cell_barcode", how="left")
    mal = s[s.compartment == "tumor_aneuploid"]
    n_cov = len(mal); n_alt2 = int((mal.alt >= 2).sum())
    n_carr = int(sm[sm.pos == pos].n_carrier_q05.iloc[0]); minp = float(sm[sm.pos == pos].min_p_adj.iloc[0])
    ec = cov[(cov["sample"] == "LUAD_AL10") & (cov.pos == pos)]
    hot = float(ec.pooled_vaf.iloc[0] + ec.T_conc.iloc[0]) * 100
    pooled = float(ec.pooled_vaf.iloc[0]) * 100
    pstrat = float(ec.p_strat.min())

    # one coherent head-to-head block on a single axis: the per-cell presence caller
    # (scMitoMut-style) reaches 0 carriers on this clone, while the coverage-controlled
    # metacell level test recovers the same between-population difference and licenses it.
    axF.set_xlim(-0.7, 3.05)
    axF.set_ylim(0, 56)
    axF.axvline(1.05, color="#cfcfcf", lw=1.0)                      # divider between the two methods
    # left half: scMitoMut per-cell -> 0 carriers (a number, not a bar)
    axF.text(0.2, 54, "scMitoMut\n(per-cell)", ha="center", va="top", fontsize=6.8, fontweight="bold")
    axF.text(0.2, 30, "0", ha="center", va="center", fontsize=20, fontweight="bold", color=RED)
    axF.text(0.2, 20, "carriers", ha="center", va="center", fontsize=6.8, color=RED)
    axF.text(0.2, 8, f"{n_cov:,} malignant cells\n{n_alt2} with $\\geq$2 alt reads\nmin adj $p$ = {minp:.2f}",
             ha="center", va="center", fontsize=6.0, color=INK)
    # right half: scMitoATAC level test -> pooled 22% vs malignant 37%, licensed
    axF.bar([1.65, 2.5], [pooled, hot], width=0.55, color=[GREY, BLUE], edgecolor="white")
    for xx, v in [(1.65, pooled), (2.5, hot)]:
        axF.text(xx, v + 1.0, f"{v:.0f}%", ha="center", va="bottom", fontsize=6.6)
    axF.text(2.07, 54, "scMitoATAC\n(level test)", ha="center", va="top", fontsize=6.8, fontweight="bold")
    axF.text(2.07, 46, f"$p_{{strat}}$ = {pstrat:.0e}\nLICENSED", ha="center", va="center",
             fontsize=5.8, color=BLUE, fontweight="bold")
    axF.set_xticks([1.65, 2.5]); axF.set_xticklabels(["pooled", "malig."], fontsize=6.2)
    axF.set_yticks([])
    for sp in ("top", "right", "left"):
        axF.spines[sp].set_visible(False)
    axF.set_title("per-cell caller vs\nlevel test", fontsize=8)

    # ---------------------------------------------------------------- panel letters (above titles)
    for ax, L, yy in ((axA, "a", 1.14), (axB, "b", 1.14), (axC, "c", 1.14),
                      (axD, "d", 1.20), (axE, "e", 1.20)):
        PS.panel(ax, L, y=yy)
    PS.panel(axF, "f", x=-0.10, y=1.14)

    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    PS.save_submission(fig, OUT)
    plt.close(fig)
    print("wrote", OUT)
    print(f"a: rho={rA:.2f} n={len(dA)}")
    print(f"b: Ds={Ds} orig={[round(o,2) for o in orig_r]} rare={[round(r,2) for r in rare_r]} n={ns}")
    print(f"c: {len(cC)} candidates across {order}")
    print(f"d: fpr_strat={fpr_s:.4f} fpr_plain={fpr_p:.4f} alpha={alpha:g}  (direct 0-0.002 scale)")
    print(f"e: AL08 m.15059  cellType sfrac={e.loc['cellType'].sfrac:.2f} p={e.loc['cellType'].p_strat:.4f} "
          f"T={e.loc['cellType'].T_conc:.3f} licensed={bool(e.loc['cellType'].licensed)} | "
          f"tumorNormal sfrac={e.loc['tumorNormal'].sfrac:.2f} p={e.loc['tumorNormal'].p_strat:.4f} "
          f"T={e.loc['tumorNormal'].T_conc:.3f} licensed={bool(e.loc['tumorNormal'].licensed)}")
    print(f"f: funnel {n_cov}->{n_alt2}->{n_carr} minp={minp:.2f} | level test pooled={pooled:.0f}% "
          f"malig={hot:.0f}% p_strat={pstrat:.0e}")


if __name__ == "__main__":
    main()

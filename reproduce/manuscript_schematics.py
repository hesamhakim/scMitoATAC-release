#!/usr/bin/env python
"""Three method schematics for the scMitoATAC manuscript, authored at final print size (coordinate system in inches
so point fonts render true; no LaTeX downscaling). Shared style via pubstyle.
Out: docs/manuscript/figures/{fig1_pipeline_overview,method_metacell_lod,method_coverage_null}.png
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle as PS
PS.apply()
ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/figures"
BLUE, RED, GREEN, INK, GREY, FILL = PS.BLUE, PS.RED, PS.GREEN, PS.INK, PS.GREY, PS.FILL


def box(ax, x, y, w, h, title, sub="", ec=BLUE, fc=FILL, ts=7.0, ss=5.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.04",
                                ec=ec, fc=fc, lw=1.1, mutation_aspect=1))
    nlines = title.count("\n") + 1
    ty = y + h - 0.08
    ax.text(x + w / 2, ty, title, ha="center", va="top", fontsize=ts, fontweight="bold", color=INK, linespacing=1.05)
    if sub:
        sy = ty - nlines * (ts / 72.0) * 1.15 - 0.05
        ax.text(x + w / 2, sy, sub, ha="center", va="top", fontsize=ss, color="#33475b", linespacing=1.15)
    return (x, y, w, h)


def arr(ax, p1, p2, color=INK, ls="-", lw=1.2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=9, color=color, ls=ls, lw=lw, shrinkA=1, shrinkB=1))


def R(b): return (b[0] + b[2], b[1] + b[3] / 2)
def L(b): return (b[0], b[1] + b[3] / 2)
def T(b): return (b[0] + b[2] / 2, b[1] + b[3])
def B(b): return (b[0] + b[2] / 2, b[1])


def pipeline():
    W, H = PS.COL, 4.3
    fig, ax = plt.subplots(figsize=(W, H)); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(0.05, H - 0.05, "Per-cell caller engine (frozen, v0.1.0)", fontsize=7.5, fontweight="bold", color=BLUE, va="top")
    bw, bh, y = 1.24, 0.82, H - 1.75
    xs = [0.05, 1.42, 2.79, 4.16, 5.53]
    labs = [("Standard input", "scATAC / Multiome\nBAM (non-enriched)"),
            ("Decoy realign\n+ pileup", "chrM+NUMT decoy;\nconsensus pileup"),
            ("Per-site null", "EB beta-binomial\nerror floor"),
            ("Mixture-LR\nfusion", "per-cell posterior\n+ detection power"),
            ("Recalibrate\n+ abstain", "isotonic;\ncallability gate")]
    A = [box(ax, xs[i], y, bw, bh, t, s, ts=6.6, ss=5.2) for i, (t, s) in enumerate(labs)]
    for p, q in zip(A, A[1:]):
        arr(ax, R(p), L(q))
    tb = box(ax, 2.79, H - 0.72, 1.24, 0.5, "External WGS truth", "specificity + accuracy", ec=GREY, fc="#f4f4f4", ts=6.2, ss=5.0)
    arr(ax, B(tb), T(A[2]), color=GREY, ls=(0, (3, 2)), lw=1.0)
    # elbow connector from the caller engine down into the between-population layer
    xd = B(A[4])[0]
    ax.plot([xd, xd, 0.825], [B(A[4])[1], 2.02, 2.02], color=INK, lw=1.2, solid_capstyle="round")
    arr(ax, (0.825, 2.02), (0.825, 1.44))
    ax.text(0.05, 2.42, "Between-population layer", fontsize=7.5, fontweight="bold", color=BLUE, va="top")
    yb = 0.55
    b0 = box(ax, 0.05, yb, 1.55, 0.86, "Metacell LOD\npooling", "pool to the depth\nthat licenses target VAF", ts=6.6, ss=5.2)
    b1 = box(ax, 1.85, yb, 1.55, 0.86, "Cluster-pooled\ndifferential", "T_conc = max cluster\n- pooled VAF", ts=6.6, ss=5.2)
    b2 = box(ax, 3.65, yb, 1.6, 0.86, "Coverage-\nstratified null", "shuffle within depth\ndeciles; positivity gate", ec=RED, ts=6.6, ss=5.2)
    b3 = box(ax, 5.45, yb, 1.4, 0.86, "Calibrated calls", "VAF or ABSTAIN", ec=GREEN, fc="#eef7ee", ts=6.6, ss=5.6)
    for p, q in zip((b0, b1, b2), (b1, b2, b3)):
        arr(ax, R(p), L(q))
    ax.text(4.45, yb - 0.12, "abstain if clusters coverage-disjoint (sfrac < 0.5)", ha="center", fontsize=5.4, color=RED, style="italic")
    PS.save(fig, f"{OUT}/fig1_pipeline_overview.png"); plt.close(fig)


def metacell_lod():
    fig, ax = plt.subplots(figsize=(PS.HALF, 2.7))
    d = np.array([50, 100, 250, 500, 1000, 2000, 5000]); floor = np.array([4.0, 2.0, 2.0, 1.0, 0.7, 0.5, 0.5])
    ax.plot(d, floor, "-o", color=BLUE, lw=1.8, ms=5, mec="white")
    for dd, ff, txt in [(100, 2.0, "2% @ 100-250x"), (500, 1.0, "1% @ 500x"), (2000, 0.5, "0.5% @ 2000x+")]:
        ax.annotate(txt, (dd, ff), xytext=(6, 9), textcoords="offset points", fontsize=6, color=INK,
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    ax.set_xscale("log"); ax.set_xlabel("pooled chrM depth  (cells x per-cell depth)")
    ax.set_ylabel("licensed floor (% VAF)")
    ax.text(0.97, 0.96, "cells k = ceil(target depth / per-cell depth)\ncells below the floor abstain",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=INK)
    PS.save(fig, f"{OUT}/method_metacell_lod.png"); plt.close(fig)


def coverage_null():
    W, H = PS.COL, 2.9
    fig, ax = plt.subplots(figsize=(W, H)); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(1.35, H - 0.12, "metacells by depth decile", fontsize=6.6, ha="center", color=INK)
    ax.text(1.35, H - 0.32, "(preserves depth, breaks cluster<->genotype)", fontsize=5.2, ha="center", color="#555", style="italic")
    rng = np.random.default_rng(0)
    for s in range(5):
        ax.axvline(0.35 + s * 0.5, ymin=0.14, ymax=0.74, color="#e3e3e3", lw=0.8)
    for _ in range(38):
        dec = rng.integers(0, 5); x = 0.35 + dec * 0.5 + rng.uniform(0.06, 0.42)
        ax.scatter(x, 0.55 + rng.uniform(0, 1.5), s=10, c=BLUE if rng.random() < 0.5 else GREY, alpha=0.8, edgecolors="none")
    ax.annotate("shuffle labels\nWITHIN each decile", (1.35, 0.5), xytext=(1.35, 0.12), ha="center", fontsize=5.8,
                color=INK, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    arr(ax, (2.75, H / 2), (3.15, H / 2))
    ok = box(ax, 3.3, H / 2 + 0.12, 3.5, 1.0, "Clusters share coverage (sfrac >= 0.5)",
             "T_conc vs the stratified null gives a licensed p;\na depth-only gain is reproduced by the null", ec=GREEN, fc="#eef7ee", ts=6.6, ss=5.4)
    no = box(ax, 3.3, 0.35, 3.5, 1.0, "Clusters coverage-disjoint (sfrac < 0.5)",
             "e.g. tumor uniformly deeper -> ABSTAIN\n(downsampling emits a still-confounded number)", ec=RED, fc="#fbeeee", ts=6.6, ss=5.4)
    arr(ax, (3.15, H / 2), L(ok), color=GREEN); arr(ax, (3.15, H / 2), L(no), color=RED)
    PS.save(fig, f"{OUT}/method_coverage_null.png"); plt.close(fig)


def caller_engine():
    """docs/reports/figures/methods1_caller.png -- the per-site statistical engine."""
    ROUT = f"{ROOT}/figures"
    W, H = PS.COL, 3.35
    fig, ax = plt.subplots(figsize=(W, H)); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(W / 2, H - 0.06, "The calibrated caller: a per-site statistical engine",
            ha="center", va="top", fontsize=9, fontweight="bold", color=INK)
    xs = [0.02, 1.39, 2.76, 4.13, 5.50]; bw, bh, y = 1.30, 0.98, 1.75
    specs = [("Raw base counts", "per (cell, position)\nA/C/G/T molecules", GREY, "#eef0f2", ""),
             ("Per-site null", "each position's own\nnoise band, fit\nfrom the data", BLUE, FILL, r"$k \sim \mathrm{BetaBin}(n,\ \mu_0,\ s_0)$"),
             ("Mixture test", "real carrier signal\nvs noise: a\nlikelihood ratio", BLUE, FILL, r"$\Lambda = L(\pi{>}0,\mu_c)/L(\pi{=}0)$"),
             ("Fused posterior", "null + mixture +\nartifact prior into\nP(carrier)", BLUE, FILL, r"$P(\mathrm{carrier}\,|\,k,n)$"),
             ("Calibrated call", "honest probability;\nabstain when\nunsure", GREEN, "#eef7ee", r"$\hat{P} = g(P)$")]
    boxes = []
    for x, (t, s, ec, fc, _) in zip(xs, specs):
        boxes.append(box(ax, x, y, bw, bh, t, s, ec=ec, fc=fc, ts=7.2, ss=5.6))
    for p, q in zip(boxes, boxes[1:]):
        arr(ax, R(p), L(q))
    for x, (_, _, _, _, m) in zip(xs, specs):
        if m:
            ax.text(x + bw / 2, y - 0.14, m, ha="center", va="top", fontsize=6.4, color=INK)
    cap = box(ax, 0.02, 0.14, W - 0.04, 1.05, "Each step defeats a different failure mode", "", ec="#b8ad93", fc="#f4f0e6", ts=7.4)
    ax.text(W / 2, 0.74, "per-site null gives SPECIFICITY      mixture test gives SENSITIVITY", ha="center", va="center", fontsize=6.4, color="#6b5f3f")
    ax.text(W / 2, 0.48, "fusion with the artifact prior gives PRECISION      calibration gives honest CONFIDENCE",
            ha="center", va="center", fontsize=6.4, color="#6b5f3f")
    PS.save(fig, f"{ROUT}/methods1_caller.png"); plt.close(fig)


def deployed_architecture():
    """docs/reports/figures/fig_new_architecture.png -- partition-agnostic layer on the frozen caller (dev labels removed)."""
    ROUT = f"{ROOT}/figures"
    W, H = PS.COL, 4.25
    fig, ax = plt.subplots(figsize=(W, H)); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(W / 2, H - 0.05, "The deployed tool: a partition-agnostic layer on a frozen caller",
            ha="center", va="top", fontsize=9, fontweight="bold", color=INK)
    ax.text(W / 2, H - 0.30, "every stage runs in sequence; the standard format lets a partition from any source enter the same scored pipeline",
            ha="center", va="top", fontsize=5.8, color="#555", style="italic")
    bw, bh = 2.15, 0.95
    xs = [0.05, 2.38, 4.71]
    top_y, bot_y = H - 1.85, H - 3.05
    top = [box(ax, xs[0], top_y, bw, bh, "Stage 0  Partitioning", "any source: external tool,\ninternal clustering, or\nuser-supplied labels", ts=7.4, ss=5.8),
           box(ax, xs[1], top_y, bw, bh, "PartitionResult\n(standard format)", "labels + cell_ids,\naxis / level, used_in_partition", ts=7.4, ss=5.8),
           box(ax, xs[2], top_y, bw, bh, "Stage 1  Feasibility", "can this compartment\nlicense a call at the\ntarget VAF?", ts=7.4, ss=5.8)]
    bot = [box(ax, xs[0], bot_y, bw, bh, "Stage 4  Licensing", "claim licensing plus the\ndouble-dipping guard", ts=7.4, ss=5.8),
           box(ax, xs[1], bot_y, bw, bh, "Stage 3  Relative vs null", "between-compartment\ndifference against a\nrandom-pool null", ts=7.4, ss=5.8),
           box(ax, xs[2], bot_y, bw, bh, "Stage 2  Metacell call", "pool by label, run the\nfrozen caller, return\nan EB point + CI", ts=7.4, ss=5.8)]
    arr(ax, R(top[0]), L(top[1])); arr(ax, R(top[1]), L(top[2]))
    arr(ax, B(top[2]), T(bot[2]))                       # Stage1 -> Stage2
    arr(ax, L(bot[2]), R(bot[1])); arr(ax, L(bot[1]), R(bot[0]))
    fc = box(ax, 0.05, 0.12, W - 0.10, 0.92, "FROZEN CALLER   (scmitoatac v0.1.0, git tag caller-v0.1.0)", "", ec=INK, fc="#eef0f2", ts=7.6)
    ax.text(W / 2, 0.55, "consensus pileup | beta-binomial site null | EB shrinkage | calibration | callability | contamination | phasing",
            ha="center", va="center", fontsize=5.6, color="#444")
    ax.text(W / 2, 0.34, "scoring math cannot change without an explicit re-freeze and a version bump",
            ha="center", va="center", fontsize=5.6, color="#444", style="italic")
    for b in bot:
        arr(ax, B(b), (B(b)[0], 1.06), color=GREY, ls=(0, (2, 2)), lw=0.9)
    PS.save(fig, f"{ROUT}/fig_new_architecture.png"); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    pipeline(); metacell_lod(); coverage_null(); caller_engine(); deployed_architecture()
    print("wrote 5 schematics at print size ->", OUT, "+ docs/reports/figures")


if __name__ == "__main__":
    main()

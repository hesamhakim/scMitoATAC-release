#!/usr/bin/env python
"""Reduced ped-HGG specificity supplement: only the two panels not duplicated in the main m9438 figure.

The full ped-HGG specificity figure (ped_hgg_specificity_fig.py) has four panels: (a) FN4 negative
t-SNE by VAF, (b) carrier-cell capture bars, (c) method-invariance (n of 13 methods significant),
(d) pan-site co-elevation verifier. Panels b (capture) and d (co-elevation) are now BOTH shown in
the main m9438 figure, so this supplement keeps ONLY the two non-duplicated panels:
  (a) FN4 negative t-SNE colored by m.9128 VAF (carriers scattered, no subpopulation), and
  (b) method-invariance bars (all 13 methods agree in the two positives, only 5/13 in the FN4 negative).
Data loading and the n_sig() helper are reused from ped_hgg_specificity_fig.py; numbers are identical.

Run:  python reproduce/supp_pedhgg_specificity_reduced.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle as PS  # noqa: E402

PS.apply()

ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DD = f"{ROOT}/work/phase18/pedhgg_deepdive"
OUT = f"{ROOT}/figures/supp_pedhgg_specificity_reduced.png"


def n_sig(tok, thr=1e-4):
    """Ported from ped_hgg_specificity_fig.py: n of the 13 carrier-cluster methods with p<thr."""
    en = pd.read_csv(f"{DD}/{tok}/enrichment.tsv", sep="\t")
    car = en[en.is_carrier_cluster]
    p = pd.to_numeric(car.carrier_MWU_p, errors="coerce")
    return int((p < thr).sum()), len(car)


def main():
    toks = ["2932", "2937", "FN4"]
    labs = ["2932\n(ped-GBM)", "2937\n(ped-GBM)", "FN4\n(normal lung)"]
    cols = [PS.RED, PS.RED, PS.GREY]

    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.1))

    # (a) FN4 negative t-SNE by VAF — carriers scattered, no tight cluster
    a = ax[0]
    ts = np.load(f"{DD}/FN4/tsne.npy"); cl = pd.read_csv(f"{DD}/FN4/clusters_vaf.tsv", sep="\t")
    ok = cl.depth.to_numpy() >= 3; vaf = cl.vaf.to_numpy()
    a.scatter(ts[~ok, 0], ts[~ok, 1], s=4, c="#dddddd")
    sc = a.scatter(ts[ok, 0], ts[ok, 1], s=14, c=vaf[ok], cmap="viridis", vmin=0, vmax=1)
    cb = plt.colorbar(sc, ax=a, fraction=0.046, pad=0.03); cb.set_label("m.9128 VAF", fontsize=7)
    cb.ax.tick_params(labelsize=6); a.set_xticks([]); a.set_yticks([])
    a.set_title("FN4 negative (normal lung):\ncarriers scattered, no subpopulation", fontsize=8)
    PS.panel(a, "a", x=-0.06, y=1.10)

    # (b) method-invariance: n of 13 methods with significant carrier enrichment (p<1e-4)
    a = ax[1]
    ns = [n_sig(t) for t in toks]
    a.bar(labs, [s / n for s, n in ns], color=cols)
    a.set_ylabel("fraction of 13 methods with p<1e-4", fontsize=8); a.set_ylim(0, 1.15)
    a.set_title("method invariance (13 methods):\npositives agree, negative does not", fontsize=8)
    PS.panel(a, "b", x=-0.20, y=1.10)
    for i, (s, n) in enumerate(ns):
        a.text(i, s / n + 0.02, f"{s}/{n}", ha="center", fontsize=7)
    a.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.86, wspace=0.34)
    PS.save(fig, OUT)
    plt.close(fig)
    print("methods significant:", dict(zip(toks, ns)))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

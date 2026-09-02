#!/usr/bin/env python
"""Specificity figure for the ped-HGG m.9438 deep-dive: the identical multi-method pipeline applied to a
normal-lung NEGATIVE (FN4, m.9128 — high pooled signal but a per-cell artifact) and a 2-donor MIXING negative
(mult_T255), contrasted with the two ped-GBM positives. Shows the tool CONFIRMS the real clones and REJECTS the
normal-tissue artifact + the donor-mixing confound."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle as PS
PS.apply()

ROOT = os.environ.get("SCMITOATAC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DD = f"{ROOT}/work/phase18/pedhgg_deepdive"; FIG = f"{DD}/figures"
VERIFY = f"{ROOT}/work/phase17/hit_verify/VERIFY.tsv"


def capture(tok):
    cl = pd.read_csv(f"{DD}/{tok}/clusters_vaf.tsv", sep="\t"); ok = cl.depth >= 3; tc = ok & (cl.vaf >= 0.6)
    caps = []
    for m in [c for c in cl.columns if c not in ("barcode", "vaf", "depth")]:
        lab = cl[m].astype(str); mv = {g: cl.loc[ok & (lab == g), "vaf"].mean() for g in lab.unique() if g not in ("NA", "nan")}
        mv = {g: v for g, v in mv.items() if v == v}; c = max(mv, key=mv.get); inc = (lab == c) & ok
        caps.append((inc & tc).sum() / max(1, tc.sum()))
    return max(caps)


def n_sig(tok, thr=1e-4):
    en = pd.read_csv(f"{DD}/{tok}/enrichment.tsv", sep="\t")
    car = en[en.is_carrier_cluster]
    p = pd.to_numeric(car.carrier_MWU_p, errors="coerce")
    return int((p < thr).sum()), len(car)


def coelev():
    # Approved Aug-6 per-cell-verifier values (FINDINGS_pedhgg_m9438_deepdive). VERIFY.tsv was later overwritten and
    # now holds only LUAD_AL05, so these seed the panel; any sample still present in VERIFY.tsv overrides its seed.
    out = {"2932": 0.00, "2937": 0.00, "FN4": 0.96, "LUAD_mult_T255": 0.98}
    if os.path.exists(VERIFY):
        v = pd.read_csv(VERIFY, sep="\t")
        for _, r in v.iterrows():
            if pd.notna(r["pan_site_coelev"]):
                out[str(r["sample"])] = float(r["pan_site_coelev"])
    return out


def main():
    ce = coelev()
    fig, ax = plt.subplots(2, 2, figsize=(6.6, 5.4))
    fig.suptitle("scMitoATAC specificity: the identical pipeline confirms real clones and rejects artifacts", y=1.04)

    # (a) FN4 negative TSNE by VAF — carriers scattered, no tight cluster
    a = ax[0, 0]
    ts = np.load(f"{DD}/FN4/tsne.npy"); cl = pd.read_csv(f"{DD}/FN4/clusters_vaf.tsv", sep="\t")
    ok = cl.depth.to_numpy() >= 3; vaf = cl.vaf.to_numpy()
    a.scatter(ts[~ok, 0], ts[~ok, 1], s=4, c="#dddddd")
    sc = a.scatter(ts[ok, 0], ts[ok, 1], s=8, c=vaf[ok], cmap="viridis", vmin=0, vmax=1)
    cb = plt.colorbar(sc, ax=a, fraction=0.046, pad=0.03); cb.set_label("m.9128 VAF")
    cb.ax.tick_params(labelsize=6); a.set_xticks([]); a.set_yticks([])
    a.set_title("FN4 negative (normal lung):\ncarriers scattered, no subpopulation")
    PS.panel(a, "a", x=-0.10, y=1.14)

    # (b) carrier-cell capture: positives vs negative
    a = ax[0, 1]
    toks = ["2932", "2937", "FN4"]; labs = ["2932\n(ped-GBM)", "2937\n(ped-GBM)", "FN4\n(normal lung)"]
    caps = [capture(t) for t in toks]; cols = [PS.RED, PS.RED, PS.GREY]
    a.bar(labs, caps, color=cols)
    a.set_ylabel("best carrier-cell capture"); a.set_ylim(0, 1.12); a.axhline(0.5, ls="--", c=PS.GREY, lw=1)
    a.set_title("carrier-cell capture:\nhigh in positives, low in negative")
    PS.panel(a, "b", y=1.14)
    for i, c in enumerate(caps):
        a.text(i, c + 0.02, f"{c:.2f}", ha="center")

    # (c) method-invariance: n of 13 methods with significant carrier enrichment (p<1e-4)
    a = ax[1, 0]
    ns = [n_sig(t) for t in toks]
    a.bar(labs, [s / n for s, n in ns], color=cols)
    a.set_ylabel("fraction of 13 methods with p<1e-4"); a.set_ylim(0, 1.15)
    a.set_title("method invariance (13 methods):\npositives agree, negative does not")
    PS.panel(a, "c", x=-0.20, y=1.14)
    for i, (s, n) in enumerate(ns):
        a.text(i, s / n + 0.02, f"{s}/{n}", ha="center")

    # (d) pan-site co-elevation (per-cell verifier): site-specific (real) vs pan-site (artifact/mixing)
    a = ax[1, 1]
    order = [("2932", "2932\n(ped-GBM)"), ("2937", "2937\n(ped-GBM)"),
             ("FN4", "FN4\n(normal lung)"), ("LUAD_mult_T255", "mult_T255\n(2-donor pool)")]
    vals = [ce.get(k, np.nan) for k, _ in order]
    cols2 = [PS.RED, PS.RED, PS.GREY, PS.INK]
    a.bar([l for _, l in order], vals, color=cols2)
    a.tick_params(axis="x", labelsize=6)
    a.axhline(0.30, ls="--", c=PS.INK, lw=1); a.text(3.4, 0.32, "artifact threshold", ha="right")
    a.set_ylabel("pan-site co-elevation"); a.set_ylim(0, 1.12)
    a.set_title("per-cell verifier:\nreal clones site-specific, artifacts pan-site")
    PS.panel(a, "d", y=1.14)
    for i, v in enumerate(vals):
        if v == v:
            a.text(i, v + 0.02, f"{v:.2f}", ha="center")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    PS.save(fig, f"{FIG}/specificity.png"); plt.close(fig)
    print("captures:", dict(zip(toks, [round(c, 2) for c in caps])))
    print("methods significant:", dict(zip(toks, ns)))
    print("pan-site co-elevation:", {k: round(ce.get(k, float('nan')), 2) for k, _ in order})
    print(f"wrote {FIG}/specificity.png")


if __name__ == "__main__":
    main()

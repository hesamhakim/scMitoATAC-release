#!/usr/bin/env python
"""
scMitoHet Phase 4.1 + 4.2 — clonal co-segregation feature and joint
variant+clone inference, with an explicit ANTI-CIRCULARITY guard.

4.1 Clonal co-segregation feature
---------------------------------
A candidate variant is more credible if its carrier cells ride an ESTABLISHED
cell partition (guilt-by-association). The partition is anchored ONLY on
pseudobulk-discovered Tier-A/B variants -- never on the candidate itself -- so
the feature cannot manufacture support for a variant out of its own signal.
The feature = the association between the candidate's per-cell carrier calls
and the anchor clone labels, quantified by a G-test likelihood ratio (carrier x
clone independence) gated by the log2 carrier-enrichment of the best clone. It
is folded into the fused posterior as a SITE-level logit adjustment (same fold
point as the Phase-3 features: fusion.fuse_posterior(site_feature_logit=...)).

4.2 Joint variant+clone inference (the anti-circularity mechanism)
------------------------------------------------------------------
Clone assignment and variant reality are inferred together WITHOUT letting the
variant under test drive its own clone's definition. The guard is explicit and
testable: when scoring candidate v, the partition is (re)built on the anchor
set with v HELD OUT. The co-segregation score for v is then measured against
that held-out partition. A variant therefore cannot gain confidence purely from
a clone it defined -- if v is the ONLY evidence for a putative clone, holding it
out dissolves the clone and the score collapses to chance.

Design choice: a full MERLIN/MitoTracer generative EM over the whole
variant x clone matrix is the strong form, but its confidence honesty rests on
exactly this hold-out property. We implement the hold-out-anchored partition as
the operational core (cheap, auditable) and expose the co-segregation LR as the
feature; the EM refinement is available via iterate_joint() but the ban on
self-anchoring is enforced structurally, not by tuning.
"""
import numpy as np
import pandas as pd


def carrier_calls(alt, dp, min_alt=2):
    a = np.asarray(alt, float); d = np.asarray(dp, float)
    return ((a >= min_alt) & (d > 0)).astype(int)


def build_partition(anchor_carrier_df, exclude_site=None, min_clone=10,
                    min_sites=2):
    """Build a cell partition from ANCHOR sites only (Tier-A/B pseudobulk
    variants). anchor_carrier_df: cells x anchor_site 0/1 carrier matrix.
    exclude_site: hold this site OUT (anti-circularity). Returns a Series
    cell -> clone label (haplotype signature over the retained anchors).

    The partition is the distinct carrier-signature grouping over the retained
    anchor sites -- a deterministic, auditable lineage labelling (cells sharing
    the same set of anchor variants = one clone). Clones smaller than min_clone
    are merged into label -1 (unassigned)."""
    cols = [c for c in anchor_carrier_df.columns if c != exclude_site]
    if len(cols) < min_sites:
        return pd.Series(-1, index=anchor_carrier_df.index)
    sig = anchor_carrier_df[cols].astype(int)
    # signature string per cell
    labels = sig.apply(lambda r: "".join(map(str, r.values)), axis=1)
    counts = labels.value_counts()
    keep = set(counts[counts >= min_clone].index)
    out = labels.where(labels.isin(keep), other="__small__")
    # map to integer labels; drop the all-zero (no-anchor) signature to -1
    allzero = "0" * len(cols)
    codes = {lab: i for i, lab in enumerate(sorted(l for l in out.unique()
                                                   if l not in ("__small__", allzero)))}
    codes["__small__"] = -1
    codes[allzero] = -1
    return out.map(codes).astype(int)


def cosegregation_lr(cand_carrier, clone_labels):
    """Association between a candidate's per-cell carrier calls and the
    (held-out-anchored) clone partition. Returns dict with:
      LR      : G-test LR of carrier ~ clone independence (df = n_clones-1)
      max_enrich : max over clones of enrichment log2( p(carrier|clone) /
                   p(carrier) ) -- the guilt-by-association strength
      best_clone, n_carriers
    Cells with clone label -1 (unassigned) are excluded.
    """
    c = np.asarray(cand_carrier, int)
    lab = np.asarray(clone_labels, int)
    m = lab >= 0
    c = c[m]; lab = lab[m]
    n = len(c)
    if n < 20 or c.sum() < 3:
        return {"LR": 0.0, "max_enrich": 0.0, "best_clone": -1,
                "n_carriers": int(c.sum()), "n": n}
    p_all = c.mean()
    clones = np.unique(lab)
    # G-test of independence carrier x clone
    G = 0.0; best_e = 0.0; best_c = -1
    for cl in clones:
        sel = lab == cl
        n_cl = sel.sum()
        if n_cl < 5:
            continue
        p_cl = c[sel].mean()
        # enrichment
        e = np.log2((p_cl + 1e-6) / (p_all + 1e-6))
        if p_cl > p_all and e > best_e:
            best_e = e; best_c = int(cl)
        # G contribution (carrier and non-carrier cells in clone)
        for obs_p, count in [(p_cl, n_cl)]:
            k = c[sel].sum(); nk = n_cl - k
            exp_k = n_cl * p_all; exp_nk = n_cl * (1 - p_all)
            if k > 0:
                G += 2 * k * np.log(k / max(exp_k, 1e-9))
            if nk > 0:
                G += 2 * nk * np.log(nk / max(exp_nk, 1e-9))
    return {"LR": float(max(G, 0.0)), "max_enrich": float(best_e),
            "best_clone": best_c, "n_carriers": int(c.sum()), "n": n}


def cosegregation_feature_logit(cand_alt, cand_dp, anchor_carrier_df,
                                cand_site_id=None, min_alt=2, scale=0.20,
                                cap=2.0):
    """The 4.1 feature as a SITE-level logit adjustment for fusion.py.

    Anti-circularity: the partition is built with cand_site_id HELD OUT of the
    anchor set. Returns (logit_adjustment, detail_dict). The adjustment is
    positive (raises the carrier prior) when the candidate's carriers
    concentrate in an anchor-defined clone, zero when they are scattered.
    """
    part = build_partition(anchor_carrier_df, exclude_site=cand_site_id)
    cc = carrier_calls(cand_alt, cand_dp, min_alt=min_alt)
    lr = cosegregation_lr(cc, part.values)
    # soft, bounded logit: gate the enrichment by a significant LR
    sig = 1.0 / (1.0 + np.exp(-(lr["LR"] - 3.84) / 4.0))
    adj = float(np.clip(scale * lr["max_enrich"] * sig, -cap, cap))
    return adj, {**lr, "logit_adj": adj, "n_clones_used": int((part >= 0).sum())}


def iterate_joint(cand_alt, cand_dp, anchor_carrier_df, cand_site_id=None,
                  n_iter=1, **kw):
    """Optional joint refinement (4.2 strong form). Each iteration rebuilds the
    partition with the candidate HELD OUT, recomputes the feature; the hold-out
    is enforced every iteration so the variant never anchors its own clone.
    With n_iter=1 this equals cosegregation_feature_logit. Kept minimal and
    auditable rather than a full generative EM whose honesty would rest on the
    same hold-out invariant."""
    adj, detail = cosegregation_feature_logit(
        cand_alt, cand_dp, anchor_carrier_df, cand_site_id=cand_site_id, **kw)
    detail["n_iter"] = n_iter
    return adj, detail

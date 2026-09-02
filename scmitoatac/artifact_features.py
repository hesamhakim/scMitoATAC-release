#!/usr/bin/env python
"""
scMitoHet Phase 3.1 + 3.2 — artifact features as CONTINUOUS covariates.

Design principle (HOLD): a true variant's carrier fraction is INDEPENDENT of
these artifact axes; an artifact's is not. So we do NOT hard-threshold. Each
feature is a continuous score in a fixed direction (higher = more artifact-like)
that DOWN-WEIGHTS the site's carrier prior in logit space inside the single
fused posterior (fusion.fuse_posterior). Calibration of the weights happens on
the injected ground truth (true carriers vs artifact-negatives), never as a cut.

------------------------------------------------------------------- 3.1 --------
Standard somatic filters, re-cast as features (Mutect2-style, "adopted"):

  strand_bias        A real mtDNA variant is seen on BOTH strands in proportion
                     to the site's natural strand coverage. An artifact (e.g.
                     one-strand PCR/RT chimera) piles on one strand. We measure
                     the ALT strand imbalance RELATIVE to the position's REF
                     background strand ratio (from read_covariate_backgrounds),
                     via a Fisher/────log-odds symmetric statistic.

  position_in_read   Sequencing/alignment artifacts cluster at read ends
                     (soft-clip boundaries, adapter remnants). Real variants are
                     uniform along the read. Feature = fraction of ALT support
                     within the terminal `end_frac` of reads, minus the site's
                     REF-background end fraction.

  base_quality       Artifacts often carry lower base quality than the site's
                     REF background. Feature = REF_bq_mean - ALT_bq_mean
                     (positive => ALT lower quality => artifact-like).

  vaf_cov_corr       A recurrent artifact's apparent VAF scales with local
                     coverage/read-count (more reads -> more chances to mis-call
                     the same way), whereas a true carrier fraction is
                     coverage-independent. Feature = per-site Spearman corr of
                     (alt/dp) vs read-count among APPARENT-ALT cells. Positive
                     corr => artifact-like. This is a SITE-level feature.

------------------------------------------------------------------- 3.2 --------
mtDNA mutational-signature prior (NOVEL). Bona fide human mtDNA somatic
mutations have a strongly skewed substitution spectrum dominated by
transitions, with a well-documented replication-strand asymmetry:
  * C>T / T>C transitions dominate (the mtDNA somatic signature; cf. the
    SBS-mtDNA / "mitochondrial mutational signature" literature).
  * Strand asymmetry: because the heavy (H) and light (L) strands replicate
    asynchronously, C>T occurs preferentially on one strand and G>A (its
    reverse-complement readout) on the other. NUMT / oxidation (G>T, 8-oxo-G)
    / formaldehyde-crosslink artifacts do NOT follow this — G>T especially is
    an oxidation hallmark, and NUMTs import a nuclear-like flat spectrum.
We score each candidate substitution by a log-likelihood ratio of the mtDNA
somatic signature vs a flat artifact spectrum; the strand of the change adds
to the score when it matches the mtDNA replication-strand expectation. Higher
score = more consistent with a real mtDNA mutation (this feature INCREASES the
carrier prior; it is the one artifact-discriminator that points "toward real").
"""
import numpy as np

BASES = "ACGT"
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}

# ---- 3.2 mtDNA somatic substitution signature ------------------------------
# Empirical relative propensities of the 6 pyrimidine-referenced substitution
# classes for bona-fide human mtDNA somatic mutations. Derived from the
# consensus of mtDNA mutational-signature reports (transition-dominated,
# T>C and C>T the top classes). Referenced to the pyrimidine of the pair so
# C>T and G>A collapse to the same class ("C>T"), etc. Normalized to sum 1.
MT_SIGNATURE = {
    "C>T": 0.42,   # dominant transition
    "T>C": 0.34,   # dominant transition
    "C>A": 0.05,
    "C>G": 0.04,
    "T>A": 0.05,
    "T>G": 0.10,   # non-trivial in mtDNA
}
# Flat null (what a NUMT / random-artifact spectrum looks like): uniform.
FLAT_SIGNATURE = {k: 1.0 / 6 for k in MT_SIGNATURE}
# G>T (oxidation, 8-oxo-dG) is a specific artifact hallmark: in pyrimidine
# reference G>T == C>A. We keep it in C>A but additionally flag the ORIENTED
# G>T / C>A readout as oxidation-suspicious in signature_score.


def _sub_class(ref, alt):
    """Collapse a ref>alt substitution to its pyrimidine-referenced class."""
    ref = ref.upper(); alt = alt.upper()
    if ref not in BASES or alt not in BASES or ref == alt:
        return None
    if ref in "CT":
        return f"{ref}>{alt}"
    # purine ref -> complement both
    return f"{COMPLEMENT[ref]}>{COMPLEMENT[alt]}"


def signature_score(ref, alt, strand_fwd_frac=None):
    """3.2 log-LR that a ref>alt substitution is a bona-fide mtDNA somatic
    mutation vs a flat artifact spectrum. Returns a scalar (higher = more
    mtDNA-signature-consistent). `strand_fwd_frac` is the ALT strand balance
    (fraction forward); a strongly one-sided readout that ALSO matches the
    oxidation orientation (G>T) is penalised further."""
    cls = _sub_class(ref, alt)
    if cls is None:
        return 0.0
    p_mt = MT_SIGNATURE.get(cls, 1e-3)
    p_flat = FLAT_SIGNATURE.get(cls, 1e-3)
    llr = float(np.log(p_mt / p_flat))
    # oxidation orientation penalty: an oriented G>T (== C>A in pyrimidine
    # reference reached from a G ref) that is strand-lopsided is 8-oxo-dG-like.
    if ref.upper() == "G" and alt.upper() == "T" and strand_fwd_frac is not None:
        imbalance = abs(strand_fwd_frac - 0.5) * 2  # 0..1
        llr -= 1.5 * imbalance
    return llr


# ---- 3.1 strand bias --------------------------------------------------------
def strand_bias_feature(alt_fwd, alt_rev, ref_fwd_frac):
    """Symmetric strand-bias statistic for ALT support, RELATIVE to the site's
    REF-background forward fraction (so naturally strand-skewed positions are
    not falsely flagged). Returns a non-negative score (0 = ALT strand ratio
    matches REF background; large = one-strand ALT pileup). Uses a log-odds
    distance with a Haldane-Anscombe 0.5 correction."""
    a_f = float(alt_fwd) + 0.5; a_r = float(alt_rev) + 0.5
    alt_fwd_frac = a_f / (a_f + a_r)
    # background odds
    rf = min(max(ref_fwd_frac, 1e-3), 1 - 1e-3)
    lo_alt = np.log(alt_fwd_frac / (1 - alt_fwd_frac))
    lo_ref = np.log(rf / (1 - rf))
    return float(abs(lo_alt - lo_ref))


# ---- 3.1 position-in-read ---------------------------------------------------
def position_in_read_feature(alt_end_frac, ref_end_frac):
    """Excess of ALT support at read ends over the site's REF background end
    fraction. Positive = ALT clusters at read ends (artifact-like)."""
    return float(max(alt_end_frac - ref_end_frac, 0.0))


# ---- 3.1 base quality -------------------------------------------------------
def base_quality_feature(alt_bq_mean, ref_bq_mean):
    """REF_bq - ALT_bq. Positive = ALT lower quality (artifact-like)."""
    return float(max(ref_bq_mean - alt_bq_mean, 0.0))


# ---- 3.1 VAF-coverage correlation (site-level) ------------------------------
def vaf_coverage_corr_feature(alt, dp, reads=None):
    """Spearman correlation of apparent VAF (alt/dp) vs read-count among cells
    with apparent ALT support. Positive corr => recurrent-artifact-like
    (apparent VAF grows with sequencing depth). Site-level scalar in [-1,1]->
    clipped at 0 (only positive is artifact-evidence). If `reads` (n_reads)
    absent, uses dp as the coverage proxy."""
    alt = np.asarray(alt, float); dp = np.asarray(dp, float)
    cov = dp > 0
    a, d = alt[cov], dp[cov]
    x = (np.asarray(reads, float)[cov] if reads is not None else d)
    frac = np.where(d > 0, a / d, 0.0)
    m = (a > 0)  # apparent-ALT cells only
    if m.sum() < 8:
        return 0.0
    fr = frac[m]; xr = x[m]
    if np.std(fr) < 1e-9 or np.std(xr) < 1e-9:
        return 0.0
    # Spearman = Pearson on ranks
    rf = np.argsort(np.argsort(fr)).astype(float)
    rx = np.argsort(np.argsort(xr)).astype(float)
    rf -= rf.mean(); rx -= rx.mean()
    denom = np.sqrt((rf**2).sum() * (rx**2).sum())
    if denom < 1e-12:
        return 0.0
    rho = float((rf * rx).sum() / denom)
    return float(max(rho, 0.0))


# ---- assembly: per-site artifact feature vector -----------------------------
FEATURE_NAMES = [
    "sig_score",        # 3.2 mtDNA signature log-LR (toward real; +weight)
    "strand_bias",      # 3.1 (toward artifact; -weight)
    "pos_in_read",      # 3.1 (toward artifact; -weight)
    "base_quality",     # 3.1 (toward artifact; -weight)
    "vaf_cov_corr",     # 3.1 (toward artifact; -weight)
]
# sign of each feature's expected effect on the carrier prior (logit space)
FEATURE_SIGN = {"sig_score": +1.0, "strand_bias": -1.0, "pos_in_read": -1.0,
                "base_quality": -1.0, "vaf_cov_corr": -1.0}


def site_artifact_features(alt, dp, ref_base, alt_base, alt_fwd, alt_rev,
                           alt_end_frac, alt_bq_mean, covar_bg, reads=None):
    """Assemble the per-site artifact feature dict. `covar_bg` is the position's
    background row from read_covariate_backgrounds (ref_fwd_frac, readpos_end_frac,
    bq_mean). Returns {name: value} using FEATURE_NAMES."""
    ref_fwd_frac = covar_bg.get("fwd_frac", 0.5) if covar_bg else 0.5
    ref_end_frac = covar_bg.get("readpos_end_frac", 0.3) if covar_bg else 0.3
    ref_bq_mean = covar_bg.get("bq_mean", 30.0) if covar_bg else 30.0
    alt_fwd_frac = (alt_fwd + 0.5) / (alt_fwd + alt_rev + 1.0)
    return {
        "sig_score": signature_score(ref_base, alt_base, strand_fwd_frac=alt_fwd_frac),
        "strand_bias": strand_bias_feature(alt_fwd, alt_rev, ref_fwd_frac),
        "pos_in_read": position_in_read_feature(alt_end_frac, ref_end_frac),
        "base_quality": base_quality_feature(alt_bq_mean, ref_bq_mean),
        "vaf_cov_corr": vaf_coverage_corr_feature(alt, dp, reads=reads),
    }

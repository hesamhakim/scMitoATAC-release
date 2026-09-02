# scMitoATAC

Calibrated inference of mitochondrial heteroplasmy from standard, unenriched single-cell ATAC and Multiome data.

Every scATAC-seq and 10x Multiome experiment sequences the mitochondrial genome as a byproduct. scMitoATAC turns those incidental reads into quantitative, calibrated heteroplasmy estimates that can be compared between cell populations, without changing the assay and without a mitochondrial enrichment protocol.

The method has two layers:

- **A per-cell calling engine** that converts aligned reads into a calibrated per-position heteroplasmy posterior and a call/abstain decision. Fragment-level consensus pileup (ATAC has no UMIs), a per-site empirical-Bayes error null, NUMT-decoy realignment, per-stratum isotonic recalibration, and explicit callability gates. This layer is version-locked at `v0.1.0` and pinned by a full-precision regression test.
- **A population layer** that sizes metacells dynamically from the sample's own measured per-cell depth, then compares pooled heteroplasmy between populations against a coverage-stratified permutation null with a positivity gate.

The design commitment is that the method abstains when the data cannot support a claim, rather than returning a number anyway.

## Install

```bash
conda env create -f environment.yml
conda activate scmitoatac
pip install -e .
```

Python 3.12 or newer. Core dependencies are numpy, scipy, pandas and pysam; the deploy layer additionally uses anndata/mudata/scanpy.

## Quick start

```bash
python examples/minimal_example.py
```

Runs in seconds on synthetic data, no downloads. It prints the depth-to-detection envelope, sizes a metacell from a sample's own per-cell depth, and then runs the between-population test in three regimes: a real shift on comparable coverage (licensed), no biological difference (abstains), and a real shift where the two populations share no coverage range (abstains, because depth and biology are not separable there).

That third case is the one worth understanding. Pooled VAF is read-weighted, so a more deeply sequenced cluster reads higher on its own. The stratified null and positivity gate exist to stop that from being reported as biology.

## What is here

| Path | Contents |
|---|---|
| `scmitoatac/` | The package. The frozen calling engine plus `scmitoatac.deploy`, the partition-agnostic population layer. |
| `tests/` | Test suite, including the regression test that locks the caller's numeric behavior. |
| `ref/` | Pre-registered artifact panels, the chrM+NUMT decoy reference, NUMT interval sets, and gnomAD artifact-prone sites. |
| `data/` | Per-figure source data for the reproduction scripts. |
| `reproduce/` | Scripts that regenerate published figures from `data/`. |
| `analysis/` | Pipeline scripts that require primary sequencing data (see below). |
| `examples/` | The minimal reproducible example. |

Run the tests with `pytest tests/`.

## Reproducing figures

```bash
python reproduce/manuscript_data_figs.py     # and the other scripts in reproduce/
```

Scripts in `reproduce/` regenerate **Figure 1, Figure 3, and Supplementary Figures S1 to S7** from the source data included here. Output lands in `figures/`.

Two honest limitations:

**Supplementary Figure S9** (19-donor ccRCC specificity) cannot be regenerated from this repository. It is built from per-donor mitochondrial genotypes in a controlled-access cohort that cannot be redistributed under its data-use agreement. `reproduce/manuscript_data_figs.py` detects the missing input and skips that panel with a notice; every other figure it builds is unaffected. The aggregate statistics reported in the paper are included in `data/ccRCC_53_summary.json`.

**Scripts in `analysis/`** need intermediates derived from primary sequencing data (alignments, per-cell pileups, embeddings) that are far too large to distribute and are not ours to redistribute in every case. They are included so the analysis is inspectable and so the figures can be rebuilt by anyone who reruns the pipeline from the accessions listed in the paper. They will not run against this repository alone.

Figure scripts resolve paths relative to the repository root. Override with `SCMITOATAC_ROOT` if you need to.

## Citing

See `CITATION.cff`. The accompanying paper is in preparation; please cite the software release until it appears.

## License

MIT. See `LICENSE`.

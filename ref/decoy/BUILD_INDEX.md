# Rebuild the chrM+NUMT decoy alignment index (self-contained)

Source of truth: `chrM_numt_decoy.fa` (git-tracked, 909 contigs: chrM 16569bp + 908 NUMT regions).

```bash
module load bwa/0.7.17 samtools/1.22.1
cd ref/decoy
samtools faidx chrM_numt_decoy.fa      # -> .fai (tracked)
bwa index chrM_numt_decoy.fa           # -> .amb .ann .bwt .pac .sa  (~0.2s)
```

**Aligner choice (Phase 0.2, logged):** `bwa/0.7.17` (BWA-MEM), NOT bwa-mem2.
Rationale: bwa-mem2 and chromap are not installed on RCC (no module, no conda env);
standard 10x scATAC BAMs are produced by cellranger-atac, which uses BWA-MEM, so
bwa-0.7.17 is the faithful match for the equivalence guardrail (Task 0.5). bwa-mem2
is only a speed reimplementation of the identical algorithm and can be swapped in
later by rebuilding the index if throughput requires it.

chrM contig name is `chrM` (length 16569, rCRS) — NOT `MT`. Contig-naming mismatch
(MT vs chrM) bit the predecessor once; assert `chrM` in every downstream slice.

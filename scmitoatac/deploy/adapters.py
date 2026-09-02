"""External partitioner adapters (Step 0). One subclass per validated tool.

Each adapter carries an evaluation-team card's validated CLI recipe + parse logic into the deployed
registry, with NO change to the frozen caller. scMitoATAC EMITS recipe() for the user to run against
their own install; the user runs it; parse() ingests the tool's native output -> PartitionResult.

First adapter: Copy-scAT (CNV / clone level). Card:
docs/deployment/partition_eval/cards/CNV_Copy-scAT.json ; parse spec: run_copyscat.R native output
`<samp>_nmf_clusters.csv` (Barcode, cluster) + a clusterNormal id flagging the non-neoplastic group.
"""
from __future__ import annotations
import os
import numpy as np
from .partition import ExternalAdapter, PartitionResult, register


@register
class CopyscatAdapter(ExternalAdapter):
    """Copy-scAT: arm-level CNV -> a coarse malignant-vs-normal (+ arm-level cluster) partition.

    axis=CNV, level=clone (it is NOT metacell-granular by itself: card median 2117 cells/group).
    Route the coarse clones through `subpool` downstream to reach metacell granularity; on its own
    it contributes the orthogonal tumour/normal carving the deployment split needs.

    Native output (validated by the eval card): `<samp>_nmf_clusters.csv` with columns
    (Barcode, cluster); identifyNonNeoplastic already assigns clusters, and a clusterNormal id flags
    the non-neoplastic group. parse() maps those cluster ids straight to PartitionResult labels.

    used_in_partition = [] : CNV consumes raw fragment-bin counts, not mtDNA variants, so the
    double-dipping guard does not apply on this axis (owner correction).
    """
    name = "Copy-scAT"
    axis = "CNV"
    level = "clone"
    synthetic = False

    def defaults(self):
        # the card's validated key_parameters
        return dict(binSize=1_000_000, minFrags=5000, cutHeight=0.4,
                    chrom_sizes="CopyscAT-own hg38 (3103 tiles, incl chrM)")

    def check_inputs(self, adata):
        """Operational gates for Copy-scAT (returns list of blocker/warning strings; empty = ready).

        Card gates: hg38; raw fragments (not binarized peaks); >=5000 fragments/cell floor.
        We can only check what the AnnData carries; the fragment-file + hg38-ref requirements are
        stated as warnings the user must satisfy when they run the emitted recipe.
        """
        msgs = []
        obs = getattr(adata, "obs", None)
        # fragment-depth floor, if a per-cell fragment count is present
        for col in ("passed_filters", "n_fragments", "nFrags", "atac_fragments"):
            if obs is not None and hasattr(obs, "columns") and col in obs.columns:
                below = int((np.asarray(obs[col]) < self.defaults()["minFrags"]).sum())
                if below:
                    msgs.append(f"WARN: {below} cells below the {self.defaults()['minFrags']}-fragment "
                                f"Copy-scAT floor (col {col!r}); they will be dropped by minFrags.")
                break
        msgs.append("INFO: Copy-scAT needs the raw fragments.tsv.gz (hg38) + CopyscAT's own hg38 refs; "
                    "the emitted recipe assumes those, not the AnnData peak matrix.")
        return msgs

    def recipe(self, params=None, inputs=None):
        """Emit the optimized CLI the user runs against their own Copy-scAT install (card cli_recipe).

        inputs: dict with 'fragments' (fragments.tsv.gz), 'out' (dir), 'samp' (sample id),
                'copyscat_refdir' (CopyscAT hg38_references/), 'chrom_sizes' (CopyscAT hg38_chrom_sizes.tsv).
        """
        p = {**self.defaults(), **(params or {})}
        i = inputs or {}
        frags = i.get("fragments", "<fragments.tsv.gz>")
        out = i.get("out", "<out_dir>")
        samp = i.get("samp", "<sample_id>")
        refdir = i.get("copyscat_refdir", "<CopyscAT hg38_references/>")
        chrom = i.get("chrom_sizes", "<CopyscAT hg38_chrom_sizes.tsv>")
        binf = int(p["binSize"] // 1000)  # process_fragment_file uses -b in bp; card shows 1000000
        return (
            f"# 1) bin raw fragments to {int(p['binSize'])//10**6}Mb tiles (MUST use CopyscAT's own "
            f"chrom_sizes -> {p['chrom_sizes']}):\n"
            f"process_fragment_file.py -i {frags} -o {out}/tiles.tsv -b {int(p['binSize'])} "
            f"-f {int(p['minFrags'])} -g {chrom}\n"
            f"# 2) run the Copy-scAT CNV pipeline (cutHeight={p['cutHeight']} for identifyNonNeoplastic):\n"
            f"run_copyscat.R {out}/tiles.tsv {out} {samp} {refdir} {int(p['minFrags'])}\n"
            f"# native output to parse: {out}/{samp}_nmf_clusters.csv (Barcode,cluster) + clusterNormal id"
        )

    def parse(self, tool_output_path, cell_ids, cluster_normal=None):
        """Ingest `<samp>_nmf_clusters.csv` (Barcode, cluster) -> PartitionResult on `cell_ids`.

        tool_output_path : path to the nmf_clusters CSV (or a dir containing exactly one *_nmf_clusters.csv).
        cell_ids         : the cell universe to align to (barcodes). Cells absent from the tool output
                           are labeled unassigned ("") and dropped from calling.
        cluster_normal   : optional cluster id Copy-scAT flagged as non-neoplastic; relabeled 'normal'.
        """
        import csv
        path = tool_output_path
        if os.path.isdir(path):
            hits = [f for f in os.listdir(path) if f.endswith("_nmf_clusters.csv")]
            if len(hits) != 1:
                raise ValueError(f"expected one *_nmf_clusters.csv in {path}, found {hits}")
            path = os.path.join(path, hits[0])
        # read Barcode -> cluster
        raw = {}
        with open(path, newline="") as fh:
            rdr = csv.reader(fh)
            header = next(rdr)
            hl = [h.strip().lower() for h in header]
            bi = hl.index("barcode") if "barcode" in hl else 0
            # cluster column: 'cluster', or Copy-scAT's native 'nmf_results.cellAssigns'/'cellassigns',
            # else the first non-barcode column.
            ci = next((k for k, h in enumerate(hl)
                       if "cluster" in h or "cellassign" in h or "assign" in h), None)
            if ci is None:
                ci = 1 if len(hl) > 1 else 0
            for row in rdr:
                if not row:
                    continue
                raw[str(row[bi]).replace("cell-", "")] = str(row[ci])
        cids = [str(c).replace("cell-", "") for c in cell_ids]
        def lab(bc):
            g = raw.get(bc)
            if g is None:
                return ""                          # unassigned -> dropped
            if cluster_normal is not None and g == str(cluster_normal):
                return "normal"
            return f"cnv{g}"
        labels = np.array([lab(bc) for bc in cids], dtype=object)
        n_assigned = int((labels != "").sum())
        n_groups = len(set(labels) - {""})
        return PartitionResult(
            labels=labels, cell_ids=np.asarray(cell_ids), method=self.name, version="github-HEAD",
            axis="CNV", level="clone",
            params=dict(self.defaults(), cluster_normal=cluster_normal),
            used_in_partition=[],                  # CNV is not variant-based
            provenance=dict(inputs=[tool_output_path], command="Copy-scAT run_copyscat.R -> parse()",
                            tool_output_path=tool_output_path,
                            n_assigned=n_assigned, n_groups=n_groups,
                            card="docs/deployment/partition_eval/cards/CNV_Copy-scAT.json"),
        )


@register
class MtdnaLineageAdapter(ExternalAdapter):
    """mtDNA-lineage clustering: MQuad (informative-variant selection) -> vireoSNP.BinomMixtureVB.

    axis=mtDNA, level=clone. Groups cells by shared mtDNA heteroplasmy (donor / founder / sub-lineage).
    Input provider is OUR frozen caller's per-cell chrM consensus (AD/DP in cellSNP mtx layout) -- NOT
    mgatk, NOT the vireo CLI (the CLI's diploid nuclear model returns uniform posteriors on continuous
    heteroplasmy; the card mandates the BinomMixtureVB Python API).

    CRITICAL -- the double-dipping guard: this is the ONE axis that CONSUMES mtDNA variants to build the
    partition. parse() populates `used_in_partition` with every variant fed to the clusterer
    ("chrM:pos ref>alt"), so licensing.py gives those positions the strictest treatment downstream
    (a variant used to define a group must not then be 'discovered' in it). Empty used_in_partition on
    this axis is a bug, so parse() refuses when it cannot recover the consumed-variant list.
    """
    name = "mtDNA-lineage"
    axis = "mtDNA"
    level = "clone"
    synthetic = False

    def defaults(self):
        return dict(mquad_selection="top30 by deltaBIC", clusterer="vireoSNP.BinomMixtureVB",
                    n_donor=2, n_init=20)

    def check_inputs(self, adata):
        """Gate: the input provider is our frozen caller (per-cell chrM AD/DP). Donor-level separation
        works at standard droplet depth; the mgatk >=20x floor is only for LOW-HET (<5%) sub-lineages."""
        msgs = ["INFO: input provider is scMitoATAC's own per-cell chrM consensus (AD/DP), not mgatk; "
                "the emitted recipe assumes cellSNP-style AD/DP mtx from the frozen caller.",
                "INFO: donor/founder separation works at standard depth; raise k / require enrichment "
                "only for low-het (<5%) within-donor sub-lineages (mgatk >=20x floor)."]
        return msgs

    def recipe(self, params=None, inputs=None):
        """Emit the CLI the user runs: MQuad variant selection -> vireoSNP.BinomMixtureVB clustering.

        inputs: dict with 'cellsnp_dir' (cellSNP.tag.AD/DP.mtx + cellSNP.base.vcf from the frozen
                caller), 'out' (dir), 'k' (n_donor). recipe assumes the mquad/vireosnp containers.
        """
        p = {**self.defaults(), **(params or {})}
        i = inputs or {}
        cs = i.get("cellsnp_dir", "<cellSNP_dir (AD/DP mtx from the frozen caller)>")
        out = i.get("out", "<out_dir>")
        k = int(i.get("k", p["n_donor"]))
        return (
            f"# 1) MQuad: select informative mtDNA variants (top-30 by deltaBIC; auto-knee is too strict):\n"
            f"mquad --cellData {cs} -o {out}/mquad --minDP 1 --nproc 4\n"
            f"# 2) vireoSNP.BinomMixtureVB clustering (Python API, NOT the vireo CLI), k={k}, n_init={p['n_init']}:\n"
            f"python -m scmitoatac_mtdna_cluster --cellsnp {cs} --mquad {out}/mquad "
            f"--k {k} --n-init {int(p['n_init'])} --top 30 --out {out}\n"
            f"# native output to parse: {out}/cluster_k{k}/labels.tsv (cell, cluster) + the cellSNP dir "
            f"(base.vcf gives the consumed variants for used_in_partition)"
        )

    def parse(self, tool_output_path, cell_ids, cellsnp_dir=None, k=None):
        """Ingest labels.tsv + the cellSNP dir -> PartitionResult with used_in_partition populated.

        tool_output_path : run dir containing cluster_k{k}/labels.tsv (cell, cluster), OR that labels.tsv.
        cellsnp_dir      : dir with cellSNP.base.vcf(.gz) -- REQUIRED to recover the consumed variants
                           (the double-dipping guard). If None, tries tool_output_path.
        k                : cluster count (to locate cluster_k{k}/labels.tsv when a run dir is given).
        """
        import csv, gzip
        # locate labels.tsv
        path = tool_output_path
        if os.path.isdir(path):
            cand = None
            if k is not None and os.path.exists(os.path.join(path, f"cluster_k{k}", "labels.tsv")):
                cand = os.path.join(path, f"cluster_k{k}", "labels.tsv")
            else:
                for root, _dirs, files in os.walk(path):
                    if "labels.tsv" in files:
                        cand = os.path.join(root, "labels.tsv"); break
            if cand is None:
                raise ValueError(f"no labels.tsv under {path} (need cluster_k*/labels.tsv)")
            path = cand
        # read cell -> cluster (tab-separated, header cell/cluster)
        raw = {}
        with open(path, newline="") as fh:
            rdr = csv.reader(fh, delimiter="\t")
            header = next(rdr); hl = [h.strip().lower() for h in header]
            ci = hl.index("cell") if "cell" in hl else 0
            gi = hl.index("cluster") if "cluster" in hl else 1
            for row in rdr:
                if row:
                    raw[str(row[ci]).replace("cell-", "")] = str(row[gi])
        # recover consumed variants from cellSNP.base.vcf -> used_in_partition (double-dipping guard)
        csd = cellsnp_dir or (tool_output_path if os.path.isdir(tool_output_path) else os.path.dirname(path))
        used = self._read_consumed_variants(csd)
        if not used:
            raise ValueError("mtDNA-lineage parse: could not recover consumed variants from "
                             f"{csd!r} (need cellSNP.base.vcf[.gz]); used_in_partition must be "
                             "populated on the mtDNA axis for the double-dipping guard.")
        cids = [str(c).replace("cell-", "") for c in cell_ids]
        # unassigned cells (absent from labels.tsv or empty) -> "" (dropped from calling)
        labels = np.array([("" if raw.get(bc) in (None, "") else f"mt{raw[bc]}") for bc in cids], dtype=object)
        n_assigned = int((labels != "").sum()); n_groups = len(set(labels) - {""})
        return PartitionResult(
            labels=labels, cell_ids=np.asarray(cell_ids), method="MQuad+vireoSNP.BinomMixtureVB",
            version="mquad0.1.8b/vireosnp0.5.9", axis="mtDNA", level="clone",
            params=dict(self.defaults(), n_variants=len(used)),
            used_in_partition=used,               # THE double-dipping guard input
            provenance=dict(inputs=[tool_output_path, csd], command="mtdna_cluster -> parse()",
                            tool_output_path=tool_output_path, cellsnp_dir=csd,
                            n_assigned=n_assigned, n_groups=n_groups,
                            card="docs/deployment/partition_eval/cards/mtDNA_MQuad_vireoSNP.json"),
        )

    @staticmethod
    def _read_consumed_variants(cellsnp_dir):
        """Return ["chrM:pos ref>alt", ...] from cellSNP.base.vcf(.gz); [] if not found."""
        import gzip
        for fn in ("cellSNP.base.vcf", "cellSNP.base.vcf.gz"):
            p = os.path.join(cellsnp_dir, fn)
            if os.path.exists(p):
                op = gzip.open(p, "rt") if p.endswith(".gz") else open(p)
                out = []
                with op as fh:
                    for line in fh:
                        if line.startswith("#"):
                            continue
                        f = line.rstrip("\n").split("\t")
                        if len(f) >= 5:
                            out.append(f"chrM:{f[1]} {f[3]}>{f[4]}")
                return out
        return []

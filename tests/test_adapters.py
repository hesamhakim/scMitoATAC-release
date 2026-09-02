#!/usr/bin/env python
"""scmitoatac.deploy.adapters -- CopyscatAdapter (external CNV) tests.

Exercises the external-adapter contract end-to-end WITHOUT running Copy-scAT: recipe() emission,
parse() of a synthetic native nmf_clusters.csv onto a cell universe (normal-flag + unassigned
handling), registry visibility, and that run() refuses (external adapters are two-step).
"""
import os, csv, tempfile, numpy as np
from scmitoatac.deploy import partition as P
from scmitoatac.deploy import adapters  # registers CopyscatAdapter on import

def _write_nmf(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["Barcode", "cluster"])
        for bc, cl in rows: w.writerow([bc, cl])

def test_registered_and_listed():
    reg = P.list_partitioners()
    assert "Copy-scAT" in reg, reg
    assert reg["Copy-scAT"]["axis"] == "CNV" and reg["Copy-scAT"]["kind"] == "external"
    assert reg["Copy-scAT"]["level"] == "clone"

def test_run_refuses_external():
    try:
        P.run("Copy-scAT", 100); assert False, "external adapter run() must refuse"
    except TypeError:
        pass

def test_recipe_emits_cli():
    a = P.get("Copy-scAT")
    r = a.recipe(inputs=dict(fragments="f.tsv.gz", out="OUT", samp="SAMPLE1",
                             copyscat_refdir="REFS", chrom_sizes="CHR.tsv"))
    assert "process_fragment_file.py" in r and "run_copyscat.R" in r
    assert "SAMPLE1_nmf_clusters.csv" in r and "-f 5000" in r        # minFrags floor threaded in
    assert "cutHeight=0.4" in r                                     # card param surfaced

def test_parse_maps_clusters_and_normal():
    cell_ids = [f"bc{i}" for i in range(10)]
    rows = [("bc0","1"),("bc1","1"),("bc2","2"),("bc3","2"),("bc4","0"),
            ("bc5","0"),("bc6","1"),("bc7","2")]  # bc8, bc9 absent -> unassigned
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "SAMPLE1_nmf_clusters.csv"); _write_nmf(p, rows)
        a = P.get("Copy-scAT")
        pr = a.parse(p, cell_ids, cluster_normal="0")   # cluster 0 = non-neoplastic
    lab = dict(zip(pr.cell_ids, pr.labels))
    assert lab["bc0"] == "cnv1" and lab["bc2"] == "cnv2"
    assert lab["bc4"] == "normal" and lab["bc5"] == "normal"   # normal-flagged
    assert lab["bc8"] == "" and lab["bc9"] == ""               # absent -> unassigned
    assert pr.axis == "CNV" and pr.level == "clone" and pr.used_in_partition == []
    assert pr.provenance["n_assigned"] == 8 and pr.provenance["n_groups"] == 3

def test_parse_real_copyscat_header():
    # Copy-scAT's actual native header is 'Barcode,nmf_results.cellAssigns' (not 'cluster').
    cell_ids = ["AAAC-1", "AAAG-1", "AAAT-1"]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "S_nmf_clusters.csv")
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["Barcode", "nmf_results.cellAssigns"])
            w.writerow(["AAAC-1", "1"]); w.writerow(["AAAG-1", "2"]); w.writerow(["AAAT-1", "1"])
        pr = P.get("Copy-scAT").parse(p, cell_ids)
    lab = dict(zip(pr.cell_ids, pr.labels))
    assert lab["AAAC-1"] == "cnv1" and lab["AAAG-1"] == "cnv2" and lab["AAAT-1"] == "cnv1"

def test_parse_from_directory():
    cell_ids = [f"bc{i}" for i in range(4)]
    with tempfile.TemporaryDirectory() as d:
        _write_nmf(os.path.join(d, "S_nmf_clusters.csv"),
                   [("bc0","1"),("bc1","1"),("bc2","2"),("bc3","2")])
        pr = P.get("Copy-scAT").parse(d, cell_ids)   # dir with one *_nmf_clusters.csv
    assert set(pr.labels) == {"cnv1", "cnv2"}

def test_parse_validates_as_clone_level():
    # coarse CNV partition is legal at clone level, flagged at metacell.
    # >=400 cells so the "few groups on a large dataset" coarse-clone flag fires (METACELL_MIN_GROUPS).
    cell_ids = [f"bc{i}" for i in range(500)]
    rows = [(f"bc{i}", str(i % 2)) for i in range(500)]  # 2 big groups (250 cells each)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "S_nmf_clusters.csv"); _write_nmf(p, rows)
        pr = P.get("Copy-scAT").parse(p, cell_ids)
    ok_clone, _ = pr.validate(granularity="clone")
    ok_mc, issues = pr.validate(granularity="metacell")
    assert ok_clone, "coarse CNV must be legal at clone level"
    # median 250 > 200 AND only 2 groups on 500 cells: metacell validation must reject it
    assert not ok_mc and any(("coarse" in i or "clones" in i) for i in issues), issues

def _write_mtdna_run(d, labels, variants):
    """Build a minimal mtDNA run dir: cluster_k2/labels.tsv + cellSNP.base.vcf."""
    import os
    os.makedirs(os.path.join(d, "cluster_k2"), exist_ok=True)
    with open(os.path.join(d, "cluster_k2", "labels.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t"); w.writerow(["cell", "cluster"])
        for cb, cl in labels: w.writerow([cb, cl])
    with open(os.path.join(d, "cellSNP.base.vcf"), "w") as fh:
        fh.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for pos, ref, alt in variants:
            fh.write(f"chrM\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\n")

def test_mtdna_registered_and_axis():
    reg = P.list_partitioners()
    assert "mtDNA-lineage" in reg and reg["mtDNA-lineage"]["axis"] == "mtDNA"
    assert reg["mtDNA-lineage"]["level"] == "clone" and reg["mtDNA-lineage"]["kind"] == "external"

def test_mtdna_recipe_uses_binommixture_not_cli():
    r = P.get("mtDNA-lineage").recipe(inputs=dict(cellsnp_dir="CS", out="OUT", k=2))
    assert "mquad" in r and "BinomMixtureVB" in r          # the card-mandated clusterer
    assert "labels.tsv" in r and "k2" in r

def test_mtdna_parse_populates_used_in_partition():
    # THE double-dipping guard: consumed variants must land in used_in_partition
    with tempfile.TemporaryDirectory() as d:
        _write_mtdna_run(d,
            labels=[("bc0","0"),("bc1","0"),("bc2","1"),("bc3","1")],
            variants=[(3594,"G","A"),(10398,"A","G"),(15301,"G","A")])
        cell_ids = ["bc0","bc1","bc2","bc3","bc4"]  # bc4 absent -> unassigned
        pr = P.get("mtDNA-lineage").parse(d, cell_ids, k=2)
    lab = dict(zip(pr.cell_ids, pr.labels))
    assert lab["bc0"] == "mt0" and lab["bc2"] == "mt1" and lab["bc4"] == ""
    assert pr.axis == "mtDNA"
    assert pr.used_in_partition == ["chrM:3594 G>A", "chrM:10398 A>G", "chrM:15301 G>A"], pr.used_in_partition
    assert pr.provenance["n_assigned"] == 4 and pr.provenance["n_groups"] == 2

def test_mtdna_parse_refuses_without_variants():
    # empty used_in_partition on the mtDNA axis is a bug -> parse must refuse
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "cluster_k2"))
        with open(os.path.join(d, "cluster_k2", "labels.tsv"), "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t"); w.writerow(["cell", "cluster"])
            w.writerow(["bc0", "0"]); w.writerow(["bc1", "1"])
        # NO cellSNP.base.vcf written
        try:
            P.get("mtDNA-lineage").parse(d, ["bc0", "bc1"], k=2)
            assert False, "must refuse when consumed variants cannot be recovered"
        except ValueError as e:
            assert "used_in_partition" in str(e) or "consumed" in str(e)

if __name__ == "__main__":
    fns = [f for f in dir() if f.startswith("test_")]
    p = 0
    for fn in fns:
        globals()[fn](); print("PASS", fn); p += 1
    print(f"\n{p}/{len(fns)} passed")

#!/usr/bin/env python3
"""Emit results/predictor_resources_v1.tsv (Supplemental_Table_S9).

Third-party predictor resources, versions and access dates. Values are curated
from the per-model documentation (`models/<model>/NOTES.md`), the pinned
environments (`models/<model>/requirements.txt`) and the score-file timestamps
under `results/`; each row's provenance is noted in the comment beside it.

Usage (PYTHONPATH=src):  python -m atlas.predictor_resources
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "predictor_resources_v1.tsv"

# predictor, version_or_resource, source, accessed
ROWS = [
    # models/alphagenome/NOTES.md; manuscript Methods (client 0.7.0, 16-kb window)
    ("AlphaGenome", "API client 0.7.0; merged-quantile splice score, 16-kb window",
     "Google DeepMind AlphaGenome API (non-commercial tier)",
     "2026-07-28; definition sweep 2026-08-04"),
    # models/spliceai/requirements.txt (spliceai==1.3.1); scores 2026-07-30
    ("SpliceAI", "1.3.1; full-precision re-score of delta scores",
     "github.com/Illumina/SpliceAI (run locally)", "2026-07-30"),
    # models/pangolin/requirements.txt (commit 5cf94b8); NOTES.md (Ensembl r112 GTF)
    ("Pangolin", "git commit 5cf94b8; Ensembl r112 GTF; full-precision re-score",
     "github.com/tkzeng/Pangolin (run locally)", "2026-07-31; full-precision 2026-08-04"),
    # models/cadd/NOTES.md (precomputed source, mirrors verified 2026-07-28)
    ("CADD", "GRCh38-v1.7 whole-genome SNV file; score = PHRED",
     "remote tabix slices, official CADD US/DE mirrors", "2026-07-28"),
    # models/alphamissense/NOTES.md (Zenodo record is canonical)
    ("AlphaMissense", "AlphaMissense_hg38.tsv.gz (no version string published)",
     "Zenodo record 8208688", "2026-07-28"),
    # models/evo2/NOTES.md; Supplementary Note S3 (env pins); scores 2026-08-03
    ("Evo2-7B", "arcinstitute/evo2_7b; evo2 0.6.0, torch 2.7.1+cu128, bf16",
     "HuggingFace (scored on rented NVIDIA A6000)", "2026-08-03"),
    # models/gpn_msa/NOTES.md
    ("GPN-MSA", "songlab/gpn-msa-hg38-scores (precomputed, sign-flipped)",
     "HuggingFace datasets", "2026-07-28"),
    # models/nt/NOTES.md; Supplementary Note S3 (env pins); scores 2026-08-01
    ("NT-v2-500M", "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species; "
     "transformers 4.46.3, torch 2.11.0+cu128",
     "HuggingFace (scored on rented NVIDIA A6000)", "2026-08-01"),
    # models/conservation/NOTES.md (UCSC track API, hg38)
    ("phyloP-100way", "hg38 phyloP100way track",
     "UCSC Genome Browser track API", "2026-07-28"),
    ("phastCons-100way", "hg38 phastCons100way track",
     "UCSC Genome Browser track API", "2026-07-28"),
    # models/gnomad/NOTES.md (GraphQL API); reference list (gnomAD v4.1)
    ("gnomAD AF (global)", "gnomAD v4.1", "gnomAD GraphQL API", "2026-07-28"),
    ("gnomAD AF (popmax)", "gnomAD v4.1", "gnomAD GraphQL API", "2026-07-28"),
    # src/atlas/metapredictors.py; batch cache dated 2026-08-03
    ("REVEL", "dbNSFP (assembly hg38), via MyVariant.info batch API",
     "myvariant.info/v1/query", "2026-08-03"),
    ("BayesDel (addAF)", "dbNSFP (assembly hg38), via MyVariant.info batch API",
     "myvariant.info/v1/query", "2026-08-03"),
    ("ClinPred", "dbNSFP (assembly hg38), via MyVariant.info batch API",
     "myvariant.info/v1/query", "2026-08-03"),
    ("MetaRNN", "dbNSFP (assembly hg38), via MyVariant.info batch API",
     "myvariant.info/v1/query", "2026-08-03"),
    ("PrimateAI", "dbNSFP (assembly hg38), via MyVariant.info batch API",
     "myvariant.info/v1/query", "2026-08-03"),
    ("VEST4", "dbNSFP (assembly hg38), via MyVariant.info batch API",
     "myvariant.info/v1/query", "2026-08-03"),
    ("ESM-1b", "dbNSFP (assembly hg38), via MyVariant.info batch API",
     "myvariant.info/v1/query", "2026-08-03"),
]


def main() -> None:
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["predictor", "version_or_resource", "source", "accessed"])
        w.writerows(ROWS)
    print(f"wrote {OUT} ({len(ROWS)} predictors)")


if __name__ == "__main__":
    main()

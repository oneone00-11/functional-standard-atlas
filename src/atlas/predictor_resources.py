#!/usr/bin/env python3
"""Emit results/predictor_resources_v1.tsv — per-predictor resources and licence terms.

Third-party predictor resources, versions, access dates and the licence that governs
the SCORE VALUES in the released matrices. Values are curated from the per-model
documentation (`models/<model>/NOTES.md`), the pinned environments
(`models/<model>/requirements.txt`) and the score-file timestamps under `results/`;
each row's provenance is noted in the comment beside it.

This table, not Supplemental_Table_S9, is the single source of truth for per-column
licence terms. S9 (`table2_model_inventory.tsv`) lists only the twelve primary
predictors; the seven dbNSFP meta-predictors -- which carry the least permissive and
least well documented terms -- appear only here. LICENSE-DATA, README.md and
CITATION.cff all point at this file rather than restating its contents.

Licence audit: 2026-08-28. Two facts that apply to whole groups of rows:

* The licence on a predictor's CODE is not necessarily the licence on its SCORES.
  SpliceAI and PrimateAI ship PolyForm-Strict code with CC BY-NC 4.0 weights, and it
  is the weights line that governs a score column. Pangolin is GPL-3.0, but running a
  GPL program does not place its output under the GPL, so `pangolin_score` is
  recorded as unencumbered by that licence.
* The seven meta-predictors are read from dbNSFP through the MyVariant.info batch
  API (`src/atlas/metapredictors.py`). dbNSFP ships an academic "a" branch and a
  commercial "c" branch with the restricted components removed; the fields requested
  include PrimateAI, VEST4 and REVEL, all academic-branch components, so these values
  come from the academic branch. MyVariant.info's own terms add no permission --
  they disclaim and pass upstream obligations through to the user.

Where no licence statement could be located, the row says so. An absent statement is
not a permissive one, and must not be read as one.

Usage (PYTHONPATH=src):  python -m atlas.predictor_resources
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "predictor_resources_v1.tsv"

_DBNSFP = "dbNSFP (assembly hg38), via MyVariant.info batch API"
_MYVARIANT = "myvariant.info/v1/query"
_NO_TERMS = "No licence terms located; supplied via the dbNSFP academic branch"

# predictor, version_or_resource, source, accessed, licence, licence_source
ROWS = [
    # models/alphagenome/NOTES.md; manuscript Methods (client 0.7.0, 16-kb window)
    ("AlphaGenome", "API client 0.7.0; merged-quantile splice score, 16-kb window",
     "Google DeepMind AlphaGenome API (non-commercial tier)",
     "2026-07-28; definition sweep 2026-08-04",
     "Non-commercial use only (API free tier)",
     "https://deepmind.google.com/science/alphagenome"),
    # models/spliceai/requirements.txt (spliceai==1.3.1); scores 2026-07-30
    ("SpliceAI", "1.3.1; full-precision re-score of delta scores",
     "github.com/Illumina/SpliceAI (run locally)", "2026-07-30",
     "CC BY-NC 4.0 (trained models); academic and non-commercial use, other use "
     "requires a commercial licence from Illumina. Code: PolyForm Strict 1.0.0",
     "https://github.com/Illumina/SpliceAI ; corroborated at "
     "https://gnomad.broadinstitute.org/terms"),
    # models/pangolin/requirements.txt (commit 5cf94b8); NOTES.md (Ensembl r112 GTF)
    ("Pangolin", "git commit 5cf94b8; Ensembl r112 GTF; full-precision re-score",
     "github.com/tkzeng/Pangolin (run locally)", "2026-07-31; full-precision 2026-08-04",
     "GPL-3.0 on the program. Scores were produced by running it; running a GPL "
     "program does not place its output under the GPL, so the column is unencumbered",
     "https://github.com/tkzeng/Pangolin"),
    # models/cadd/NOTES.md (precomputed source, mirrors verified 2026-07-28)
    ("CADD", "GRCh38-v1.7 whole-genome SNV file; score = PHRED",
     "remote tabix slices, official CADD US/DE mirrors", "2026-07-28",
     "Non-commercial use only; commercial use requires a licence from the "
     "University of Washington",
     "https://cadd.gs.washington.edu/download"),
    # models/alphamissense/NOTES.md (Zenodo record is canonical)
    ("AlphaMissense", "AlphaMissense_hg38.tsv.gz (no version string published)",
     "Zenodo record 8208688", "2026-07-28",
     "CC BY-NC-SA 4.0 (non-commercial and share-alike)",
     "https://zenodo.org/records/8208688"),
    # models/evo2/NOTES.md; Supplementary Note S3 (env pins); scores 2026-08-03
    ("Evo2-7B", "arcinstitute/evo2_7b; evo2 0.6.0, torch 2.7.1+cu128, bf16",
     "HuggingFace (scored on rented NVIDIA A6000)", "2026-08-03",
     "Apache-2.0",
     "https://huggingface.co/arcinstitute/evo2_7b"),
    # models/gpn_msa/NOTES.md
    ("GPN-MSA", "songlab/gpn-msa-hg38-scores (precomputed, sign-flipped)",
     "HuggingFace datasets", "2026-07-28",
     "MIT",
     "https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores"),
    # models/nt/NOTES.md; Supplementary Note S3 (env pins); scores 2026-08-01
    ("NT-v2-500M", "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species; "
     "transformers 4.46.3, torch 2.11.0+cu128",
     "HuggingFace (scored on rented NVIDIA A6000)", "2026-08-01",
     "CC BY-NC-SA 4.0 (non-commercial and share-alike)",
     "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"),
    # models/conservation/NOTES.md (UCSC track API, hg38)
    ("phyloP-100way", "hg38 phyloP100way track",
     "UCSC Genome Browser track API", "2026-07-28",
     "Free for public and commercial use (UCSC track data)",
     "https://genome.ucsc.edu/license/"),
    ("phastCons-100way", "hg38 phastCons100way track",
     "UCSC Genome Browser track API", "2026-07-28",
     "Free for public and commercial use (UCSC track data)",
     "https://genome.ucsc.edu/license/"),
    # models/gnomad/NOTES.md (GraphQL API); reference list (gnomAD v4.1)
    ("gnomAD AF (global)", "gnomAD v4.1", "gnomAD GraphQL API", "2026-07-28",
     "CC0 1.0 (public domain dedication); attribution requested, not required",
     "https://gnomad.broadinstitute.org/terms"),
    ("gnomAD AF (popmax)", "gnomAD v4.1", "gnomAD GraphQL API", "2026-07-28",
     "CC0 1.0 (public domain dedication); attribution requested, not required",
     "https://gnomad.broadinstitute.org/terms"),
    # src/atlas/metapredictors.py; batch cache dated 2026-08-03
    ("REVEL", _DBNSFP, _MYVARIANT, "2026-08-03",
     "Non-commercial use only; other use by arrangement with the authors",
     "https://sites.google.com/site/revelgenomics/downloads"),
    ("BayesDel (addAF)", _DBNSFP, _MYVARIANT, "2026-08-03",
     _NO_TERMS,
     "https://fenglab.chpc.utah.edu/BayesDel/ (host unreachable 2026-08-28)"),
    ("ClinPred", _DBNSFP, _MYVARIANT, "2026-08-03",
     _NO_TERMS,
     "https://sites.google.com/site/clinpred/ (download page states no terms)"),
    ("MetaRNN", _DBNSFP, _MYVARIANT, "2026-08-03",
     _NO_TERMS,
     "https://github.com/Chang-Li2019/MetaRNN (no LICENSE file in the repository)"),
    ("PrimateAI", _DBNSFP, _MYVARIANT, "2026-08-03",
     "CC BY-NC 4.0 (trained models); academic and non-commercial use, other use "
     "requires a commercial licence from Illumina. Code: PolyForm Strict 1.0.0",
     "https://github.com/Illumina/PrimateAI"),
    ("VEST4", _DBNSFP, _MYVARIANT, "2026-08-03",
     "Non-commercial use only; excluded from the dbNSFP commercial branch",
     "https://sites.google.com/site/jpopgen/dbNSFP"),
    ("ESM-1b", _DBNSFP, _MYVARIANT, "2026-08-03",
     "MIT (code and pretrained weights)",
     "https://github.com/facebookresearch/esm"),
]

HEADER = ["predictor", "version_or_resource", "source", "accessed",
          "licence", "licence_source"]

# Columns whose terms are more restrictive than the CC BY 4.0 granted to the derived
# data. Quoted by LICENSE-DATA and README.md; keep in sync with ROWS.
RESTRICTED = ["AlphaGenome", "CADD", "AlphaMissense", "NT-v2-500M", "SpliceAI",
              "REVEL", "PrimateAI", "VEST4"]
NO_TERMS_LOCATED = ["BayesDel (addAF)", "ClinPred", "MetaRNN"]


def main() -> None:
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(HEADER)
        w.writerows(ROWS)
    print(f"wrote {OUT} ({len(ROWS)} predictors; "
          f"{len(RESTRICTED)} more restrictive than CC BY 4.0, "
          f"{len(NO_TERMS_LOCATED)} with no terms located)")


if __name__ == "__main__":
    main()

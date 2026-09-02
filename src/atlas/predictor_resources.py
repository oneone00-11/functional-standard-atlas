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


# ---------------------------------------------------------------------------
# Training provenance
# ---------------------------------------------------------------------------
# Whether the benchmarked predictors have a hidden dependency on the functional
# data they are scored against. Two dependencies must be kept apart, because
# only the first is the one usually discussed:
#
#   ClinVar circularity  -- a predictor trained on clinical assertions, scored
#                           against clinical assertions. This paper avoids it by
#                           construction: the standard is assay measurement.
#   assay overlap        -- a predictor trained on MAVE/DMS/SGE measurements, then
#                           scored against measurements of the same kind.
#
# There is also an indirect path that neither column alone captures: a predictor
# trained on ClinVar or HGMD can inherit assay evidence, because PS3/BS3
# classifications are themselves made partly from functional assays, including
# the SGE experiments in this atlas. Rows carrying clinical labels are marked for
# that path rather than as clean.
#
# `training_source` is held to the same standard as `licence_source`: the
# project's own documentation or the publication that describes the training set.
# Where a row states an assessment rather than a documented fact, it says so.

_INHERITED = ("Indirect: clinical labels can inherit PS3/BS3 evidence derived "
              "from functional assays, including SGE")
_NONE_DOC = "None documented"

# predictor -> (training_data, clinical_labels, mave_sge_overlap, training_source)
TRAINING = {
    "AlphaGenome": (
        "Multi-task supervision on functional genomics tracks (expression, "
        "splicing, chromatin) from ENCODE/GTEx-class data; no variant labels",
        "No", _NONE_DOC,
        "https://deepmind.google.com/science/alphagenome"),
    "SpliceAI": (
        "GENCODE-annotated splice junctions on pre-mRNA sequence; no variant labels",
        "No", _NONE_DOC,
        "https://doi.org/10.1016/j.cell.2018.12.015"),
    "Pangolin": (
        "Splice-site usage quantified from RNA-seq across four species and "
        "multiple tissues; no variant labels",
        "No", _NONE_DOC,
        "https://doi.org/10.1186/s13059-022-02664-4"),
    "CADD": (
        "Simulated de novo variants versus fixed derived human alleles, a proxy "
        "contrast with no disease labels",
        "No", _NONE_DOC,
        "https://cadd.gs.washington.edu/info"),
    "AlphaMissense": (
        "Protein language model fine-tuned on a population-frequency proxy "
        "(gnomAD common variants versus unobserved), plus structural context",
        "No", "None documented; DMS sets were used for evaluation, not training",
        "https://doi.org/10.1126/science.adg7492"),
    "Evo2-7B": (
        "Self-supervised next-token prediction over genomes; no labels of any kind",
        "No", _NONE_DOC,
        "https://github.com/ArcInstitute/evo2"),
    "GPN-MSA": (
        "Self-supervised masked language modelling over a whole-genome vertebrate "
        "alignment; no labels of any kind",
        "No", _NONE_DOC,
        "https://huggingface.co/songlab/gpn-msa-hg38-scores"),
    "NT-v2-500M": (
        "Self-supervised masked language modelling over multi-species genomes; "
        "never trained on variant effects",
        "No", _NONE_DOC,
        "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"),
    "phyloP-100way": (
        "Not trained. Per-base substitution rate under a neutral model fitted to a "
        "100-species alignment",
        "No", _NONE_DOC,
        "https://genome.ucsc.edu/cgi-bin/hgTrackUi?db=hg38&g=cons100way"),
    "phastCons-100way": (
        "Not trained. Phylo-HMM conserved-element posterior on the same alignment",
        "No", _NONE_DOC,
        "https://genome.ucsc.edu/cgi-bin/hgTrackUi?db=hg38&g=cons100way"),
    "gnomAD AF (global)": (
        "Not a predictor and not trained; an observed allele frequency",
        "No", _NONE_DOC,
        "https://gnomad.broadinstitute.org/help/what-is-gnomad"),
    "gnomAD AF (popmax)": (
        "Not a predictor and not trained; an observed allele frequency",
        "No", _NONE_DOC,
        "https://gnomad.broadinstitute.org/help/what-is-gnomad"),
    "REVEL": (
        "Ensemble trained on HGMD disease mutations versus rare neutral variants "
        "from ESP/ARIC, over thirteen component scores",
        "Yes (HGMD disease assertions)", _INHERITED,
        "https://doi.org/10.1016/j.ajhg.2016.08.016"),
    "BayesDel (addAF)": (
        "Metascore trained on clinically classified variants (ClinVar/HGMD) "
        "against controls, integrating allele frequency in the addAF form",
        "Yes (ClinVar/HGMD assertions)", _INHERITED,
        "https://doi.org/10.1002/humu.23158"),
    "ClinPred": (
        "Trained on ClinVar pathogenic versus gnomAD benign missense variants",
        "Yes (ClinVar assertions)", _INHERITED,
        "https://doi.org/10.1016/j.ajhg.2018.08.005"),
    "MetaRNN": (
        "Recurrent ensemble trained on ClinVar-derived pathogenic and benign sets "
        "with gnomAD frequency features",
        "Yes (ClinVar assertions)", _INHERITED,
        "https://doi.org/10.1186/s13073-022-01120-z"),
    "PrimateAI": (
        "Deep network trained on common missense variation observed in six "
        "non-human primate species; no human disease labels",
        "No", _NONE_DOC,
        "https://doi.org/10.1038/s41588-018-0167-z"),
    "VEST4": (
        "Random forest trained on HGMD disease mutations versus common variants "
        "from ESP",
        "Yes (HGMD disease assertions)", _INHERITED,
        "https://doi.org/10.1186/1471-2164-14-S3-S3"),
    "ESM-1b": (
        "Self-supervised masked language modelling over UniRef protein sequences; "
        "variant scores are zero-shot pseudo-likelihoods, with no labels",
        "No", _NONE_DOC,
        "https://github.com/facebookresearch/esm"),
}

HEADER = ["predictor", "version_or_resource", "source", "accessed",
          "licence", "licence_source",
          "training_data", "contains_clinical_labels",
          "possible_mave_sge_overlap", "training_source"]

# Columns whose terms are more restrictive than the CC BY 4.0 granted to the derived
# data. Quoted by LICENSE-DATA and README.md; keep in sync with ROWS.
RESTRICTED = ["AlphaGenome", "CADD", "AlphaMissense", "NT-v2-500M", "SpliceAI",
              "REVEL", "PrimateAI", "VEST4"]
NO_TERMS_LOCATED = ["BayesDel (addAF)", "ClinPred", "MetaRNN"]


def main() -> None:
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(HEADER)
        for row in ROWS:
            w.writerow(list(row) + list(TRAINING[row[0]]))
    print(f"wrote {OUT} ({len(ROWS)} predictors; "
          f"{len(RESTRICTED)} more restrictive than CC BY 4.0, "
          f"{len(NO_TERMS_LOCATED)} with no terms located)")


if __name__ == "__main__":
    main()

"""The current file name of every product, written once.

A module that reads or writes a product of the current generation takes its name from
here instead of writing it out. When a product's content changes, its version number
goes up (CONVENTIONS.md rule 1): the new name is written here, the old file stays where it
is, byte for byte, and every module follows.

    from atlas import paths
    paths.path("score_matrix")      # REPO/results/score_matrix_atlas_v3.parquet
    paths.name("score_matrix")      # score_matrix_atlas_v3.parquet, for an --out directory
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
FROZEN = REPO / "data" / "frozen" / "frozen-matrix-v1.parquet"

# product -> its current file name under results/
NAMES: dict[str, str] = {
    # inputs whose current version is their first
    "alphagenome_official_scores": "alphagenome_official_scores_v1.parquet",
    "alphagenome_official_provenance": "alphagenome_official_scores_v1.provenance.json",
    "alphagenome_atlas_scores": "alphagenome_atlas_scores_v1.parquet",
    "alphagenome_atlas_provenance": "alphagenome_atlas_scores_v1.provenance.json",
    "clinvar_status": "clinvar_status_v1.parquet",
    "clinvar_status_provenance": "clinvar_status_v1.provenance.json",
    # the data layer
    "score_matrix": "score_matrix_atlas_v3.parquet",
    "score_matrix_provenance": "score_matrix_atlas_v3.provenance.json",
    "assay_aux": "assay_aux_v2.parquet",
    "assay_aux_provenance": "assay_aux_v2.provenance.json",
    "external_scores": "external_scores_v2.parquet",
    "external_scores_provenance": "external_scores_v2.provenance.json",
    "predictor_resources": "predictor_resources_v2.tsv",
}


def name(key: str, **fields: str) -> str:
    """The file name alone (for a module writing into an --out directory)."""
    return NAMES[key].format(**fields)


def path(key: str, results: Path = RESULTS, **fields: str) -> Path:
    """The product's path under results/ (or under another results directory)."""
    return Path(results) / name(key, **fields)

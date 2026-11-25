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
    "consequence": "consequence_v1.parquet",
    # the data layer
    "score_matrix": "score_matrix_atlas_v3.parquet",
    "score_matrix_provenance": "score_matrix_atlas_v3.provenance.json",
    "assay_aux": "assay_aux_v2.parquet",
    "assay_aux_provenance": "assay_aux_v2.provenance.json",
    "external_scores": "external_scores_v2.parquet",
    "external_scores_provenance": "external_scores_v2.provenance.json",
    "predictor_resources": "predictor_resources_v2.tsv",
    # the main grid
    "eval_ext": "eval_ext_v2.tsv",
    "eval_ext_per_gene": "eval_ext_per_gene_v2.tsv",
    "reliability": "reliability_v2.tsv",
    "reliability_summary": "reliability_v2.json",
    "reliability_corroboration": "reliability_corroboration_v2.tsv",
    "reliability_neutral_check": "reliability_neutral_check_v1.tsv",
    "external_reliability": "external_reliability_v2.tsv",
    "external_reliability_summary": "external_reliability_v2.json",
    "matched_ceilings": "matched_ceilings_v2.tsv",
    "matched_ceilings_summary": "matched_ceilings_v2.json",
    "alphagenome_definition_change": "alphagenome_definition_change_v1.tsv",
    "canonical_site_saturation": "canonical_site_saturation_v1.tsv",
    # the other analyses that read the matrix
    "tie_audit": "tie_audit_v2.tsv",
    "power": "power_v2.tsv",
    "head_to_head": "head_to_head_v2.tsv",
    "head_to_head_pvalues": "head_to_head_pvalues_v2.tsv",
    "logo": "logo_v2.tsv",
    "robustness_summary": "robustness_v2.json",
    "attenuation_simulation": "attenuation_simulation_v3.tsv",
    "attenuation_simulation_summary": "attenuation_simulation_v3.json",
    "attenuation_simulation_curves": "attenuation_simulation_v3_curves.tsv",
    "rna_readout": "rna_readout_v2.tsv",
    "rna_readout_summary": "rna_readout_v2.json",
    "clinical_evidence": "clinical_evidence_v2.tsv",
    "clinical_evidence_summary": "clinical_evidence_v2.json",
    "ensemble_logo": "ensemble_logo_v2.tsv",
    "ensemble_pooled": "ensemble_pooled_v2.tsv",
    "ensemble_summary": "ensemble_v2.json",
    "model_correlation": "model_correlation_{stratum}_v2.tsv",
    # MaveDB
    "mavedb_survey": "mavedb_survey_v2.tsv",
    "mavedb_metadata": "mavedb_metadata_v2.tsv",
    "mavedb_ceilings": "mavedb_ceilings_v2.tsv",
    "mavedb_accounting": "mavedb_accounting_v2.tsv",
    "mavedb_ceiling_k_sensitivity": "mavedb_ceiling_k_sensitivity_v2.tsv",
    "mavedb_estimator_comparison": "mavedb_estimator_comparison_v2.tsv",
    "mavedb_survey_reaudit": "mavedb_survey_reaudit_v2.tsv",
    "mavedb_survey_reaudit_summary": "mavedb_survey_reaudit_v2.json",
}


def name(key: str, **fields: str) -> str:
    """The file name alone (for a module writing into an --out directory)."""
    return NAMES[key].format(**fields)


def path(key: str, results: Path = RESULTS, **fields: str) -> Path:
    """The product's path under results/ (or under another results directory)."""
    return Path(results) / name(key, **fields)

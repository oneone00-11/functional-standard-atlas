"""The predictors, their groups, and the one selection overlap: written once, here.

Every module that runs over predictors takes the list from here; none keeps its own copy.

PREDICTORS  the 23 of the current generation: the 19 of v5, with ``alphagenome`` now the
            official definition (the merged splicing score, results/score_matrix_atlas_v3
            from A3.1), and the four columns of the AlphaGenome Atlas (A3.2). The three
            AlphaGenome scorers beside the merged score (``alphagenome_splice_sites``,
            ...) are auxiliary columns, not predictors.
EXTERNAL    the 13 scored for the external genes: v1's nine and the Atlas's four.

SELECTION_OVERLAP: the Atlas's merged score (``avi``) was selected on a validation set that
includes BRCA1 and RAD51C experiments (results/alphagenome_atlas_scores_v1.provenance.json).
Those two cells are computed as usual and carry a flag; no quantity across genes uses
them: a pooled rho, a median, a ranking, a leave-one-gene-out estimate, a paired
comparison. The companion paper treats them the same way.
"""

from __future__ import annotations

import pandas as pd

# v5's nineteen, in v5's order (atlas.evaluate_ext.MODELS until this generation)
V5 = [
    "alphagenome", "spliceai_ds", "pangolin_score", "cadd", "alphamissense",
    "evo2", "gpn_msa", "nucleotide_transformer", "phylop100way",
    "phastcons100way", "gnomad_af_global", "gnomad_af_popmax",
    # dbNSFP meta-predictors (missense-only; atlas.metapredictors)
    "revel", "bayesdel_addaf", "clinpred", "metarnn", "primateai", "vest4", "esm1b",
]
ATLAS = ["avi", "avi_splice_sites", "avi_splice_site_usage", "avi_splice_junctions"]
PREDICTORS = V5 + ATLAS

# not predictors: the official definition's three scorers, kept beside the merged score
AUXILIARY = ["alphagenome_splice_sites", "alphagenome_splice_site_usage",
             "alphagenome_splice_junctions"]

# groups
SPLICE_FAMILY = ["spliceai_ds", "pangolin_score", "alphagenome",
                 "avi_splice_sites", "avi_splice_site_usage", "avi_splice_junctions"]
META = ["revel", "bayesdel_addaf", "clinpred", "metarnn", "primateai", "vest4", "esm1b"]
ALLELE_FREQUENCY = ["gnomad_af_global", "gnomad_af_popmax"]
GROUPS = {"splice_family": SPLICE_FAMILY, "meta": META, "allele_frequency": ALLELE_FREQUENCY}

# the external genes' columns (results/external_scores_v2.parquet)
EXTERNAL = ["spliceai_ds", "pangolin_score", "cadd", "gpn_msa", "phylop100way",
            "phastcons100way", "gnomad_af_global", "gnomad_af_popmax", "alphagenome"] + ATLAS

# the five of v5's nineteen that score other variants than SNVs; the other fourteen
# score SNVs only, so restricting the analyses to SNVs leaves their numbers unchanged
SCORE_NON_SNV_IN_V5 = ["alphagenome", "phylop100way", "phastcons100way",
                       "gnomad_af_global", "gnomad_af_popmax"]
SNV_ONLY_IN_V5 = [m for m in V5 if m not in SCORE_NON_SNV_IN_V5]

SELECTION_OVERLAP: dict[str, list[str]] = {"avi": ["BRCA1", "RAD51C"]}


def overlap_genes(*models: str) -> set[str]:
    """The genes a quantity across genes over these predictors leaves out."""
    return {g for m in models for g in SELECTION_OVERLAP.get(m, [])}


def is_overlap(model: str, gene: str) -> bool:
    """Does this predictor x gene cell carry the selection-overlap flag?"""
    return gene in SELECTION_OVERLAP.get(model, [])


def drop_overlap(df: pd.DataFrame, *models: str, gene: str = "gene") -> pd.DataFrame:
    """The rows a quantity across genes over these predictors may use."""
    out = overlap_genes(*models)
    return df[~df[gene].isin(out)] if out else df

"""Supplemental Table S11 must actually answer the hidden-dependency question.

Whether the benchmarked predictors have an information overlap with the
functional data they are scored against. A table that answers it only for some
rows would be worse than none, because the gaps read as "no overlap".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "figures"))

from atlas.predictor_resources import ROWS, TRAINING   # noqa: E402

# Trained on clinical assertions, so they carry the indirect path: a PS3/BS3
# classification can itself rest on a functional assay. Pinned by name because
# the count is quoted in the Discussion, and because an earlier draft of that
# paragraph put all seven meta-predictors and CADD in this set.
CLINICALLY_LABELLED = {"REVEL", "BayesDel (addAF)", "ClinPred", "MetaRNN", "VEST4"}


def test_every_scored_predictor_has_a_provenance_row():
    missing = [p for p, *_ in ROWS if p not in TRAINING]
    assert not missing, f"predictors with no training provenance: {missing}"
    extra = set(TRAINING) - {p for p, *_ in ROWS}
    assert not extra, f"provenance rows for predictors that are not scored: {extra}"


def test_every_row_cites_a_source_url():
    """Held to the same standard as licence_source: an assessment with no
    documentation behind it is not admissible in a supplementary table."""
    bad = [p for p, v in TRAINING.items() if not str(v[3]).startswith("https://")]
    assert not bad, f"training claims with no source URL: {bad}"


def test_every_field_is_populated():
    blank = [f"{p}[{i}]" for p, v in TRAINING.items()
             for i in range(4) if not str(v[i]).strip()]
    assert not blank, f"empty provenance fields: {blank}"


def test_clinical_label_column_matches_the_audited_set():
    marked = {p for p, v in TRAINING.items() if str(v[1]).startswith("Yes")}
    assert marked == CLINICALLY_LABELLED, (
        f"clinical-label set drifted: {marked ^ CLINICALLY_LABELLED}")


def test_clinically_labelled_rows_declare_the_indirect_path():
    """The reason these rows matter is not that they are ClinVar-trained; it is
    that a clinical label can inherit assay evidence. A row marked 'Yes' that
    claims no overlap has lost that reasoning."""
    for p in CLINICALLY_LABELLED:
        assert "Indirect" in TRAINING[p][2], f"{p} carries labels but claims no path"


def test_self_supervised_rows_claim_no_overlap():
    for p in ("Evo2-7B", "GPN-MSA", "NT-v2-500M", "ESM-1b"):
        assert TRAINING[p][1] == "No" and "None documented" in TRAINING[p][2], p


def test_the_table_renders_with_every_predictor():
    pytest.importorskip("pandas")
    from build_tables import table_training_provenance

    df = table_training_provenance()
    assert len(df) == len(ROWS) == 19
    assert list(df.columns) == ["predictor", "training_data",
                                "contains_clinical_labels",
                                "possible_mave_sge_overlap", "training_source"]

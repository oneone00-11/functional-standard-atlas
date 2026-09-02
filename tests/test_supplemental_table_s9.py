"""Pin Supplemental Table S9 (model inventory) to the pipeline.

S9 is the table the manuscript points at for third-party predictor resources:
"Third-party predictor resources and their versions are listed in
Supplemental_Table_S9". The delivered S9 has nineteen rows -- every scored
column. The generator had twelve.

The seven dbNSFP meta-predictors were joined in by hand during the submission
pass and never went back into the build, so a rebuild returned a shorter table
than the one the manuscript cites, and nothing noticed. That is the same defect
as MAIN_FIGS (tests/test_figure_numbering.py), the hand-maintained Table 2
(tests/test_table2_territory_guide.py) and the stale table1: a delivered product
decoupled from the code with no test watching. This file is the watch for S9.

`figures/build_tables.table2()` now renders all nineteen from
`atlas.predictor_resources` -- the curated source of truth for version, source,
access date and licence -- adding the editorial scope/definition strings and the
two columns that must be computed from the matrix.

SOURCE OF TRUTH: the submitted manuscript's Supplemental Table S9 (nineteen
predictors; n_scored and coverage_pct as delivered). When the atlas is rescored,
update EXPECTED_COVERAGE here and the delivered table together.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "figures"))

from atlas.predictor_resources import NO_TERMS_LOCATED, RESTRICTED, ROWS  # noqa: E402
from atlas.supplement import TABLES  # noqa: E402
from build_tables import MODELS, table2  # noqa: E402

# Every scored column, in delivered order.
EXPECTED_PREDICTORS = [
    "AlphaGenome", "CADD", "AlphaMissense", "GPN-MSA", "NT-v2-500M", "Evo2-7B",
    "phyloP-100way", "phastCons-100way", "gnomAD AF (global)", "gnomAD AF (popmax)",
    "SpliceAI", "Pangolin", "REVEL", "BayesDel (addAF)", "ClinPred", "MetaRNN",
    "PrimateAI", "VEST4", "ESM-1b",
]

EXPECTED_COLUMNS = [
    "model", "version", "scope", "score_definition", "source", "accessed",
    "n_scored", "coverage_pct", "licence", "licence_source",
]

# model -> n_scored, as printed in the delivered S9.
EXPECTED_COVERAGE = {
    "AlphaGenome": 64178, "CADD": 46392, "AlphaMissense": 26022, "GPN-MSA": 46392,
    "NT-v2-500M": 46392, "Evo2-7B": 46392, "phyloP-100way": 64178,
    "phastCons-100way": 64178, "gnomAD AF (global)": 10545,
    "gnomAD AF (popmax)": 10545, "SpliceAI": 46392, "Pangolin": 46392,
    "REVEL": 26021, "BayesDel (addAF)": 28903, "ClinPred": 26391, "MetaRNN": 26329,
    "PrimateAI": 25818, "VEST4": 27779, "ESM-1b": 21385,
}


@pytest.fixture(scope="module")
def s9():
    """`table2()` reads the score matrix, which is gitignored. Without it these
    tests cannot run -- and they should say so rather than erroring, so that a
    bare clone reports a clean skip instead of five tracebacks."""
    try:
        return table2()
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"score matrix not built in this checkout ({exc.__class__.__name__})")


def test_s9_covers_every_scored_predictor(s9):
    """Nineteen rows, not twelve. This is the assertion that would have caught it."""
    assert list(s9["model"]) == EXPECTED_PREDICTORS, (
        f"S9 has {len(s9)} rows, the manuscript cites {len(EXPECTED_PREDICTORS)}. "
        "The seven dbNSFP meta-predictors belong in MODELS in figures/build_tables.py; "
        "do not re-join them by hand into the delivered xlsx.")


def test_s9_column_set_is_what_was_delivered(s9):
    assert list(s9.columns) == EXPECTED_COLUMNS, (
        f"S9 columns changed: {list(s9.columns)} != {EXPECTED_COLUMNS}")


def test_s9_coverage_matches_the_delivered_table(s9):
    for model, n in EXPECTED_COVERAGE.items():
        got = int(s9.loc[s9["model"] == model, "n_scored"].iloc[0])
        assert got == n, (
            f"{model}: n_scored is {got}, the delivered S9 prints {n}. "
            "If the atlas was rescored, update EXPECTED_COVERAGE and the manuscript together.")


def test_every_row_declares_a_licence_with_a_source(s9):
    """LICENSE-DATA, README and CITATION.cff all defer to this table."""
    for _, row in s9.iterrows():
        assert row["licence"].strip(), f"{row['model']}: empty licence"
        assert re.match(r"https?://", row["licence_source"]), (
            f"{row['model']}: licence_source is not a URL ({row['licence_source']!r})")


def test_licence_classification_matches_the_audit(s9):
    """The 8 restricted and 3 no-terms columns, derived from the rendered table."""
    restricted, no_terms = set(), set()
    for _, row in s9.iterrows():
        if row["licence"].startswith("No licence terms located"):
            no_terms.add(row["model"])
        elif re.search(r"Non-commercial|NC 4\.0|NC-SA", row["licence"]):
            restricted.add(row["model"])
    assert restricted == set(RESTRICTED), f"restricted set drifted: {restricted ^ set(RESTRICTED)}"
    assert no_terms == set(NO_TERMS_LOCATED), (
        f"no-terms set drifted: {no_terms ^ set(NO_TERMS_LOCATED)}")


def test_the_supplement_ships_the_nineteen_row_table():
    """Guard the mapping itself, not just the generator behind it."""
    target = TABLES["Supplemental_Table_S9_model_inventory.tsv"]
    assert target == "table2_model_inventory.tsv", (
        f"S9 now maps to {target}; whatever it maps to must carry all "
        f"{len(EXPECTED_PREDICTORS)} predictors and the licence columns.")
    built = REPO / "results" / target
    if not built.exists():
        pytest.skip(f"{target} not built in this checkout")
    header = built.read_text().splitlines()[0].split("\t")
    n_rows = len(built.read_text().strip().splitlines()) - 1
    assert n_rows == len(EXPECTED_PREDICTORS), (
        f"the shipped {target} has {n_rows} rows, expected {len(EXPECTED_PREDICTORS)}")
    assert header == EXPECTED_COLUMNS, f"the shipped {target} header is {header}"


def test_models_and_predictor_resources_agree():
    """The two lists must name the same predictors, or table2() joins wrongly.

    Their ORDER differs on purpose: MODELS is in delivered-table order, ROWS in
    curation order. table2() joins by name, so only the sets have to match.
    """
    in_models = [label for _, label, *_ in MODELS]
    in_rows = [p for p, *_ in ROWS]
    assert len(in_models) == len(set(in_models)), "duplicate predictor in MODELS"
    assert len(in_rows) == len(set(in_rows)), "duplicate predictor in ROWS"
    assert set(in_models) == set(in_rows), (
        "MODELS (figures/build_tables.py) and ROWS (atlas.predictor_resources) "
        "have diverged; table2() joins licence and access date by predictor name. "
        f"Difference: {set(in_models) ^ set(in_rows)}")

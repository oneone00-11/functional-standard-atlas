"""Pin manuscript Table 1 (atlas composition) to the pipeline.

Third instance of the same failure mode: MAIN_FIGS drifted
from the manuscript's figure order, Table 2 had no generator at all, and
results/table1_atlas_composition.tsv went stale -- it held BRCA1 as
2,868 / 137 / 527 while both the classifier and the manuscript say
2,859 / 143 / 530, predating the fix for the region classifier that mis-filed
UTR-intronic variants as coding (see tests/test_hardening.py).

The stale file was the shipped one, so the archive disagreed with the paper it
was archived to support. Regenerating reproduces the manuscript exactly; this
test is what stops it drifting back.

SOURCE OF TRUTH: the submitted manuscript's Table 1. If the atlas is rebuilt and
these change, the docx changes with them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "figures"))

FROZEN = REPO / "data" / "frozen" / "frozen-matrix-v1.parquet"

# gene -> (variants, SNVs, coding/UTR, splice +-1-2, 3-10, 11-50, >50)
EXPECTED_BRCA1 = (3893, 3893, 2859, 143, 530, 361, 0)
EXPECTED_TOTALS = (64178, 46392, 52523, 1056, 3955, 5961, 683)

COLS = ["n_variants", "n_snv", "n_coding_or_utr", "n_splice_1_2",
        "n_splice_3_10", "n_splice_11_50", "n_splice_deep"]


@pytest.fixture(scope="module")
def table1_df():
    if not FROZEN.exists():
        pytest.skip("frozen matrix not present; run the pipeline first")
    from build_tables import table1
    return table1().set_index("gene")


def test_brca1_row_matches_the_manuscript(table1_df):
    """The row the region-classifier fix moved; the stale file had 2,868/137/527."""
    got = tuple(int(table1_df.loc["BRCA1", c]) for c in COLS)
    assert got == EXPECTED_BRCA1, (
        f"BRCA1 row drifted from Table 1: {got} != {EXPECTED_BRCA1}. "
        "If the atlas was rebuilt, update the manuscript's Table 1 too.")


def test_column_totals_match_the_manuscript(table1_df):
    got = tuple(int(table1_df.loc["Total", c]) for c in COLS)
    assert got == EXPECTED_TOTALS, (
        f"Table 1 totals drifted: {got} != {EXPECTED_TOTALS}")


def test_region_columns_partition_the_atlas(table1_df):
    """Regions are mutually exclusive and exhaustive, so they must sum to the total.

    Independent of the pinned constants above: catches a classifier change that
    moved variants between regions even if someone updated the constants.
    """
    t = table1_df.loc["Total"]
    regions = sum(int(t[c]) for c in COLS[2:])
    assert regions == int(t["n_variants"]), (
        f"region columns sum to {regions}, atlas holds {int(t['n_variants'])}")

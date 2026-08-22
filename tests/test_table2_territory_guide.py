"""Pin manuscript Table 2 (territory selection guide) to the pipeline.

Table 2 was maintained by hand while every other table and figure came from a
committed script, and it drifted: three held-out rho cells disagreed with the
leave-one-gene-out medians they quote (0.468/0.467/0.119 against 0.473/0.466/
0.116), and the caption said the medians were over seven folds when two strata
have only six -- VHL contributes no splice +-1-2 variants and BRCA2 no splice
11-50 bp.

README promises that every figure and table is produced by a committed script.
`figures/build_tables.table2_territory_guide()` now produces this one, and the
tests below are what keep the promise true: the first derives the numbers a
second way, the second pins them to the values the manuscript prints.

SOURCE OF TRUTH for the printed values: the submitted manuscript's Table 2.
When the analysis changes, update EXPECTED here and the docx together.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "figures"))

from build_tables import GUIDE_ROWS, table2_territory_guide  # noqa: E402

LOGO = REPO / "results" / "ensemble_logo_v1.tsv"

# territory -> (held-out rho as printed, LOGO folds behind it, ceiling as printed)
# "" means the row recommends no predictor, so it quotes no rho.
EXPECTED = {
    "Missense coding":            ("",      "",  "0.90"),
    "General coding / UTR":       ("0.440",  7,  "0.90"),
    "Splice region (any offset)": ("0.473",  7,  "0.91"),
    "Splice 3-10 bp":             ("0.466",  7,  "0.89"),
    "Splice 11-50 bp":            ("0.116",  6,  "0.71 (0.58-0.79)"),
    "Splice +-1-2":               ("",      "",  "0.77 (0.59-0.79)"),
    "Deep intronic >50 bp":       ("",      "",  "0.30"),
    "Whole-gene, single choice":  ("0.412",  7,  "0.89"),
}


@pytest.fixture(scope="module")
def guide() -> pd.DataFrame:
    if not LOGO.exists():
        pytest.skip("ensemble_logo_v1.tsv not built; run the pipeline first")
    return table2_territory_guide().set_index("territory")


def test_every_held_out_rho_is_the_logo_median(guide):
    """Recompute each rho straight from ensemble_logo_v1, independently."""
    logo = pd.read_csv(LOGO, sep="\t")
    for territory, stratum, col, _, _ in GUIDE_ROWS:
        if col is None:
            continue
        v = logo.loc[logo["stratum"] == stratum, col].dropna()
        assert guide.loc[territory, "held_out_rho"] == f"{v.median():.3f}", (
            f"{territory}: table says {guide.loc[territory, 'held_out_rho']}, "
            f"LOGO median of {col} over {stratum} is {v.median():.3f}")


def test_matches_the_values_the_manuscript_prints(guide):
    """Catch pipeline drift away from the submitted Table 2."""
    for territory, (rho, folds, ceiling) in EXPECTED.items():
        row = guide.loc[territory]
        assert row["held_out_rho"] == rho, (
            f"{territory}: held-out rho drifted, {row['held_out_rho']} != {rho}. "
            "Update Table 2 in the manuscript and EXPECTED here together.")
        assert row["logo_folds"] == folds, (
            f"{territory}: fold count drifted, {row['logo_folds']} != {folds}. "
            "The caption states the number of folds; check it too.")
        assert row["ceiling"] == ceiling, (
            f"{territory}: ceiling drifted, {row['ceiling']} != {ceiling}")


def test_fold_counts_are_not_all_seven(guide):
    """The caption said 'seven folds' for rows that have six.

    Two strata lose a gene (VHL has no splice +-1-2 variants, BRCA2 no splice
    11-50 bp). This asserts the asymmetry still exists, so that if a future
    build makes every stratum seven-fold, whoever sees this fail also fixes the
    caption rather than leaving it stale in the other direction.
    """
    folds = {t: guide.loc[t, "logo_folds"] for t in EXPECTED
             if guide.loc[t, "logo_folds"] != ""}
    assert folds["Splice 11-50 bp"] == 6, folds
    assert set(folds.values()) == {6, 7}, (
        f"fold counts changed shape: {folds}. Table 2's caption quotes a fold "
        "count and must be re-checked.")

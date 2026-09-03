"""The published attenuation table must regenerate from a default run.

reliability.py declares nineteen models and skipped any that the matrix did not
carry, in silence. Its --matrix default was score_matrix_atlas_v1.parquet, which
has twelve columns. Running the module as documented therefore produced a table
missing all seven dbNSFP meta-predictors -- 48 of 257 rows -- and nothing said
so. The only symptom was that Supplemental Table S3 stopped regenerating, which
is exactly the kind of drift the artefact manifest exists to catch and did.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

RESULTS = REPO / "results"


def test_the_default_matrix_carries_every_declared_model():
    pytest.importorskip("pyarrow")
    import pandas as pd

    from atlas.reliability import MODELS

    default = RESULTS / "score_matrix_atlas_v2.parquet"
    if not default.exists():
        pytest.skip("score matrix not built in this checkout")
    cols = set(pd.read_parquet(default, columns=None).columns)
    missing = [m for m in MODELS if m not in cols]
    assert not missing, (
        "the default matrix does not carry every declared model, so a default "
        f"run would silently drop them: {missing}")


def test_the_default_is_the_matrix_with_the_meta_predictors():
    """v1 has twelve columns and v2 nineteen. Pointing the default at v1 is what
    made the published table unreproducible."""
    import inspect

    from atlas import reliability

    src = inspect.getsource(reliability.main)
    assert "score_matrix_atlas_v2.parquet" in src
    assert 'default=str(RESULTS / "score_matrix_atlas_v1.parquet")' not in src


def test_the_attenuation_table_covers_the_meta_predictors():
    """Regression on the delivered content: the three genes with a validated
    error model each carry all seven meta-predictors."""
    import pandas as pd

    f = RESULTS / "attenuation_v1.tsv"
    if not f.exists():
        pytest.skip("results/ not built in this checkout")
    d = pd.read_csv(f, sep="\t")
    meta = {"revel", "bayesdel_addaf", "clinpred", "metarnn", "primateai",
            "vest4", "esm1b"}
    present = meta & set(d.model)
    assert present == meta, f"missing from the attenuation table: {sorted(meta - present)}"

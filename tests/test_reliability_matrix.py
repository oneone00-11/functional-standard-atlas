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


def test_the_replicate_ceiling_is_preferred_where_both_are_validated(monkeypatch):
    """The comment always said so; sorting the method names kept the standard-error
    estimate. v5's data has no gene x stratum with both validated."""
    import pandas as pd
    from atlas import external_reliability, reliability as r

    rel = pd.DataFrame([
        {"gene": "G", "stratum": "all", "method": "se", "n": 100, "ceiling": 0.5,
         "reliability": 0.25, "status": "validated"},
        {"gene": "G", "stratum": "all", "method": "replicates", "n": 100, "ceiling": 0.8,
         "reliability": 0.64, "status": "validated"},
        {"gene": "H", "stratum": "all", "method": "se", "n": 100, "ceiling": 0.7,
         "reliability": 0.49, "status": "validated"},
    ])
    monkeypatch.setattr(r, "MODELS", ["m"])
    monkeypatch.setattr(r, "per_gene_rho", lambda mat, model: pd.DataFrame(
        {"gene": ["G", "H"], "stratum": ["all", "all"], "model": [model] * 2,
         "n": [100, 100], "rho": [0.4, 0.35]}))
    for order in (rel, rel.iloc[::-1]):
        out = r.realisation_table(pd.DataFrame({"m": [1.0]}), order).set_index("gene")
        assert out.loc["G", "method"] == "replicates" and out.loc["G", "ceiling"] == 0.8
        assert out.loc["H", "method"] == "se" and out.loc["H", "realisation"] == 0.35 / 0.7
    assert external_reliability.METHOD_PREFERENCE is r.METHOD_PREFERENCE

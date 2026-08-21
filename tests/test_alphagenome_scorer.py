"""Tests for the AlphaGenome scorer (offline, --mock mode only)."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCORER = REPO / "models" / "alphagenome" / "score.py"

sys.path.insert(0, str(SCORER.parent))
from score import extract_merged_splicing_score  # noqa: E402


@pytest.fixture()
def toy_variants(tmp_path):
    df = pd.DataFrame(
        {
            "variant_id": ["v1", "v2", "v3"],
            "chrom": ["17", "17", "3"],
            "pos": [43051129, 43092573, 10183776],
            "ref": ["G", "A", "AA"],
            "alt": ["C", "G", "A"],
            "transcript": ["NM_007294.4"] * 3,
            "hgvs_c": ["c.5278-12C>G", "c.594+2T>C", "c.80del"],
        }
    )
    p = tmp_path / "variants.parquet"
    df.to_parquet(p, index=False)
    return p


def run_scorer(input_path, output_path, cache_dir):
    return subprocess.run(
        [
            sys.executable,
            str(SCORER),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--cache-dir",
            str(cache_dir),
            "--mock",
        ],
        capture_output=True,
        text=True,
    )


def test_mock_scorer_contract_and_determinism(toy_variants, tmp_path):
    out1, out2 = tmp_path / "s1.parquet", tmp_path / "s2.parquet"
    cache = tmp_path / "cache"

    r1 = run_scorer(toy_variants, out1, cache)
    assert r1.returncode == 0, r1.stderr
    df1 = pd.read_parquet(out1)

    # Contract: all input columns preserved + exactly one score column.
    assert "alphagenome" in df1.columns
    assert len(df1) == 3
    assert set(["variant_id", "chrom", "pos", "ref", "alt"]) <= set(df1.columns)

    # Deterministic: second run (now cache-assisted) gives identical scores.
    r2 = run_scorer(toy_variants, out2, cache)
    assert r2.returncode == 0, r2.stderr
    df2 = pd.read_parquet(out2)
    pd.testing.assert_series_equal(df1["alphagenome"], df2["alphagenome"])

    # Cache file exists and was used on the second run.
    lines = (cache / "scores.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert "3 from cache" in r2.stdout

    # Run metadata recorded (client version, score definition, timestamp).
    log = json.loads((out1.parent / "run_log.json").read_text())
    assert log["client_package"] == "alphagenome==0.7.0"
    assert log["mock"] is True


def test_extract_merged_splicing_score_formula():
    tidy = pd.DataFrame(
        {
            "output_type": ["SPLICE_SITES", "SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS"],
            "score": [0.4, 0.7, 0.2, 1.5],
        }
    )
    # max(SPLICE_SITES)=0.7 + max(SPLICE_SITE_USAGE)=0.2 + max(SPLICE_JUNCTIONS)/5=0.3
    assert extract_merged_splicing_score(tidy) == pytest.approx(1.2)


def test_extract_merged_splicing_score_fails_loudly_on_bad_shape():
    tidy = pd.DataFrame({"unexpected": [1], "columns": [2]})
    with pytest.raises(ValueError, match="unexpected tidy_scores columns"):
        extract_merged_splicing_score(tidy)

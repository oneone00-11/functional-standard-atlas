"""Tests for the UCSC conservation scorer (offline, --mock mode only)."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCORER = REPO / "models" / "conservation" / "score.py"


def toy_variants(tmp_path):
    df = pd.DataFrame(
        {
            "variant_id": ["v1", "v2", "v3"],
            "chrom": ["17", "17", "3"],
            "pos": [43051129, 43092573, 10183776],
            "ref": ["G", "A", "T"],
            "alt": ["C", "G", "A"],
            "transcript": ["NM_007294.4"] * 3,
            "hgvs_c": ["c.5278-12C>G", "c.594+2T>C", "c.80G>A"],
        }
    )
    p = tmp_path / "variants.parquet"
    df.to_parquet(p, index=False)
    return p


def run_scorer(input_path, output_path, cache_dir, track="phyloP100way"):
    return subprocess.run(
        [sys.executable, str(SCORER), "--input", str(input_path),
         "--output", str(output_path), "--track", track,
         "--cache-dir", str(cache_dir), "--mock"],
        capture_output=True, text=True,
    )


def test_mock_scorer_contract_and_determinism(tmp_path):
    inp = toy_variants(tmp_path)
    out1, out2 = tmp_path / "s1.parquet", tmp_path / "s2.parquet"
    cache = tmp_path / "cache"

    r1 = run_scorer(inp, out1, cache)
    assert r1.returncode == 0, r1.stderr
    df1 = pd.read_parquet(out1)

    # Contract: input columns preserved + exactly one score column per run.
    assert "phylop100way" in df1.columns
    assert "phastcons100way" not in df1.columns
    assert len(df1) == 3
    assert {"variant_id", "chrom", "pos", "ref", "alt"} <= set(df1.columns)

    # Deterministic.
    r2 = run_scorer(inp, out2, cache)
    assert r2.returncode == 0, r2.stderr
    pd.testing.assert_series_equal(
        df1["phylop100way"], pd.read_parquet(out2)["phylop100way"]
    )

    # Run metadata recorded per track.
    log = json.loads((out1.parent / "run_log_phylop100way.json").read_text())
    assert log["track"] == "phyloP100way"
    assert log["mock"] is True


def test_track_choice_controls_column_name(tmp_path):
    inp = toy_variants(tmp_path)
    out = tmp_path / "s.parquet"
    r = run_scorer(inp, out, tmp_path / "cache", track="phastCons100way")
    assert r.returncode == 0, r.stderr
    df = pd.read_parquet(out)
    assert "phastcons100way" in df.columns
    assert "phylop100way" not in df.columns

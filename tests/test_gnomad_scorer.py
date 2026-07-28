"""Tests for the gnomAD AF scorer (offline, --mock mode only)."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCORER = REPO / "models" / "gnomad" / "score.py"


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


def run_scorer(input_path, output_path, cache_dir, metric="global"):
    return subprocess.run(
        [sys.executable, str(SCORER), "--input", str(input_path),
         "--output", str(output_path), "--metric", metric,
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

    assert "gnomad_af_global" in df1.columns
    assert "gnomad_af_popmax" not in df1.columns
    assert len(df1) == 3
    assert (df1["gnomad_af_global"] <= 0).all()  # flipped AF

    r2 = run_scorer(inp, out2, cache)
    assert r2.returncode == 0, r2.stderr
    pd.testing.assert_series_equal(
        df1["gnomad_af_global"], pd.read_parquet(out2)["gnomad_af_global"]
    )

    log = json.loads((SCORER.parent / "run_log_gnomad_af_global.json").read_text())
    assert log["dataset"] == "gnomad_r4"
    assert log["mock"] is True


def test_metric_choice_controls_column_name(tmp_path):
    inp = toy_variants(tmp_path)
    out = tmp_path / "s.parquet"
    r = run_scorer(inp, out, tmp_path / "cache", metric="popmax")
    assert r.returncode == 0, r.stderr
    df = pd.read_parquet(out)
    assert "gnomad_af_popmax" in df.columns
    assert "gnomad_af_global" not in df.columns

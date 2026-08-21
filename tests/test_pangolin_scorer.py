"""Tests for the Pangolin scorer (offline, --mock mode only)."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCORER = REPO / "models" / "pangolin" / "score.py"


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


def run_scorer(input_path, output_path, cache_dir):
    return subprocess.run(
        [sys.executable, str(SCORER), "--input", str(input_path),
         "--output", str(output_path), "--cache-dir", str(cache_dir), "--mock"],
        capture_output=True, text=True,
    )


def test_mock_scorer_contract_and_determinism(tmp_path):
    inp = toy_variants(tmp_path)
    out1, out2 = tmp_path / "s1.parquet", tmp_path / "s2.parquet"
    cache = tmp_path / "cache"

    r1 = run_scorer(inp, out1, cache)
    assert r1.returncode == 0, r1.stderr
    df1 = pd.read_parquet(out1)

    for col in ("pangolin_score", "pangolin_gain", "pangolin_loss"):
        assert col in df1.columns
    assert len(df1) == 3
    assert {"variant_id", "chrom", "pos", "ref", "alt"} <= set(df1.columns)

    r2 = run_scorer(inp, out2, cache)
    assert r2.returncode == 0, r2.stderr
    pd.testing.assert_series_equal(
        df1["pangolin_score"], pd.read_parquet(out2)["pangolin_score"]
    )

    log = json.loads((out1.parent / "run_log.json").read_text())
    assert log["mock"] is True
    assert log["score_definition"].startswith("max(splice gain")


def test_parse_pangolin_csv(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("pangolin_score", SCORER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    csv = tmp_path / "out.csv"
    csv.write_text(
        "CHROM,POS,REF,ALT,Pangolin\n"
        # two transcript records: pick max(gain,|loss|) = 0.50 (loss record)
        "17,43045705,T,A,ENST1|100:0.20|-50:-0.50|Warnings: none,extra,commas\n"
        # gain record wins
        "17,43045706,A,C,ENST1|10:0.42|-5:-0.10|Warnings: none\n"
        # line without any score record -> skipped
        "17,43045707,G,T,no_scores_here\n"
    )
    scores = mod.parse_pangolin_csv(csv)
    assert scores[("17", 43045705, "T", "A")] == (0.50, 0.20, -0.50)
    assert scores[("17", 43045706, "A", "C")] == (0.42, 0.42, -0.10)
    assert ("17", 43045707, "G", "T") not in scores

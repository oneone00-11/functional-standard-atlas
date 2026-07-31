"""Tests for the Evo2 scorer (offline, --mock mode only)."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCORER = REPO / "models" / "evo2" / "score.py"


def toy_variants(tmp_path):
    df = pd.DataFrame(
        {
            "variant_id": ["v1", "v2", "v3"],
            "chrom": ["17", "17", "3"],
            "pos": [43051129, 43092573, 10183776],
            "ref": ["G", "A", "T"],
            "alt": ["C", "G", "A"],
        }
    )
    p = tmp_path / "variants.parquet"
    df.to_parquet(p, index=False)
    return p


def test_mock_scorer_contract_and_determinism(tmp_path):
    inp = toy_variants(tmp_path)
    out1, out2 = tmp_path / "s1.parquet", tmp_path / "s2.parquet"
    cache = tmp_path / "cache"

    def run(out):
        return subprocess.run(
            [sys.executable, str(SCORER), "--input", str(inp),
             "--output", str(out), "--cache-dir", str(cache), "--mock"],
            capture_output=True, text=True,
        )

    r1 = run(out1)
    assert r1.returncode == 0, r1.stderr
    df1 = pd.read_parquet(out1)

    assert "evo2" in df1.columns
    assert len(df1) == 3
    assert {"variant_id", "chrom", "pos", "ref", "alt"} <= set(df1.columns)

    r2 = run(out2)
    assert r2.returncode == 0, r2.stderr
    pd.testing.assert_series_equal(df1["evo2"], pd.read_parquet(out2)["evo2"])

    log = json.loads((SCORER.parent / "run_log.json").read_text())
    assert log["mock"] is True
    assert "LL(ref window)" in log["score_definition"]


def test_extract_window():
    import importlib.util

    spec = importlib.util.spec_from_file_location("evo2_score", SCORER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    chrom = "ACGT" * 5000  # 20,000 bp
    w = mod.extract_window(chrom, 10_000)
    assert w is not None and len(w) == mod.WINDOW_BP
    assert w[mod.WINDOW_BP // 2] == chrom[10_000]
    # near chromosome start: window still valid but shorter/shifted
    w0 = mod.extract_window(chrom, 100)
    assert w0 is not None and len(w0) >= mod.MIN_WINDOW_BP
    # N in window -> rejected
    bad = chrom[:9000] + "N" + chrom[9001:]
    assert mod.extract_window(bad, 9000) is None
    # tiny chromosome -> rejected
    assert mod.extract_window("ACGT" * 100, 200) is None

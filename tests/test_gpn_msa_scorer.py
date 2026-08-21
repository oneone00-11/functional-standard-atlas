"""Tests for the GPN-MSA scorer (offline, --mock mode + pure-function units)."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCORER = REPO / "models" / "gpn_msa" / "score.py"

sys.path.insert(0, str(SCORER.parent))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("gpn_msa_score", SCORER)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
byte_ranges_for, read_tbi, reg2bins = _mod.byte_ranges_for, _mod.read_tbi, _mod.reg2bins


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

    assert "gpn_msa" in df1.columns
    assert len(df1) == 3
    assert {"variant_id", "chrom", "pos", "ref", "alt"} <= set(df1.columns)

    r2 = run_scorer(inp, out2, cache)
    assert r2.returncode == 0, r2.stderr
    pd.testing.assert_series_equal(df1["gpn_msa"], pd.read_parquet(out2)["gpn_msa"])

    log = json.loads((out1.parent / "run_log.json").read_text())
    assert log["mock"] is True
    assert log["score_definition"].startswith("gpn_msa = -raw")


def test_reg2bins_and_byte_ranges_are_sane():
    # The real pinned index must cover the BRCA1 region cheaply.
    refs = read_tbi(SCORER.parent / "gpn_scores.tsv.bgz.tbi")
    assert "17" in refs
    ranges = byte_ranges_for(refs, [("17", 43045703, 43124119)])
    # bgzf head + EOF + the region chunks; region data should be ~1-2 MB.
    assert ranges[0][0] == 0
    total = sum(e - s for s, e in ranges)
    assert total < 10_000_000
    assert reg2bins(43045703, 43124119)[0] == 0

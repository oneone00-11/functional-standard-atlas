"""Tests for the SpliceAI scorer (offline, --mock mode only)."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCORER = REPO / "models" / "spliceai" / "score.py"


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

    assert "spliceai_ds" in df1.columns
    assert len(df1) == 3
    assert {"variant_id", "chrom", "pos", "ref", "alt"} <= set(df1.columns)

    r2 = run_scorer(inp, out2, cache)
    assert r2.returncode == 0, r2.stderr
    pd.testing.assert_series_equal(df1["spliceai_ds"], pd.read_parquet(out2)["spliceai_ds"])

    log = json.loads((SCORER.parent / "run_log.json").read_text())
    assert log["mock"] is True
    assert log["score_definition"].startswith("max(DS_AG")


def test_parse_spliceai_vcf(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("spliceai_score", SCORER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    vcf = tmp_path / "out.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "17\t43045705\t.\tT\tA\t.\t.\tSpliceAI=A|BRCA1|0.00|0.10|0.00|0.20|-7|-18|-43|3\n"
        "17\t43045706\t.\tA\tC\t.\t.\tSpliceAI=C|BRCA1|0.01|0.02|0.03|0.04|-7|-18|-43|3\n"
    )
    scores = mod.parse_spliceai_vcf(vcf)
    assert scores[("17", 43045705, "T", "A")] == 0.20
    assert scores[("17", 43045706, "A", "C")] == 0.04

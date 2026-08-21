"""Tests for the NT scorer (offline, --mock mode only)."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCORER = REPO / "models" / "nt" / "score.py"


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

    assert "nucleotide_transformer" in df1.columns
    assert len(df1) == 3
    assert {"variant_id", "chrom", "pos", "ref", "alt"} <= set(df1.columns)

    r2 = run(out2)
    assert r2.returncode == 0, r2.stderr
    pd.testing.assert_series_equal(
        df1["nucleotide_transformer"], pd.read_parquet(out2)["nucleotide_transformer"]
    )

    log = json.loads((out1.parent / "run_log.json").read_text())
    assert log["mock"] is True
    assert "logP(ref 6-mer" in log["score_definition"]


def test_masked_llr_kmer_logic():
    """Pure-python parts of masked_llr: frame alignment + ref/alt 6-mer build."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("nt_score", SCORER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # variant 'G' at pos0=10 inside a synthetic chromosome; KMER=6
    chrom = "AACCGGTTAA" + "G" + "TTCCAAGGTT" * 50
    pos0 = 10
    # replicate the alignment logic (no model needed)
    start = max(0, pos0 - mod.WINDOW_BP // 2)
    var_off = pos0 - start
    start += var_off % mod.KMER
    var_off = pos0 - start
    tok_idx, within = var_off // mod.KMER, var_off % mod.KMER
    window = chrom[start:].upper()
    ref_kmer = window[tok_idx * mod.KMER:(tok_idx + 1) * mod.KMER]
    assert ref_kmer[within] == "G"
    alt_kmer = ref_kmer[:within] + "C" + ref_kmer[within + 1:]
    assert len(alt_kmer) == mod.KMER and alt_kmer != ref_kmer
    assert alt_kmer[:within] == ref_kmer[:within]

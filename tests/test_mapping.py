"""Guardrail tests for the frozen mapping matrix (data/frozen/frozen-matrix-v1.parquet).

Rules honored here (CONVENTIONS.md):
  - Expected reference alleles are fetched LIVE from Ensembl REST (GRCh38)
    during the test; no expected coordinate is hardcoded.
  - Tests skip (not fail) when the frozen matrix has not been built yet, so
    `make fetch && make test` keeps working on a clean clone.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import pytest

from atlas.mapping import parse_c_hgvs

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN = REPO_ROOT / "data" / "frozen" / "frozen-matrix-v1.parquet"
RAW_MAVEDB = REPO_ROOT / "data" / "raw" / "mavedb"

ENSEMBL_SEQ = "https://rest.ensembl.org/sequence/region/human"

pytestmark = pytest.mark.skipif(not FROZEN.exists(), reason="frozen matrix not built")


@pytest.fixture(scope="module")
def matrix() -> pd.DataFrame:
    return pd.read_parquet(FROZEN)


def test_variant_ids_unique(matrix):
    assert matrix["variant_id"].is_unique


def test_row_counts_match_scores_csv(matrix):
    """Every dataset contributes exactly as many rows as its scores.csv."""
    counts = matrix.groupby("urn").size().to_dict()
    assert counts, "matrix is empty"
    for urn, n in counts.items():
        csv_path = RAW_MAVEDB / urn.replace(":", "_") / "scores.csv"
        with open(csv_path, "rb") as fh:
            expected = sum(1 for _ in fh) - 1  # minus header
        assert n == expected, f"{urn}: matrix {n} rows != scores.csv {expected}"


def test_orientation_is_negative_score(matrix):
    """functional_pathogenicity must equal -score from scores.csv."""
    urn = sorted(matrix["urn"].unique())[0]
    raw = pd.read_csv(
        RAW_MAVEDB / urn.replace(":", "_") / "scores.csv",
        usecols=["accession", "score"],
    ).set_index("accession")
    sample = matrix[matrix["urn"] == urn].head(50)
    merged = sample.join(raw, on="variant_id")
    assert (
        abs(merged["functional_pathogenicity"] + merged["score"]) < 1e-9
    ).all()


def _ensembl_ref_bases(chrom: str, pos: int, length: int) -> str:
    """Live-query the GRCh38 reference bases for chrom:pos..pos+length-1."""
    import requests

    url = f"{ENSEMBL_SEQ}/{chrom}:{pos}..{pos + length - 1}:1?content-type=text/plain"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text.strip().upper()


def test_ref_alleles_match_grch38_live(matrix):
    """Spot-check: mapped ref alleles must equal the live GRCh38 reference.

    A seeded random sample of mapped variants is checked; expected bases come
    from a live Ensembl REST query, never from hardcoded values.
    """
    requests = pytest.importorskip("requests")
    ok = matrix[matrix["mapping_status"] == "ok"]
    rng = random.Random(20260724)
    per_urn = []
    for urn, grp in ok.groupby("urn"):
        idx = rng.sample(list(grp.index), min(3, len(grp)))
        per_urn.extend(idx)
    assert per_urn, "no mapped variants to check"
    try:
        for idx in per_urn:
            row = matrix.loc[idx]
            expected = _ensembl_ref_bases(row["chrom"], int(row["pos"]), len(row["ref"]))
            assert row["ref"] == expected, (
                f"{row['variant_id']} {row['hgvs_c']}: matrix ref {row['ref']} != "
                f"GRCh38 {expected} at {row['chrom']}:{row['pos']}"
            )
    except requests.RequestException as exc:
        pytest.skip(f"Ensembl REST unreachable: {exc}")


# --- pure-unit tests for the HGVS parser (no data, no network) -------------


@pytest.mark.parametrize(
    "hgvs,kind,start,end,ref,alt",
    [
        ("c.5565A>T", "snv", "5565", None, "A", "T"),
        ("c.-19-1G>T", "snv", "-19-1", None, "G", "T"),
        ("c.*43T>G", "snv", "*43", None, "T", "G"),
        ("c.7436-10T>A", "snv", "7436-10", None, "T", "A"),
        ("c.-69del", "del", "-69", None, "", ""),
        ("c.188_189dup", "dup", "188", "189", "", ""),
        ("c.437+15_437+23dup", "dup", "437+15", "437+23", "", ""),
        ("c.249_250delinsAG", "delins", "249", "250", "", "AG"),
        ("c.376-24_376-23insAA", "ins", "376-24", "376-23", "", "AA"),
        ("c.1201_1203delinsGAG", "delins", "1201", "1203", "", "GAG"),
    ],
)
def test_parse_c_hgvs(hgvs, kind, start, end, ref, alt):
    pv = parse_c_hgvs(hgvs)
    assert pv.kind == kind
    assert pv.start.describe() == start
    assert (pv.end.describe() if pv.end else None) == end
    assert pv.ref == ref
    assert pv.alt == alt

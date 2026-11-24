"""atlas.alphagenome_official: the input's order and the trial's draw; the release's table
read to the bit; the two comparisons. Offline: small tables built here."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas import alphagenome_official as ao  # noqa: E402


def matrix() -> pd.DataFrame:
    rows = [("a", "G1", "17", 10, "A", "G", "c.1A>G"),
            ("b", "G1", "17", 11, "AC", "A", "c.2+1del"),
            ("c", "G1", "17", 12, "C", "T", "c.3C>T"),
            ("d", "G2", "2", 20, "T", "TT", "c.40+2dup"),
            ("e", "G2", "2", 21, "GTA", "GC", "c.40+1_40+3delinsC"),
            ("f", "G2", "2", 22, "G", "A", "c.40+15G>A"),
            ("g", "G2", "2", 23, "CAT", "C", "c.41del")]
    m = pd.DataFrame(rows, columns=["variant_id", "gene", "chrom", "pos", "ref", "alt", "hgvs_c"])
    extra = pd.DataFrame({"variant_id": [f"s{i}" for i in range(12)], "gene": "G3", "chrom": "1",
                          "pos": range(100, 112), "ref": "A", "alt": "C",
                          "hgvs_c": [f"c.{i}A>C" for i in range(12)]})
    more = pd.DataFrame({"variant_id": [f"o{i}" for i in range(8)], "gene": "G3", "chrom": "1",
                         "pos": range(200, 208), "ref": "AT", "alt": "A",
                         "hgvs_c": [f"c.{i}del" for i in range(8)]})
    return pd.concat([m, extra, more], ignore_index=True)


def test_kinds():
    assert ao.kind("A", "G") == "snv"
    assert ao.kind("AC", "A") == "deletion"
    assert ao.kind("T", "TT") == "insertion"
    assert ao.kind("GTA", "GC") == "complex"


def test_the_input_puts_the_snvs_first_each_part_in_the_matrix_order():
    t = ao.score_input(matrix())
    snv = t["kind"] == "snv"
    assert snv.iloc[:int(snv.sum())].all() and not snv.iloc[int(snv.sum()):].any()
    assert t.loc[snv, "variant_id"].tolist()[:3] == ["a", "c", "f"]
    assert t.loc[~snv, "variant_id"].tolist()[:4] == ["b", "d", "e", "g"]
    assert t["pos"].dtype == "int64" and t["chrom"].map(type).eq(str).all()


def test_the_trial_draws_ten_snvs_and_a_canonical_deletion_insertion_and_complex():
    m = matrix()
    t = ao.score_input(m)
    hg = m.set_index("variant_id")["hgvs_c"]
    a = ao.trial_input(t, hg, seed=1)
    assert len(a) == 20 and (a["kind"] == "snv").sum() == 10
    roles = dict(zip(a["trial_role"], a["variant_id"]))
    assert roles["canonical_deletion"] == "b"          # the only deletion at +-1-2
    assert roles["canonical_insertion"] == "d"
    assert roles["canonical_complex"] == "e"
    assert a["variant_id"].is_unique
    pd.testing.assert_frame_equal(a, ao.trial_input(t, hg, seed=1))   # fixed by the seed
    with pytest.raises(SystemExit, match="insertion"):
        ao.trial_input(t[t["variant_id"] != "d"].reset_index(drop=True), hg, seed=1)


def test_the_release_table_is_read_round_trip(tmp_path):
    vals = ["0.09389648586511612", "0.1", "1e-05", "0.30000000000000004"]
    p = tmp_path / "all_scores.tsv"
    pd.DataFrame({"variant_id": ["a", "b", "c", "d"], "alphagenome_splicing": vals}).to_csv(
        p, sep="\t", index=False)
    got = ao.read_release(p)["alphagenome_splicing"].tolist()
    assert got == [float(v) for v in vals]


def test_bitwise_equality_counts_nan_as_equal():
    a = pd.Series([1.0, np.nan, 0.1 + 0.2])
    b = pd.Series([1.0, np.nan, 0.3])
    assert ao.bitwise_equal(a, b).tolist() == [True, True, False]


def test_segments_split_where_the_scoring_paused():
    times = pd.Series(["2026-11-24T03:54:03+00:00", "2026-11-24T03:54:04+00:00",
                       "2026-11-24T03:54:05+00:00", "2026-11-24T07:41:22+00:00",
                       "2026-11-24T07:41:23+00:00", "2026-11-24T07:50:00+00:00",
                       "2026-11-24T15:33:50+00:00"])
    s = ao.segments(times.sample(frac=1, random_state=0))          # in any order
    assert [x["n"] for x in s] == [3, 3, 1]                         # 8.6 min stays in one
    assert s[1]["first_scored_utc"] == "2026-11-24T07:41:22+00:00"
    assert s[1]["last_scored_utc"] == "2026-11-24T07:50:00+00:00"
    assert [x["segment"] for x in s] == [1, 2, 3]


def _scores(ids, shift=0.0):
    return pd.DataFrame({"variant_id": ids,
                         **{c: [0.25 + shift] * len(ids) for c in ao.COMPONENTS},
                         ao.COMPOSITE: [0.5] * len(ids), "scored_utc": ["t"] * len(ids)})


def test_the_release_comparison_is_row_by_row_to_the_bit():
    inp = ao.score_input(matrix())
    t = ao.build_table(inp, _scores(inp["variant_id"].tolist()))
    rel = _scores(["a", "c"]).rename(columns={ao.COMPOSITE: ao.RELEASE_COMPOSITE})
    s = pd.DataFrame({"variant_id": ["a", "c"]})
    rows = ao.release_concordance(t, rel, s)
    assert len(rows) == 4 and all(r["holds"] for r in rows)
    rel.loc[1, "SPLICE_SITES"] = 0.25 + 2 ** -40
    rows = {r["column"]: r for r in ao.release_concordance(t, rel, s)}
    assert not rows["alphagenome_splice_sites"]["holds"]
    assert rows["alphagenome_splice_sites"]["n_differing"] == 1
    assert rows[ao.COMPOSITE]["holds"]


def test_the_v5_comparison_gives_rho_overall_by_gene_and_stratum():
    m = matrix()
    inp = ao.score_input(m)
    sc = _scores(inp["variant_id"].tolist())
    sc[ao.COMPOSITE] = np.arange(len(sc), dtype=float)
    t = ao.build_table(inp, sc)
    v5 = pd.DataFrame({"variant_id": inp["variant_id"], "alphagenome": np.arange(len(inp)) * 2.0})
    rows = pd.DataFrame(ao.v5_concordance(t, v5, m.set_index("variant_id")["hgvs_c"]))
    allrow = rows[(rows["scope"] == "all") & (rows["gene"] == "all") & (rows["stratum"] == "all")]
    assert allrow["spearman_rho"].item() == pytest.approx(1.0)
    assert set(rows["scope"]) == {"all", "snv"}
    assert {"splice_1_2", "coding_or_utr"} <= set(rows["stratum"])

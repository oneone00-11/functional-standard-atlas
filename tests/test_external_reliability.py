"""Guards for atlas.external_reliability: the v5 estimators applied to six external
deposits.

The arithmetic tests run anywhere. The tests that read the deposits or the delivered
tables skip when `data/raw/mavedb_external/` or `results/` is absent (a bare clone).
"""
from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas import external_reliability as er  # noqa: E402
from atlas.evaluate import classify_region  # noqa: E402
from atlas.reliability import MIN_N, spearman_brown  # noqa: E402

V5_TSV = REPO / "results" / "reliability_v1.tsv"
V5_JSON = REPO / "results" / "reliability_v1.json"
OUT_TSV = REPO / "results" / "external_reliability_v1.tsv"
OUT_JSON = REPO / "results" / "external_reliability_v1.json"
MANE = REPO / "data" / "raw" / "reference" / "mane" / "MANE.GRCh38.v1.5.summary.txt.gz"


# --------------------------------------------------------------------------
# Spearman-Brown: the k argument must leave the two-replicate default alone
# --------------------------------------------------------------------------
def test_spearman_brown_default_is_the_two_replicate_formula():
    for r in (0.0, 0.2, 0.5, 0.9):
        assert spearman_brown(r) == spearman_brown(r, 2) == pytest.approx(2 * r / (1 + r))
    assert spearman_brown(0.5, 3) == pytest.approx(0.75)
    assert spearman_brown(0.5, 1) == pytest.approx(0.5)
    assert math.isnan(spearman_brown(float("nan")))


def test_spearman_brown_default_still_gives_the_published_v5_values():
    """Locks k = 2 against the delivered v5 table: every replicate row's reliability
    is the two-replicate step-up of its own replicate rho, to the last bit."""
    if not V5_TSV.exists():
        pytest.skip("results/reliability_v1.tsv not present")
    v5 = pd.read_csv(V5_TSV, sep="\t")
    rep = v5[v5["method"] == "replicates"]
    assert len(rep) > 0
    for _, row in rep.iterrows():
        assert spearman_brown(row["rep_rho"]) == row["reliability"], row.to_dict()


# --------------------------------------------------------------------------
# names and territories
# --------------------------------------------------------------------------
def test_transcript_prefix_is_removed_before_classification():
    """classify_region is anchored at `c.`: a prefixed name is filed as coding."""
    name = "NM_006565.4:c.782-23T>A"
    assert classify_region(name) == "coding_or_utr"          # the pitfall
    bare = er.hgvs_c(name, "NM_006565.4")
    assert bare == "c.782-23T>A"
    assert classify_region(bare) == "splice_11_50"
    assert classify_region(er.hgvs_c("ENST00000246868.7:c.625-1G>T",
                                     "ENST00000246868.7")) == "splice_1_2"


@pytest.mark.parametrize("bad", ["NM_006565.3:c.782-23T>A", "c.782-23T>A",
                                 "NC_000016.10:g.67611928T>A", "NM_006565.4:n.100A>G"])
def test_a_name_on_any_other_transcript_is_an_error(bad):
    with pytest.raises(ValueError):
        er.hgvs_c(bad, "NM_006565.4")


def test_every_registered_transcript_is_mane_select():
    if not MANE.exists():
        pytest.skip("MANE summary not present")
    select = {}
    with gzip.open(MANE, "rt") as fh:
        header = fh.readline().lstrip("#").rstrip("\n").split("\t")
        for line in fh:
            rec = dict(zip(header, line.rstrip("\n").split("\t")))
            if rec["MANE_status"] == "MANE Select":
                select[rec["symbol"]] = {rec["RefSeq_nuc"], rec["Ensembl_nuc"]}
    for urn, spec in er.EXTERNAL.items():
        assert spec["transcript"] in select[spec["gene"]], (urn, spec)


# --------------------------------------------------------------------------
# error-model checks
# --------------------------------------------------------------------------
def _se_frame(n=200, seed=0, ratio=1.96):
    rng = np.random.default_rng(seed)
    score = rng.normal(0, 1, n)
    se = rng.uniform(0.05, 0.2, n)
    return pd.DataFrame({"score": score, "se": se,
                         "lo": score - ratio * se, "hi": score + ratio * se})


def test_check_se_accepts_an_interval_the_se_reproduces():
    chk = er.check_se(_se_frame(), "se", ("lo", "hi"))
    assert chk["status"] == "validated" and chk["rows_checked"] == 200


def test_check_se_rejects_an_interval_built_on_another_scale():
    df = _se_frame(ratio=1.645)                       # a 90% interval
    assert er.check_se(df, "se", ("lo", "hi"))["status"] == "ci_mismatch"
    df = _se_frame()
    df.loc[0, "lo"] -= 0.5                            # one bad lower bound is enough
    assert er.check_se(df, "se", ("lo", "hi"))["status"] == "ci_mismatch"


def test_replicate_summary_tells_mean_from_median():
    reps = ("a", "b", "c")
    r = pd.DataFrame({"a": [1.0, 2.0, 5.0], "b": [2.0, 4.0, 1.0], "c": [6.0, 3.0, 3.0]})
    assert er.replicate_summary(r.assign(score=r.mean(axis=1)), reps) == "mean"
    assert er.replicate_summary(r.assign(score=r.median(axis=1)), reps) == "median"
    assert er.replicate_summary(r.assign(score=r["a"]), reps) == "other"


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------
def test_se_rows_use_the_v5_formula_and_the_minimum_count():
    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame({"score": rng.normal(0, 2, n), "se": np.full(n, 1.0),
                       "region": ["splice_3_10"] * n})
    df.loc[: MIN_N - 2, "region"] = "splice_1_2"     # 29 rows: below MIN_N
    rows = pd.DataFrame(er.se_rows(df, "G", "urn:x", "se", "validated"))
    assert "splice_1_2" not in set(rows["stratum"])
    allrow = rows[rows["stratum"] == "all"].iloc[0]
    expect = 1 - 1.0 / df["score"].var()
    assert allrow["reliability"] == pytest.approx(expect)
    assert allrow["ceiling"] == pytest.approx(math.sqrt(expect))
    assert allrow["role"] == "main"

    noisy = df.assign(se=10.0)                       # error swamps the spread
    r = pd.DataFrame(er.se_rows(noisy, "G", "urn:x", "se", "validated"))
    assert (r["reliability"] == 0).all() and (r["ceiling"] == 0).all()

    bad = pd.DataFrame(er.se_rows(df, "G", "urn:x", "se", "ci_mismatch"))
    assert set(bad["role"]) == {"excluded"}


def _replicates(n=300, seed=2):
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 1, n)
    d = pd.DataFrame({f"r{i}": truth + rng.normal(0, 0.8, n) for i in (1, 2, 3)})
    return d.assign(region="splice_3_10")


def test_a_median_score_gives_two_bounding_rows_and_no_main_row():
    reps = ("r1", "r2", "r3")
    d = _replicates()
    rows = pd.DataFrame(er.replicate_rows(d, "G", "urn:x", reps, "median"))
    assert "main" not in set(rows["role"])
    one = rows[(rows["stratum"] == "all") & (rows["role"] == "lower_bound")].iloc[0]
    three = rows[(rows["stratum"] == "all") & (rows["role"] == "upper_bound")].iloc[0]
    rbar = np.mean([d["r1"].corr(d["r2"], method="spearman"),
                    d["r1"].corr(d["r3"], method="spearman"),
                    d["r2"].corr(d["r3"], method="spearman")])
    assert one["k"] == 1 and one["reliability"] == pytest.approx(rbar)
    assert three["k"] == 3 and three["reliability"] == pytest.approx(3 * rbar / (1 + 2 * rbar))
    assert set(rows["status"]) == {"median_of_replicates"}


def test_a_mean_score_enters_the_main_estimate_at_k_equal_to_the_replicate_count():
    rows = pd.DataFrame(er.replicate_rows(_replicates(), "G", "urn:x",
                                          ("r1", "r2", "r3"), "mean"))
    assert set(rows["role"]) == {"main"} and set(rows["k"]) == {3}
    assert set(rows["status"]) == {"validated"}


# --------------------------------------------------------------------------
# combining with v5
# --------------------------------------------------------------------------
def _v5_like():
    return pd.DataFrame([
        {"gene": "A", "stratum": "all", "method": "replicates", "n": 50, "ceiling": 0.80,
         "status": "validated"},
        {"gene": "A", "stratum": "all", "method": "se", "n": 50, "ceiling": 0.90,
         "status": "derived_from_replicates"},
        {"gene": "B", "stratum": "all", "method": "se", "n": 50, "ceiling": 0.70,
         "status": "validated"},
        {"gene": "C", "stratum": "all", "method": "se", "n": 50, "ceiling": 0.99,
         "status": "unvalidated"},
    ])


def test_v5_primary_route_takes_replicates_and_drops_unvalidated_rows():
    p = er.v5_primary(_v5_like())
    assert sorted(p["gene"]) == ["A", "B"]
    assert p.set_index("gene").loc["A", "method"] == "replicates"
    assert p.set_index("gene").loc["A", "ceiling"] == 0.80


def test_v5_primary_route_prefers_replicates_when_both_estimators_are_validated():
    both = pd.concat([_v5_like(), pd.DataFrame([
        {"gene": "E", "stratum": "all", "method": "se", "n": 50, "ceiling": 0.95,
         "status": "validated"},
        {"gene": "E", "stratum": "all", "method": "replicates", "n": 50,
         "ceiling": 0.85, "status": "validated"}])], ignore_index=True)
    p = er.v5_primary(both).set_index("gene")
    assert p.loc["E", "method"] == "replicates" and p.loc["E", "ceiling"] == 0.85


def test_on_the_delivered_v5_table_the_selection_matches_realisation_table():
    """atlas.reliability.realisation_table sorts method names in descending order,
    which would keep 'se' if a gene x stratum had two validated rows. None does, so
    both selections give the same rows on the published table."""
    if not V5_TSV.exists():
        pytest.skip("results/reliability_v1.tsv not present")
    v5 = pd.read_csv(V5_TSV, sep="\t")
    validated = v5[v5["status"] == "validated"]
    assert not validated.duplicated(["gene", "stratum"]).any()
    theirs = (validated.sort_values("method", ascending=False)
                       .drop_duplicates(["gene", "stratum"], keep="first"))
    key = ["gene", "stratum", "method", "ceiling"]
    mine = er.v5_primary(v5)[key].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(mine, theirs[key].sort_values(key).reset_index(drop=True))


def test_bounded_rows_stay_out_of_the_main_estimate():
    ext = pd.DataFrame([
        {"gene": "D", "stratum": "all", "n": 60, "rep_rho": np.nan, "k": np.nan,
         "reliability": 0.36, "ceiling": 0.60, "status": "validated", "role": "main"},
        {"gene": "T", "stratum": "all", "n": 60, "rep_rho": 0.5, "k": 1,
         "reliability": 0.25, "ceiling": 0.50, "status": "median_of_replicates",
         "role": "lower_bound"},
        {"gene": "T", "stratum": "all", "n": 60, "rep_rho": 0.5, "k": 3,
         "reliability": 0.81, "ceiling": 0.90, "status": "median_of_replicates",
         "role": "upper_bound"},
    ])
    s = er.summarise(ext, _v5_like())
    main = s["main"][0]
    assert main["genes"] == ["A", "B", "D"] and main["n_genes"] == 3
    assert main["ceiling_median"] == pytest.approx(0.70)
    assert main["ceiling_min"] == pytest.approx(0.60) and main["ceiling_max"] == pytest.approx(0.80)
    low = s["main_plus_bounded_deposit"]["lower_bound"][0]
    high = s["main_plus_bounded_deposit"]["upper_bound"][0]
    assert low["genes"] == high["genes"] == ["A", "B", "D", "T"]
    assert low["ceiling_min"] == pytest.approx(0.50) and high["ceiling_max"] == pytest.approx(0.90)
    assert s["v5_three_genes"][0]["genes"] == ["A", "B"]


def test_the_v5_side_reproduces_the_published_v5_summary():
    """The three-gene figures in the new summary are v5's, not a recomputation."""
    if not (V5_TSV.exists() and V5_JSON.exists()):
        pytest.skip("v5 results not present")
    mine = {r["stratum"]: r for r in
            er.stratum_summary(er.v5_primary(pd.read_csv(V5_TSV, sep="\t")))}
    published = json.loads(V5_JSON.read_text())["ceiling_by_stratum"]
    assert len(published) == len(mine)
    for p in published:
        m = mine[p["stratum"]]
        assert m["n_genes"] == p["genes"]
        assert m["ceiling_median"] == pytest.approx(p["ceiling_median"], abs=1e-15)
        assert m["ceiling_min"] == pytest.approx(p["ceiling_min"], abs=1e-15)
        assert m["ceiling_max"] == pytest.approx(p["ceiling_max"], abs=1e-15)


# --------------------------------------------------------------------------
# the delivered tables
# --------------------------------------------------------------------------
def _need_inputs():
    if not (er.RAW / "MANIFEST.json").exists():
        pytest.skip("data/raw/mavedb_external not present")
    if not (OUT_TSV.exists() and OUT_JSON.exists() and V5_TSV.exists()):
        pytest.skip("results not present")


def test_the_delivered_table_regenerates_from_the_deposits():
    _need_inputs()
    ext, checks = er.external_table()
    # round_trip: the default C parser can be one ulp off on 17-digit values
    delivered = pd.read_csv(OUT_TSV, sep="\t", float_precision="round_trip")
    pd.testing.assert_frame_equal(ext.reset_index(drop=True), delivered,
                                  check_dtype=False, check_exact=True)
    summary = json.loads(OUT_JSON.read_text())
    again = er.summarise(ext, pd.read_csv(V5_TSV, sep="\t"))
    assert json.loads(json.dumps(again["main"], default=str)) == summary["main"]
    assert summary["checks"] == json.loads(json.dumps(checks, default=str))


def test_the_delivered_checks_hold():
    """Every SE deposit reconciles with its interval, TP53's score is the median of its
    replicates (so it is bounded, not estimated), and the main estimate holds the
    three v5 genes plus the five SE deposits."""
    _need_inputs()
    s = json.loads(OUT_JSON.read_text())
    for urn, rec in s["checks"].items():
        if er.EXTERNAL[urn]["method"] == "se":
            assert rec["se_check"]["status"] == "validated", urn
        else:
            assert rec["score_is"].startswith("median of"), urn
    main_genes = set().union(*[set(r["genes"]) for r in s["main"]])
    assert main_genes == {"BRCA1", "BARD1", "PALB2", "CTCF", "SFPQ", "RAD51D",
                          "XRCC2", "SBDS"}
    assert {r["gene"] for r in s["bounded_rows"]} == {"TP53"}

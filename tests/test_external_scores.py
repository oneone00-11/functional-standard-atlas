"""Guards for atlas.external_scores: the four checks of the variant tables, the scoring
input, and the comparison with the companion release. Synthetic data only; nothing here
reads the network, the companion repository or data/raw."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas import external_scores as xs  # noqa: E402

TX = "NM_000001.1"


def _table(rows):
    """Rows of a release variant table: (variant_urn, c. name, stratum, score)."""
    return pd.DataFrame([{"gene": "GENE1", "variant_urn": u, "hgvs_nt": f"{TX}:{c}",
                          "transcript": TX, "stratum": s, "score": sc} for u, c, s, sc in rows])


def _deposit(rows):
    """A deposit as atlas.external_reliability.load_deposit returns it."""
    from atlas.evaluate import classify_region
    d = pd.DataFrame([{"accession": u, "hgvs_c": c, "score": sc} for u, c, sc in rows])
    d["region"] = d["hgvs_c"].map(classify_region)
    return d


DEP = _deposit([("u#1", "c.100+1G>A", -0.30), ("u#2", "c.100+5G>T", -0.10),
                ("u#3", "c.101-20C>T", 0.02), ("u#4", "c.101-80A>G", 0.01),
                ("u#5", "c.100+3_100+5del", -0.20), ("u#6", "c.50A>G", 0.00)])
TAB = _table([("u#1", "c.100+1G>A", "pm12", -0.30), ("u#2", "c.100+5G>T", "s3_10", -0.10),
              ("u#3", "c.101-20C>T", "s11_50", 0.02), ("u#4", "c.101-80A>G", "out_of_scope_deep", 0.01)])


# --------------------------------------------------------------------------
# the four checks
# --------------------------------------------------------------------------
def test_check_one_finds_a_missing_variant_and_a_different_score():
    ok = xs.check_urns_and_scores(TAB, DEP)
    assert ok == {"rows": 4, "missing": [], "score_differs": []}
    bad = TAB.copy()
    bad.loc[0, "variant_urn"] = "u#9"
    bad.loc[1, "score"] = -0.1000001
    res = xs.check_urns_and_scores(bad, DEP)
    assert res["missing"] == ["u#9"] and res["score_differs"] == ["u#2"]


def test_check_two_compares_the_stratum_on_the_bare_c_name():
    assert xs.check_strata(TAB)["differs"] == []
    bad = TAB.assign(stratum=["s3_10", "s3_10", "s11_50", "out_of_scope_deep"])
    assert xs.check_strata(bad)["differs"] == ["u#1"]
    with pytest.raises(ValueError):                  # another transcript is an error
        xs.check_strata(TAB.assign(transcript="NM_999.1"))


def test_check_three_wants_exactly_the_intronic_snvs_and_only_deletions_besides():
    res = xs.check_completeness(TAB, DEP)
    assert res["deposit_intronic_snv"] == 4 and res["table_rows"] == 4
    assert res["only_in_deposit"] == [] and res["only_in_table"] == []
    assert res["other_intronic"] == 1 and res["other_intronic_not_deletion"] == []
    short = xs.check_completeness(TAB.iloc[1:], DEP)
    assert short["only_in_deposit"] == ["u#1"]
    dup = _deposit([("u#7", "c.100+2dup", 0.0)])
    res = xs.check_completeness(TAB, pd.concat([DEP, dup], ignore_index=True))
    assert res["other_intronic_not_deletion"] == ["c.100+2dup"]


def _meta(abnormal_hi=-0.05, normal_lo=-0.02):
    fc = [{"functionalClassification": "abnormal", "label": "Functionally Abnormal",
           "range": [None, abnormal_hi], "inclusiveLowerBound": False, "inclusiveUpperBound": True},
          {"functionalClassification": "not_specified", "label": "Indeterminate",
           "range": [abnormal_hi, normal_lo], "inclusiveLowerBound": False,
           "inclusiveUpperBound": True},
          {"functionalClassification": "normal", "label": "Functionally Normal",
           "range": [normal_lo, None], "inclusiveLowerBound": False, "inclusiveUpperBound": False}]
    return {"scoreCalibrations": [
        {"title": "ExCALIBR calibration", "investigatorProvided": True, "functionalClassifications": []},
        {"title": "Investigator-provided functional classes", "investigatorProvided": True,
         "functionalClassifications": fc}]}


def test_check_four_reads_the_investigator_classes_with_their_bounds():
    src, cls = xs.deposit_classes(_meta(), DEP)
    assert src.endswith("Investigator-provided functional classes")
    assert list(cls) == ["abnormal", "abnormal", "normal", "normal", "abnormal", "normal"]
    edge = _deposit([("e#1", "c.1+1G>A", -0.05), ("e#2", "c.1+1G>C", -0.02)])
    assert list(xs.deposit_classes(_meta(), edge)[1]) == ["abnormal", "not_specified"]
    res = xs.check_direction(TAB, DEP, _meta())
    assert res["deposit"]["abnormal_lower"] and res["table_rows"]["abnormal_lower"]


def test_check_four_catches_a_reversed_scale_and_uses_the_column_without_calibrations():
    labels = ["functionally_abnormal", "indeterminate", "functionally_normal",
              "functionally_normal", "functionally_abnormal", "functionally_normal"]
    labelled = DEP.assign(functional_consequence=labels)
    src, cls = xs.deposit_classes({"scoreCalibrations": []}, labelled)
    assert "functional_consequence" in src
    assert list(cls) == ["abnormal", "indeterminate", "normal", "normal", "abnormal", "normal"]
    no_cal = {"scoreCalibrations": []}
    assert xs.check_direction(TAB, labelled, no_cal)["deposit"]["abnormal_lower"]
    reversed_ = labelled.assign(score=-labelled["score"])       # the same labels, scale flipped
    res = xs.check_direction(TAB.assign(score=-TAB["score"]), reversed_, no_cal)
    assert not res["deposit"]["abnormal_lower"] and not res["table_rows"]["abnormal_lower"]
    with pytest.raises(xs.CheckFailed):
        xs.deposit_classes({"scoreCalibrations": []}, DEP)


# --------------------------------------------------------------------------
# the scoring input
# --------------------------------------------------------------------------
def _input(**over):
    v = pd.DataFrame({"variant_id": ["1-100-A-G", "1-105-C-T"], "gene": ["G", "G"],
                      "chrom": ["1", "1"], "pos": [100, 105], "ref": ["A", "C"],
                      "alt": ["G", "T"], "variant_urn": ["u#1", "u#2"],
                      "stratum": ["pm12", "s3_10"]})
    for k, val in over.items():
        v[k] = val
    return v


def test_the_scoring_input_is_keyed_by_chrom_pos_ref_alt_and_holds_snvs_only():
    assert list(xs.score_input(_input())["variant_id"]) == ["1-100-A-G", "1-105-C-T"]
    with pytest.raises(xs.CheckFailed):
        xs.score_input(_input(variant_id=["1-100-A-G", "1-100-A-G"]))
    with pytest.raises(xs.CheckFailed):
        xs.score_input(_input(pos=[100, 106]))
    with pytest.raises(xs.CheckFailed):
        xs.score_input(_input(variant_id=["1-100-AT-G", "1-105-C-T"], ref=["AT", "C"]))


def test_the_release_files_include_every_table_and_the_run_logs():
    files = xs.companion_files()
    for g in xs.GENES:
        assert f"{xs.EXT_DIR}/{g.lower()}_variants.parquet" in files
        for run in xs.COMPARISON_RUNS:
            assert f"{xs.EXT_DIR}/{g.lower()}_{run}.parquet" in files
    assert f"{xs.AG_DIR}/NOTEBOOK_SOURCE.json" in files
    assert all(f"{xs.RUN_LOGS}/{n}" in files for n in xs.RELEASE_RUN_LOGS)
    assert len(files) == len(set(files))


# --------------------------------------------------------------------------
# the comparison with the release
# --------------------------------------------------------------------------
def _scored(values):
    """A delivered table with one score column per CONCORDANCE entry set to values."""
    n = len(values)
    t = pd.DataFrame({"variant_id": [f"1-{i}-A-G" for i in range(n)],
                      "gene": ["CTCF"] * n, "stratum": ["splice_3_10"] * n})
    for col in xs.CONCORDANCE:
        t[col] = values
    return t


def test_the_lookup_columns_must_equal_the_release_row_by_row():
    t = _scored([0.1, 0.2, np.nan, 0.4])
    same = lambda run, col: t.set_index("variant_id")[[c for c in xs.CONCORDANCE][0]]  # noqa: E731
    conc = xs.concordance(t, release=same)
    all_ = conc[conc["gene"] == "all"].set_index("column")
    assert all_.loc["cadd", "holds"] and all_.loc["cadd", "equal_rows"] == 4   # NaN = NaN
    shifted = lambda run, col: same(run, col) + 1e-12  # noqa: E731
    conc = xs.concordance(t, release=shifted).set_index(["gene", "column"])
    assert not conc.loc[("all", "cadd"), "holds"]
    assert conc.loc[("all", "spliceai_ds"), "holds"]                  # within 0.001


def test_splice_columns_stop_beyond_a_thousandth_and_alphagenome_is_reported():
    t = _scored([0.1, 0.2, 0.3, 0.4])
    off = lambda run, col: t.set_index("variant_id")["cadd"] + np.array([0, 0, 0, 0.0015])  # noqa: E731
    conc = xs.concordance(t, release=off).set_index(["gene", "column"])
    assert not conc.loc[("all", "spliceai_ds"), "holds"]
    assert math.isclose(conc.loc[("all", "spliceai_ds"), "max_abs_diff"], 0.0015, rel_tol=1e-9)
    ag = conc.loc[("all", "alphagenome_official")]
    assert ag["holds"] is None and ag["spearman_rho"] == pytest.approx(1.0)


def test_rows_outside_three_to_fifty_are_not_compared():
    t = _scored([0.1, 0.2, 0.3, 0.4]).assign(stratum=["splice_1_2", "splice_3_10",
                                                      "splice_11_50", "splice_deep"])
    rel = lambda run, col: t.set_index("variant_id")["cadd"]  # noqa: E731
    conc = xs.concordance(t, release=rel)
    assert conc[conc["gene"] == "all"]["rows"].unique().tolist() == [2]


def test_scored_counts_by_gene_stratum_and_column():
    t = _scored([0.1, np.nan, 0.3, 0.4])
    for c in xs.SCORE_COLUMNS + list(xs.AG_COMPONENTS.values()):
        if c not in t.columns:
            t[c] = 0.5
    t["stratum"] = ["splice_1_2", "splice_1_2", "splice_3_10", "splice_3_10"]
    c = xs.scored_counts(t)
    assert c["CTCF"]["splice_1_2"]["rows"] == 2 and c["CTCF"]["splice_1_2"]["cadd"] == 1
    assert c["CTCF"]["splice_3_10"]["alphagenome_splice_sites"] == 2

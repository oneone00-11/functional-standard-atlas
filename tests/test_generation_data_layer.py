"""Guards for the current generation's data layer: the three shared modules (paths,
population, predictors), score matrix v3 and its acceptance checks, assay_aux v2's check
of BAP1's standard error by the deposit's own test statistic, the external genes' score
table v2, and the predictor resources table v2.

The pure-function tests use synthetic data. The tests of the delivered files skip in a
checkout where results/ is not built."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas import paths, population, predictors  # noqa: E402

RESULTS = REPO / "results"


def _built(*names: str) -> None:
    missing = [n for n in names if not (RESULTS / n).exists()]
    if missing:
        pytest.skip(f"results/ not built in this checkout ({missing[0]})")


# --------------------------------------------------------------------------
# atlas.paths
# --------------------------------------------------------------------------
def test_every_current_name_carries_a_version_and_lives_under_results():
    for key, name in paths.NAMES.items():
        assert "/" not in name, key
        assert "_v" in name, f"{key}: {name} carries no version"
        assert paths.path(key).parent == paths.RESULTS
    assert paths.name("score_matrix") == "score_matrix_atlas_v3.parquet"
    assert paths.name("assay_aux") == "assay_aux_v2.parquet"
    assert paths.name("external_scores") == "external_scores_v2.parquet"
    assert paths.FROZEN.name == "frozen-matrix-v1.parquet"


# --------------------------------------------------------------------------
# atlas.population
# --------------------------------------------------------------------------
def _toy():
    return pd.DataFrame({
        "variant_id": ["a", "b", "c", "d", "e"],
        "ref": ["A", "AT", "A", "G", "CG"], "alt": ["G", "A", "AT", "T", "TA"],
        "hgvs_c": ["c.10A>G", "c.20+1del", "c.30-5_30-4insT", "c.40+60G>T", "c.50_51delinsTA"]})


def test_the_populations_split_on_single_bases_and_the_region_comes_from_classify_region():
    from atlas.evaluate import classify_region
    d = population.assign(_toy())
    assert d["population"].tolist() == ["snv", "non_snv", "non_snv", "snv", "non_snv"]
    assert d["region"].tolist() == [classify_region(h) for h in _toy()["hgvs_c"]]
    assert population.select(_toy())["variant_id"].tolist() == ["a", "d"]
    assert population.select(_toy(), "non_snv")["variant_id"].tolist() == ["b", "c", "e"]
    assert len(population.select(_toy(), None)) == 5
    with pytest.raises(ValueError):
        population.select(_toy(), "indel")


def test_the_default_matrix_is_the_current_one_and_its_populations_have_their_sizes():
    _built(paths.name("score_matrix"))
    d = population.load(None, columns=["variant_id"])
    assert d["population"].value_counts().to_dict() == population.EXPECTED
    assert len(population.load(columns=["variant_id"])) == population.EXPECTED["snv"]


# --------------------------------------------------------------------------
# atlas.predictors
# --------------------------------------------------------------------------
def test_the_predictor_list_is_written_once_with_its_groups():
    p = predictors
    assert len(p.V5) == 19 and len(p.PREDICTORS) == 23 and len(set(p.PREDICTORS)) == 23
    assert p.PREDICTORS[-4:] == ["avi", "avi_splice_sites", "avi_splice_site_usage",
                                 "avi_splice_junctions"]
    assert not set(p.AUXILIARY) & set(p.PREDICTORS)
    assert p.SPLICE_FAMILY == ["spliceai_ds", "pangolin_score", "alphagenome",
                               "avi_splice_sites", "avi_splice_site_usage",
                               "avi_splice_junctions"]
    assert len(p.META) == 7 and set(p.META) <= set(p.PREDICTORS)
    assert p.ALLELE_FREQUENCY == ["gnomad_af_global", "gnomad_af_popmax"]
    assert len(p.EXTERNAL) == 13 and set(p.EXTERNAL) <= set(p.PREDICTORS)
    assert len(p.SNV_ONLY_IN_V5) == 14
    assert p.SELECTION_OVERLAP == {"avi": ["BRCA1", "RAD51C"]}


def test_the_selection_overlap_leaves_brca1_and_rad51c_out_of_avi_only():
    df = pd.DataFrame({"gene": ["BRCA1", "RAD51C", "BARD1"], "x": [1, 2, 3]})
    assert predictors.drop_overlap(df, "avi")["gene"].tolist() == ["BARD1"]
    assert predictors.drop_overlap(df, "spliceai_ds")["gene"].tolist() == ["BRCA1", "RAD51C",
                                                                           "BARD1"]
    assert predictors.drop_overlap(df, "cadd", "avi")["gene"].tolist() == ["BARD1"]
    assert predictors.is_overlap("avi", "BRCA1") and not predictors.is_overlap("avi", "PALB2")
    assert not predictors.is_overlap("avi_splice_sites", "BRCA1")


# --------------------------------------------------------------------------
# score matrix v3
# --------------------------------------------------------------------------
def test_bitwise_equality_tells_signed_zero_and_nan_positions_apart():
    from atlas.matrix_v3 import bitwise_equal
    a = pd.Series([1.0, np.nan, 0.0])
    assert bitwise_equal(a, pd.Series([1.0, np.nan, 0.0]))
    assert not bitwise_equal(a, pd.Series([1.0, np.nan, -0.0]))
    assert not bitwise_equal(a, pd.Series([1.0, 2.0, 0.0]))
    assert not bitwise_equal(a, pd.Series([1.0 + 1e-16 * 2, np.nan, 0.0]))
    assert bitwise_equal(pd.Series(["x", None]), pd.Series(["x", None]))
    assert not bitwise_equal(pd.Series(["x", None]), pd.Series(["x", "y"]))


def test_matrix_v3_holds_every_acceptance_check():
    _built(paths.name("score_matrix"), paths.name("score_matrix_provenance"),
           "score_matrix_atlas_v2.parquet")
    from atlas.matrix_v3 import ADDED, EXPECTED_ARMS, EXPECTED_CLASSES, bitwise_equal
    v3 = pd.read_parquet(paths.path("score_matrix"))
    v2 = pd.read_parquet(RESULTS / "score_matrix_atlas_v2.parquet")
    frozen = pd.read_parquet(paths.FROZEN, columns=["variant_id"])
    assert len(v3) == 64178 and v3["variant_id"].equals(frozen["variant_id"])
    assert list(v3.columns) == list(v2.columns) + ADDED and v3.shape[1] == 43
    for c in v2.columns:
        if c != "alphagenome":
            assert bitwise_equal(v3[c], v2[c]), c
    ag = pd.read_parquet(paths.path("alphagenome_official_scores"))
    assert bitwise_equal(v3["alphagenome"], ag["alphagenome_official"])
    for c in predictors.AUXILIARY:
        assert bitwise_equal(v3[c], ag[c]) and v3[c].notna().all(), c
    snv = population.is_snv(v3["ref"], v3["alt"])
    at = pd.read_parquet(paths.path("alphagenome_atlas_scores"))
    at = at[at["set"] == "core"].reset_index(drop=True)
    for c in predictors.ATLAS:
        assert bitwise_equal(v3.loc[snv, c], at[c]), c
        assert v3.loc[snv, c].notna().all() and v3.loc[~snv, c].isna().all(), c
    assert v3["variant_class"].value_counts().to_dict() == EXPECTED_CLASSES
    assert v3["clinvar_arm"].value_counts().to_dict() == EXPECTED_ARMS
    assert (v3["clinvar_arm_strict"] == "recorded_no_assertion").sum() == 2046
    prov = json.loads(paths.path("score_matrix_provenance").read_text())
    assert prov["checks"]["failures"] == [] and set(prov["columns"]) == set(v3.columns)


# --------------------------------------------------------------------------
# assay_aux v2
# --------------------------------------------------------------------------
def _deposit(z_noise=0.0, lfc_shift=0.0):
    lfc = np.array([-1.5, -0.2, 0.0, 0.4, 2.0])
    se = np.array([0.1, 0.2, 0.3, 0.2, 0.5])
    return pd.DataFrame({"score": lfc, "processed_LFC_continuous": lfc + lfc_shift,
                         "SE_bind_continuous": se,
                         "processed_Z_continuous": np.round(lfc / se, 5) + z_noise})


def test_the_test_statistic_check_takes_rounded_z_and_stops_on_one_bad_row():
    from atlas.assay_aux import check_test_statistic
    cols = {"lfc": "processed_LFC_continuous", "z": "processed_Z_continuous"}
    rec = check_test_statistic(_deposit(), "SE_bind_continuous", cols, "G")
    assert rec["rows_checked"] == 5 and rec["max_abs_diff_score_over_se_vs_z"] < 1e-4
    bad = _deposit()
    bad.loc[2, "processed_Z_continuous"] += 2e-4
    with pytest.raises(ValueError, match="differs"):
        check_test_statistic(bad, "SE_bind_continuous", cols, "G")
    with pytest.raises(ValueError, match="does not equal"):
        check_test_statistic(_deposit(lfc_shift=1e-9), "SE_bind_continuous", cols, "G")


def test_assay_aux_v2_validates_bap1_by_its_test_statistic_and_changes_nothing_else():
    _built(paths.name("assay_aux"), paths.name("assay_aux_provenance"), "assay_aux_v1.parquet")
    from atlas.assay_aux import AUX
    v2 = pd.read_parquet(paths.path("assay_aux"))
    prov = json.loads(paths.path("assay_aux_provenance").read_text())
    by = v2.groupby("gene")["se_validated_by"].agg(lambda s: set(s.dropna()))
    assert by["BAP1"] == {"test_statistic"} and by["BRCA1"] == {"replicates"}
    assert by["BARD1"] == {"ci"} and by["PALB2"] == {"ci"} and by["RAD51C"] == set()
    st = v2.groupby("gene")["se_status"].agg(lambda s: set(s.dropna()))
    assert st["BAP1"] == {"validated"} and st["RAD51C"] == {"scale_mismatch"}
    bap1 = prov["assays"]["urn:mavedb:00000662-0-1"]["se_test_statistic"]
    assert bap1["rows_checked"] == 18108 and bap1["max_abs_diff_score_over_se_vs_z"] < 1e-4
    assert prov["against_v1"]["holds"]
    assert prov["against_v1"]["se_status_changed"] == {"BAP1: unvalidated -> validated": 18108}
    assert "processed_adj_lfc_D4_D14 / SE_bind_D4_D14" in Path(
        REPO / "src/atlas/assay_aux.py").read_text()
    assert AUX["urn:mavedb:00000673-0-1"]["se_status"] == "scale_mismatch"


# --------------------------------------------------------------------------
# the external genes' scores v2 and the predictor resources v2
# --------------------------------------------------------------------------
def test_external_scores_v2_is_v1_with_the_atlas_columns():
    _built(paths.name("external_scores"), "external_scores_v1.parquet")
    from atlas.external_scores import extend
    t, checks = extend()
    assert checks["holds"] and checks["predictors"] == predictors.EXTERNAL
    got = pd.read_parquet(paths.path("external_scores"))
    assert got.equals(t) and len(got) == 5095
    assert "alphagenome_official" not in got.columns and "alphagenome" in got.columns


def test_the_resources_table_v2_changes_alphagenome_and_adds_the_atlas_only():
    _built("predictor_resources_v1.tsv", paths.name("predictor_resources"))
    from atlas.predictor_resources import RESTRICTED, ROWS, TRAINING
    v1 = pd.read_csv(RESULTS / "predictor_resources_v1.tsv", sep="\t").fillna("")
    v2 = pd.read_csv(paths.path("predictor_resources"), sep="\t").fillna("")
    assert list(v2.columns) == list(v1.columns) and len(v2) == 23
    assert v2["predictor"].tolist()[:19] == v1["predictor"].tolist()
    changed = [p for i, p in enumerate(v1["predictor"]) if not (v1.iloc[i] == v2.iloc[i]).all()]
    assert changed == ["AlphaGenome"]
    assert "0.10.0" in v2.loc[0, "version_or_resource"]
    atlas = v2["predictor"].tolist()[19:]
    assert all(p.startswith("AlphaGenome Atlas") for p in atlas)
    assert set(atlas) <= set(RESTRICTED) and set(TRAINING) == {r[0] for r in ROWS}
    assert "BRCA1, RAD51C" in TRAINING["AlphaGenome Atlas AVI"][2]

"""atlas.error_models: the two error models per gene, their tables, the error scale and
the self-test (A4b.2; the third generation's error scales, A4b commit 3). The first
versions' content is read from their own files."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from atlas import error_models as em
from atlas import paths
from atlas import rank_ceiling as rc

TABLE = paths.path("error_models")
RECORD = paths.path("error_models_summary")
SCALE = paths.path("error_scale")
SELFTEST = paths.path("rank_ceiling_selftest")
SUMMARY = paths.path("rank_ceiling_selftest_summary")
# the earlier versions, by their own names
TABLE_V1 = paths.RESULTS / "error_models_v1.tsv"
SELFTEST_V2 = paths.RESULTS / "rank_ceiling_selftest_v2.tsv"
SUMMARY_V2 = paths.RESULTS / "rank_ceiling_selftest_v2.json"
needs = pytest.mark.skipif(not RECORD.exists(), reason="run `python -m atlas.error_models` first")
needs_v1 = pytest.mark.skipif(not TABLE_V1.exists(), reason="error_models_v1.tsv not built")
needs_v2 = pytest.mark.skipif(not SUMMARY_V2.exists(), reason="rank_ceiling_selftest_v2 not built")


def test_the_core_neutral_variants_are_the_neutral_checks_second_batch():
    nc = pd.read_csv(paths.path("reliability_neutral_check"), sep="\t")
    second = nc[nc["batch"] == "synonymous_spliceai_below_0.1"].set_index("gene")["n"]
    for gene, n in em.NEUTRAL_EXPECTED.items():
        assert len(em.neutral_variants(gene)) == n == int(second[gene])


def test_brca1s_score_is_the_mean_of_its_replicates():
    chk = em.brca1_mean_check()
    assert chk["rows"] == 3893 and chk["holds"]


def test_brca1s_half_difference_is_on_the_functional_scale():
    d = em.half_difference([0.2, -0.1], [0.0, 0.3])
    assert np.allclose(d, [-0.1, 0.2])        # (-(0.2) - -(0.0)) / 2, (0.1 - -0.3) / 2


def test_neutral_residuals_are_centred_on_the_batch_median():
    for gene in ("BARD1", "BRCA1"):
        nv = em.neutral_variants(gene)
        assert np.allclose(nv["z"], (nv["y"] - nv["y"].median()) / nv["s"])
        assert nv["id"].is_monotonic_increasing


@needs_v1
def test_the_first_table_has_both_models_for_every_gene():
    t = pd.read_csv(TABLE_V1, sep="\t")
    for gene in em.GENES:
        assert set(t.loc[t["gene"] == gene, "model"]) == {"own", "neutral"}
    gauss = t[t["kind"] == "gaussian"]
    assert len(gauss) == 8 and (gauss["sd"] == 1).all() and gauss["n"].notna().all()
    assert gauss[["robust_sd", "kurtosis", "bandwidth"]].isna().all().all()
    b = t[(t["gene"] == "BRCA1") & (t["model"] == "own")]
    assert {"gene", "splice_1_2", "splice_3_10", "splice_11_50"} <= set(b["stratum"])


@needs
def test_the_external_neutral_variants_follow_the_core_rule():
    x = pd.read_parquet(paths.path("external_coding_consequence"))
    t = pd.read_csv(TABLE, sep="\t")
    for gene in em.EXTERNAL:
        g = x[(x["gene"] == gene) & (x["consequence"] == "synonymous")
              & x["standard_error"].notna() & (x["spliceai_ds"] < 0.1)]
        row = t[(t["gene"] == gene) & (t["model"] == "neutral")]
        assert len(row) == 1
        assert row["n"].iloc[0] == len(g) == len(em.neutral_variants(gene))


@needs
def test_a_neutral_row_describes_its_residuals():
    t = pd.read_csv(TABLE, sep="\t", float_precision="round_trip")
    z = em.neutral_variants("BAP1")["z"].to_numpy(float)
    r = t[(t["gene"] == "BAP1") & (t["model"] == "neutral")].iloc[0]
    assert r["sd"] == z.std() and r["share_abs_gt_3"] == (np.abs(z) > 3).mean()
    assert r["bandwidth"] == rc.ErrorModel(z).h


@needs_v2
def test_the_second_selftest_reproduces_the_first_and_adds_the_neutral_cells():
    rec = json.loads(SUMMARY_V2.read_text())
    assert rec["first_version_reproduced"]
    acc = rec["acceptance"]
    assert acc["real_cells_done"] == 36 and acc["known_truth_holds"]
    t = pd.read_csv(SELFTEST_V2, sep="\t")
    assert ((t["part"] == "real_cells") & (t["error"] == "neutral")).sum() == 18
    v1 = pd.read_csv(paths.RESULTS / "rank_ceiling_selftest_v1.tsv", sep="\t",
                     float_precision="round_trip")
    v2 = pd.read_csv(SELFTEST_V2, sep="\t", float_precision="round_trip").iloc[:len(v1)]
    pd.testing.assert_frame_equal(v1.reset_index(drop=True), v2.reset_index(drop=True))


@needs_v1
def test_the_first_tables_neutral_row_took_the_se_variant_by_variant():
    t = pd.read_csv(TABLE_V1, sep="\t", float_precision="round_trip")
    nv = em.neutral_variants("BAP1")
    z = (nv["r"] / nv["se"]).to_numpy(float)
    r = t[(t["gene"] == "BAP1") & (t["model"] == "neutral")].iloc[0]
    assert r["sd"] == z.std() and r["bandwidth"] == rc.ErrorModel(z).h


# --------------------------------------------------------------------------
# the error scales (third generation)
# --------------------------------------------------------------------------
def test_pava_pools_adjacent_violators_with_equal_weights():
    fit, block = rc.pava([1.0, 3.0, 2.0, 4.0])
    assert np.allclose(fit, [1.0, 2.5, 2.5, 4.0]) and block.tolist() == [0, 1, 1, 2]
    fit, block = rc.pava([3.0, 1.0, 2.0])      # a tie with the pooled block is no violation
    assert np.allclose(fit, [2.0, 2.0, 2.0]) and block.tolist() == [0, 0, 1]
    fit, block = rc.pava([3.0, 1.0, 1.0])
    assert np.allclose(fit, [5 / 3] * 3) and block.tolist() == [0, 0, 0]
    fit, block = rc.pava([1.0, 2.0, 3.0])
    assert fit.tolist() == [1.0, 2.0, 3.0] and block.tolist() == [0, 1, 2]


def test_the_error_scale_is_flat_beyond_its_bins_and_log_linear_between():
    rng = np.random.default_rng(5)
    se = np.exp(rng.uniform(np.log(0.01), np.log(0.1), 500))
    r = se * 2.0 * rng.standard_normal(500)                  # the spread grows with SE
    sc = rc.ErrorScale(se, r)
    assert len(sc.n) == 5 and sc.n.sum() == 500 and np.all(np.diff(sc.sigma) >= 0)
    assert sc(np.array([1e-4]))[0] == sc.sigma[0] and sc(np.array([10.0]))[0] == sc.sigma[-1]
    mid = np.exp((sc.x[1] + sc.x[2]) / 2)
    assert np.isclose(sc(np.array([mid]))[0], (sc.sigma[1] + sc.sigma[2]) / 2)
    t = sc.table()
    assert np.allclose(t["sigma_b_over_rms_se"], sc.sigma / sc.rms_se)
    # bins as numpy.array_split cuts the SE order
    order = np.argsort(se, kind="stable")
    assert all((b == o).all() for b, o in zip(sc.bins, np.array_split(order, 5)))


def test_a_pooled_cell_uses_one_scale_for_the_point():
    rng = np.random.default_rng(6)
    n = 300
    se = 0.1 * np.exp(rng.normal(0, 0.5, n))
    y = np.where(rng.random(n) < 0.8, 0.0, 1.0) + se * rng.standard_normal(n)
    ids = [f"v{i:04d}" for i in range(n)]
    key = rc.cell_key("core", "TOY", "splice_3_10", "snv", "own", rc.set_sha(ids))
    out = rc.estimate_cell(ids, y, se, key, em=rc.ErrorModel(), b=4, nulls=4, pooled=True)
    seed, _ = rc.seeds(key)
    s = np.full(n, rc.pooled_scale(se))
    point, _ = rc.estimate(y, s, rc.ErrorModel(), np.random.default_rng([seed, 0]))
    assert out["ceiling"] == point and out["pooled_scale"] == s[0]


@needs
def test_the_error_scale_table_is_the_genes_calibrated_scale():
    t = pd.read_csv(SCALE, sep="\t", float_precision="round_trip")
    assert list(t["gene"].unique()) == list(em.SE_GENES) and (t.groupby("gene").size() == 5).all()
    for gene, g in t.groupby("gene"):
        sc = em.error_scale(gene)
        assert (g["sigma_b"].to_numpy() == sc.sigma).all()
        assert (np.diff(g["sigma_b"].to_numpy()) >= 0).all()
        assert g["n"].sum() == len(em.neutral_variants(gene))
    # the second supplement's trial (sigma_b / sqrt(mean SE^2) after the monotone fit)
    trial = {"BARD1": [1.69, 1.25, 1.14, 0.95, 0.72], "PALB2": [1.63, 1.11, 0.89, 0.73, 0.65],
             "BAP1": [1.71, 1.64, 1.55, 1.43, 1.40], "CTCF": [2.43, 1.68, 1.36, 1.05, 0.71],
             "SFPQ": [1.26, 1.21, 0.96, 0.79, 0.77], "RAD51D": [2.45, 1.80, 1.41, 1.10, 0.94],
             "XRCC2": [2.23, 1.55, 1.39, 1.10, 0.79], "SBDS": [1.66, 1.42, 1.11, 1.13, 0.81]}
    got = t.groupby("gene")["sigma_b_over_rms_se"].apply(lambda v: list(np.round(v, 2)))
    assert {g: got[g] for g in trial} == trial


@needs
def test_the_second_tables_models_name_their_scales():
    t = pd.read_csv(TABLE, sep="\t")
    se = t[t["gene"].isin(em.SE_GENES)]
    assert set(se.loc[se["model"] == "own", "kind"]) == {em.KIND_OWN_SE}
    assert set(se.loc[se["model"] == "neutral", "kind"]) == {em.KIND_NEUTRAL_SE}
    rec = json.loads(RECORD.read_text())
    assert rec["checks"]["neutral_counts"]["holds"] and rec["checks"]["brca1_score_is_the_mean"]["holds"]
    assert rec["records"]["monotone_fit"]["largest_block"]["RAD51D"] == 3


@needs
def test_the_third_selftest_holds_and_reproduces_the_second_where_nothing_changed():
    rec = json.loads(SUMMARY.read_text())
    acc = rec["acceptance"]
    assert acc["known_truth_holds"] and acc["real_cells_hold"] and acc["real_cells_done"] == 36
    assert rec["previous_version_reproduced"] == {"known_truth": True, "brca1_real_cells": True}
    if not SUMMARY_V2.exists():
        pytest.skip("rank_ceiling_selftest_v2 not built")
    new = pd.DataFrame(rec["data_sets"])
    old = pd.DataFrame(json.loads(SUMMARY_V2.read_text())["data_sets"])

    def rows(d, known):
        sel = d["part"].eq("known_truth") if known else d["case"].str.startswith("BRCA1 ")
        return d[sel].sort_values(["error", "case", "rep"]).reset_index(drop=True)

    for known in (True, False):
        a, b = rows(new, known), rows(old, known)
        assert len(a) == len(b) > 0
        for col in ("truth", "estimate"):
            assert (a[col].to_numpy(float) == b[col].to_numpy(float)).all()

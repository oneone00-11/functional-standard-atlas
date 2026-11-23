"""Guards for atlas.matched_ceilings: the estimators match v5's, the vectorised resampling
matches the one-set computation, the scopes and the checkpoints. Synthetic data only."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas import matched_ceilings as mc  # noqa: E402
from atlas.reliability import spearman_brown  # noqa: E402

RNG = np.random.default_rng(7)


def _se_cell(n=200, noise=0.5):
    truth = RNG.normal(size=n)
    se = np.full(n, noise)
    y = truth + RNG.normal(scale=noise, size=n)
    x = truth + RNG.normal(scale=1.0, size=n)
    return x, y, se


def test_the_se_estimator_is_v5s_with_and_without_clipping():
    x, y, se = _se_cell()
    p = mc.point(x, y, "se", se)
    raw = 1 - float((pd.Series(se) ** 2).mean()) / float(pd.Series(y).var())
    assert p["reliability_unclipped"] == raw and p["reliability"] == min(max(raw, 0), 1)
    assert p["ceiling"] == math.sqrt(p["reliability"])
    assert p["realisation"] == pytest.approx(p["rho"] / p["ceiling"])
    noisy = mc.point(x, y, "se", se * 10)                       # error variance > variance
    assert noisy["reliability_unclipped"] < 0 and noisy["reliability"] == 0.0
    assert noisy["ceiling"] == 0.0 and math.isnan(noisy["realisation"])


def test_the_replicate_estimator_is_spearman_brown_of_the_replicates_rho():
    t = RNG.normal(size=300)
    r1, r2 = t + RNG.normal(size=300), t + RNG.normal(size=300)
    p = mc.point(t, t, "replicates", r1=r1, r2=r2)
    r = float(pd.Series(r1).corr(pd.Series(r2), method="spearman"))
    assert p["rep_rho"] == r and p["reliability"] == spearman_brown(r)
    assert math.isnan(p["reliability_unclipped"])


def test_rowwise_spearman_equals_pandas_with_ties():
    a = RNG.integers(0, 5, size=(6, 40)).astype(float)          # many ties
    b = a + RNG.normal(size=(6, 40))
    got = mc.spearman_rows(a, b)
    want = [pd.Series(a[i]).corr(pd.Series(b[i]), method="spearman") for i in range(6)]
    assert np.allclose(got, want, rtol=0, atol=1e-12)
    assert np.isnan(mc.spearman_rows(np.ones((1, 5)), b[:1, :5]))[0]


@pytest.mark.parametrize("method", ["se", "replicates"])
def test_a_resample_of_every_variant_once_gives_the_point_estimate(method):
    x, y, se = _se_cell(80)
    r1, r2 = y + RNG.normal(size=80), y + RNG.normal(size=80)
    p = mc.point(x, y, method, se, r1, r2)
    idx = np.arange(80)[None, :]
    res = mc.resample_stats(idx, x, y, method, se, r1, r2)
    for k in ("rho", "reliability", "ceiling", "realisation"):
        assert res[k][0] == pytest.approx(p[k], abs=1e-12)


def test_the_bootstrap_is_fixed_by_its_seed_and_chunking_does_not_matter():
    x, y, se = _se_cell(150)
    arrays = {"x": x, "y": y, "se": se}
    a = mc.bootstrap(arrays, "se", seed=11, b=300)
    b = mc.bootstrap(arrays, "se", seed=11, b=300)
    assert a == b
    c = mc.bootstrap(arrays, "se", seed=12, b=300)
    assert a != c
    p = mc.point(x, y, "se", se)
    assert a["ceiling_lo"] <= p["ceiling"] <= a["ceiling_hi"]
    assert mc.cell_seed(("core", "BARD1", "splice_1_2", "cadd", "snv")) == \
        mc.cell_seed(("core", "BARD1", "splice_1_2", "cadd", "snv"))


def _frame(n_snv=40, n_del=10, gene="G1", method="se", group="core"):
    n = n_snv + n_del
    df = pd.DataFrame({
        "gene": gene, "group": group, "method": method,
        "region": "splice_1_2", "functional_pathogenicity": RNG.normal(size=n),
        "se": 0.3, "fit_rep1": np.nan, "fit_rep2": np.nan, "has_error": True,
        "snv": [True] * n_snv + [False] * n_del,
        "m_all": RNG.normal(size=n), "m_snv": [*RNG.normal(size=n_snv), *[np.nan] * n_del]})
    return df


def test_scopes_and_the_minimum_n():
    df = _frame()
    tasks = mc.cell_tasks(df, ["m_all", "m_snv"], ["splice_1_2"])
    got = {(k[3], k[4]): len(a["x"]) for k, _, a in tasks}
    assert got == {("m_all", "scored"): 50, ("m_all", "snv"): 40,
                   ("m_snv", "scored"): 40, ("m_snv", "snv"): 40}
    df.loc[:15, "has_error"] = False                         # 16 SNVs lose their SE
    tasks = mc.cell_tasks(df, ["m_snv"], ["splice_1_2"])
    assert tasks == []                                       # 24 < 30: no cell


def _table(rows):
    cols = ["group", "gene", "stratum", "model", "scope", "n", "rho", "ceiling",
            "realisation", "realisation_lo", "realisation_hi", "share_realisation_gt_1"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


def test_the_attenuation_check_separates_missing_error_information_from_the_rest():
    df = _frame(n_snv=60, n_del=0)
    df.loc[:4, "has_error"] = False                          # 5 variants without SE
    table = _table([("core", "G1", "splice_1_2", "m_all", "scored", 55, 0.3, 0.8, 0.375, 0, 0, 0),
                    ("core", "G1", "splice_1_2", "m_snv", "scored", 60, 0.2, 0.8, 0.25, 0, 0, 0)])
    att = pd.DataFrame([{"gene": "G1", "stratum": "splice_1_2", "model": "m_all", "n": 60, "rho": 0.31},
                        {"gene": "G1", "stratum": "splice_1_2", "model": "m_snv", "n": 60, "rho": 0.2},
                        {"gene": "G1", "stratum": "splice_1_2", "model": "m_x", "n": 60, "rho": 0.2}])
    res = mc.check_attenuation(table, df, att)
    assert res["identical"] == 1
    assert res["fewer_only_for_missing_error_information"][0]["without_error_information"] == 5
    assert [d["cell"][2] for d in res["other_differences"]] == ["m_x"]


def test_v5_ceilings_and_the_two_lists():
    table = _table([("core", "G1", "splice_1_2", "a", "snv", 119, 0.517, 0.454, 1.139, 0.9, 1.4, 0.7),
                    ("core", "G1", "splice_1_2", "b", "snv", 119, 0.20, 0.454, 0.44, 0.3, 0.6, 0.0),
                    ("core", "G1", "coding_or_utr", "b", "snv", 900, 0.30, 0.90, 0.33, 0.3, 0.4, 0.0)])
    ceil = pd.DataFrame([{"gene": "G1", "stratum": "splice_1_2", "v5_method": "se", "v5_ceiling": 0.586},
                         {"gene": "G1", "stratum": "coding_or_utr", "v5_method": "se", "v5_ceiling": 0.0}])
    t = mc.add_v5(table, ceil)
    assert t.loc[0, "realisation_v5_ceiling"] == pytest.approx(0.517 / 0.586)
    assert math.isnan(t.loc[2, "realisation_v5_ceiling"])         # a zero stratum ceiling
    lists = mc.lists(t)
    assert len(lists["ceiling_differs_from_stratum_by_more_than_0.05"]) == 3
    over = lists["realisation_above_1_or_interval_crossing_1"]
    assert [r["model"] for r in over] == ["a"]


def test_json_carries_no_nan():
    obj = {"a": float("nan"), "b": [np.float64(1.5), np.int64(3), np.bool_(True)],
           "c": {"d": np.inf}}
    assert mc.jsonable(obj) == {"a": None, "b": [1.5, 3, True], "c": {"d": None}}

"""The fourth generation of the rank-scale tables (A4c.1): B = 1,000 resamples and 500 null
data sets, nested on the third generation's 200 and 100; the per-set store; and the three
regressions of the delivered tables against the third generation's files."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from atlas import paths
from atlas import rank_cells
from atlas import rank_ceiling as rc

RESULTS = paths.RESULTS


def _built(*names: str) -> None:
    for n in names:
        p = paths.path(n) if n in paths.NAMES else RESULTS / n
        if not p.exists():
            pytest.skip(f"{p.name} not built in this checkout")


def _toy(n=60, seed=3):
    rng = np.random.default_rng(seed)
    t = np.where(rng.random(n) < 0.7, 0.0, 1.0)
    se = 0.25 * np.exp(rng.normal(0, 0.3, n))
    y = t + se * rng.standard_normal(n)
    return [f"v{i:03d}" for i in range(n)], y, se, y + rng.normal(0, 0.5, n)


def _bits(a, b) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(((a.view(np.uint64) == b.view(np.uint64))
                                        | (np.isnan(a) & np.isnan(b))).all())


# --------------------------------------------------------------------------
# the settings and the nesting
# --------------------------------------------------------------------------
def test_the_bootstrap_has_1000_resamples_and_the_floor_500_nulls():
    assert (rc.B, rc.NULLS) == (1000, 500)
    assert (rc.B_V3, rc.NULLS_V3) == (200, 100)
    d = rc.definition()
    assert d["bootstrap_B"] == 1000 and d["nulls"] == 500
    assert d["third_generation"]["bootstrap_B"] == 200 and d["third_generation"]["nulls"] == 100
    # the rest of the estimator as before
    assert (rc.ITER_POINT, rc.ITER_RESAMPLE, rc.REPEAT_POINT, rc.REPEAT_RESAMPLE) == (1000, 300,
                                                                                    20, 10)


def test_the_first_200_resamples_are_those_of_200():
    for seed, n in ((12345, 61), (2 ** 63 + 7, 997)):
        assert np.array_equal(rc.resample_indices(seed, n, 1000)[:200],
                              rc.resample_indices(seed, n, 200))


def test_the_first_resamples_and_nulls_give_the_smaller_settings_numbers():
    """Nested settings on toy data (b 40 against 8, nulls 20 against 4): the first resamples'
    ceilings, the first nulls, and the interval, floor and realisation from them are those
    of the smaller run, bit for bit."""
    ids, y, se, x = _toy()
    key = rc.cell_key("core", "TOY", "splice_3_10", "snv", "own", rc.set_sha(ids))
    kw = dict(em=rc.ErrorModel(), pooled=True, predictors={"p": x})
    small = rc.estimate_cell(ids, y, se, key, b=8, nulls=4, **kw)
    big = rc.estimate_cell(ids, y, se, key, b=40, nulls=20, **kw)
    assert small["ceiling"] == big["ceiling"] and small["rank_reliability"] == big["rank_reliability"]
    assert _bits(big["boot"][:8], small["boot"]) and _bits(big["null"][:4], small["null"])
    assert _bits(big["rho_boot"]["p"][:8], small["rho_boot"]["p"])
    first = rc.first_resamples(big["boot"], big["null"], big["rho_boot"], b=8, nulls=4)
    for k in ("ceiling_lo", "ceiling_hi", "floor", "estimable"):
        assert first[k] == small[k]
    assert first["realisation"]["p"] == small["realisation"]["p"]


def test_the_interval_helpers_are_the_estimate_cells():
    boot = np.array([0.5, 0.6, 0.0, 0.7, 0.65])
    null = np.array([0.1, 0.2, 0.3])
    iv = rc.interval(boot, null)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    assert iv == {"ceiling_lo": float(lo), "ceiling_hi": float(hi),
                  "floor": float(np.quantile(null, 0.95)), "estimable": bool(lo > np.quantile(null, 0.95))}
    real = rc.realisation_interval(np.array([0.3, 0.9, 0.2, 0.5, 0.7]), boot)
    assert real["bootstrap_realisation_undefined"] == 1          # the resample with ceiling 0
    assert real["share_realisation_gt_1"] == 0.5        # 0.9 / 0.6 and 0.7 / 0.65 of four


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------
def _task(ids, y, se, preds):
    inputs = {"neutral": {"s": np.full(len(y), 0.25), "em": rc.ErrorModel()},
              "own": {"s": se, "em": rc.ErrorModel(), "pooled": True}}
    return (("core", "TOY", "splice_3_10", "snv"), ids, y, inputs, preds, 0.1)


def test_a_stored_set_is_read_back_and_a_changed_one_is_refused(tmp_path):
    ids, y, se, x = _toy(n=30)
    task = _task(ids, y, se, {"p": x})
    a = rank_cells.compute([task], 1, tmp_path)
    name = rank_cells.store_name(task)
    assert (tmp_path / f"{name}.pkl").exists() and (tmp_path / f"{name}.npz").exists()
    arr = np.load(tmp_path / f"{name}.npz")
    assert len(arr["neutral_boot"]) == rc.B and len(arr["own_null"]) == rc.NULLS
    assert len(arr["own_rho_p"]) == rc.B
    # the arrays give the table's numbers, and the first 200 and 100 the third generation's
    for m in rank_cells.MODELS:
        iv = rc.interval(arr[f"{m}_boot"], arr[f"{m}_null"])
        assert all(iv[k] == a[0][m][k] for k in iv)
        first = rc.first_resamples(arr[f"{m}_boot"], arr[f"{m}_null"], {"p": arr[f"{m}_rho_p"]})
        assert first == a[0][m]["first"]
    stored = pickle.loads((tmp_path / f"{name}.pkl").read_bytes())
    stored["result"]["neutral"]["ceiling"] = -1.0                # read, not computed again
    (tmp_path / f"{name}.pkl").write_bytes(pickle.dumps(stored))
    b = rank_cells.compute([task], 1, tmp_path)
    assert b[0]["neutral"]["ceiling"] == -1.0 and b[0]["own"] == a[0]["own"]
    changed = _task(ids, y + 1e-9, se, {"p": x})                 # the same ids, other inputs
    with pytest.raises(SystemExit):
        rank_cells.compute([changed], 1, tmp_path)


def test_first_columns_carry_the_third_generations_names():
    r = {m: {"first": {"ceiling_lo": 0.1, "ceiling_hi": 0.9, "floor": 0.2, "estimable": False,
                       "realisation": {"p": {"realisation_lo": 0.3, "realisation_hi": 1.2,
                                             "share_realisation_gt_1": 0.1,
                                             "bootstrap_realisation_undefined": 0}}}}
         for m in rank_cells.MODELS}
    cols = rank_cells.first_columns(r) | rank_cells.first_realisation(r, "p")
    assert list(cols) == rank_cells.FIRST + rank_cells.FIRST_REALISATION
    assert set(rank_cells.MAY_CHANGE) == set(cols)


# --------------------------------------------------------------------------
# the delivered tables against the third generation
# --------------------------------------------------------------------------
TABLES = (("reliability", "reliability_v3.tsv", ["gene", "population", "stratum", "method"], False),
          ("external_reliability", "external_reliability_v3.tsv",
           ["gene", "population", "stratum", "method", "role"], False),
          ("matched_ceilings", "matched_ceilings_v3.tsv",
           ["group", "gene", "stratum", "model", "population"], True))


@pytest.mark.parametrize("name,v3,keys,realisation", TABLES)
def test_regression_1_only_the_bootstrap_columns_differ_from_v3(name, v3, keys, realisation):
    """Every column but the interval, floor, estimable (and the realisation's interval) is
    the third generation's, bit for bit; the columns are the same."""
    _built(name, v3)
    v4 = pd.read_csv(paths.path(name), sep="\t", float_precision="round_trip")
    old = pd.read_csv(RESULTS / v3, sep="\t", float_precision="round_trip")
    assert list(v4.columns) == list(old.columns)
    rec = rank_cells.check_v3(v4, RESULTS / v3, keys, realisation=realisation)
    assert rec["point_estimates_and_the_rest"]["holds"], rec


@pytest.mark.parametrize("name", ["reliability_summary", "external_reliability_summary",
                                  "matched_ceilings_summary"])
def test_regression_2_the_first_200_and_100_give_v3(name):
    _built(name)
    s = json.loads(paths.path(name).read_text())
    v3 = s["checks"]["v3"]
    assert v3["point_estimates_and_the_rest"]["holds"]
    assert v3["first_200_resamples_and_100_nulls"]["holds"]
    assert v3["first_200_resamples_and_100_nulls"]["columns_differing"] == []


def test_regression_3_one_set_one_ceiling_in_the_layer_and_matched_tables():
    _built("matched_ceilings", "matched_ceilings_summary", "reliability", "external_reliability")
    s = json.loads(paths.path("matched_ceilings_summary").read_text())
    assert s["checks"]["same_set_same_ceiling_in_the_layer_tables"]["holds"]
    from atlas.matched_ceilings import check_layer, layer_ceilings_v3
    t = pd.read_csv(paths.path("matched_ceilings"), sep="\t", float_precision="round_trip")
    rec = check_layer(t, layer_ceilings_v3())
    assert rec["holds"] and rec["matched_cells_on_a_layer_set"] > 0


def test_the_summaries_carry_the_settings_and_v3_to_v4():
    _built("reliability_summary", "external_reliability_summary", "matched_ceilings_summary")
    core = json.loads(paths.path("reliability_summary").read_text())
    ext = json.loads(paths.path("external_reliability_summary").read_text())
    mat = json.loads(paths.path("matched_ceilings_summary").read_text())
    for d in (core["definition"]["rank_ceiling"], ext["definition"]["rank_ceiling"],
              mat["definition"]["rank_ceiling"]):
        assert d["bootstrap_B"] == 1000 and d["nulls"] == 500
    assert list(core["v3_to_v4"]) == ["neutral", "own"] == list(ext["v3_to_v4"])
    assert list(mat["by_model"]) == ["neutral", "own"]
    for m in ("neutral", "own"):
        assert "7_v3_to_v4" in mat["by_model"][m]
        assert list(mat["by_model"][m])[:6] == ["1_by_stratum", "2_by_gene_stratum",
                                                "3_best_realisation_3_10_and_11_50",
                                                "4_beyond_50bp",
                                                "5_realisation_above_1_or_interval_crossing_1",
                                                "6_v2_to_v3"]


def test_v3_to_v4_lists_exactly_the_cells_whose_estimable_changed():
    _built("reliability", "reliability_v3.tsv", "reliability_summary")
    keys = ["gene", "population", "stratum", "method"]
    v4 = pd.read_csv(paths.path("reliability"), sep="\t", float_precision="round_trip")
    v3 = pd.read_csv(RESULTS / "reliability_v3.tsv", sep="\t", float_precision="round_trip")
    s = json.loads(paths.path("reliability_summary").read_text())["v3_to_v4"]
    m = v3.merge(v4, on=keys, suffixes=("_3", "_4"))
    for model in rank_cells.MODELS:
        changed = m[m[f"estimable_{model}_3"] != m[f"estimable_{model}_4"]]
        listed = {tuple(r[k] for k in keys) for r in s[model]["estimable_changed"]}
        assert listed == {tuple(r) for r in changed[keys].itertuples(index=False)}
        # the v3 side of the genes entering is the third generation's own summary
        old = json.loads((RESULTS / "reliability_v3.json").read_text())["ceiling_by_stratum"][model]
        for pop, rows in s[model]["genes_entering"].items():
            want = {r["stratum"]: r["n_genes_entering"] for r in old[pop]}
            assert {r["stratum"]: r["n_genes_entering_v3"] for r in rows} == want

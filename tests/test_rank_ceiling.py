"""atlas.rank_ceiling: the rank-scale ceiling, its error models and its self-test (A4b.1)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from atlas import paths
from atlas import rank_ceiling as rc

SELFTEST = paths.path("rank_ceiling_selftest")
SUMMARY = paths.path("rank_ceiling_selftest_summary")
needs_selftest = pytest.mark.skipif(not SUMMARY.exists(),
                                    reason="run `python -m atlas.rank_ceiling selftest` first")


def _sparse(rng, n):
    t = np.where(rng.random(n) < 0.85, 0.0, rng.normal(1, 0.3, n))
    s = 0.1 * np.exp(rng.normal(0, 0.5, n))
    return t, s


def test_a_small_known_truth_is_recovered_with_gaussian_errors():
    rng = np.random.default_rng(11)
    diffs = []
    for _ in range(4):
        t, s = _sparse(rng, 600)
        y = t + s * rng.standard_normal(len(t))
        est, _ = rc.estimate(y, s, rc.ErrorModel(), rng)
        diffs.append(est - rc.true_ceiling(t, s, rc.gaussian_noise, rng))
    assert abs(np.mean(diffs)) <= 0.06


def test_a_small_known_truth_is_recovered_with_heavy_tailed_errors():
    rng = np.random.default_rng(12)
    diffs = []
    for _ in range(4):
        n = 500
        t = rng.normal(0, 1, n)
        s = 0.35 * np.exp(rng.normal(0, 0.3, n))
        y = t + s * rc.heavy_noise(rng, n)
        em = rc.ErrorModel(rc.heavy_noise(rng, 1300))
        est, _ = rc.estimate(y, s, em, rng)
        diffs.append(est - rc.true_ceiling(t, s, rc.heavy_noise, rng))
    assert abs(np.mean(diffs)) <= 0.06


def test_where_most_variants_have_no_effect_the_ceiling_is_far_below_the_variance_formula():
    rng = np.random.default_rng(13)
    t, s = _sparse(rng, 1500)
    y = t + s * rng.standard_normal(len(t))
    est, _ = rc.estimate(y, s, rc.ErrorModel(), rng)
    variance_formula = np.sqrt(1 - (s ** 2).mean() / y.var(ddof=1))
    assert variance_formula > 0.9 and est < 0.7


def test_the_empirical_density_integrates_to_one():
    rng = np.random.default_rng(14)
    em = rc.ErrorModel(rc.heavy_noise(rng, 800))
    step = em.grid[1] - em.grid[0]
    assert abs(em.dens.sum() * step - 1) < 1e-12
    assert abs(np.trapezoid(em.dens, em.grid) - 1) < 1e-3
    assert len(em.grid) == rc.KDE_POINTS


def test_the_draws_have_the_residual_variance_plus_the_bandwidth_squared():
    rng = np.random.default_rng(15)
    z = rc.heavy_noise(rng, 1000)
    em = rc.ErrorModel(z)
    d = em.draw(400_000, rng)
    assert d.var() == pytest.approx(z.var() + em.h ** 2, rel=0.02)
    # the density's variance is the same
    step = em.grid[1] - em.grid[0]
    mean = (em.grid * em.dens).sum() * step
    var = ((em.grid - mean) ** 2 * em.dens).sum() * step
    assert var == pytest.approx(z.var() + em.h ** 2, rel=0.01)


def test_the_residuals_are_not_clipped():
    z = np.concatenate([np.random.default_rng(16).standard_normal(500), [20.0]])
    em = rc.ErrorModel(z)
    assert em.grid[-1] > 20
    assert em.pdf(np.array([20.0]))[0] > 1e-6          # the outlier keeps its mass
    assert em.pdf(np.array([em.grid[-1] + 1]))[0] == rc.FLOOR_L
    draws = em.draw(200_000, np.random.default_rng(17))
    assert (draws > 15).mean() == pytest.approx(1 / 501, rel=0.25)


def test_the_bandwidth_is_silvermans():
    z = np.random.default_rng(18).standard_normal(400)
    q75, q25 = np.percentile(z, [75, 25])
    assert rc.ErrorModel(z).h == 0.9 * min(z.std(), (q75 - q25) / 1.34) * 400 ** (-0.2)


def _cell(rng, n=200, effect=True):
    t = (np.where(rng.random(n) < 0.7, 0.0, rng.normal(1, 0.3, n)) if effect
         else np.zeros(n))
    s = 0.1 * np.exp(rng.normal(0, 0.5, n))
    ids = [f"v{i:04d}" for i in range(n)]
    return ids, t + s * rng.standard_normal(n), s


def test_the_null_floor_is_positive_and_a_cell_without_effect_is_not_estimable():
    ids, y, s = _cell(np.random.default_rng(19), effect=False)
    key = rc.cell_key("core", "G", "splice_3_10", "snv", "own", rc.set_sha(ids))
    out = rc.estimate_cell(ids, y, s, key, em=rc.ErrorModel(), b=20, nulls=20)
    assert out["floor"] > 0
    assert not out["estimable"]


def test_the_same_variants_in_another_order_give_the_same_ceiling():
    ids, y, s = _cell(np.random.default_rng(20))
    perm = np.random.default_rng(21).permutation(len(ids))
    key = rc.cell_key("core", "G", "splice_3_10", "snv", "own", rc.set_sha(ids))
    assert rc.set_sha(np.asarray(ids)[perm]) == rc.set_sha(ids)
    x = y + np.random.default_rng(22).normal(0, 0.3, len(y))
    a = rc.estimate_cell(ids, y, s, key, em=rc.ErrorModel(), b=10, nulls=10,
                         predictors={"x": x})
    b = rc.estimate_cell(np.asarray(ids)[perm], y[perm], s[perm], key, em=rc.ErrorModel(),
                         b=10, nulls=10, predictors={"x": x[perm]})
    for k in ("ceiling", "ceiling_lo", "ceiling_hi", "floor", "rank_reliability"):
        assert a[k] == b[k]
    assert a["realisation"] == b["realisation"]
    assert np.array_equal(a["boot"], b["boot"])


def test_residuals_of_the_cell_are_rebuilt_in_each_resample():
    rng = np.random.default_rng(23)
    ids, y, _ = _cell(rng)
    d = rng.normal(0, 0.1, len(ids))
    key = rc.cell_key("core", "BRCA1", "splice_3_10", "snv", "own", rc.set_sha(ids))
    out = rc.estimate_cell(ids, y, np.ones(len(ids)), key, residuals=d, b=5, nulls=5)
    assert np.isfinite(out["ceiling"]) and out["ceiling_lo"] <= out["ceiling_hi"]
    with pytest.raises(ValueError):
        rc.estimate_cell(ids, y, np.ones(len(ids)), key, em=rc.ErrorModel(), residuals=d)


def test_a_perfect_predictor_realises_one_over_the_ceiling_in_each_resample():
    ids, y, s = _cell(np.random.default_rng(24))
    key = rc.cell_key("core", "G", "splice_3_10", "snv", "own", rc.set_sha(ids))
    out = rc.estimate_cell(ids, y, s, key, em=rc.ErrorModel(), b=10, nulls=5,
                           predictors={"y": y})
    real = 1 / out["boot"]
    assert out["realisation"]["y"]["realisation_lo"] == pytest.approx(np.quantile(real, 0.025))
    assert out["realisation"]["y"]["share_realisation_gt_1"] == 1.0


@needs_selftest
def test_the_delivered_selftest_holds_its_acceptance():
    rec = json.loads(SUMMARY.read_text())
    acc = rec["acceptance"]
    assert acc["known_truth_holds"] and acc["known_truth_signal_cases"] == 6
    assert acc["real_cells_hold"] and acc["real_cells_bias_above_0.05"] == 0
    t = pd.read_csv(SELFTEST, sep="\t")
    assert (t["part"] == "known_truth").sum() == 10
    assert (t["part"] == "real_cells").sum() == 18


@needs_selftest
def test_a_delivered_known_truth_data_set_regenerates():
    rec = json.loads(SUMMARY.read_text())
    row = next(r for r in rec["data_sets"] if r["part"] == "known_truth"
               and r["error"] == "heavy_tailed" and r["case"] == "sparse_3pct" and r["rep"] == 0)
    new = rc._known_task(("heavy_tailed", "sparse_3pct", 0))
    assert new["estimate"] == row["estimate"] and new["truth"] == row["truth"]


@needs_selftest
def test_the_summary_is_the_mean_of_its_data_sets():
    rec = json.loads(SUMMARY.read_text())
    rows = pd.DataFrame(rec["data_sets"])
    t = pd.read_csv(SELFTEST, sep="\t", float_precision="round_trip")
    for r in t.itertuples(index=False):
        d = rows[(rows["part"] == r.part) & (rows["error"] == r.error) & (rows["case"] == r.case)]
        assert len(d) == r.datasets == 8
        assert d["difference"].mean() == pytest.approx(r.mean_bias, abs=1e-15)

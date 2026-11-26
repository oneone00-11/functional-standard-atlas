"""atlas.lof_end (the second supplement): the amplifier h(t), the sensitivity reducing to the
neutral point estimate with kappa* = 1, the level table and the sensitivity table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas import control_classes as cc
from atlas import lof_end as le
from atlas import paths
from atlas import rank_ceiling as rc

RESULTS = paths.RESULTS


def _built(*names: str) -> None:
    for n in names:
        p = paths.path(n) if n in paths.NAMES else RESULTS / n
        if not p.exists():
            pytest.skip(f"{p.name} not built in this checkout")


# --------------------------------------------------------------------------
# the loss-of-function end
# --------------------------------------------------------------------------
def test_the_amplifier_at_the_two_levels_and_the_midpoint():
    h = le.amplifier(3.0, 0.0, 2.0)
    assert h([0.0, 1.0, 2.0]).tolist() == [1.0, 2.0, 3.0]
    assert h([-1.0, 5.0]).tolist() == [1.0, 3.0]                       # clipped
    assert le.amplifier(1.0, 0.0, 2.0)([0.0, 1.0, 2.0, 9.0]).tolist() == [1.0] * 4


def test_with_kappa_one_the_sensitivity_is_the_neutral_point_estimate_bit_for_bit():
    rng = np.random.default_rng(8)
    n = 300
    t = np.where(rng.random(n) < 0.8, 0.0, 1.0)
    s = 0.1 * np.exp(rng.normal(0, 0.4, n))
    y = t + s * rng.standard_normal(n)
    for em in (rc.ErrorModel(), rc.ErrorModel(rng.standard_t(5, 500))):
        a = rc.estimate(y, s, em, np.random.default_rng([11, 0]))
        b = le.estimate_grown(y, s, em, le.amplifier(1.0, 0.0, 1.0), np.random.default_rng([11, 0]))
        assert a == b


def test_the_level_table_counts_the_variants_above_the_threshold():
    _built("lof_level_by_stratum", "control_classes")
    t = pd.read_csv(paths.path("lof_level_by_stratum"), sep="\t")
    assert set(t["gene"]) == set(cc.GENES)
    assert np.allclose(t["lof_share"], t["n_lof_level"] / t["n"])
    few = t["n_lof_level"] < le.MIN_LOF
    assert t.loc[few, "sd_lof_level"].isna().all() and t.loc[~few, "sd_lof_level"].notna().all()


def test_the_sensitivity_table_takes_the_layer_tables_neutral_estimate():
    _built("rank_ceiling_lof_sensitivity", "reliability", "external_reliability")
    t = pd.read_csv(paths.path("rank_ceiling_lof_sensitivity"), sep="\t", float_precision="round_trip")
    layer = pd.concat([pd.read_csv(paths.path(k), sep="\t", float_precision="round_trip")
                       for k in ("reliability", "external_reliability")])
    layer = layer[(layer["population"] == "snv") & layer["ceiling_neutral"].notna()]
    layer = layer.set_index(["gene", "stratum"])          # one row per cell (TP53's bounds out)
    assert layer.index.is_unique
    for r in t.itertuples(index=False):
        assert r.ceiling_neutral == layer.loc[(r.gene, r.stratum), "ceiling_neutral"]
        assert r.set_sha == layer.loc[(r.gene, r.stratum), "set_sha"]
    assert (t["kappa_star"] == t["kappa"].clip(lower=1)).all()
    assert np.allclose(t["difference"], t["ceiling_neutral_lof_end"] - t["ceiling_neutral"], atol=0)
    assert (t["n"] >= le.MIN_N).all()



"""atlas.control_classes: the neutral and nonsense classes of the twelve genes, their
levels, spreads and position structure (A4b, the second supplement)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas import control_classes as cc
from atlas import error_models as em
from atlas import paths
from atlas import rank_ceiling as rc

TABLE = paths.path("control_classes")
needs = pytest.mark.skipif(not TABLE.exists(), reason="run `python -m atlas.control_classes` first")


def _class(positions, alts, y, se=None):
    d = pd.DataFrame({"hgvs_c": [f"c.{p}G>{a}" for p, a in zip(positions, alts)],
                      "y": np.asarray(y, float)})
    d["se"] = np.nan if se is None else np.asarray(se, float)
    return d


def test_the_cds_order_is_position_then_alternative_base():
    order = cc.cds_order(["c.10G>T", "c.2A>G", "c.10G>A", "c.10G>C"])
    assert order.tolist() == [1, 2, 3, 0]
    with pytest.raises(ValueError):
        cc.cds_order(["c.10+1G>A"])


def test_the_local_median_leaves_the_variant_out_and_takes_fewer_at_the_ends():
    v = np.arange(1.0, 8.0)                                # 1 ... 7
    loc = cc.local_median(v, k=2)
    # windows: [2,3] [1,3,4] [1,2,4,5] [2,3,5,6] [3,4,6,7] [4,5,7] [5,6]
    assert loc.tolist() == [2.5, 3.0, 3.0, 4.0, 5.0, 5.0, 5.5]
    loc5 = cc.local_median(np.arange(12.0))
    assert loc5[0] == np.median([1, 2, 3, 4, 5]) and loc5[6] == np.median([1, 2, 3, 4, 5, 7, 8, 9, 10, 11])


def test_the_columns_on_toy_classes():
    neutral = _class(range(1, 8), "ACGTACG", [0.0, 0.1, -0.1, 0.2, -0.2, 0.05, -0.05])
    nonsense = _class(range(20, 25), "AAAAA", [1.0, 1.4, 0.6, 1.2, 0.8])
    c = cc.class_columns(neutral, nonsense)
    assert c["n_neutral"] == 7 and c["n_nonsense"] == 5
    assert c["t_N"] == 0.0 and c["t_L"] == 1.0 and c["gap"] == 1.0 and c["lof_threshold"] == 0.5
    assert np.isclose(c["sd_neutral"], 1.4826 * 0.1) and np.isclose(c["sd_nonsense"], 1.4826 * 0.2)
    assert np.isclose(c["sd_nonsense_over_gap"], 1.4826 * 0.2) and np.isclose(c["spread_ratio"], 2.0)
    assert np.isnan(c["kappa"]) and c["kappa_basis"] == ""
    # the position structure, by hand for the nonsense class (already in CDS order)
    y = np.array([1.0, 1.4, 0.6, 1.2, 0.8])
    loc = cc.local_median(y)
    assert np.isclose(c["local_sd_ratio_nonsense"], rc.robust_sd(y - loc) / rc.robust_sd(y))
    assert np.isclose(c["local_corr_nonsense"], np.corrcoef(y, loc)[0, 1])
    assert cc.class_columns(neutral, nonsense, kappa_is_spread=True)["kappa"] == c["spread_ratio"]


def test_kappa_divides_out_the_scale():
    rng = np.random.default_rng(3)
    se_n, se_l = rng.uniform(0.01, 0.05, 200), rng.uniform(0.01, 0.05, 100)
    yn = se_n * rng.standard_normal(200)                    # noise of scale SE
    yl = 1.0 + 3.0 * se_l * rng.standard_normal(100)        # three times as wide
    neutral = _class(range(1, 201), "A" * 200, yn, se_n)
    nonsense = _class(range(300, 400), "T" * 100, yl, se_l)
    flat = cc.class_columns(neutral, nonsense, scale=lambda se: np.ones(len(se)))
    assert flat["kappa_basis"] == "sigma(SE)"
    assert np.isclose(flat["kappa"], flat["sd_nonsense"] / flat["sd_neutral"])
    by_se = cc.class_columns(neutral, nonsense, scale=lambda se: np.asarray(se))
    assert 2.4 < by_se["kappa"] < 3.6                       # the SE divided out: about 3


@needs
def test_the_table_has_the_twelve_genes_and_the_error_models_neutral_variants():
    t = pd.read_csv(TABLE, sep="\t", float_precision="round_trip").set_index("gene")
    assert list(t.index) == list(cc.GENES)
    for gene in em.GENES:
        nv = em.neutral_variants(gene)
        assert t.loc[gene, "n_neutral"] == len(nv) and t.loc[gene, "t_N"] == nv["y"].median()
    assert t.loc[["RAD51C", "BRCA2", "VHL"], "kappa"].isna().all()
    assert t.loc["BRCA1", "kappa"] == t.loc["BRCA1", "spread_ratio"]
    assert set(t.loc[list(em.SE_GENES), "kappa_basis"]) == {"sigma(SE)"}
    assert (t["gap"] > 0).all() and np.allclose(t["lof_threshold"], (t["t_N"] + t["t_L"]) / 2)
    lv = cc.levels(TABLE)
    assert (lv["t_L"] == t["t_L"]).all()


@needs
def test_the_table_agrees_with_the_trial_of_the_instruction():
    # the second supplement's table: n, n, nonsense spread / gap, spread_ratio,
    # local correlation (nonsense, neutral), kappa
    trial = {"BRCA1": (468, 138, 0.155, 1.11, -0.20, 0.10, 1.11),
             "BARD1": (1231, 320, 0.281, 4.10, 0.46, 0.42, 3.02),
             "PALB2": (1807, 425, 0.424, 3.90, 0.64, 0.33, 2.95),
             "BAP1": (1322, 218, 0.320, 6.92, 0.67, 0.25, 5.19),
             "RAD51C": (688, 153, 0.459, 4.56, 0.77, 0.15, None),
             "BRCA2": (1218, 342, 0.124, 1.52, 0.21, 0.14, None),
             "VHL": (367, 60, 0.557, 9.75, 0.82, -0.10, None),
             "CTCF": (914, 222, 0.408, 4.39, 0.68, 0.66, 3.46),
             "SFPQ": (717, 182, 0.374, 6.58, 0.78, 0.16, 4.15),
             "RAD51D": (582, 86, 0.465, 7.25, 0.83, 0.19, 5.19),
             "XRCC2": (460, 121, 0.151, 3.28, 0.53, 0.67, 3.08),
             "SBDS": (372, 107, 0.209, 3.01, 0.88, 0.32, 2.46)}
    t = pd.read_csv(TABLE, sep="\t", float_precision="round_trip").set_index("gene")
    for gene, (nn, nl, sg, sr, cl, cn, k) in trial.items():
        r = t.loc[gene]
        assert (r["n_neutral"], r["n_nonsense"]) == (nn, nl)
        assert round(r["sd_nonsense_over_gap"], 3) == sg and round(r["spread_ratio"], 2) == sr
        assert round(r["local_corr_nonsense"], 2) == cl and round(r["local_corr_neutral"], 2) == cn
        assert (np.isnan(r["kappa"]) if k is None else round(r["kappa"], 2) == k)

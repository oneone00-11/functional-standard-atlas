"""atlas.neutral_sensitivity (A4c.2): the neutral ceiling under other neutral classes, SE
bins and the loss-of-function end, beside the main model; the three checks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from atlas import error_models as em
from atlas import lof_end
from atlas import neutral_sensitivity as ns
from atlas import paths
from atlas import rank_ceiling as rc

RESULTS = paths.RESULTS
# the neutral classes' sizes: every synonymous SNV with a score and error information, and
# those with spliceai_ds below 0.2, 0.1, 0.05 and 0.02 (the instruction's trial gives the same)
SIZES = {"BRCA1": (579, 537, 468, 405, 305), "BARD1": (1346, 1298, 1231, 1112, 924),
         "PALB2": (1919, 1862, 1807, 1741, 1597), "BAP1": (1543, 1449, 1322, 1138, 860),
         "CTCF": (952, 926, 914, 880, 795), "SFPQ": (763, 731, 717, 674, 580),
         "RAD51D": (678, 633, 582, 498, 350), "XRCC2": (481, 476, 460, 424, 385),
         "SBDS": (437, 398, 372, 331, 282)}


def _built(*names: str) -> None:
    for n in names:
        p = paths.path(n) if n in paths.NAMES else RESULTS / n
        if not p.exists():
            pytest.skip(f"{p.name} not built in this checkout")


def _inputs() -> None:
    _built("score_matrix", "assay_aux", "external_coding_consequence")


def _bits(a, b) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool((a.view(np.uint64) == b.view(np.uint64)).all())


# --------------------------------------------------------------------------
# the neutral classes and their models
# --------------------------------------------------------------------------
def test_the_neutral_classes_at_the_five_thresholds():
    _inputs()
    for gene, want in SIZES.items():
        got = tuple(len(ns.neutral_class(gene, t)) for t in ns.THRESHOLDS.values())
        assert got == want, gene
        assert want[2] == len(em._neutral_frame(gene))            # 0.1: the main model's


def test_the_explicit_construction_is_the_main_model():
    """Check 3 at the inputs: five bins and the threshold 0.1, written out, give the main
    model's sigma(SE) and z, bit for bit."""
    _inputs()
    for gene in em.GENES:
        scale, model = ns.neutral_spec(gene, 0.1, 5)
        assert _bits(model.z, em.neutral_model(gene).z), gene
        if gene == "BRCA1":
            assert scale is None
        else:
            main = em.error_scale(gene)
            assert _bits(scale.x, main.x) and _bits(scale.sigma, main.sigma), gene


def test_another_class_makes_its_own_scale_and_shape():
    _inputs()
    scale, model = ns.neutral_spec("BAP1", None, 5)
    nv = ns.neutral_class("BAP1", None)
    assert len(model.z) == len(nv) == SIZES["BAP1"][0]
    assert _bits(model.z, (nv["r"] / scale(nv["se"].to_numpy(float))).to_numpy(float))
    three, _ = ns.neutral_spec("BAP1", 0.1, 3)
    assert len(three.x) == 3 and np.all(np.diff(three.sigma) >= 0)


def test_the_lof_end_with_kappa_star_1_is_the_point_estimate_on_toy_data():
    rng = np.random.default_rng(5)
    y = np.where(rng.random(70) < 0.7, 0.0, 2.0) + 0.3 * rng.standard_normal(70)
    s = np.full(70, 0.3)
    m = rc.ErrorModel(rng.standard_normal(200))
    a = rc.estimate(y, s, m, np.random.default_rng([11, 0]))[0]
    b = lof_end.estimate_grown(y, s, m, lof_end.amplifier(1.0, 0.0, 2.0),
                               np.random.default_rng([11, 0]))[0]
    assert a == b


# --------------------------------------------------------------------------
# the delivered table
# --------------------------------------------------------------------------
def test_the_three_checks_hold():
    _built("rank_ceiling_sensitivity_summary")
    c = json.loads(paths.path("rank_ceiling_sensitivity_summary").read_text())["checks"]
    for name in ("base_with_the_own_neutral_seed_is_the_layer_ceiling",
                 "lof_end_with_kappa_star_1_is_base", "explicit_five_bins_threshold_0.1_is_base"):
        assert c[name]["holds"] and c[name]["cells"] == 57, name
    assert c["base_with_the_own_neutral_seed_is_the_layer_ceiling"]["seeds_differing"] == []


def test_the_table_has_57_cells_and_takes_the_layer_tables_columns():
    _built("rank_ceiling_sensitivity", "reliability", "external_reliability",
           "rank_ceiling_lof_sensitivity")
    t = pd.read_csv(paths.path("rank_ceiling_sensitivity"), sep="\t", float_precision="round_trip")
    assert list(t.columns) == ns.COLUMNS and len(t) == 57
    lof = pd.read_csv(paths.path("rank_ceiling_lof_sensitivity"), sep="\t",
                      float_precision="round_trip")
    assert t[["gene", "stratum", "set_sha"]].equals(lof[["gene", "stratum", "set_sha"]])
    layer = pd.concat([pd.read_csv(paths.path(k), sep="\t", float_precision="round_trip")
                       for k in ("reliability", "external_reliability")], ignore_index=True)
    layer = layer[(layer["population"] == "snv") & layer["ceiling_neutral"].notna()]
    layer = layer.set_index(["gene", "stratum"])
    for r in t.itertuples():
        L = layer.loc[(r.gene, r.stratum)]
        for c in ("ceiling_neutral", "floor_neutral", "lof_share"):
            assert getattr(r, c) == L[c], (r.gene, r.stratum, c)
        assert bool(r.estimable_neutral) == bool(L["estimable_neutral"])
        assert bool(r.neutral_in_range) == bool(L["neutral_in_range"])
    for e in ns.ALTERNATIVES:
        d = t[f"d_{e}"].dropna()
        assert np.array_equal(d.to_numpy(), (t.loc[d.index, e] - t.loc[d.index, "base"]).to_numpy())


def test_brca1_has_no_bins_and_the_others_have_every_estimate():
    _built("rank_ceiling_sensitivity")
    t = pd.read_csv(paths.path("rank_ceiling_sensitivity"), sep="\t", float_precision="round_trip")
    b = t[t["gene"] == "BRCA1"]
    assert b[["bins_3", "bins_8", "d_bins_3", "d_bins_8"]].isna().all().all()
    assert b[["base", "syn_all", "ds_lt_0_2", "ds_lt_0_05", "ds_lt_0_02", "lof_end"]].notna().all().all()
    assert t[t["gene"] != "BRCA1"][ns.ESTIMATES].notna().all().all()
    assert (t["kappa_star"] == t["kappa"].clip(lower=1.0)).all()
    # kappa* = 1: the grown estimate is base, under the row's seed
    one = t[t["kappa_star"] == 1.0]
    assert (one["lof_end"] == one["base"]).all()


def test_the_summary_counts_the_classes_and_spans_the_differences():
    _built("rank_ceiling_sensitivity", "rank_ceiling_sensitivity_summary")
    s = json.loads(paths.path("rank_ceiling_sensitivity_summary").read_text())
    assert {g: tuple(v.values()) for g, v in s["neutral_class_sizes"].items()} == SIZES
    t = pd.read_csv(paths.path("rank_ceiling_sensitivity"), sep="\t", float_precision="round_trip")
    enter = t["estimable_neutral"] & t["neutral_in_range"]
    assert s["cells"] == {"all": 57, "entering": int(enter.sum())}
    for rows, part in ((s["differences_all_cells"], t), (s["differences_cells_entering"], t[enter])):
        for rec in rows:
            d = part[part["stratum"] == rec["stratum"]]
            for e in ns.ALTERNATIVES:
                v = d[f"d_{e}"].dropna()
                if len(v):
                    assert rec[f"d_{e}"]["median"] == pytest.approx(float(v.median()), abs=0)
                    assert rec[f"d_{e}"]["min"] == float(v.min())

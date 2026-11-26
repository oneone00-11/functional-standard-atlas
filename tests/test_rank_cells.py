"""atlas.rank_cells and the third generation's tables (A4b.4): the rank-scale ceilings of
the layer and matched cells under both error models, the neutral model's range, G17 and
G18, and the regressions against the second generation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from atlas import control_classes as cc
from atlas import error_models as em
from atlas import paths
from atlas import rank_cells
from atlas import rank_ceiling as rc

RESULTS = paths.RESULTS


def _built(*names: str) -> None:
    for n in names:
        p = paths.path(n) if n in paths.NAMES else RESULTS / n
        if not p.exists():
            pytest.skip(f"{p.name} not built in this checkout")


def _toy(n=80, seed=1):
    rng = np.random.default_rng(seed)
    t = np.where(rng.random(n) < 0.7, 0.0, 1.0)
    se = 0.2 * np.exp(rng.normal(0, 0.3, n))
    y = t + se * rng.standard_normal(n)
    ids = [f"t{i:03d}" for i in range(n)]
    return ids, y, se, y + rng.normal(0, 0.5, n)


def _task(ids, y, se, preds):
    inputs = {"neutral": {"s": np.full(len(y), 0.2), "em": rc.ErrorModel()},
              "own": {"s": se, "em": rc.ErrorModel(), "pooled": True}}
    return (("core", "TOY", "splice_3_10", "snv"), ids, y, inputs, preds, 0.1)


# --------------------------------------------------------------------------
# the module
# --------------------------------------------------------------------------
def test_the_models_inputs_follow_the_error_models():
    se = np.array([0.01, 0.02, 0.05])
    neu = rank_cells.model_inputs("BARD1", "neutral", 3, se=se)
    assert np.array_equal(neu["s"], em.error_scale("BARD1")(se)) and neu["em"].name == "neutral"
    own = rank_cells.model_inputs("BARD1", "own", 3, se=se)
    assert own["pooled"] and own["em"].z is None and np.array_equal(own["s"], se)
    d = np.array([0.1, -0.2, 0.05])
    b = rank_cells.model_inputs("BRCA1", "own", 3, d=d)
    assert np.array_equal(b["residuals"], d) and (b["s"] == 1).all()
    assert (rank_cells.model_inputs("BRCA1", "neutral", 3)["s"] == 1).all()
    with pytest.raises(ValueError):
        rank_cells.model_inputs("VHL", "own", 3)


@pytest.mark.skipif(not paths.path("control_classes").exists(), reason="control classes not built")
def test_the_lof_share_counts_the_scores_above_the_threshold():
    thr = cc.levels().loc["PALB2", "lof_threshold"]
    y = np.array([thr - 0.01, thr, thr + 0.01, thr + 1.0])
    assert rank_cells.lof_share("PALB2", y) == 0.5          # strictly above


def test_a_set_is_estimated_under_both_models_with_its_own_key():
    ids, y, se, x = _toy()
    r = rank_cells.run(_task(ids, y, se, {"p": x}))
    sha = rc.set_sha(ids)
    for model, kw in (("neutral", {"em": rc.ErrorModel()}),
                      ("own", {"em": rc.ErrorModel(), "pooled": True})):
        s = np.full(len(y), 0.2) if model == "neutral" else se
        want = rc.estimate_cell(ids, y, s, rc.cell_key("core", "TOY", "splice_3_10", "snv", model,
                                                       sha), predictors={"p": x}, **kw)
        assert r[model]["ceiling"] == want["ceiling"] and r[model]["floor"] == want["floor"]
        assert r[model]["realisation"]["p"] == want["realisation"]["p"]
    cols = rank_cells.columns(r)
    assert list(cols) == rank_cells.COLUMNS and cols["neutral_in_range"]
    rho = 0.4
    real = rank_cells.realisation(r, "p", rho)
    assert real["realisation_neutral"] == rho / r["neutral"]["ceiling"]


def test_one_set_gets_the_same_ceiling_whatever_predictors_it_carries():
    ids, y, se, x = _toy(seed=2)
    a = rank_cells.run(_task(ids, y, se, {}))
    b = rank_cells.run(_task(list(reversed(ids)), y[::-1], se[::-1], {"p": x[::-1], "q": -x[::-1]}))
    for m in rank_cells.MODELS:
        for k in ("ceiling", "ceiling_lo", "ceiling_hi", "floor", "rank_reliability"):
            assert a[m][k] == b[m][k]


def test_a_cell_enters_a_summary_when_estimable_and_for_neutral_in_range():
    t = pd.DataFrame({"estimable_neutral": [True, True, False], "estimable_own": [True, True, False],
                      "neutral_in_range": [True, False, True]})
    assert rank_cells.usable(t, "neutral").tolist() == [True, False, False]
    assert rank_cells.usable(t, "own").tolist() == [True, True, False]


# --------------------------------------------------------------------------
# the delivered tables
# --------------------------------------------------------------------------
def test_the_layer_tables_keep_v2s_estimate_and_add_both_models():
    _built("reliability", "reliability_v2.tsv", "external_reliability", "external_reliability_v2.tsv")
    from atlas.reliability import V2_ESTIMATE, _same
    for name, keys in (("reliability", ["gene", "population", "stratum", "method"]),
                       ("external_reliability", ["gene", "population", "stratum", "method", "role"])):
        t = pd.read_csv(paths.path(name), sep="\t", float_precision="round_trip")
        v2 = pd.read_csv(RESULTS / f"{name}_v2.tsv", sep="\t", float_precision="round_trip")
        v2 = v2[v2["role"] != "cross_check"]
        a = t.sort_values(keys, kind="stable").reset_index(drop=True)
        b = v2.sort_values(keys, kind="stable").reset_index(drop=True)
        assert len(a) == len(b)
        for c in V2_ESTIMATE:
            assert _same(a[f"{c}_v2"], b[c]), (name, c)
        assert "ceiling_inflated" not in t and "inflation_factor" not in t
        ranked = t[t["ceiling_neutral"].notna()]
        assert (ranked["role"] == "main").all()
        assert (ranked["neutral_in_range"] == (ranked["lof_share"] <= 0.5)).all()
        for m in rank_cells.MODELS:
            assert (ranked[f"floor_{m}"] > 0).all()
            assert (ranked[f"estimable_{m}"] == (ranked[f"ceiling_{m}_lo"] > ranked[f"floor_{m}"])).all()
    core = pd.read_csv(paths.path("reliability"), sep="\t", float_precision="round_trip")
    b = core[core["gene"] == "BRCA1"]
    assert np.allclose(b["ceiling_replicate_k1"], np.sqrt(b["rep_rho"].clip(lower=0)), rtol=0, atol=0)
    assert core.loc[core["gene"] != "BRCA1", "ceiling_replicate_k1"].isna().all()
    ext = pd.read_csv(paths.path("external_reliability"), sep="\t")
    assert ext.loc[ext["gene"] == "TP53", "ceiling_neutral"].isna().all()
    for name in ("reliability_summary", "external_reliability_summary"):
        rec = json.loads(paths.path(name).read_text())
        assert rec["checks"]["v2_computed_again"]["holds"]


def test_the_matched_table_has_no_external_all_rows_and_its_regressions_hold():
    _built("matched_ceilings", "matched_ceilings_summary", "matched_ceilings_v2.tsv")
    t = pd.read_csv(paths.path("matched_ceilings"), sep="\t", float_precision="round_trip")
    assert not ((t["group"] == "external") & (t["stratum"] == "all")).any()          # G18
    s = json.loads(paths.path("matched_ceilings_summary").read_text())
    assert s["checks"]["v2_computed_again"]["holds"]
    assert s["checks"]["same_set_same_ceiling_in_the_layer_tables"]["holds"]
    none = t["method"] == "none"
    assert t.loc[none, ["ceiling_neutral", "ceiling_own", "lof_share"]].isna().all().all()
    r = t[~none]
    for m in rank_cells.MODELS:
        ok = r[f"ceiling_{m}"] > 0
        assert np.array_equal(r.loc[ok, f"realisation_{m}"].to_numpy(),        # an empty rho
                              (r.loc[ok, "rho"] / r.loc[ok, f"ceiling_{m}"]).to_numpy(),  # (G17)
                              equal_nan=True)                                    # gives an empty one
    assert list(s["by_model"]) == ["neutral", "own"]


def test_g17_the_best_predictor_scored_ninety_percent_of_the_cell():
    _built("matched_ceilings_summary")
    s = json.loads(paths.path("matched_ceilings_summary").read_text())
    for m in rank_cells.MODELS:
        for rec in s["by_model"][m]["2_by_gene_stratum"]:
            for k in ("best", "best_splice_family"):
                if rec[k] is not None:
                    assert rec[k]["model"] in rec["predictors_scoring_90pct"]
                    assert rec[k]["n"] >= 0.9 * rec["n_variants"]


def test_g17_the_grid_differs_from_v2_in_the_constant_cell_alone():
    _built("eval_ext", "eval_ext_v2.tsv")
    v2 = (RESULTS / "eval_ext_v2.tsv").read_text().splitlines()
    v3 = paths.path("eval_ext").read_text().splitlines()
    assert len(v2) == len(v3)
    cut = [line.split("\t")[:10] for line in v3]
    differ = [i for i, (a, b) in enumerate(zip(v2, cut)) if a.split("\t") != b]
    assert len(differ) == 1
    row = v3[differ[0]].split("\t")
    assert row[:3] == ["snv", "bayesdel_addaf", "splice_1_2"] and row[-1] == "BAP1"
    g = pd.read_csv(paths.path("eval_ext"), sep="\t")
    flagged = g[g["constant_scores"].fillna("").astype(str) != ""]
    assert len(flagged) == 1 and flagged["k_genes"].iloc[0] == 5 and flagged["pooled_rho"].notna().all()

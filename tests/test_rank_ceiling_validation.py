"""atlas.rank_ceiling_validation: predicting the agreement of replicates from the combined
score and its error (A4b.3), four predictions, the pooled replicate one the acceptance."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from atlas import paths
from atlas import rank_ceiling as rc
from atlas import rank_ceiling_validation as rv

TABLE = paths.path("rank_ceiling_validation")
RECORD = paths.path("rank_ceiling_validation_summary")
needs = pytest.mark.skipif(not RECORD.exists(),
                           reason="run `python -m atlas.rank_ceiling_validation` first")


def test_synonymous_variants_are_recognised_by_their_protein_name():
    h = pd.Series(["p.Ala12=", "p.(=)", "p.Leu5Pro", "_wt", "p.(Ser3=)", np.nan])
    assert rv.is_synonymous(h).tolist() == [True, True, False, False, True, False]


def test_the_replicate_shape_is_the_deviation_from_the_mean_over_its_scale():
    R = np.array([[1.0, 3.0], [2.0, 2.0], [0.0, 4.0]])
    z = rv.replicate_shape(R, 0.5).reshape(3, 2)
    assert np.allclose(z[:, 0], [-2.0, 0.0, -4.0]) and np.allclose(z.sum(axis=1), 0)
    zv = rv.replicate_shape(R, np.array([1.0, 1.0, 2.0])).reshape(3, 2)
    assert np.allclose(zv[:, 1], [1.0, 0.0, 1.0])
    R3 = np.array([[0.0, 1.0, 2.0]])
    assert np.allclose(rv.replicate_shape(R3, 1.0), np.array([-1, 0, 1]) / np.sqrt(2))


def test_the_neutral_model_of_a_deposit_is_calibrated_on_its_synonymous_variants():
    rng = np.random.default_rng(4)
    n = 200
    s = np.exp(rng.uniform(np.log(0.02), np.log(0.2), n))
    score = s * rng.standard_normal(n)
    hp = np.where(np.arange(n) < 120, "p.Gly7=", "p.Gly7Arg")
    q = {"x": pd.DataFrame({"score": score, "hgvs_pro": hp}), "s": s}
    spec, rec = rv.neutral_spec(q)
    syn = np.arange(n) < 120
    r = score[syn] - np.median(score[syn])
    assert rec["n"] == 120 and np.allclose(spec["z"], r / spec["scale"](s[syn]))
    few = {"x": pd.DataFrame({"score": score[:5], "hgvs_pro": ["p.=" ] * 5}), "s": s[:5]}
    text, rec = rv.neutral_spec(few)
    assert isinstance(text, str) and text.startswith("not computable") and rec["n"] == 5


@needs
def test_the_replicate_prediction_holds_for_every_deposit_and_brca1_cell():
    rec = json.loads(RECORD.read_text())
    acc = rec["acceptance"]
    assert acc["holds"] and acc["rows_applying"] == acc["rows_passing"] == 15
    t = pd.read_csv(TABLE, sep="\t", float_precision="round_trip")
    ok = t[t["acceptance_applies"]]
    assert (ok["pred_minus_obs_replicate"].abs() <= rv.TOL).all()
    for m in rv.MODELS:
        d = t[f"pred_{m}"] - t["r_obs"]
        assert np.allclose(d, t[f"pred_minus_obs_{m}"], equal_nan=True)


@needs
def test_the_four_predictions_are_where_they_can_be():
    t = pd.read_csv(TABLE, sep="\t", float_precision="round_trip")
    dep, brca1 = t[t["urn"] != rv.BRCA1], t[t["urn"] == rv.BRCA1]
    assert len(dep) == 9 and set(brca1["population"]) == {"snv"}
    assert dep[["pred_replicate", "pred_per_variant", "pred_own"]].notna().all().all()
    assert brca1[["pred_per_variant", "pred_own"]].isna().all().all()
    assert brca1[["pred_replicate", "pred_neutral"]].notna().all().all()
    tb = dep["urn"].str.contains("00001212")
    assert dep.loc[tb, "pred_neutral"].isna().all() and dep.loc[~tb, "pred_neutral"].notna().all()
    rec = json.loads(RECORD.read_text())
    for urn in rv.DEPOSITS:
        row = dep[dep["urn"] == urn].iloc[0]
        assert row["pooled_s"] == rec["checks"][urn]["pooled_s"]

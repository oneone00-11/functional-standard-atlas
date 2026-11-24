"""atlas.mavedb_survey_reaudit: RE_ERROR split into its four alternatives; the strata and
their sizes; the draw; the numeric checks; the Wilson interval; the verdicts' rules."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas import mavedb_survey_reaudit as ra  # noqa: E402
from atlas.mavedb_survey import RE_ERROR  # noqa: E402


def test_the_four_parts_are_re_error_itself():
    parts = ra.ERROR_PARTS
    assert list(parts) == ["conventional", "interval_prefix", "sigma", "err"]
    names = ["se", "score_sd", "variance", "ci_lower", "upper_ci", "score_95CI_low",
             "dG_95CI_high", "sigma", "raw_sigma", "sigma1_uncorr", "fitting_err", "err",
             "score", "hgvs_pro", "sem_x", "series", "error", "ci95", "pval"]
    for n in names:
        assert bool(RE_ERROR.search(n)) == any(p.search(n) for p in parts.values()), n
    assert "|".join(ra.split_alternation(RE_ERROR.pattern)) == RE_ERROR.pattern


def test_the_stratum_is_the_first_part_in_order():
    s = ra.error_stratum
    assert s(["score_95CI_low", "se"]) == "conventional"
    assert s(["score_95CI_low", "raw_sigma"]) == "interval_prefix"
    assert s(["raw_sigma", "fitting_err"]) == "sigma"
    assert s(["fitting_err"]) == "err"
    assert s(["score"]) is None


def test_an_unknown_alternative_stops_the_split():
    with pytest.raises(SystemExit):
        ra.error_parts(RE_ERROR.pattern + r"|(^|_)spread($|_)")


@pytest.mark.skipif(not ra.SURVEY.exists(), reason="results/mavedb_survey_v1.tsv not built")
def test_the_survey_gives_the_four_strata_their_sizes():
    s = pd.read_csv(ra.SURVEY, sep="\t")
    got = ra.survey_strata(s).value_counts().to_dict()
    assert {k: got[k] for k in ra.STRATA} == {"conventional": 228, "interval_prefix": 1608,
                                              "sigma": 582, "err": 12}
    assert got["none"] == 322 and got["replicates"] == 22


def _survey():
    cols = {"conventional": "score;se", "interval_prefix": "score;score_95CI_low;score_95CI_high",
            "sigma": "score;sigma", "err": "score;err"}
    rows = []
    for k, n in ra.SIZES.items():
        ec = ";".join(c for c in cols[k].split(";") if c != "score")
        rows += [{"urn": f"{k}-{i}", "permits": "errors", "error_columns": ec, "human": i % 2 == 0,
                  "title": "t", "n_variants": 10, "all_columns": cols[k],
                  "replicate_columns": ""} for i in range(n)]
    rows += [{"urn": f"none-{i}", "permits": "none", "error_columns": "", "human": True,
              "title": "t", "n_variants": 10, "all_columns": "score", "replicate_columns": ""}
             for i in range(322)]
    return pd.DataFrame(rows)


def test_the_draw_is_142_fixed_by_the_seed():
    s = _survey()
    a = ra.draw(s)
    assert a["stratum"].value_counts().to_dict() == {"conventional": 30, "interval_prefix": 30,
                                                     "sigma": 30, "err": 12, "none": 40}
    pd.testing.assert_frame_equal(a, ra.draw(s))
    assert not a["urn"].duplicated().any()
    with pytest.raises(SystemExit, match="strata"):
        ra.draw(s.iloc[1:])


def test_numeric_checks_spread_and_intervals():
    df = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0], "se": [0.1, 0.2, np.nan, 0.3],
                       "score_95CI_low": [0.5, 1.5, 2.5, 4.5], "score_95CI_high": [1.5, 2.5, 3.5, 5.0],
                       "dG": [1.0, 2.0, 3.0, 4.6], "dG_95CI_low": [0.0, 1.0, 2.0, 4.5],
                       "dG_95CI_high": [2.0, 3.0, 4.0, 5.0], "raw_sigma": [0.1, -0.1, 0.2, 0.2]})
    c = ra.numeric_checks(df, ["se", "score_95CI_low", "score_95CI_high", "dG_95CI_low",
                               "dG_95CI_high", "raw_sigma"])
    assert c["columns"]["se"]["numeric_share"] == 0.75 and c["columns"]["se"]["non_negative"]
    assert not c["columns"]["raw_sigma"]["non_negative"] and c["columns"]["raw_sigma"]["negative"] == 1
    iv = {(i["lower"], i["against"]): i for i in c["intervals"]}
    assert iv[("score_95CI_low", "score")]["inside"] == 3 and not iv[("score_95CI_low", "score")]["holds"]
    assert iv[("dG_95CI_low", "dG")]["holds"]            # the stem's own column
    assert ra._stem("score_95CI_low") == "score" and ra._stem("ci_lower") == ""


def test_wilson():
    lo, hi = ra.wilson(30, 30)
    assert lo == pytest.approx(0.8865, abs=1e-4) and hi == pytest.approx(1.0)
    lo, hi = ra.wilson(15, 30)
    assert (lo, hi) == pytest.approx((0.3315, 0.6685), abs=1e-4)


def _v():
    q = {"source": "MaveDB record methodText", "quote": "the standard error of the score"}
    return {"families": {"F": {"columns": {"se": "uncertainty_of_score"}, "evidence": [q],
                               "basis": "text"}},
            "deposits": {"u1": {"family": "F"},
                         "u2": {"columns": {"raw_sigma": "uncertainty_of_other"}, "evidence": [q],
                                "basis": "relation"},
                         "u3": {"verdict": "missed_uncertainty", "missed_columns": ["spread"],
                                "evidence": [q], "basis": "header"},
                         "u4": {"columns": {"se": "maybe"}, "evidence": [q], "basis": "text"},
                         "u5": {"columns": {"se": "uncertainty_of_score"}, "basis": "text",
                                "evidence": [{"source": "x", "quote": " ".join(["w"] * 26)}]},
                         "u6": {"columns": {"se": "uncertainty_of_score"}, "evidence": [q]}}}


def test_verdicts_follow_the_rules():
    v = _v()
    assert ra.resolve("u1", "conventional", ["se"], v)["verdict"] == "right"
    assert ra.resolve("u2", "sigma", ["raw_sigma"], v)["verdict"] == "wrong"
    assert ra.resolve("u3", "none", [], v)["missed_columns"] == ["spread"]
    with pytest.raises(SystemExit, match="without a verdict"):
        ra.resolve("u2", "sigma", ["raw_sigma", "sigma"], v)
    with pytest.raises(SystemExit, match="bad"):
        ra.resolve("u4", "conventional", ["se"], v)
    with pytest.raises(SystemExit, match="26 words"):
        ra.resolve("u5", "conventional", ["se"], v)
    with pytest.raises(SystemExit, match="no verdict"):
        ra.resolve("u9", "conventional", ["se"], v)
    with pytest.raises(SystemExit, match="basis"):
        ra.resolve("u6", "conventional", ["se"], v)
    assert ra.resolve("u1", "conventional", ["se"], v)["basis"] == "text"


def test_the_summary_weights_the_strata_by_size():
    s = _survey()
    rows = []
    for k in ra.STRATA:
        for i in range(ra.DRAW[k]):
            right = not (k == "sigma" and i < 15)
            rows.append({"stratum": k, "verdict": "right" if right else "wrong",
                         "basis": "header" if k == "err" else "text",
                         "replicates_reproduce_score": None, "survey_note": "",
                         "column_verdicts": '{"x": "%s"}' % ("uncertainty_of_score" if right
                                                             else "uncertainty_of_other")})
    rows += [{"stratum": "none", "basis": "text", "survey_note": "", "column_verdicts": "{}",
              "verdict": "missed_uncertainty" if i < 4 else "missed_replicates" if i < 6 else "none",
              "replicates_reproduce_score": i == 4} for i in range(40)]
    summ = ra.summarise(pd.DataFrame(rows), s)
    w = summ["weighted"]["proportion"]
    assert w == pytest.approx((228 + 1608 + 582 * 0.5 + 12) / 2430)
    assert summ["implied_permits"]["all"]["permits"] == pytest.approx(0 + 2430 * w)
    assert summ["none"]["missed"] == 5            # four, and the one replicate set that reconciles
    assert summ["none"]["proportion"] == pytest.approx(5 / 40)
    assert summ["none"]["implied_missed"]["all"][0] == pytest.approx(322 * 5 / 40)
    assert summ["strata"]["err"]["right_not_on_header_alone"] == 0
    assert summ["strata"]["sigma"]["right_not_on_header_alone"] == 15


def test_a_quotation_must_be_in_its_source(tmp_path):
    (tmp_path / "records").mkdir()
    rec = {"methodText": "Scores were averaged and the\nstandard error \u2013 of the mean \u2013 kept."}
    (tmp_path / "records" / "urn_mavedb_1.json").write_text(__import__("json").dumps(rec))
    assert ra.quote_found("the standard error - of the mean", "mavedb:score_set:methodText",
                          "urn:mavedb:1", tmp_path)
    assert not ra.quote_found("the standard deviation", "mavedb:score_set:methodText",
                              "urn:mavedb:1", tmp_path)
    with pytest.raises(SystemExit, match="unknown evidence source"):
        ra.source_text("web:x", "urn:mavedb:1", tmp_path)


def test_relations_recompute_what_a_verdict_rests_on():
    df = pd.DataFrame({"score": [1.0, 2.0, 4.0], "lfc": [0.5, 1.0, 2.0], "se": [0.5, 0.5, 0.5],
                       "r1": [1.0, 1.0, 3.0], "r2": [1.0, 3.0, 5.0], "sd": [0.0, 1.0, 1.0]})
    got = {r["relation"]: r for r in ra.relations(df, [
        {"equal": ["score", "lfc / se"]}, {"row_mean": ["score", ["r1", "r2"]]},
        {"row_std": ["sd", ["r1", "r2"], 0]}, {"ratio": ["lfc", "score"]},
        {"halfwidth": ["r2", "score", "se"]}])}
    assert got["equal"]["share_holding"] == 1.0 and got["row_mean"]["share_holding"] == 1.0
    assert got["row_std"]["share_holding"] == 1.0 and got["ratio"]["constant"] == 0.5
    assert got["halfwidth"]["max"] == pytest.approx(2.0)

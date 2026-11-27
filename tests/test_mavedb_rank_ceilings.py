"""atlas.mavedb_rank_ceilings (A4c.3): the rank-scale ceiling of the 820 MaveDB deposits with
a ceiling, beside the variance-scale ceiling computed again from each score table."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from atlas import mavedb_ceilings as mc
from atlas import mavedb_rank_ceilings as mr
from atlas import paths

RESULTS = paths.RESULTS


def _built(*names: str) -> None:
    for n in names:
        p = paths.path(n) if n in paths.NAMES else RESULTS / n
        if not p.exists():
            pytest.skip(f"{p.name} not built in this checkout")


def _bits(a, b) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(((a.view(np.uint64) == b.view(np.uint64))
                                        | (np.isnan(a) & np.isnan(b))).all())


def _toy(n=80, seed=2):
    rng = np.random.default_rng(seed)
    t = rng.normal(size=n)
    reps = {f"rep{k}": t + 0.4 * rng.standard_normal(n) for k in (1, 2, 3)}
    df = pd.DataFrame({"accession": [f"a{i:03d}" for i in range(n)], **reps})
    df["score"] = df[["rep1", "rep2", "rep3"]].mean(axis=1)
    df["se"] = 0.4 / np.sqrt(3) * np.exp(rng.normal(0, 0.2, n))
    df["ci_lower"] = df["score"] - 1.959963984540054 * df["se"]
    df["ci_upper"] = df["score"] + 1.959963984540054 * df["se"]
    df.loc[3, "rep2"] = np.nan                                   # an incomplete row
    df.loc[5, "score"] = np.nan                                  # a row without a score
    return df


# --------------------------------------------------------------------------
# one deposit, on toy tables
# --------------------------------------------------------------------------
def test_the_replicates_se_is_their_sd_over_root_k_on_the_complete_rows():
    df = _toy()
    se, gap = mr.replicate_errors(df, "rep1;rep2;rep3")
    reps = df[["rep1", "rep2", "rep3"]]
    ok = reps.notna().all(axis=1) & df["score"].notna()
    want = reps.std(axis=1, ddof=1) / np.sqrt(3)
    assert se[~ok].isna().all() and np.allclose(se[ok], want[ok], rtol=0, atol=0)
    assert se[ok].iloc[0] == pytest.approx(float(np.std(reps.iloc[0], ddof=1) / np.sqrt(3)))
    assert gap == pytest.approx(float((df["score"] - reps.mean(axis=1)).abs()[ok].max()))
    x, _ = mr.variant_set(df, "replicates", "rep1;rep2;rep3")
    assert len(x) == int(ok.sum()) == len(df) - 2


def test_the_variance_ceiling_is_mavedb_ceilings_estimate():
    df = _toy()
    for method, column, cols in (("se", "se", ["se"]),
                                 ("ci", "ci_lower+ci_upper", ["ci_lower", "ci_upper"])):
        got = mr.variance_ceiling(df, method, column)
        want = mc.from_table(df[["accession", "score"] + cols], "u", "errors", "")
        assert want["method"] == method and want["error_column"] == column
        assert got["n"] == want["n_variants"] and got["status"] == want["status"] == "ok"
        assert got["ceiling"] == want["ceiling"]
    got = mr.variance_ceiling(df, "replicates", "rep1;rep2;rep3")
    want = mc.from_table(df, "u", "replicates", "rep1;rep2;rep3")
    assert got["n"] == want["n_variants"] and got["ceiling"] == want["ceiling"]


def test_the_floor_and_above_floor():
    df = _toy(n=120)
    x, _ = mr.variant_set(df, "se", "se")
    r = mr.rank_ceiling("urn:toy", x, nulls=20)
    assert len(r["null"]) == 20
    assert r["floor_own"] == float(np.quantile(r["null"], 0.95))
    assert r["above_floor"] == (r["ceiling_rank_own"] > r["floor_own"])
    again = mr.rank_ceiling("urn:toy", x.iloc[::-1], nulls=20)          # the order does not matter
    assert again["ceiling_rank_own"] == r["ceiling_rank_own"] and _bits(again["null"], r["null"])


# --------------------------------------------------------------------------
# the delivered table
# --------------------------------------------------------------------------
def test_the_trials_70_deposits_bit_for_bit():
    _built("mavedb_rank_ceilings", "mavedb_rank_ceiling_trial")
    t = pd.read_csv(paths.path("mavedb_rank_ceilings"), sep="\t",
                    float_precision="round_trip").set_index("urn")
    tr = pd.read_csv(paths.path("mavedb_rank_ceiling_trial"), sep="\t",
                     float_precision="round_trip").set_index("urn")
    assert len(tr) == 70
    here = t.loc[tr.index]
    for c in ("n", "pooled_se", "ceiling_rank_own", "rank_reliability_own"):
        assert _bits(here[c], tr[c]), c
    assert (here["seed"].astype(float) == tr["seed"].astype(float)).all()
    assert here["reproduces_v2"].all()


def test_three_of_the_trials_deposits_computed_again_from_the_cache():
    _built("mavedb_rank_ceiling_trial")
    tr = pd.read_csv(paths.path("mavedb_rank_ceiling_trial"), sep="\t",
                     float_precision="round_trip").set_index("urn")
    dep = mr.deposits().set_index("urn")
    for urn in tr.sort_values("n").index[:3]:
        if mr.table_path(urn) is None:
            pytest.skip("the score tables are not cached here")
        o = mr.one(dep.loc[urn].to_dict() | {"urn": urn})
        assert o["reproduces_v2"] and o["status"] == "ok"
        for c in ("n", "pooled_se", "ceiling_rank_own"):
            assert _bits(o[c], tr.loc[urn, c]), (urn, c)


def test_the_table_and_its_columns():
    _built("mavedb_rank_ceilings", "mavedb_ceilings")
    t = pd.read_csv(paths.path("mavedb_rank_ceilings"), sep="\t", float_precision="round_trip")
    assert list(t.columns) == mr.COLUMNS and len(t) == 820
    v2 = pd.read_csv(paths.path("mavedb_ceilings"), sep="\t", float_precision="round_trip")
    v2 = v2[v2["status"] == "ok"].set_index("urn")
    assert set(t["urn"]) == set(v2.index)
    ok = t[t["status"] == "ok"]
    assert (ok["above_floor"] == (ok["ceiling_rank_own"] > ok["floor_own"])).all()
    assert np.array_equal(ok["difference"].to_numpy(),
                          (ok["ceiling_rank_own"] - ok["ceiling_variance_here"]).to_numpy())
    rep = ok[ok["reproduces_v2"]]
    assert ((rep["ceiling_variance_here"] - rep["ceiling_v2"]).abs() <= 1e-9).all()
    assert (rep["n_variance_here"] == rep["n_v2"]).all()
    assert (t["table_sha256"].dropna().str.len() == 16).all()
    assert t.loc[t["method"] != "replicates", "replicate_mean_max_abs_diff"].isna().all()


def test_the_summary_uses_the_deposits_that_reproduce_v2():
    _built("mavedb_rank_ceilings", "mavedb_rank_ceilings_summary")
    t = pd.read_csv(paths.path("mavedb_rank_ceilings"), sep="\t", float_precision="round_trip")
    s = json.loads(paths.path("mavedb_rank_ceilings_summary").read_text())
    used = t[(t["status"] == "ok") & t["reproduces_v2"].astype("boolean").fillna(False)]
    assert s["summary"]["used"] == len(used)
    assert sum(r["deposits"] for r in s["summary"]["by_stratum_and_method"]) == len(used)
    assert (s["summary"]["not_computed"]["n"] + s["summary"]["not_reproducing_v2"]["n"]
            + len(used)) == len(t)
    assert len(s["summary"]["replicates"]) == int((used["method"] == "replicates").sum())
    assert s["checks"]["trial_70"]["holds"]
    for stratum, rec in s["summary"]["by_stratum_error_methods"].items():
        d = used[(used["stratum"] == stratum) & used["method"].isin(mr.ERROR_METHODS)]
        assert rec["deposits"] == len(d)
        assert rec["difference"]["median"] == float(d["difference"].median())
        assert rec["not_above_the_floor"] == int((~d["above_floor"]).sum())
    assert len(s["tables"]) == int(t["table_source"].notna().sum())
    assert all(len(v["sha256"]) == 64 for v in s["tables"].values())

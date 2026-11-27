"""atlas.deposit_classes' AUROC table (A4c, K19): the predictors against the deposits' own
calls, on toy data and on the delivered table; the two tables beside it unchanged."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from atlas import clinical_evidence as ce
from atlas import deposit_classes as dc
from atlas import paths, predictors
from atlas.external_reliability import RAW as EXTERNAL_RAW

RESULTS = paths.RESULTS
KEYS = ["gene", "group", "stratum", "model"]


def _built(*names: str) -> None:
    for n in names:
        p = paths.path(n) if n in paths.NAMES else RESULTS / n
        if not p.exists():
            pytest.skip(f"{p.name} not built in this checkout")


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(paths.path(name), sep="\t", float_precision="round_trip")


def _bits(a, b) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(((a.view(np.uint64) == b.view(np.uint64))
                                        | (np.isnan(a) & np.isnan(b))).all())


# --------------------------------------------------------------------------
# toy data
# --------------------------------------------------------------------------
def test_the_auroc_counts_a_tie_as_half_a_pair():
    rng = np.random.default_rng(3)
    score = rng.integers(0, 5, 80).astype(float)                 # many ties
    ab = rng.random(80) < 0.4
    pos, neg = score[ab], score[~ab]
    pairs = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    assert dc.auroc(ab, score) == pytest.approx(pairs / (len(pos) * len(neg)), abs=1e-12)
    assert dc.auroc(ab, -score) == pytest.approx(1 - dc.auroc(ab, score), abs=1e-12)
    assert dc.auroc(ab, np.ones(80)) == 0.5                      # every pair tied


def test_a_class_below_ten_leaves_the_auroc_empty():
    score = np.arange(40.0)
    for k in (9, 31):                                            # 9 abnormal; 9 normal
        ab = np.zeros(40, bool)
        ab[-k:] = True
        assert np.isnan(dc.auroc(ab, score))
        assert all(np.isnan(v) for v in dc.auroc_interval(dc.auroc(ab, score), k, 40 - k))
    ab = np.zeros(40, bool)
    ab[-10:] = True
    assert dc.auroc(ab, score) == 1.0                            # 10 and 30: computed
    assert all(np.isnan(v) for v in dc.auroc_interval(1.0, 10, 30))   # no logit at 1


def test_the_interval_is_hanley_mcneil_on_the_logit_scale():
    auc, n_pos, n_neg = 0.8, 50, 200
    lo, hi = dc.auroc_interval(auc, n_pos, n_neg)
    logit = lambda p: math.log(p / (1 - p))  # noqa: E731
    se = math.sqrt(ce.hanley_mcneil_var(auc, n_pos, n_neg)) / (auc * (1 - auc))
    assert lo < auc < hi
    assert logit(hi) - logit(auc) == pytest.approx(dc.CI_Z * se)
    assert logit(auc) - logit(lo) == pytest.approx(dc.CI_Z * se)


# --------------------------------------------------------------------------
# the delivered tables
# --------------------------------------------------------------------------
def test_the_auroc_table_has_within_class_rhos_rows_n_and_rho():
    _built("deposit_call_auroc", "within_class_rho")
    t, w = _read("deposit_call_auroc"), _read("within_class_rho")
    assert list(t.columns) == dc.AUROC_COLUMNS
    assert len(t) == len(w) and (t[KEYS].astype(str).to_numpy() == w[KEYS].astype(str).to_numpy()).all()
    for a, b in (("n", "n_all"), ("n_abnormal", "n_abnormal"), ("n_normal", "n_normal"),
                 ("rho", "rho_all")):
        assert _bits(t[a], w[b]), a
    small = (t["n_abnormal"] < dc.MIN_CLASS) | (t["n_normal"] < dc.MIN_CLASS)
    assert t.loc[small, ["auroc", "auroc_lo", "auroc_hi"]].isna().all().all()
    assert t.loc[~small, "auroc"].notna().all()
    iv = t[t["auroc_lo"].notna()]
    assert ((iv["auroc_lo"] < iv["auroc"]) & (iv["auroc"] < iv["auroc_hi"])).all()
    assert (t["selection_overlap"] == [predictors.is_overlap(m, g)
                                       for m, g in zip(t["model"], t["gene"])]).all()
    assert not ((t["group"] == "external") & (t["stratum"] == "all")).any()


def test_the_two_tables_computed_again_have_the_delivered_bytes():
    _built("deposit_classes", "within_class_rho", "assay_aux", "external_scores")
    bap1 = paths.REPO / "data" / "raw" / "mavedb" / "urn_mavedb_00000662-0-1" / "scores.csv"
    if not bap1.exists() or not EXTERNAL_RAW.exists():
        pytest.skip("the deposits' score tables are not here")
    core = dc.core_variants()
    dep, sc = dc.external_variants()
    assert (dc.class_table(core, dep).to_csv(sep="\t", index=False).encode("utf-8")
            == paths.path("deposit_classes").read_bytes())
    assert (dc.within_class_table(core, sc).to_csv(sep="\t", index=False).encode("utf-8")
            == paths.path("within_class_rho").read_bytes())


def test_the_summary_leaves_selection_overlap_out_and_spans_the_genes():
    _built("deposit_call_auroc", "deposit_call_auroc_summary")
    s = json.loads(paths.path("deposit_call_auroc_summary").read_text())
    assert all(c["holds"] for c in s["checks"].values())
    t = _read("deposit_call_auroc")
    flag = t["selection_overlap"].astype(bool)
    assert flag.any()                                     # avi on RAD51C
    for rec in s["by_stratum_and_predictor"]:
        d = t[(t["stratum"] == rec["stratum"]) & (t["model"] == rec["model"])]
        if rec["group"] != "all":
            d = d[d["group"] == rec["group"]]
        assert rec["left_out_selection_overlap"] == d.loc[flag[d.index], "gene"].tolist()
        d = d[~flag[d.index]]
        for col, key, sub in (("auroc", "auroc", d), ("rho", "rho", d),
                              ("rho", "rho_genes_with_auroc", d[d["auroc"].notna()])):
            v = sub[col].dropna()
            q = rec[key]
            assert q["n_genes"] == len(v) and q["genes"] == sub.loc[v.index, "gene"].tolist()
            if len(v):
                assert (q["median"], q["min"], q["max"]) == (float(v.median()), float(v.min()),
                                                             float(v.max()))

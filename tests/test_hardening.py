"""Guardrails for the manuscript-hardening analyses.

These cover the statistics that new claims rest on, plus a regression test for
the region-classifier defect that mis-filed UTR-intronic variants as coding.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from atlas.clinical_evidence import (acmg_tier, average_precision,
                                     lr_at_specificity, roc_auc)
from atlas.consequence import classify_variant, translate
from atlas.evaluate import classify_region, splice_offset
from atlas.reliability import spearman_brown
from atlas.robustness import max_spearman_given_ties, min_detectable_rho


# --------------------------------------------------------------------------
# region classifier
# --------------------------------------------------------------------------
@pytest.mark.parametrize("hgvs_c,expected", [
    ("c.5565A>T", "coding_or_utr"),
    ("c.-27T>G", "coding_or_utr"),          # 5'UTR, not intronic
    ("c.*104A>G", "coding_or_utr"),         # 3'UTR, not intronic
    ("c.5467+20C>A", "splice_11_50"),
    ("c.5278-12C>G", "splice_11_50"),
    ("c.4675+1G>A", "splice_1_2"),
    ("c.441+5G>C", "splice_3_10"),
    ("c.302-99A>G", "splice_deep"),
    # regression: an intron inside the 5'UTR is intronic, not coding. An earlier
    # pattern only allowed a `*` prefix and mis-filed these nine BRCA1 variants.
    ("c.-19-1G>T", "splice_1_2"),
    ("c.-19-2A>G", "splice_1_2"),
    ("c.-19-3A>C", "splice_3_10"),
    # and the 3'UTR equivalent must keep working
    ("c.*104+5A>G", "splice_3_10"),
])
def test_classify_region(hgvs_c, expected):
    assert classify_region(hgvs_c) == expected


def test_splice_offset_is_nan_for_non_intronic():
    assert math.isnan(splice_offset("c.5565A>T"))
    assert splice_offset("c.-19-2A>G") == 2


# --------------------------------------------------------------------------
# tie-aware Spearman ceiling
# --------------------------------------------------------------------------
def test_tie_ceiling_no_ties_is_one():
    assert max_spearman_given_ties(np.arange(50, dtype=float)) == pytest.approx(1.0)


def test_tie_ceiling_all_tied_is_zero():
    assert max_spearman_given_ties(np.zeros(50)) == 0.0


def test_tie_ceiling_balanced_dichotomy():
    # a balanced dichotomy admits at most sqrt(3)/2, approached as n grows
    x = np.r_[np.zeros(5000), np.ones(5000)]
    assert max_spearman_given_ties(x) == pytest.approx(math.sqrt(3) / 2, abs=1e-4)
    small = max_spearman_given_ties(np.r_[np.zeros(50), np.ones(50)])
    assert small == pytest.approx(math.sqrt(3) / 2, abs=1e-3)


def test_tie_ceiling_is_attained():
    """The ceiling must actually be reachable by some target vector."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, 60).astype(float)
    ceiling = max_spearman_given_ties(x)
    from scipy import stats
    order = np.argsort(x, kind="stable")
    y = np.empty(60)
    y[order] = np.arange(60)
    assert stats.spearmanr(x, y).statistic == pytest.approx(ceiling, abs=1e-9)


# --------------------------------------------------------------------------
# classification metrics (implemented locally instead of via scikit-learn)
# --------------------------------------------------------------------------
def _brute_force_auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0)
               for a, b in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))


def test_roc_auc_matches_brute_force_with_ties():
    rng = np.random.default_rng(7)
    for _ in range(25):
        y = (rng.random(40) < 0.4).astype(float)
        if y.sum() in (0, len(y)):
            continue
        s = rng.integers(0, 3, 40).astype(float)  # heavy ties
        assert roc_auc(y, s) == pytest.approx(_brute_force_auc(y, s), abs=1e-12)


def test_roc_auc_perfect_and_inverted():
    y = np.r_[np.zeros(10), np.ones(10)]
    assert roc_auc(y, np.arange(20, dtype=float)) == pytest.approx(1.0)
    assert roc_auc(y, -np.arange(20, dtype=float)) == pytest.approx(0.0)


def test_average_precision_perfect_and_baseline():
    y = np.r_[np.ones(5), np.zeros(15)]
    assert average_precision(y, np.r_[np.ones(5), np.zeros(15)]) == pytest.approx(1.0)
    # all scores tied -> precision is the prevalence everywhere
    assert average_precision(y, np.zeros(20)) == pytest.approx(0.25)


def test_lr_at_specificity_and_tier():
    y = np.r_[np.ones(100), np.zeros(100)]
    s = np.r_[np.full(100, 2.0), np.full(100, 1.0)]  # perfectly separated
    lr, _ = lr_at_specificity(y, s, 0.95)
    assert lr > 18.7 and acmg_tier(lr) in ("Strong", "Very strong")
    # an uninformative score sits at LR ~ 1
    rng = np.random.default_rng(1)
    lr_null, _ = lr_at_specificity(y, rng.normal(size=200), 0.95)
    assert 0.2 < lr_null < 3.0
    assert acmg_tier(1.0) == "below supporting"


# --------------------------------------------------------------------------
# reliability
# --------------------------------------------------------------------------
def test_spearman_brown_monotone_and_endpoints():
    assert spearman_brown(1.0) == pytest.approx(1.0)
    assert spearman_brown(0.0) == pytest.approx(0.0)
    assert spearman_brown(0.5) == pytest.approx(2 / 3)
    assert spearman_brown(0.4) < spearman_brown(0.6)


def test_min_detectable_rho_shrinks_with_n():
    assert min_detectable_rho(50) > min_detectable_rho(500) > min_detectable_rho(5000)
    assert 0 < min_detectable_rho(1000) < 0.15


# --------------------------------------------------------------------------
# protein consequence
# --------------------------------------------------------------------------
def test_codon_table_spot_checks():
    assert translate("ATG") == "M"
    assert translate("TGA") == "*"
    assert translate("TTT") == "F"
    assert translate("GGG") == "G"


def test_classify_variant_against_a_synthetic_transcript():
    #        0123456789
    # CDS starts at 3: ATG AAA TGG TAA
    seq = "GGG" + "ATGAAATGGTAA"
    cds_start, cds_end = 3, 3 + 12
    f = lambda h: classify_variant(h, seq, cds_start, cds_end)  # noqa: E731
    assert f("c.4A>G") == "missense"       # AAA(K) -> AGA(R)
    assert f("c.6A>G") == "synonymous"     # AAA(K) -> AAG(K)
    assert f("c.7T>A") == "missense"       # TGG(W) -> AGG(R)
    assert f("c.1A>T") == "start_lost"
    assert f("c.9G>A") == "nonsense"       # TGG(W) -> TGA(*)
    assert f("c.4C>G") == "ref_mismatch"   # reference base disagrees
    assert f("c.5565A>T") == "beyond_cds"
    assert f("c.4_5del") == "indel_or_complex"
    assert f("c.-1G>A") == "utr5"
    assert f("c.*1A>G") == "utr3"
    assert f("c.4+1G>A") == "intronic"


def test_matched_realisation_uses_a_matched_broad_scope_set():
    """The coding-vs-±1-2 comparison must not silently compare different predictors."""
    from pathlib import Path

    from atlas.matched_realisation import MISSENSE_ONLY, compare

    results = Path(__file__).resolve().parents[1] / "results"
    if not (results / "attenuation_v1.tsv").exists():
        pytest.skip("attenuation_v1.tsv not built")
    table, prov = compare(results)
    matched = prov["summary"]["matched_models"]

    assert matched, "no predictor carries an estimate in both strata"
    assert not set(matched) & set(MISSENSE_ONLY), (
        "missense-only meta-predictors leaked into a splice-territory comparison")
    # both metrics must summarise the identical predictor set, or the ratio between
    # observed and corrected is not a like-for-like statement
    assert set(table["n_models"]) == {len(matched)}

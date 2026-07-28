"""Tests for atlas.evaluate: region classifier, Fisher-z, DL pooling, end-to-end."""

import math

import numpy as np
import pandas as pd
import pytest

from atlas.evaluate import classify_region, dl_pool, evaluate, fisher_z, per_gene_spearman


# ------------------------------------------------------------- region rules
@pytest.mark.parametrize(
    "hgvs,expected",
    [
        ("c.5278-12C>G", "splice_region"),
        ("c.5467+20C>A", "splice_region"),
        ("c.594+2T>C", "splice_region"),
        ("c.5565A>T", "coding_or_utr"),
        ("c.-9G>C", "coding_or_utr"),
        ("c.*3G>A", "coding_or_utr"),
    ],
)
def test_classify_region(hgvs, expected):
    assert classify_region(hgvs) == expected


# ------------------------------------------------------------------ fisher z
def test_fisher_z_roundtrip_and_variance():
    z, var = fisher_z(0.75, 100)
    assert math.tanh(z) == pytest.approx(0.75)
    assert var == pytest.approx(1 / 97)


def test_fisher_z_clamps_extreme_rho():
    z, _ = fisher_z(1.0, 50)
    assert math.isfinite(z)


# ------------------------------------------------------------------ DL pool
def test_dl_pool_hand_computed():
    # Two studies, equal precision: pooled must be the mean; tau2 = 0 when Q < k-1.
    res = dl_pool([0.5, 0.7], [0.04, 0.04])
    assert res["pooled"] == pytest.approx(0.6)
    assert res["tau2"] == 0.0
    assert res["se"] == pytest.approx(math.sqrt(1 / (1 / 0.04 + 1 / 0.04)))


def test_dl_pool_tau2_positive_with_heterogeneity():
    res = dl_pool([0.1, 0.9, 0.5], [0.01, 0.01, 0.01])
    assert res["tau2"] > 0
    assert res["i2_pct"] > 50


def test_dl_pool_rejects_empty():
    with pytest.raises(ValueError):
        dl_pool([], [])


# --------------------------------------------------------------- end to end
def _toy_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for gene, strength in (("GENEA", 0.8), ("GENEB", 0.6)):
        fp = rng.normal(size=200)
        noise = rng.normal(size=200)
        score = strength * fp + math.sqrt(1 - strength**2) * noise
        for i in range(200):
            rows.append(
                {
                    "variant_id": f"{gene}#{i}",
                    "gene": gene,
                    "hgvs_c": "c.100+5G>A" if i % 2 else "c.101A>G",
                    "functional_pathogenicity": fp[i],
                    "toymodel": score[i],
                }
            )
    return pd.DataFrame(rows)


def test_evaluate_recovers_simulated_correlation():
    result = evaluate(_toy_df(), "toymodel")
    all_ = result["strata"]["all"]
    assert all_["n_genes_used"] == 2
    # Simulated per-gene Pearson ~0.8/0.6; Spearman pooled should land in between.
    rho = all_["pooled"]["pooled_rho"]
    assert 0.55 < rho < 0.85
    assert all_["pooled"]["rho_ci_lo"] < rho < all_["pooled"]["rho_ci_hi"]
    # Splice stratum uses only the intronic half of the variants.
    splice = result["strata"]["splice_region"]
    assert splice["n_variants"] == 200


def test_per_gene_min_n_flag():
    df = _toy_df()
    df = df[df["gene"] == "GENEA"].head(10)
    pg = per_gene_spearman(df, "toymodel", min_n=30)
    assert pg.iloc[0]["note"] == "below min_n"
    assert math.isnan(pg.iloc[0]["rho"])

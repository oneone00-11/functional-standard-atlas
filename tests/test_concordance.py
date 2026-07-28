"""Tests for atlas.concordance."""

import numpy as np
import pandas as pd
import pytest

from atlas.concordance import concordance


def _frames(n=120):
    rng = np.random.default_rng(1)
    old = pd.DataFrame(
        {
            "chrom": ["17"] * n,
            "pos": range(1000, 1000 + n),
            "ref": ["A"] * n,
            "alt": ["G"] * n,
            "gene": ["GENEA"] * (n // 2) + ["GENEB"] * (n - n // 2),
            "oldscore": rng.normal(size=n),
        }
    )
    new = pd.DataFrame(
        {
            "chrom": ["17"] * n,
            "pos": range(1000, 1000 + n),
            "ref": ["A"] * n,
            "alt": ["G"] * n,
            "newscore": old["oldscore"] * 2 + rng.normal(scale=0.05, size=n),
        }
    )
    return old, new


def test_concordance_high_for_monotone_related_scores():
    old, new = _frames()
    res = concordance(old, "oldscore", new, "newscore")
    assert res["n_shared"] == 120
    assert res["overall_rho"] > 0.95
    assert set(res["per_gene_rho"]) == {"GENEA", "GENEB"}


def test_concordance_rejects_disjoint_inputs():
    old, new = _frames()
    new["pos"] = new["pos"] + 5000
    with pytest.raises(ValueError, match="no shared variants"):
        concordance(old, "oldscore", new, "newscore")

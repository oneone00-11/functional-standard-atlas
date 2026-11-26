"""atlas.deposit_classes (A4b.5): the class bound on toy data and against the trial, and the
within-class rho."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas import deposit_classes as dc
from atlas import paths
from atlas import rank_ceiling as rc

RESULTS = paths.RESULTS


def _built(*names: str) -> None:
    for n in names:
        p = paths.path(n) if n in paths.NAMES else RESULTS / n
        if not p.exists():
            pytest.skip(f"{p.name} not built in this checkout")


# --------------------------------------------------------------------------
# the class bound
# --------------------------------------------------------------------------
def test_the_class_bound_on_toy_data():
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    ab = np.array([True, True, False, False, False, False])
    # the ordering: 5, 4, then four ties below them
    want = rc.spearman(np.array([5.0, 4.0, -1.0, -1.0, -1.0, -1.0]), y)
    assert dc.class_bound(y, ab) == want
    assert dc.class_bound(y, np.ones(6, bool)) == pytest.approx(1.0)
    assert np.isnan(dc.class_bound(y, np.zeros(6, bool)))
    # abnormal variants out of order with the measurement do not raise the bound
    assert dc.class_bound(y, ~ab) < dc.class_bound(y, ab)


def test_the_class_table_agrees_with_the_trial():
    _built("deposit_classes")
    t = pd.read_csv(paths.path("deposit_classes"), sep="\t").set_index(["gene", "stratum"])
    trial = {("BARD1", "splice_3_10"): (70, 473, 0.618), ("BARD1", "splice_11_50"): (25, 1458, 0.189),
             ("PALB2", "splice_3_10"): (104, 501, 0.655), ("PALB2", "splice_11_50"): (35, 540, 0.427),
             ("BAP1", "splice_3_10"): (146, 768, 0.681), ("BAP1", "splice_11_50"): (137, 1500, 0.494),
             ("XRCC2", "splice_11_50"): (3, 294, 0.174), ("SBDS", "splice_11_50"): (0, 258, None)}
    for k, (ab, n, bound) in trial.items():
        r = t.loc[k]
        assert (int(r["n_abnormal"]), int(r["n"])) == (ab, n)
        assert np.isnan(r["class_bound"]) if bound is None else round(r["class_bound"], 3) == bound
    assert not {"BRCA1", "BRCA2", "VHL"} & set(t.index.get_level_values("gene"))


def test_within_class_rho_leaves_small_classes_empty_and_external_all_out():
    _built("within_class_rho")
    w = pd.read_csv(paths.path("within_class_rho"), sep="\t")
    assert not ((w["group"] == "external") & (w["stratum"] == "all")).any()
    for part in ("all", "normal", "abnormal"):
        small = w[f"n_{part}"] < dc.MIN_N
        assert w.loc[small, f"rho_{part}"].isna().all()
    assert (w["n_normal"] + w["n_abnormal"] == w["n_all"]).all()



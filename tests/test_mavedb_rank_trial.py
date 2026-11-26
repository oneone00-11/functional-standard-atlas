"""atlas.mavedb_rank_trial (A4b.5): the error column read as the ceilings table reads it,
and the 70 cached deposits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas import mavedb_rank_trial as mt
from atlas import paths

RESULTS = paths.RESULTS


def _built(*names: str) -> None:
    for n in names:
        p = paths.path(n) if n in paths.NAMES else RESULTS / n
        if not p.exists():
            pytest.skip(f"{p.name} not built in this checkout")


# --------------------------------------------------------------------------
# the MaveDB trial
# --------------------------------------------------------------------------
def test_the_trial_reads_the_error_column_as_the_ceilings_table_does():
    df = pd.DataFrame({"lo": [0.0, 1.0], "hi": [1.96 * 2, 3.0]})
    assert np.allclose(mt.errors(df, "ci", "lo+hi"), [1.96 * 2 / (2 * 1.959963984540054),
                                                      2.0 / (2 * 1.959963984540054)])
    df = pd.DataFrame({"sd": [0.5, "x"]})
    assert mt.errors(df, "sd_as_se", "sd").iloc[0] == 0.5 and np.isnan(mt.errors(df, "sd_as_se", "sd").iloc[1])
    with pytest.raises(ValueError):
        mt.errors(df, "replicates", "a;b")


def test_the_trial_covers_the_cached_deposits():
    _built("mavedb_rank_ceiling_trial")
    t = pd.read_csv(paths.path("mavedb_rank_ceiling_trial"), sep="\t")
    assert len(t) == 70 and t["stratum"].value_counts().to_dict() == {
        "human": 63, "other_natural": 4, "designed_stability": 3}
    assert t["method"].value_counts().to_dict() == {"se": 36, "sd_as_se": 31, "ci": 3}
    assert np.allclose(t["difference"], t["ceiling_rank_own"] - t["ceiling_v2"])

"""atlas.error_models: the two error models per gene, their table, and the self-test's
second version (A4b.2)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from atlas import error_models as em
from atlas import paths
from atlas import rank_ceiling as rc

TABLE = paths.path("error_models")
RECORD = paths.path("error_models_summary")
SELFTEST_V2 = paths.path("rank_ceiling_selftest")
SUMMARY_V2 = paths.path("rank_ceiling_selftest_summary")
needs = pytest.mark.skipif(not RECORD.exists(), reason="run `python -m atlas.error_models` first")


def test_the_core_neutral_variants_are_the_neutral_checks_second_batch():
    nc = pd.read_csv(paths.path("reliability_neutral_check"), sep="\t")
    second = nc[nc["batch"] == "synonymous_spliceai_below_0.1"].set_index("gene")["n"]
    for gene, n in em.NEUTRAL_EXPECTED.items():
        assert len(em.neutral_variants(gene)) == n == int(second[gene])


def test_brca1s_score_is_the_mean_of_its_replicates():
    chk = em.brca1_mean_check()
    assert chk["rows"] == 3893 and chk["holds"]


def test_brca1s_half_difference_is_on_the_functional_scale():
    d = em.half_difference([0.2, -0.1], [0.0, 0.3])
    assert np.allclose(d, [-0.1, 0.2])        # (-(0.2) - -(0.0)) / 2, (0.1 - -0.3) / 2


def test_neutral_residuals_are_centred_on_the_batch_median():
    for gene in ("BARD1", "BRCA1"):
        nv = em.neutral_variants(gene)
        assert np.allclose(nv["z"], (nv["y"] - nv["y"].median()) / nv["s"])
        assert nv["id"].is_monotonic_increasing


@needs
def test_the_table_has_both_models_for_every_gene():
    t = pd.read_csv(TABLE, sep="\t")
    for gene in em.GENES:
        assert set(t.loc[t["gene"] == gene, "model"]) == {"own", "neutral"}
    gauss = t[t["kind"] == "gaussian"]
    assert len(gauss) == 8 and (gauss["sd"] == 1).all() and gauss["n"].notna().all()
    assert gauss[["robust_sd", "kurtosis", "bandwidth"]].isna().all().all()
    b = t[(t["gene"] == "BRCA1") & (t["model"] == "own")]
    assert {"gene", "splice_1_2", "splice_3_10", "splice_11_50"} <= set(b["stratum"])


@needs
def test_the_external_neutral_variants_follow_the_core_rule():
    x = pd.read_parquet(paths.path("external_coding_consequence"))
    t = pd.read_csv(TABLE, sep="\t")
    for gene in em.EXTERNAL:
        g = x[(x["gene"] == gene) & (x["consequence"] == "synonymous")
              & x["standard_error"].notna() & (x["spliceai_ds"] < 0.1)]
        row = t[(t["gene"] == gene) & (t["model"] == "neutral")]
        assert len(row) == 1
        assert row["n"].iloc[0] == len(g) == len(em.neutral_variants(gene))


@needs
def test_a_neutral_row_describes_its_residuals():
    t = pd.read_csv(TABLE, sep="\t", float_precision="round_trip")
    z = em.neutral_variants("BAP1")["z"].to_numpy(float)
    r = t[(t["gene"] == "BAP1") & (t["model"] == "neutral")].iloc[0]
    assert r["sd"] == z.std() and r["share_abs_gt_3"] == (np.abs(z) > 3).mean()
    assert r["bandwidth"] == rc.ErrorModel(z).h


@needs
def test_the_second_selftest_reproduces_the_first_and_adds_the_neutral_cells():
    rec = json.loads(SUMMARY_V2.read_text())
    assert rec["first_version_reproduced"]
    acc = rec["acceptance"]
    assert acc["real_cells_done"] == 36 and acc["known_truth_holds"]
    t = pd.read_csv(SELFTEST_V2, sep="\t")
    assert ((t["part"] == "real_cells") & (t["error"] == "neutral")).sum() == 18
    v1 = pd.read_csv(paths.RESULTS / "rank_ceiling_selftest_v1.tsv", sep="\t",
                     float_precision="round_trip")
    v2 = pd.read_csv(SELFTEST_V2, sep="\t", float_precision="round_trip").iloc[:len(v1)]
    pd.testing.assert_frame_equal(v1.reset_index(drop=True), v2.reset_index(drop=True))

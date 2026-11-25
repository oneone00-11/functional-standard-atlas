"""Guards for the other analyses that read the matrix, in the current generation:
robustness (the splice family's pairs beside the eighteen claims), the attenuation
simulation (v3: SNV marginals, the trials and the seed unchanged), the RNA readout, the
classification and the ensemble. Each keeps the 14 predictors that score only SNVs where
v5 had them; avi's BRCA1 and RAD51C cells enter no quantity across genes."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas import paths, predictors  # noqa: E402

RESULTS = REPO / "results"
SNV14 = set(predictors.SNV_ONLY_IN_V5)


def _built(*keys_or_names):
    for k in keys_or_names:
        p = paths.path(k) if k in paths.NAMES else RESULTS / k
        if not p.exists():
            pytest.skip(f"results/ not built in this checkout ({p.name})")


def _read(p):
    return pd.read_csv(p, sep="\t", float_precision="round_trip")


def _same(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], cols: list[str]) -> None:
    a, b = a.set_index(keys), b.set_index(keys)
    for k in a.index:
        for c in cols:
            x, y = a.loc[k, c], b.loc[k, c]
            if isinstance(x, str) or isinstance(y, str):
                assert str(x) == str(y), (k, c)
            else:
                assert (math.isnan(float(x)) and math.isnan(float(y))) or \
                    np.float64(x).view(np.uint64) == np.float64(y).view(np.uint64), (k, c)


# --------------------------------------------------------------------------
# robustness
# --------------------------------------------------------------------------
def test_the_claimed_pairs_come_first_and_the_family_pairs_are_not_repeated():
    from atlas import robustness as r
    assert r.HEAD_TO_HEAD[:18] == r.CLAIMED and len(r.CLAIMED) == 18
    assert len(r.FAMILY_PAIRS) == 3 * 21 - 5
    seen = {(frozenset((a, b)), s) for a, b, s in r.HEAD_TO_HEAD}
    assert len(seen) == len(r.HEAD_TO_HEAD)
    assert r.FAMILY_COLUMNS == predictors.SPLICE_FAMILY + ["avi"]


def test_avi_leaves_brca1_and_rad51c_out_of_the_tie_audit_and_the_power():
    from atlas import robustness as r
    rng = np.random.default_rng(5)
    rows = [{"gene": g, "hgvs_c": f"c.{i + 1}A>G", "functional_pathogenicity": rng.normal(),
             "avi": rng.normal(), "cadd": rng.normal()}
            for g in ("BRCA1", "RAD51C", "BARD1") for i in range(40)]
    mat = pd.DataFrame(rows)
    old = r.MODELS
    try:
        r.MODELS = ["avi", "cadd"]
        t = r.tie_audit(mat).set_index(["model", "stratum"])
        p = r.power_table(mat).set_index(["model", "stratum"])
    finally:
        r.MODELS = old
    assert t.loc[("avi", "all"), "n"] == 40 and t.loc[("cadd", "all"), "n"] == 120
    assert p.loc[("avi", "all"), "k_genes"] == 1 and p.loc[("cadd", "all"), "k_genes"] == 3


def test_robustness_keeps_v5s_numbers_for_the_snv_only_predictors():
    _built("tie_audit", "power", "logo", "head_to_head", "tie_audit_v1.tsv", "power_v1.tsv",
           "logo_v1.tsv", "head_to_head_v1.tsv")
    for key, old, keys in (("tie_audit", "tie_audit_v1.tsv", ["model", "stratum"]),
                           ("power", "power_v1.tsv", ["model", "stratum"]),
                           ("logo", "logo_v1.tsv", ["model", "stratum"])):
        a = _read(RESULTS / old)
        a = a[a["model"].isin(SNV14)]
        _same(a, _read(paths.path(key)), keys, [c for c in a.columns if c not in keys])
    h1 = _read(RESULTS / "head_to_head_v1.tsv")
    h1 = h1[h1["model_a"].isin(SNV14) & h1["model_b"].isin(SNV14)]
    assert len(h1) == 14
    _same(h1, _read(paths.path("head_to_head")), ["model_a", "model_b", "stratum"],
          [c for c in h1.columns if c not in ("model_a", "model_b", "stratum")])
    assert len(_read(paths.path("head_to_head"))) == 76


# --------------------------------------------------------------------------
# the attenuation simulation
# --------------------------------------------------------------------------
def test_the_simulation_resamples_snv_marginals_with_v2s_grid():
    _built("attenuation_simulation", "attenuation_simulation_summary",
           "attenuation_simulation_v2.json", "attenuation_simulation_v2.tsv")
    v2 = json.loads((RESULTS / "attenuation_simulation_v2.json").read_text())
    v3 = json.loads(paths.path("attenuation_simulation_summary").read_text())
    for k in ("trials_per_cell", "seed", "reliability_targets", "rho_true",
              "score_dependent_calibration"):
        assert v2[k] == v3[k], k
    a, b = _read(RESULTS / "attenuation_simulation_v2.tsv"), _read(paths.path("attenuation_simulation"))
    assert a.shape == b.shape and (a["trials"].to_numpy() == b["trials"].to_numpy()).all()
    from atlas.simulate_attenuation import sim_cells
    cells = sim_cells(paths.path("score_matrix"))
    m = pd.read_parquet(paths.path("score_matrix"), columns=["gene", "ref", "alt"])
    snv = m[(m["ref"].str.len() == 1) & (m["alt"].str.len() == 1)].groupby("gene").size()
    assert sum(len(c["marginal"]) for c in cells if c["gene"] == "BAP1") <= snv["BAP1"]


# --------------------------------------------------------------------------
# the RNA readout, the classification, the ensemble
# --------------------------------------------------------------------------
def test_the_rna_readout_and_the_classification_keep_v5s_numbers():
    _built("rna_readout", "rna_readout_v1.tsv", "clinical_evidence", "clinical_evidence_v1.tsv")
    a = _read(RESULTS / "rna_readout_v1.tsv")
    a = a[a["model"].isin(SNV14 | {"(assay itself)"})]
    b = _read(paths.path("rna_readout"))
    _same(a, b, ["gene", "comparison", "model"], ["n", "rho", "z", "var"])
    assert set(zip(b.loc[b["selection_overlap"], "gene"], b.loc[b["selection_overlap"], "model"])) \
        == {("BRCA1", "avi")}
    c1 = _read(RESULTS / "clinical_evidence_v1.tsv")
    c1 = c1[c1["model"].isin(SNV14)]
    c2 = _read(paths.path("clinical_evidence"))
    _same(c1, c2, ["model", "stratum"], [c for c in c1.columns if c not in ("model", "stratum")])
    assert set(c2.loc[c2["model"] == "avi", "excluded_genes"]) == {"BRCA1;RAD51C"}
    assert (c2.loc[c2["model"] == "avi", "k_genes"] == 2).all()        # BARD1 and PALB2


def test_the_ensemble_takes_the_atlas_splice_scorers_but_not_avi():
    from atlas import ensemble as e
    assert e.PANEL_BROAD == ["cadd", "evo2", "gpn_msa", "phylop100way"]
    assert e.PANEL_SPLICE == ["alphagenome", "spliceai_ds", "pangolin_score"]
    assert set(e.ATLAS_SPLICE) <= set(e.CANDIDATES) and "avi" not in e.CANDIDATES
    _built("ensemble_pooled", "ensemble_pooled_v1.tsv")
    a = _read(RESULTS / "ensemble_pooled_v1.tsv")
    a = a[a["score"].isin(SNV14)]
    _same(a, _read(paths.path("ensemble_pooled")), ["stratum", "score"], ["k_genes", "n", "pooled_rho"])
    for st in ("all", "coding_or_utr", "splice_region"):
        assert paths.path("model_correlation", stratum=st).exists()

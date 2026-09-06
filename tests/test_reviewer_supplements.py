"""Guardrails for the reviewer-requested supplementary analyses.

Pins the official numbers of the analyses added in response to review:

* the hardened attenuation simulation (denser reliability grid, SE-based
  estimator alongside the replicate chain, three noise scenarios, and the
  RMSE-crossover brackets that locate the 0.45 correction bound);
* the BRCA1 estimator-consistency swap and the matched-set leave-one-gene-out
  table;
* the MaveDB deposit ledger, the k = 1/2/3 SE=SD/sqrt(k) sensitivity, and the
  two-direction audit's adjudication status;
* the Steiger-versus-bootstrap P comparison for the manuscript's ordering
  claims.

File-level tests skip on a checkout without results/, exactly as
tests/test_analysis_claims.py does; the pure-function tests run everywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

RESULTS = REPO / "results"

from atlas.robustness import bootstrap_p_two_sided                       # noqa: E402
from atlas.simulate_attenuation import (RELIABILITY_TARGETS_V2,          # noqa: E402
                                       _ranks_average, _ranks_ordinal,
                                       batch_cell_v2, calibrate_score_dependent,
                                       crossover_brackets)


def _need(name: str) -> Path:
    p = RESULTS / name
    if not p.exists():
        pytest.skip(f"{name} not built in this checkout")
    return p


# --------------------------------------------------------------------------
# simulation primitives
# --------------------------------------------------------------------------
def test_ranks_average_matches_scipy_with_ties():
    from scipy import stats

    rng = np.random.default_rng(7)
    a = rng.choice([0.1, 0.2, 0.3, 0.7], size=(40, 25))   # heavy ties
    got = _ranks_average(a)
    want = stats.rankdata(a, axis=1) - 1.0                # 0-based like ours
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_ranks_ordinal_on_continuous_values():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(30, 50))
    got = _ranks_ordinal(a)
    order = a.argsort(axis=1)
    want = np.empty_like(got)
    np.put_along_axis(want, order,
                      np.broadcast_to(np.arange(50, dtype=float), a.shape), axis=1)
    np.testing.assert_array_equal(got, want)


def test_calibrate_score_dependent_recovers_the_shape():
    rng = np.random.default_rng(3)
    t = rng.normal(0.0, 2.0, 50_000)
    se = 0.5 + 0.25 * np.abs(t)
    cal = calibrate_score_dependent(se, t)
    assert cal["a"] == pytest.approx(0.5, abs=1e-2)
    assert cal["b"] == pytest.approx(0.25, abs=1e-2)
    # q = b*sd/a = 0.25*2/0.5 = 1.0
    assert cal["q"] == pytest.approx(1.0, abs=0.05)


def test_crossover_brackets_directions():
    curves = pd.DataFrame({
        "noise": "g", "estimator": "replicates",
        "stratum": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
        "reliability_target": [0.1, 0.2, 0.3, 0.4] * 3,
        "achieved_ceiling": [0.3, 0.4, 0.5, 0.6] * 3,
        "rmse_uncorrected": [0.30, 0.25, 0.20, 0.15] * 3,
        "rmse_corrected": [0.35, 0.26, 0.15, 0.10] * 3,      # crosses 0.2-0.3
    })
    curves.loc[curves["stratum"] == "b", "rmse_corrected"] = [0.2, 0.15, 0.1, 0.05]
    curves.loc[curves["stratum"] == "c", "rmse_corrected"] = [0.9, 0.8, 0.7, 0.6]
    out = {r["stratum"]: r for r in crossover_brackets(curves)}
    assert out["a"]["direction"] == "crosses"
    assert out["a"]["crossover_lo"] == 0.2 and out["a"]["crossover_hi"] == 0.3
    assert out["a"]["achieved_ceiling_lo"] == 0.4
    assert out["b"]["direction"] == "always_helps_on_grid"
    assert out["c"]["direction"] == "never_helps_on_grid"


def test_bootstrap_p_two_sided():
    rng = np.random.default_rng(11)
    all_pos = np.abs(rng.normal(size=4000)) + 0.01
    assert bootstrap_p_two_sided(all_pos) == 0.0
    centred = rng.normal(size=4000)
    assert bootstrap_p_two_sided(centred) == pytest.approx(1.0, abs=0.05)
    shifted = rng.normal(loc=3.0, size=4000)
    p = bootstrap_p_two_sided(shifted)
    assert 0.0 <= p <= 0.05


def test_batch_cell_v2_is_seed_deterministic():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    marginal = np.linspace(-1.0, 1.0, 200) ** 3
    targets = np.array([0.25, 0.8])
    a = batch_cell_v2(marginal, len(marginal), 0.3, targets, 0.386, 20, rng1)
    b = batch_cell_v2(marginal, len(marginal), 0.3, targets, 0.386, 20, rng2)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["noise"]) == {"gaussian", "score_dependent", "bounded"}
    # the SE estimator is fed the known error, so at high reliability it must
    # land near the target while the rank-based replicate chain reads low
    hi = a[a["reliability_target"] == 0.8]
    assert hi["ceiling_hat_se"].mean() > hi["ceiling_hat_replicates"].mean()


# --------------------------------------------------------------------------
# simulation v2 outputs (the 0.45-boundary evidence)
# --------------------------------------------------------------------------
def test_simulation_v2_grid_shape():
    summary = pd.read_csv(_need("attenuation_simulation_v2.tsv"), sep="\t")
    assert sorted(summary["reliability_target"].unique()) == RELIABILITY_TARGETS_V2
    assert set(summary["noise"]) == {"gaussian", "score_dependent", "bounded"}
    assert set(summary["estimator"]) == {"replicates", "se"}
    # 3 noise x 2 estimators x 5 strata x 10 targets x 6 true rho
    assert len(summary) == 3 * 2 * 5 * 10 * 6


def test_simulation_v2_gaussian_replicate_matches_v1_anchors():
    """The v2 gaussian/replicate chain must reproduce the v1 reference points:
    achieved ceiling ~0.277 at target 0.10 with the correction hurting, and
    ~0.451 at target 0.25 with it helping."""
    summary = pd.read_csv(_need("attenuation_simulation_v2.tsv"), sep="\t")
    g = (summary[(summary["noise"] == "gaussian") & (summary["estimator"] == "replicates")]
         .groupby("reliability_target")
         .agg(ceil=("ceiling_hat", "mean"), ru=("rmse_uncorrected", "mean"),
              rc=("rmse_corrected", "mean")))
    assert g.loc[0.10, "ceil"] == pytest.approx(0.277, abs=0.01)
    assert g.loc[0.10, "rc"] > g.loc[0.10, "ru"]            # correction hurts
    assert g.loc[0.10, "rc"] / g.loc[0.10, "ru"] == pytest.approx(1.5, abs=0.25)
    assert g.loc[0.25, "ceil"] == pytest.approx(0.451, abs=0.01)
    assert g.loc[0.25, "rc"] < g.loc[0.25, "ru"]            # correction helps


def test_simulation_v2_pooled_crossover_brackets_the_045_bound():
    """The precise crossover location for the manuscript: pooled over strata,
    the correction starts paying between reliability targets 0.15 and 0.20,
    i.e. between achieved ceilings ~0.34 and ~0.40 -- just under 0.45."""
    doc = json.loads(_need("attenuation_simulation_v2.json").read_text())
    pooled = {(r["noise"], r["estimator"]): r
              for r in doc["crossover_pooled_over_strata"]}
    r = pooled[("gaussian", "replicates")]
    assert r["direction"] == "crosses"
    assert r["crossover_lo"] == pytest.approx(0.15)
    assert r["crossover_hi"] == pytest.approx(0.20)
    assert 0.30 < r["achieved_ceiling_lo"] < r["achieved_ceiling_hi"] < 0.45
    # the SE chain crosses in the same target bracket under Gaussian noise
    s = pooled[("gaussian", "se")]
    assert (s["crossover_lo"], s["crossover_hi"]) == (0.15, 0.20)


def test_simulation_v2_stratum_crossovers_are_not_uniform():
    """coding_or_utr improves everywhere on the grid; splice_1_2 only crosses
    higher up -- the stratum dependence the review asked to be made explicit."""
    doc = json.loads(_need("attenuation_simulation_v2.json").read_text())
    by_stratum = {(r["noise"], r["estimator"], r["stratum"]): r
                  for r in doc["crossover"]}
    coding = by_stratum[("gaussian", "replicates", "coding_or_utr")]
    assert coding["direction"] == "always_helps_on_grid"
    splice = by_stratum[("gaussian", "replicates", "splice_1_2")]
    assert splice["direction"] == "crosses"
    assert splice["crossover_lo"] >= 0.25            # v1 bracket: 0.25-0.50
    assert splice["crossover_hi"] <= 0.50


def test_simulation_v2_se_estimator_reads_higher_ceiling_at_low_reliability():
    """At a 0.10 target the SE chain lands closer to the truth than the
    rank-based replicate chain (0.287 vs 0.274 achieved ceiling)."""
    curves = pd.read_csv(_need("attenuation_simulation_v2_curves.tsv"), sep="\t")
    g = curves[(curves["noise"] == "gaussian") & (curves["reliability_target"] == 0.10)]
    ceil = g.groupby("estimator")["achieved_ceiling"].mean()
    assert ceil["se"] > ceil["replicates"]
    assert ceil["replicates"] == pytest.approx(0.277, abs=0.01)


def test_simulation_v2_bounded_noise_exposes_se_misspecification():
    """Under truncation the nominal SE overstates the true error, so the SE
    chain's reliability collapses at low targets and the correction explodes --
    the signature the bounded scenario was built to show."""
    curves = pd.read_csv(_need("attenuation_simulation_v2_curves.tsv"), sep="\t")
    b = curves[(curves["noise"] == "bounded") & (curves["estimator"] == "se")]
    low = b[b["reliability_target"] == 0.25]
    assert low["achieved_ceiling"].mean() < 0.15
    # pooled over strata the correction explodes (mean RMSE ~2.2 vs 0.25),
    # driven by the strata whose ceilings collapse to ~0
    assert low["rmse_corrected"].mean() > 4 * low["rmse_uncorrected"].mean()
    assert low["rmse_corrected"].max() > 1.0
    hi = b[b["reliability_target"] == 0.95]
    assert hi["achieved_ceiling"].mean() == pytest.approx(0.975, abs=0.02)


def test_simulation_v2_noise_calibration_is_recorded():
    doc = json.loads(_need("attenuation_simulation_v2.json").read_text())
    cal = doc["score_dependent_calibration"]
    assert cal["a"] == pytest.approx(0.170, abs=0.01)
    assert cal["b"] == pytest.approx(0.072, abs=0.01)
    assert cal["q"] == pytest.approx(0.386, abs=0.02)


# --------------------------------------------------------------------------
# BRCA1 estimator consistency + matched LOO
# --------------------------------------------------------------------------
def test_brca1_estimator_swap_official_numbers():
    doc = json.loads(_need("estimator_consistency_v1.json").read_text())
    r = doc["matched_realisation_ratio"]
    assert r["published_brca1_replicates"] == pytest.approx(1.431, abs=0.002)
    assert r["brca1_swapped_to_se"] == pytest.approx(1.451, abs=0.002)
    assert r["n_matched_models"] == 9
    lo, hi = doc["cross_check"]["delta_range"]
    assert lo == pytest.approx(0.0491, abs=0.0005)
    assert hi == pytest.approx(0.0619, abs=0.0005)
    med = doc["ceiling_medians_by_stratum"]["splice_1_2"]
    assert med["published_brca1_replicates"] == pytest.approx(0.772, abs=0.002)
    assert med["brca1_swapped_to_se"] == pytest.approx(0.794, abs=0.002)


def test_brca1_cross_check_table():
    cross = pd.read_csv(_need("estimator_consistency_v1.tsv"), sep="\t")
    assert len(cross) == 6
    coding = cross[cross["stratum"] == "coding_or_utr"].iloc[0]
    assert coding["delta_se_minus_replicates"] == pytest.approx(0.0491, abs=0.0005)
    assert (cross["delta_se_minus_replicates"] > 0).all()


def test_matched_realisation_loo_table():
    loo = pd.read_csv(_need("matched_realisation_loo_v1.tsv"), sep="\t")
    assert len(loo) == 8                       # 4 scopes x rho/realisation
    got = {(r["scope"], r["metric"]): r["median_ratio"] for _, r in loo.iterrows()}
    assert got[("full (all 3 genes)", "realisation")] == pytest.approx(1.431, abs=0.005)
    assert got[("drop BARD1", "realisation")] == pytest.approx(1.246, abs=0.005)
    assert got[("drop BARD1", "rho")] == pytest.approx(1.421, abs=0.005)
    assert got[("drop PALB2", "realisation")] == pytest.approx(1.610, abs=0.005)
    assert got[("drop PALB2", "rho")] == pytest.approx(2.413, abs=0.005)
    # the reversal: dropping BRCA1 flips the corrected ratio below 1
    assert got[("drop BRCA1", "realisation")] == pytest.approx(0.780, abs=0.005)
    assert got[("drop BRCA1", "realisation")] < 1.0
    assert got[("drop BRCA1", "rho")] == pytest.approx(1.087, abs=0.005)


# --------------------------------------------------------------------------
# MaveDB ledger, k-sensitivity, audit status
# --------------------------------------------------------------------------
def test_mavedb_accounting_ledger():
    ledger = pd.read_csv(_need("mavedb_accounting_v1.tsv"), sep="\t")
    got = dict(zip(ledger["metric"], ledger["count"]))
    assert got["published score sets"] == 2803
    assert got["human score sets"] == 1202
    assert got["human permitting an estimate"] == 991
    assert got["paper counting class: human ceiling candidates"] == 728
    assert got["ceilings computed (paper class)"] == 674
    frac = dict(zip(ledger["metric"], ledger["fraction"]))
    assert frac["human permitting an estimate"] == pytest.approx(0.824, abs=0.002)
    assert frac["ceilings computed (paper class)"] == pytest.approx(0.926, abs=0.002)
    assert frac["ceilings computed / all published"] == pytest.approx(0.240, abs=0.002)


def test_mavedb_k_sensitivity_table():
    ks = pd.read_csv(_need("mavedb_ceiling_k_sensitivity_v1.tsv"), sep="\t")
    sd = ks[ks["view"] == "sd_as_se subset"].set_index("k")
    assert sd["n_rows"].unique().tolist() == [606]
    assert sd.loc[1, "below_0.45"] == pytest.approx(0.039, abs=0.005)
    assert sd.loc[2, "below_0.45"] == pytest.approx(0.002, abs=0.005)
    assert sd.loc[3, "below_0.45"] == pytest.approx(0.005, abs=0.005)
    whole = ks[ks["view"] == "whole distribution (excl designed_stability)"].set_index("k")
    assert whole.loc[1, "below_0.45"] == pytest.approx(0.033, abs=0.005)
    assert whole.loc[2, "below_0.45"] == pytest.approx(0.004, abs=0.005)
    assert whole.loc[3, "below_0.45"] == pytest.approx(0.007, abs=0.005)
    assert set(ks["k"]) == {1, 2, 3}


def test_mavedb_audit_status():
    status = pd.read_csv(_need("mavedb_survey_audit_status_v1.tsv"), sep="\t")
    assert len(status) == 131
    counts = status["permits"].value_counts().to_dict()
    assert counts == {"none": 60, "errors": 25, "replicates": 18,
                      "replicates_unverified": 28}
    # nothing has been adjudicated by this module; the false-positive direction
    # (25 'errors' rows) is explicitly pending manual adjudication
    assert (status["adjudication"] == "pending manual adjudication").all()
    fp = status[status["permits"] == "errors"]
    assert len(fp) == 25
    assert fp["direction"].str.contains("false-positive").all()


# --------------------------------------------------------------------------
# Steiger vs bootstrap P
# --------------------------------------------------------------------------
def test_head_to_head_pvalue_comparison():
    pv = pd.read_csv(_need("head_to_head_pvalues_v1.tsv"), sep="\t")
    pairs = set(zip(pv["model_a"], pv["model_b"], pv["stratum"]))
    assert ("evo2", "cadd", "all") in pairs
    am = pv[pv["model_a"] == "alphamissense"]
    assert len(am) == 8                        # the eight missense pairs
    assert pv["boot_p"].between(0, 1).all()
    # at 0.05 the two tests tell the same story on every claimed pair
    assert pv["agree_at_0.05"].all()
    evo2_cadd = pv[(pv["model_a"] == "evo2") & (pv["model_b"] == "cadd")].iloc[0]
    assert evo2_cadd["steiger_p"] == pytest.approx(0.028, abs=0.01)
    assert evo2_cadd["boot_p"] <= 0.01
    # every AlphaMissense pair is bootstrap-significant
    assert (am["boot_p"] <= 0.001).all()


def test_head_to_head_v1_carries_both_p_values():
    h2h = pd.read_csv(_need("head_to_head_v1.tsv"), sep="\t")
    assert {"steiger_p", "boot_p"} <= set(h2h.columns)
    assert h2h["boot_p"].between(0, 1).all()

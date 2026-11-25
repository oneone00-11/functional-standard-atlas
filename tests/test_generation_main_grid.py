"""Guards for the current generation's main grid: the pooled grid by population
(atlas.evaluate_ext), the layer ceilings with their intervals and the neutral-variant
check (atlas.reliability, atlas.external_reliability), the matched ceilings
(atlas.matched_ceilings), the two small tables; and the four regressions against v5 and
A2. Pure-function tests use synthetic data; tests of the delivered files skip where
results/ is not built."""
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

from atlas import evaluate_ext as ee  # noqa: E402
from atlas import matched_ceilings as mc  # noqa: E402
from atlas import paths, population, predictors  # noqa: E402
from atlas import reliability as rl  # noqa: E402

RESULTS = REPO / "results"
RNG = np.random.default_rng(11)


def _built(*keys_or_names: str) -> None:
    for k in keys_or_names:
        p = paths.path(k) if k in paths.NAMES else RESULTS / k
        if not p.exists():
            pytest.skip(f"results/ not built in this checkout ({p.name})")


def _bits(a, b) -> bool:
    a, b = float(a), float(b)
    return (math.isnan(a) and math.isnan(b)) or \
        np.float64(a).view(np.uint64) == np.float64(b).view(np.uint64)


# --------------------------------------------------------------------------
# the grid
# --------------------------------------------------------------------------
def _matrix(n=40, genes=("BRCA1", "RAD51C", "BARD1"), kind="snv"):
    rows = []
    for g in genes:
        t = RNG.normal(size=n)
        for i in range(n):
            rows.append({"variant_id": f"{g}#{i}", "gene": g,
                         "ref": "A" if kind == "snv" else "AT", "alt": "G" if kind == "snv" else "A",
                         "hgvs_c": f"c.{i + 10}A>G", "functional_pathogenicity": t[i],
                         "consequence": "missense", "clinvar_arm": "unrecorded",
                         "clinvar_arm_strict": "unrecorded",
                         "variant_class": "snv" if kind == "snv" else "deletion",
                         "avi": t[i] + RNG.normal(), "cadd": t[i] + RNG.normal()})
    return pd.DataFrame(rows)


def test_the_strata_of_each_population():
    snv = population.select(_matrix())
    m = ee.stratum_masks(snv, "snv")
    assert list(m) == ee.TERRITORIES + ee.CONSEQUENCE + ee.CLINVAR
    non = population.select(_matrix(kind="del"), "non_snv")
    assert list(ee.stratum_masks(non, "non_snv")) == ee.TERRITORIES + ee.CLASSES


def test_avi_pools_without_brca1_and_rad51c_and_lists_them():
    d = population.select(_matrix())
    rows, genes = ee.evaluate_population(d.assign(alphagenome=np.nan), "snv")
    grid, pg = pd.DataFrame(rows), pd.DataFrame(genes)
    a = grid[(grid["model"] == "avi") & (grid["stratum"] == "all")].iloc[0]
    c = grid[(grid["model"] == "cadd") & (grid["stratum"] == "all")].iloc[0]
    assert a["k_genes"] == 1 and math.isnan(a["pooled_rho"]) and a["n"] == 40
    assert a["excluded_genes"] == "BRCA1;RAD51C"
    assert c["k_genes"] == 3 and c["excluded_genes"] == "" and c["n"] == 120
    flagged = pg[pg["selection_overlap"]]
    assert set(zip(flagged["model"], flagged["gene"])) == {("avi", "BRCA1"), ("avi", "RAD51C")}


def test_non_snv_rows_only_for_predictors_scoring_a_non_snv():
    d = _matrix(kind="del").assign(cadd=np.nan)
    rows, _ = ee.evaluate_population(population.select(d, None), "non_snv")
    assert {r["model"] for r in rows} == {"avi"}


def test_regression_one_and_three_per_gene_values_equal_v5():
    """The 14 predictors that score only SNVs, and the conservation and allele-frequency
    columns in the three genes with SNVs only: per-gene n and rho equal v5's to the bit,
    against the most precise delivered file (attenuation_v1.tsv, then eval_<model>.json)."""
    _built("eval_ext_per_gene", "attenuation_v1.tsv")
    pg = pd.read_csv(paths.path("eval_ext_per_gene"), sep="\t", float_precision="round_trip")
    pg = pg[pg["population"] == "snv"].set_index(["model", "stratum", "gene"])
    att = pd.read_csv(RESULTS / "attenuation_v1.tsv", sep="\t", float_precision="round_trip")
    checked = 0
    for r in att.itertuples(index=False):
        if r.model in predictors.SNV_ONLY_IN_V5 or (
                r.model in predictors.SCORE_NON_SNV_IN_V5[1:] and r.gene == "BRCA1"):
            o = pg.loc[(r.model, r.stratum, r.gene)]
            assert int(o["n"]) == int(r.n) and _bits(o["rho"], r.rho), (r.model, r.stratum, r.gene)
            checked += 1
    for m in predictors.V5:
        f = RESULTS / f"eval_{m}.json"
        if not f.exists():
            continue
        genes = None if m in predictors.SNV_ONLY_IN_V5 else (
            {"BRCA1", "BRCA2", "VHL"} if m != "alphagenome" else set())
        for st, rec in json.loads(f.read_text())["strata"].items():
            if st not in ee.TERRITORIES:
                continue
            for g in rec["per_gene"]:
                rho = g.get("rho")
                if rho is None or math.isnan(rho) or (genes is not None and g["gene"] not in genes):
                    continue                      # below the minimum n in v5 too
                o = pg.loc[(m, st, g["gene"])]
                assert int(o["n"]) == g["n"] and _bits(o["rho"], g["rho"]), (m, st, g["gene"])
                checked += 1
    assert checked > 400


def test_regression_two_pooled_values_equal_v5():
    _built("eval_ext", "eval_ext_v1.tsv")
    v1 = pd.read_csv(RESULTS / "eval_ext_v1.tsv", sep="\t", float_precision="round_trip")
    v2 = pd.read_csv(paths.path("eval_ext"), sep="\t", float_precision="round_trip")
    v1 = v1.set_index(["model", "stratum"])
    v2 = v2[v2["population"] == "snv"].set_index(["model", "stratum"])
    for m in predictors.SNV_ONLY_IN_V5:
        for st in ee.TERRITORIES + ee.CONSEQUENCE:
            for c in ("k_genes", "n", "pooled_rho", "ci_lo", "ci_hi", "i2_pct"):
                assert _bits(v1.loc[(m, st), c], v2.loc[(m, st), c]), (m, st, c)


def test_the_grid_has_23_predictors_by_16_snv_strata_and_the_non_snv_strata():
    _built("eval_ext")
    g = pd.read_csv(paths.path("eval_ext"), sep="\t")
    snv = g[g["population"] == "snv"]
    assert len(snv) == 23 * 16 and set(snv["model"]) == set(predictors.PREDICTORS)
    non = g[g["population"] == "non_snv"]
    assert set(non["model"]) == set(predictors.SCORE_NON_SNV_IN_V5)
    assert {"deletion", "complex"} <= set(non["stratum"]) and "splice_deep" not in set(non["stratum"])
    assert set(g.loc[g["model"] == "avi", "excluded_genes"]) == {"BRCA1;RAD51C"}


# --------------------------------------------------------------------------
# the layer ceilings
# --------------------------------------------------------------------------
def test_the_layer_bootstrap_is_seeded_and_brackets_the_estimate():
    y = RNG.normal(size=300)
    se = np.full(300, 0.5)
    a = rl.bootstrap_layer("se", 5, y=y, se=se, b=400)
    assert a == rl.bootstrap_layer("se", 5, y=y, se=se, b=400)
    rel = 1 - 0.25 / np.var(y, ddof=1)
    assert a["reliability_lo"] <= rel <= a["reliability_hi"]
    t = RNG.normal(size=200)
    reps = [t + RNG.normal(size=200) for _ in range(3)]
    one = rl.bootstrap_layer("replicates", 9, reps=reps, k=1, b=300)
    three = rl.bootstrap_layer("replicates", 9, reps=reps, k=3, b=300)
    assert three["reliability_lo"] > one["reliability_lo"]          # same resamples, k higher


def test_the_inflated_ceiling_multiplies_the_error_variance():
    assert rl.inflated_ceiling(0.2, 1.0, 1.0) == pytest.approx(math.sqrt(0.8))
    assert rl.inflated_ceiling(0.2, 1.0, 3.0) == pytest.approx(math.sqrt(0.4))
    assert rl.inflated_ceiling(0.2, 1.0, 6.0) == 0.0
    assert math.isnan(rl.inflated_ceiling(0.2, 1.0, None))


def test_the_neutral_check_reads_synonymous_snvs_with_an_se():
    n = 60
    mat = pd.DataFrame({"variant_id": [f"v{i}" for i in range(n)], "gene": "BAP1",
                        "ref": "A", "alt": "G", "hgvs_c": [f"c.{i + 1}A>G" for i in range(n)],
                        "functional_pathogenicity": RNG.normal(scale=2.0, size=n),
                        "consequence": ["synonymous"] * 50 + ["missense"] * 10,
                        "spliceai_ds": [0.05] * 40 + [0.5] * 20})
    aux = pd.DataFrame({"variant_id": mat["variant_id"], "gene": "BAP1", "se": 1.0,
                        "se_status": "validated", "se_validated_by": "test_statistic"})
    nc = rl.neutral_check(mat, aux)
    b = nc[nc["gene"] == "BAP1"].set_index("batch")
    assert b.loc["synonymous", "n"] == 50 and b.loc["synonymous_spliceai_below_0.1", "n"] == 40
    want = mat["functional_pathogenicity"][:40].var()
    assert b.loc["synonymous_spliceai_below_0.1", "ratio"] == pytest.approx(want)
    assert rl.inflation_factors(nc)["BAP1"] == pytest.approx(max(1.0, want))


def test_the_delivered_layer_ceilings():
    _built("reliability", "reliability_summary", "reliability_neutral_check")
    t = pd.read_csv(paths.path("reliability"), sep="\t")
    main = t[t["role"] == "main"]
    assert set(main["gene"]) == {"BRCA1", "BARD1", "PALB2", "BAP1"}
    assert not ({"RAD51C", "BRCA2", "VHL"} & set(t["gene"]))
    assert set(main.loc[main["gene"] == "BRCA1", "method"]) == {"replicates"}
    assert (t["estimable"] == (t["ceiling_lo"] > 0)).all()
    se = t[t["method"] == "se"]
    assert (se["reliability"] == se["reliability_unclipped"].clip(0, 1)).all()
    s = json.loads(paths.path("reliability_summary").read_text())
    assert s["instruction_trial"]["differ_by_more_than_tolerance"] == []
    assert s["inflation_factor"]["BAP1"] == pytest.approx(3.409, abs=1e-3)
    infl = t[t["ceiling_inflated"].notna()]
    assert set(infl["gene"]) == {"BARD1", "PALB2", "BAP1"}


def test_external_snvs_are_named_by_their_c_name():
    from atlas.external_reliability import deposit_population
    names = pd.Series(["c.10A>G", "c.100+5G>T", "c.-19-1G>T", "c.*104+5A>G", "c.79_81del",
                       "c.376_377delinsAC", "c.12dup"])
    assert deposit_population(names).tolist() == ["snv"] * 4 + ["non_snv"] * 3


def test_the_external_summary_combines_the_core_main_rows():
    from atlas.external_reliability import summarise_v2
    core = pd.DataFrame([{"gene": "A", "population": "snv", "stratum": "all", "ceiling": 0.8,
                          "estimable": True, "role": "main"},
                         {"gene": "A", "population": "snv", "stratum": "all", "ceiling": 0.9,
                          "estimable": True, "role": "cross_check"}])
    ext = pd.DataFrame([{"gene": "D", "population": "snv", "stratum": "all", "n": 50,
                         "rep_rho": np.nan, "k": np.nan, "reliability": 0.36, "ceiling": 0.6,
                         "ceiling_lo": 0.0, "ceiling_hi": 0.7, "estimable": False,
                         "role": "main", "status": "validated"}])
    s = summarise_v2(ext, core)
    m = s["main"][0]
    assert m["genes"] == ["A", "D"] and m["genes_estimable"] == ["A"]
    assert m["ceiling_median"] == pytest.approx(0.7)


# --------------------------------------------------------------------------
# the matched ceilings
# --------------------------------------------------------------------------
def test_a_gene_without_an_error_model_gets_rho_and_its_interval_only():
    x, y = RNG.normal(size=80), RNG.normal(size=80)
    p = mc.point(x, y, "none")
    assert p["rho"] == float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    assert all(math.isnan(p[k]) for k in ("reliability", "ceiling", "realisation"))
    b = mc.bootstrap({"x": x, "y": y}, "none", seed=3, b=200)
    assert b["rho_lo"] < p["rho"] < b["rho_hi"] and math.isnan(b["ceiling_lo"])


def test_the_non_snv_scope_takes_the_other_variants():
    n = 70
    df = pd.DataFrame({"gene": "G", "group": "core", "method": "se", "region": "splice_1_2",
                       "functional_pathogenicity": RNG.normal(size=n), "se": 0.3,
                       "fit_rep1": np.nan, "fit_rep2": np.nan, "has_error": True,
                       "snv": [True] * 35 + [False] * 35, "m": RNG.normal(size=n)})
    tasks = mc.cell_tasks(df, ["m"], ["splice_1_2"], scopes=("snv", "non_snv"))
    assert {(k[4], len(a["x"])) for k, _, a in tasks} == {("snv", 35), ("non_snv", 35)}


def test_finish_adds_the_flag_the_inflated_ceiling_and_the_breakeven():
    raw = pd.DataFrame([
        {"group": "core", "gene": "BAP1", "stratum": "all", "model": "avi", "scope": "snv",
         "n": 50, "method": "se", "rho": 0.5, "rep_rho": np.nan, "reliability": 0.9,
         "reliability_unclipped": 0.9, "ceiling": math.sqrt(0.9), "realisation": 0.5 / math.sqrt(0.9),
         "seed": 1, "bootstrap_B": 10, "ceiling_lo": 0.8},
        {"group": "core", "gene": "BRCA1", "stratum": "all", "model": "avi", "scope": "snv",
         "n": 50, "method": "replicates", "rho": 0.5, "rep_rho": 0.7, "reliability": 0.82,
         "reliability_unclipped": np.nan, "ceiling": 0.9, "realisation": 0.55, "seed": 2,
         "bootstrap_B": 10, "ceiling_lo": 0.0}])
    t = mc.finish(raw.reindex(columns=list(raw.columns) + [
        c for c in mc.COLUMNS_V2 if c not in raw.columns and c != "population"]), {"BAP1": 3.0})
    b, r = t.iloc[0], t.iloc[1]
    assert b["ceiling_inflated"] == pytest.approx(math.sqrt(1 - 3 * 0.1))
    assert b["breakeven_inflation"] == pytest.approx((1 - 0.25) / 0.1)
    assert b["estimable"] and not r["estimable"]
    assert r["selection_overlap"] and not b["selection_overlap"]
    assert math.isnan(r["ceiling_inflated"]) and math.isnan(r["breakeven_inflation"])


def test_regression_four_matched_ceilings_equal_a2s():
    """Core BRCA1, BARD1, PALB2 x the 18 predictors whose definition did not change, and
    the external five x their nine columns: n, rho, reliability, ceiling and realisation
    to the bit; intervals too, except where the seed's key changed (the external
    alphagenome_official cells, now named alphagenome)."""
    _built("matched_ceilings", "matched_ceilings_v1.tsv")
    t = pd.read_csv(paths.path("matched_ceilings"), sep="\t", float_precision="round_trip")
    reg = mc.regression_v1(t)
    assert reg["holds"] and reg["point_bitwise"] and reg["rows_compared"] == 422
    assert {(d["cell"][0], d["cell"][3]) for d in reg["interval_differ"]} <= \
        {("external", "alphagenome")}


def test_the_matched_summary_leaves_flagged_and_unestimable_cells_out():
    _built("matched_ceilings", "matched_ceilings_summary")
    t = pd.read_csv(paths.path("matched_ceilings"), sep="\t")
    s = json.loads(paths.path("matched_ceilings_summary").read_text())
    assert set(t["population"]) <= {"snv", "non_snv"}
    assert set(t.loc[t["selection_overlap"], "gene"]) == {"BRCA1", "RAD51C"}
    assert set(t.loc[t["selection_overlap"], "model"]) == {"avi"}
    assert (t.loc[t["method"] == "none", "ceiling"].isna()).all()
    for rec in s["2_by_gene_stratum"]:
        for k in ("best", "best_splice_family"):
            assert rec[k] is None or not (rec[k]["model"] == "avi" and rec["gene"] in
                                          ("BRCA1", "RAD51C"))
    assert all(c["ceiling_lo"] > 0 for c in s["5_realisation_above_1_or_interval_crossing_1"]["cells"])
    per = s["1_by_stratum"]["splice_3_10"]["all"]["per_predictor"]["avi"]
    assert per["rho"]["n_genes"] == len(set(t[(t["model"] == "avi") & (t["stratum"] == "splice_3_10")
                                             & (t["population"] == "snv")]["gene"]) - {"BRCA1", "RAD51C"})


# --------------------------------------------------------------------------
# the two small tables
# --------------------------------------------------------------------------
def test_the_definition_change_table_sets_the_two_definitions_side_by_side():
    _built("alphagenome_definition_change", "score_matrix_atlas_v2.parquet")
    t = pd.read_csv(paths.path("alphagenome_definition_change"), sep="\t")
    assert set(t["stratum"]) == set(ee.TERRITORIES) and set(t["population"]) == {"snv"}
    pooled = t[t["gene"] == "pooled"]
    assert len(pooled) == 7
    g = t[t["gene"] != "pooled"].dropna(subset=["rho_older", "rho_official"])
    assert np.allclose(g["rho_difference"], g["rho_official"] - g["rho_older"], rtol=0, atol=1e-15)
    grid = pd.read_csv(paths.path("eval_ext"), sep="\t", float_precision="round_trip")
    new = grid[(grid["population"] == "snv") & (grid["model"] == "alphagenome")].set_index("stratum")
    for r in pooled.itertuples():
        assert r.rho_official == pytest.approx(new.loc[r.stratum, "pooled_rho"], abs=1e-12)


def test_the_saturation_table_has_the_twelve_genes():
    _built("canonical_site_saturation")
    t = pd.read_csv(paths.path("canonical_site_saturation"), sep="\t")
    assert len(t) == 12 and set(t["group"]) == {"core", "external"}
    assert t.loc[t["gene"].isin(["BAP1", "VHL", "BRCA2"]), "n_abnormal"].isna().all()
    assert (t["spliceai_ds_n_ge_0.5"] <= t["spliceai_ds_n_ge_0.2"]).all()

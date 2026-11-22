"""Guards for atlas.proteingym_pilot: the pairing rules, the data check, the tiers and
the realisation arithmetic. Synthetic data only; nothing here reads the network or
results/_scratch (CONVENTIONS.md rule 6)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas import proteingym_pilot as pg  # noqa: E402


# --------------------------------------------------------------------------
# names, titles, versions
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name, want", [
    ("Findlay, Gregory M", "findlay"), ("Findlay", "findlay"), ("Findlay GM", "findlay"),
    ("Findlay G M", "findlay"), ("Kozek KA", "kozek"), ("Muhammad A", "muhammad"),
    ("O'Neill MJ", "oneill"), ("van Loggerenberg, Warren", "vanloggerenberg"),
    ("van Loggerenberg", "vanloggerenberg"), ("van Loggerenberg W", "vanloggerenberg"),
    ("Jiménez-Ochoa", "jimenezochoa"), ("Jimenez-Ochoa, Ana", "jimenezochoa")])
def test_surnames_compare_across_the_spellings(name, want):
    assert pg.surname(name) == want


def test_titles_match_after_normalisation_and_not_otherwise():
    a = "Accurate classification of BRCA1 variants with saturation genome editing."
    assert pg.title_match(a, "accurate classification of BRCA1 variants with saturation-genome editing")
    assert not pg.title_match(a, "Massively parallel functional analysis of BRCA1 RING domain variants")
    assert not pg.title_match("", a)


def test_a_title_cut_short_matches_only_as_a_long_prefix():
    full = ("High-throughput functional mapping of variants in an arrhythmia gene, KCNE1, "
            "reveals novel biology")
    assert pg.title_match(full, "High-throughput functional mapping of variants in an arrhythmia gene, ")
    assert pg.title_match("High-throughput discovery of trafficking-deficient variants in the K", 
                          "High-throughput discovery of trafficking-deficient variants in the KCNH2 channel")
    assert not pg.title_match(full, "High-throughput functional")              # too short to tell
    assert not pg.title_match(full, "High-throughput functional mapping of variants in an "
                                    "arrhythmia gene, SCN5A")


def test_versions_come_from_mavedb_links_not_from_urn_numbers():
    old, new = "urn:mavedb:00000097-0-1", "urn:mavedb:00000097-0-2"
    assert pg.current_versions([old, new], {old: new}) == [new]
    sorts = ["urn:mavedb:00000069-a-1", "urn:mavedb:00000069-a-2"]     # two sorts, no link
    assert pg.current_versions(sorts, {}) == sorts
    chain = {"x-1": "x-2", "x-2": "x-3"}
    assert pg.current_versions(["x-1", "x-3"], chain) == ["x-3"]       # middle not a candidate
    assert pg.current_versions(["x-1"], chain) == ["x-1"]              # successor not a candidate


def test_the_supersede_map_reads_both_sides_of_a_link():
    deps = pd.DataFrame([
        {"urn": "u-2", "superseded_by": None, "supersedes": "u-1"},
        {"urn": "v-1", "superseded_by": "v-2", "supersedes": None}])
    assert pg.supersede_map(deps) == {"u-1": "u-2", "v-1": "v-2"}


def test_dms_suffix():
    assert pg.dms_suffix("P53_HUMAN_Giacomelli_2018_WT_Nutlin") == "WT_Nutlin"
    assert pg.dms_suffix("BRCA1_HUMAN_Findlay_2018") == ""


# --------------------------------------------------------------------------
# pairing: candidates by protein and paper, then the data decide
# --------------------------------------------------------------------------
def _pg(**kw):
    row = {"DMS_id": "GENE1_HUMAN_Smith_2020", "UniProt_ID": "GENE1_HUMAN",
           "first_author": "Smith", "year": 2020, "title": "A deep scan of gene one",
           "raw_DMS_phenotype_name": "score", "selection_assay": "growth",
           "selection_type": "Growth"}
    row.update(kw)
    return row


def _dep(urn, genes=("GENE1",), acc=(), author="Smith, Ann", year=2020,
         title="A deep scan of gene one", st="GENE1 scores", desc="",
         superseded_by=None, supersedes=None, kind="primary", pubs=None):
    if pubs is None:
        pubs = [{"kind": kind, "first_author": author, "year": year, "title": title,
                 "doi": "", "id": "PubMed:1"}]
    return {"urn": urn, "title": st, "description": desc, "experiment": urn.rsplit("-", 1)[0],
            "accessions": list(acc), "genes": list(genes), "pubs": pubs,
            "superseded_by": superseded_by, "supersedes": supersedes}


UP = pd.DataFrame([{"accession": "P00001", "entry_name": "GENE1_HUMAN",
                    "gene_primary": "GENE1", "gene_synonym": "G1"}])


def _pair(exps, deps, up=UP, data=None):
    """Candidates and decision, with the data check stubbed: `data` maps a URN to
    (|rho|, coverage) or to the reason its data cannot be compared; by default every
    candidate reproduces ProteinGym's scores."""
    data = data or {}

    def check(dms, urn):
        v = data.get(urn, (1.0, 1.0))
        if isinstance(v, str):
            return {"check": v}
        return {"check": "checked", "abs_rho": v[0], "rho": v[0], "coverage": v[1],
                "shared_variants": 100, "pg_single_mutants": 100}
    exps = pd.DataFrame(exps)
    cands = pg.candidate_table(exps, pd.DataFrame(deps), up)
    pairs, checked = pg.pair(exps, cands, check)
    return pairs.set_index("DMS_id"), checked


def test_protein_paper_and_data_together_make_a_pair():
    res, checked = _pair([_pg()], [_dep("urn:mavedb:00000001-a-1", acc=["P00001"])])
    r = res.loc["GENE1_HUMAN_Smith_2020"]
    assert r["status"] == "paired" and r["urn"] == "urn:mavedb:00000001-a-1"
    assert "UniProt accession" in r["basis"] and "paper: title" in r["basis"]
    assert "|rho| 1.0000 on 100 of 100" in r["basis"]
    assert checked["reproduces"].tolist() == [True]


def test_gene_symbol_stands_in_when_the_deposit_carries_no_accession():
    res, _ = _pair([_pg()], [_dep("urn:mavedb:00000001-a-1")])
    assert res.iloc[0]["status"] == "paired" and "protein: gene" in res.iloc[0]["basis"]


@pytest.mark.parametrize("change", [
    {"author": "Jones, Bo", "title": "Something else entirely"},     # another paper
    {"year": 2017, "title": "Something else entirely"},              # same author, 3 years off
    {"genes": ("OTHER",)},                                           # another protein
])
def test_a_deposit_of_another_paper_or_protein_is_no_candidate(change):
    res, checked = _pair([_pg()], [_dep("urn:mavedb:00000001-a-1", **change)])
    assert res.iloc[0]["status"] == "no_match" and res.iloc[0]["urn"] == "" and checked.empty


def test_the_entry_name_mnemonic_is_used_only_when_uniprot_has_no_record():
    dep = [_dep("urn:mavedb:00000001-a-1", genes=("P53",))]
    known = pd.DataFrame([{"accession": "P04637", "entry_name": "P53_HUMAN",
                           "gene_primary": "TP53", "gene_synonym": None}])
    exp = [_pg(DMS_id="P53_HUMAN_Smith_2020", UniProt_ID="P53_HUMAN")]
    assert _pair(exp, dep, up=known)[0].iloc[0]["status"] == "no_match"   # P53 is not a gene
    assert _pair(exp, dep)[0].iloc[0]["status"] == "paired"                # not in UniProt


def test_author_and_year_within_one_suffice_without_a_title():
    res, _ = _pair([_pg()], [_dep("urn:mavedb:00000001-a-1", author="Smith A", year=2021,
                                  title="Preprint title differs")])
    assert res.iloc[0]["status"] == "paired" and "first author + year" in res.iloc[0]["basis"]


def test_a_secondary_publication_counts_and_is_named():
    pubs = [{"kind": "primary", "first_author": "Jones, Bo", "year": 2018,
             "title": "A method paper", "doi": "", "id": "PubMed:2"},
            {"kind": "secondary", "first_author": "Smith, Ann", "year": 2020,
             "title": "A deep scan of gene one", "doi": "", "id": "PubMed:1"}]
    res, _ = _pair([_pg()], [_dep("urn:mavedb:00000001-a-1", pubs=pubs)])
    assert res.iloc[0]["status"] == "paired"
    assert "paper: title (secondary publication)" in res.iloc[0]["basis"]


def test_a_deposit_with_no_publication_pairs_on_the_data_alone():
    dep = [_dep("urn:mavedb:00000001-a-1", pubs=[])]
    res, checked = _pair([_pg()], dep)
    assert res.iloc[0]["status"] == "paired" and "paper: none recorded" in res.iloc[0]["basis"]
    assert _pair([_pg()], dep, data={"urn:mavedb:00000001-a-1": (0.80, 1.0)})[0] \
        .iloc[0]["status"] == "not_reproduced"
    blind = _pair([_pg()], dep, data={"urn:mavedb:00000001-a-1": "no protein-level single changes"})
    assert blind[0].iloc[0]["status"] == "not_reproduced"       # nothing else ties it


def test_the_data_decide_between_two_sorts_of_one_experiment():
    # urn:mavedb:00000069-a-1 and -a-2 were two sorts; ProteinGym's scores are a-1's
    deps = [_dep("urn:mavedb:00000001-a-1"), _dep("urn:mavedb:00000001-a-2")]
    res, _ = _pair([_pg()], deps, data={"urn:mavedb:00000001-a-2": (0.778, 1.0)})
    assert res.iloc[0]["status"] == "paired" and res.iloc[0]["urn"] == "urn:mavedb:00000001-a-1"
    both, _ = _pair([_pg()], deps)                           # both reproduce, no words
    assert both.iloc[0]["status"] == "ambiguous" and both.iloc[0]["urn"] == ""
    assert both.iloc[0]["candidates"] == "urn:mavedb:00000001-a-1;urn:mavedb:00000001-a-2"


def test_phenotype_words_decide_only_between_candidates_that_reproduce():
    deps = [_dep("urn:mavedb:00000001-a-1", st="GENE1 abundance"),
            _dep("urn:mavedb:00000001-b-1", st="GENE1 activity")]
    exp = [_pg(DMS_id="GENE1_HUMAN_Smith_2020_activity")]
    ok, _ = _pair(exp, deps)
    assert ok.iloc[0]["status"] == "paired" and ok.iloc[0]["urn"] == "urn:mavedb:00000001-b-1"
    assert "phenotype words (1 shared, next 0)" in ok.iloc[0]["basis"]
    data, _ = _pair(exp, deps, data={"urn:mavedb:00000001-b-1": (0.5, 1.0)})
    assert data.iloc[0]["urn"] == "urn:mavedb:00000001-a-1"     # the words do not overrule


def test_a_candidate_must_cover_most_of_proteingyms_mutants():
    dep = [_dep("urn:mavedb:00000001-a-1")]
    half, _ = _pair([_pg()], dep, data={"urn:mavedb:00000001-a-1": (1.0, 0.5)})
    assert half.iloc[0]["status"] == "not_reproduced"
    assert "coverage 0.50" in half.iloc[0]["basis"]


def test_reproduces_at_the_thresholds():
    at = {"check": "checked", "abs_rho": pg.DATA_MATCH, "coverage": pg.DATA_COVERAGE}
    assert pg.reproduces(at)
    assert not pg.reproduces({**at, "abs_rho": pg.DATA_MATCH - 1e-9})
    assert not pg.reproduces({**at, "coverage": pg.DATA_COVERAGE - 1e-9})
    assert not pg.reproduces({**at, "abs_rho": float("nan")})
    assert not pg.reproduces({"check": "no protein-level single changes"})


def test_an_uncheckable_candidate_with_a_matching_paper_is_paired_and_flagged():
    blind = "no protein-level single changes"
    one, _ = _pair([_pg()], [_dep("urn:mavedb:00000001-a-1")],
                   data={"urn:mavedb:00000001-a-1": blind})
    assert one.iloc[0]["status"] == "paired" and f"data not checked ({blind})" in one.iloc[0]["basis"]
    two, _ = _pair([_pg()], [_dep("urn:mavedb:00000001-a-1"), _dep("urn:mavedb:00000001-b-1")],
                   data={"urn:mavedb:00000001-a-1": blind, "urn:mavedb:00000001-b-1": blind})
    assert two.iloc[0]["status"] == "ambiguous"
    # a checked candidate that reproduces wins over an uncheckable one
    mixed, _ = _pair([_pg()], [_dep("urn:mavedb:00000001-a-1"), _dep("urn:mavedb:00000001-b-1")],
                     data={"urn:mavedb:00000001-a-1": blind})
    assert mixed.iloc[0]["urn"] == "urn:mavedb:00000001-b-1"


def test_when_no_candidate_reproduces_the_closest_is_reported():
    res, _ = _pair([_pg()], [_dep("urn:mavedb:00000001-a-1"), _dep("urn:mavedb:00000001-b-1")],
                   data={"urn:mavedb:00000001-a-1": (0.68, 1.0), "urn:mavedb:00000001-b-1": (0.81, 1.0)})
    assert res.iloc[0]["status"] == "not_reproduced" and res.iloc[0]["urn"] == ""
    assert res.iloc[0]["basis"].startswith("closest urn:mavedb:00000001-b-1: |rho| 0.8100")


def test_an_older_version_gives_way_to_the_newer_one():
    deps = [_dep("urn:mavedb:00000001-a-1"),
            _dep("urn:mavedb:00000001-a-2", supersedes="urn:mavedb:00000001-a-1")]
    res, checked = _pair([_pg()], deps)
    assert res.iloc[0]["status"] == "paired" and res.iloc[0]["urn"] == "urn:mavedb:00000001-a-2"
    assert checked["urn"].tolist() == ["urn:mavedb:00000001-a-2"]


def test_a_deposit_claimed_by_two_experiments_is_paired_with_neither():
    exps = [_pg(), _pg(DMS_id="GENE1_HUMAN_Smith_2020_rep")]
    res, _ = _pair(exps, [_dep("urn:mavedb:00000001-a-1")])
    assert set(res["status"]) == {"deposit_claimed_twice"} and set(res["urn"]) == {""}


# --------------------------------------------------------------------------
# the data check
# --------------------------------------------------------------------------
def test_protein_change_reads_single_missense_only():
    assert pg.protein_change("p.Ala23Pro") == ("A", 23, "P")
    for bad in ("p.Ala23=", "p.Ala23Ter", "p.[Ala23Pro;Gly24Cys]", "p.Ala23del", "_wt", "NA"):
        assert pg.protein_change(bad) is None


def test_the_data_check_finds_the_numbering_shift_and_a_perfect_match():
    seq = "MKTAYIAKQR"
    rng = np.random.default_rng(0)
    pg_rows, dep_rows = [], []
    for i, wt in enumerate(seq, 1):
        for mt in "ACDE":
            if mt == wt:
                continue
            s = float(rng.normal())
            pg_rows.append({"mutant": f"{wt}{i}{mt}", "DMS_score": s})
            dep_rows.append({"hgvs_pro": f"p.{_three(wt)}{i + 5}{_three(mt)}", "score": -2 * s})
    pg_rows.append({"mutant": "M1A:K2C", "DMS_score": 9.0})        # a double: ignored
    dep = pg.deposit_single_mutants(pd.DataFrame(dep_rows))
    res = pg.data_check(pd.DataFrame(pg_rows), dep, seq, offset_candidates=(0, 5, -5))
    assert res["offset"] == -5 and res["wild_type_agreement"] == 1.0
    assert res["abs_rho"] == pytest.approx(1.0) and res["rho"] < 0
    assert res["shared_variants"] == len(dep)


def test_the_data_check_scans_every_shift_when_the_given_ones_fail():
    seq = "MKTAYIAKQRQISFVKSHFSRQ"
    pg_rows = [{"mutant": f"{wt}{i}A", "DMS_score": float(i)} for i, wt in enumerate(seq, 1) if wt != "A"]
    dep = pg.deposit_single_mutants(pd.DataFrame(
        [{"hgvs_pro": f"p.{_three(wt)}{i + 37}Ala", "score": float(i)}
         for i, wt in enumerate(seq, 1) if wt != "A"]))
    res = pg.data_check(pd.DataFrame(pg_rows), dep, seq, offset_candidates=(0,))
    assert res["offset"] == -37 and res["wild_type_agreement"] == 1.0
    assert res["abs_rho"] == pytest.approx(1.0)


def _three(aa):
    return {v: k for k, v in pg.AA3.items()}[aa]


def test_codon_level_rows_are_averaged_per_protein_change():
    d = pg.deposit_single_mutants(pd.DataFrame(
        [{"hgvs_pro": "p.Ala2Pro", "score": 1.0}, {"hgvs_pro": "p.Ala2Pro", "score": 3.0},
         {"hgvs_pro": "p.Ala2=", "score": 0.0}]))
    assert len(d) == 1 and d["score"].iloc[0] == 2.0


# --------------------------------------------------------------------------
# tiers and the interval check
# --------------------------------------------------------------------------
def test_tiers():
    assert pg.tier_of("replicates") == pg.tier_of("ci") == "strict"
    assert pg.tier_of("se", "validated") == "strict"
    for check in ("no_interval", "ci_mismatch", ""):
        assert pg.tier_of("se", check) == "loose"
    assert pg.tier_of("sd_as_se") == pg.tier_of("variance") == "loose"
    with pytest.raises(ValueError):
        pg.tier_of("something")


def test_check_deposit_se():
    rng = np.random.default_rng(3)
    s, se = rng.normal(size=100), rng.uniform(0.1, 0.3, 100)
    good = pd.DataFrame({"score": s, "se": se, "lower_ci": s - 1.96 * se, "upper_ci": s + 1.96 * se})
    assert pg.check_deposit_se(good, "se") == "validated"
    assert pg.check_deposit_se(good.assign(upper_ci=s + 1.645 * se), "se") == "ci_mismatch"
    assert pg.check_deposit_se(good[["score", "se"]], "se") == "no_interval"


# --------------------------------------------------------------------------
# realisation and the three analyses
# --------------------------------------------------------------------------
def test_realisation_divides_only_where_the_ceiling_clears_the_bound():
    rho = pd.DataFrame({"m": [0.4, 0.4, 0.4]}, index=["a", "b", "c"])
    ceil = pd.Series([0.8, 0.45, 0.44], index=["a", "b", "c"])
    r = pg.realisation(rho, ceil)
    assert list(r.index) == ["a", "b"]
    assert r.loc["a", "m"] == pytest.approx(0.5) and r.loc["b", "m"] == pytest.approx(0.4 / 0.45)


def _toy():
    exps = [f"e{i}" for i in range(12)]
    ceil = pd.Series(np.linspace(0.5, 0.95, 12), index=exps)
    ceil.iloc[0] = 0.30                                   # below the bound
    models = {f"m{k}": ceil * (0.9 - 0.05 * k) for k in range(12)}
    models["m11"] = pd.Series(np.linspace(0.6, 0.1, 12), index=exps)   # anti-correlated
    return pd.DataFrame(models), ceil


def test_analyse_ranks_on_the_usable_experiments_and_reports_the_rest():
    rho, ceil = _toy()
    out = pg.analyse(rho, ceil)
    one, two, three = out["1_ceiling_vs_rho"], out["2_realisation"], out["3_rankings"]
    assert one["best_model"] == "m0" and one["top_models"][:3] == ["m0", "m1", "m2"]
    assert one["n_experiments_with_ceiling"] == 12 and one["n_experiments_ge_bound"] == 11
    assert one["ceiling_vs_best_model"]["all"]["rho"] == pytest.approx(1.0)
    assert out["below_bound"] == [{"DMS_id": "e0", "ceiling": 0.30}]
    # m0 realises 0.9 of the ceiling everywhere it is divided
    assert two["m0"]["median"] == pytest.approx(0.9) and two["m0"]["share_ge_0.9"] == 1.0
    assert two["m0"]["n"] == 11 and two["m3"]["share_ge_0.8"] == 0.0
    assert three["top_by_mean_rho"][0]["model"] == "m0"
    assert -1 <= three["kendall_tau"] <= 1 and three["n_models"] == 12


def test_kendall_tau_is_one_when_realisation_preserves_the_order():
    rho, ceil = _toy()
    rho = rho.drop(columns="m11")
    assert pg.analyse(rho, ceil)["3_rankings"]["kendall_tau"] == pytest.approx(1.0)


def test_realisation_can_reorder_models_that_mean_rho_ranks_otherwise():
    # A scores 0.6 on two easy experiments; B scores 0.45 on two hard ones and 0.5
    # on the easy ones. Mean rho prefers A; relative to the ceilings, B is closer.
    exps = ["easy1", "easy2", "hard1", "hard2"]
    ceil = pd.Series([0.95, 0.95, 0.5, 0.5], index=exps)
    rho = pd.DataFrame({"A": [0.9, 0.9, 0.1, 0.1], "B": [0.5, 0.5, 0.45, 0.45]}, index=exps)
    three = pg.analyse(rho, ceil)["3_rankings"]
    assert three["top_by_mean_rho"][0]["model"] == "A"
    assert three["top_by_mean_realisation"][0]["model"] == "B"
    assert three["kendall_tau"] == pytest.approx(-1.0)


def test_a_model_missing_rho_on_a_usable_experiment_is_left_out_of_the_rankings():
    rho, ceil = _toy()
    rho.loc["e5", "m1"] = np.nan
    out = pg.analyse(rho, ceil)
    assert out["models_dropped_for_missing_rho"] == ["m1"]
    assert "m1" not in out["1_ceiling_vs_rho"]["top_models"]
    assert out["3_rankings"]["n_models"] == 11
    rho.loc["e5", "m1"] = 0.5                       # e0 is below the bound: no effect
    rho.loc["e0", "m1"] = np.nan
    assert pg.analyse(rho, ceil)["models_dropped_for_missing_rho"] == []


def test_model_columns_refuse_a_non_numeric_column():
    sp = pd.DataFrame({"DMS ID": ["x"], "Selection Type": ["Growth"], "m": [0.3]}).set_index("DMS ID")
    assert pg.model_columns(sp) == ["m"]
    with pytest.raises(ValueError):
        pg.model_columns(sp.assign(note=["text"]))
    with pytest.raises(ValueError):                      # a count is not a correlation
        pg.model_columns(sp.assign(n_mutants=[9576]))
    meta = sp.assign(**{"Number of Mutants": [9576]})
    assert pg.model_columns(meta) == ["m"]


# --------------------------------------------------------------------------
# fetching: a rerun resumes, an outage stops, part of the records is refused
# --------------------------------------------------------------------------
class _Resp:
    content = b"{}"


def _offline(monkeypatch, tmp_path, bad=()):
    """Point the module at tmp_path and replace the network with a stub."""
    calls = []

    def get(url, **kw):
        calls.append(url)
        if any(url.endswith(b) for b in bad):
            raise requests.HTTPError(f"504 for {url}")
        return _Resp()
    monkeypatch.setattr(pg, "INPUTS", tmp_path)
    monkeypatch.setattr(pg, "PAUSE", 0)
    monkeypatch.setattr(pg, "_get", get)
    return calls


def test_a_rerun_fetches_only_what_is_missing_and_reports_failures(monkeypatch, tmp_path):
    calls = _offline(monkeypatch, tmp_path, bad=("00000003-a-1",))
    (tmp_path / "mavedb").mkdir()
    (tmp_path / "mavedb" / "urn_mavedb_00000001-a-1.json").write_text("{}")
    urns = ["urn:mavedb:00000001-a-1", "urn:mavedb:00000002-a-1", "urn:mavedb:00000003-a-1"]
    failed = pg.fetch_mavedb_records(urns)
    assert [c.rsplit("/", 1)[1] for c in calls] == urns[1:]          # the stored one skipped
    assert len(failed) == 1 and failed[0].startswith("urn:mavedb:00000003-a-1")
    assert (tmp_path / "mavedb" / "urn_mavedb_00000002-a-1.json").exists()
    assert "mavedb/urn_mavedb_00000002-a-1.json" in pg.load_manifest(
        tmp_path / "MANIFEST.json")["files"]
    with pytest.raises(RuntimeError, match="rerun with --fetch"):
        pg._all_or_raise(failed, "record")


def test_an_outage_stops_the_fetch_instead_of_walking_the_list(monkeypatch, tmp_path):
    urns = [f"urn:mavedb:{i:08d}-a-1" for i in range(1, 21)]
    calls = _offline(monkeypatch, tmp_path, bad=tuple(u.split(":")[-1] for u in urns))
    with pytest.raises(RuntimeError, match="in a row"):
        pg.fetch_mavedb_records(urns)
    assert len(calls) == pg.GIVE_UP_AFTER


def test_the_analysis_refuses_to_pair_on_part_of_the_records(monkeypatch, tmp_path):
    ceilings = tmp_path / "ceilings.tsv"
    pd.DataFrame({"urn": ["urn:mavedb:00000001-a-1", "urn:mavedb:00000002-a-1",
                          "urn:mavedb:00000009-a-1"],
                  "stratum": ["human", "other_natural", "designed_stability"]}).to_csv(
        ceilings, sep="\t", index=False)
    _offline(monkeypatch, tmp_path)
    monkeypatch.setattr(pg, "CEILINGS", ceilings)
    (tmp_path / "mavedb").mkdir()
    (tmp_path / "mavedb" / "urn_mavedb_00000001-a-1.json").write_text("{}")
    assert pg.missing_records() == ["urn:mavedb:00000002-a-1"]      # designed: not a candidate
    with pytest.raises(FileNotFoundError, match="1 MaveDB records missing"):
        pg.run()


# --------------------------------------------------------------------------
# the data check as `pair` calls it, the accounting, and the guards of run()
# --------------------------------------------------------------------------
def test_the_data_check_reads_proteingym_and_the_deposit_once_each(tmp_path):
    import zipfile
    seq = "MKTAYIAKQR"
    pg_rows = [{"mutant": f"{wt}{i}{mt}", "DMS_score": float(i * 10 + k)}
               for i, wt in enumerate(seq, 1) for k, mt in enumerate("CDE") if mt != wt]
    archive = tmp_path / "dms.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("DMS_ProteinGym_substitutions/X_HUMAN_Smith_2020.csv",
                   pd.DataFrame(pg_rows).to_csv(index=False))
    ref = pd.DataFrame([{"DMS_id": "X_HUMAN_Smith_2020", "DMS_filename": "X_HUMAN_Smith_2020.csv",
                         "raw_mut_offset": np.nan, "target_seq": seq}])
    tables = {"same": pd.DataFrame({"hgvs_pro": [f"p.{_three(r['mutant'][0])}{r['mutant'][1:-1]}"
                                                 f"{_three(r['mutant'][-1])}" for r in pg_rows],
                                    "score": [-r["DMS_score"] for r in pg_rows]}),
              "nucleotide only": pd.DataFrame({"hgvs_nt": ["c.1A>G"], "score": [0.1]})}
    reads = []

    def scores(urn):
        reads.append(urn)
        if urn not in tables:
            raise FileNotFoundError(urn)
        return tables[urn]
    check = pg.DataCheck(ref, archive=archive, scores=scores)
    res = check("X_HUMAN_Smith_2020", "same")
    assert res["check"] == "checked" and res["abs_rho"] == pytest.approx(1.0)
    assert res["coverage"] == 1.0 and pg.reproduces(res)
    check("X_HUMAN_Smith_2020", "same")
    assert reads == ["same"]                                       # read once
    assert check("X_HUMAN_Smith_2020", "nucleotide only")["check"] == "no protein-level single changes"
    assert check("X_HUMAN_Smith_2020", "absent")["check"] == "scores not fetched"


def test_accounting_sorts_the_experiments_and_bounds_the_tiers():
    ref = pd.DataFrame([
        {"DMS_id": "A_HUMAN_Tsuboyama_2023_1ABC", "UniProt_ID": "A_HUMAN",
         "first_author": "Tsuboyama", "year": 2023},
        {"DMS_id": "GENE1_HUMAN_Smith_2020", "UniProt_ID": "GENE1_HUMAN",
         "first_author": "Smith", "year": 2020},
        {"DMS_id": "GENE2_HUMAN_Lee_2021", "UniProt_ID": "GENE2_HUMAN",
         "first_author": "Lee", "year": 2021},
        {"DMS_id": "GENE3_HUMAN_Kim_2022", "UniProt_ID": "GENE3_HUMAN",
         "first_author": "Kim", "year": 2022}])
    pairs = pd.DataFrame({"DMS_id": ref["DMS_id"],
                          "status": ["no_match", "paired", "no_match", "no_match"]})
    up = pd.DataFrame([{"accession": f"P0000{i}", "entry_name": f"GENE{i}_HUMAN",
                        "gene_primary": f"GENE{i}", "gene_synonym": None} for i in (1, 2, 3)])
    ceilings = pd.DataFrame([
        {"urn": "u1", "status": "ok", "stratum": "human", "method": "se", "error_column": "se"},
        {"urn": "u2", "status": "degenerate", "stratum": "human", "method": "se",
         "error_column": "se"}])
    survey = pd.DataFrame([{"urn": "u1", "all_columns": "score;se;ci_lower;ci_upper"},
                           {"urn": "u2", "all_columns": "score;se"}])
    metadata = pd.DataFrame([{"urn": "u1", "target_genes": "GENE1"},
                             {"urn": "u2", "target_genes": "GENE2"}])
    a = pg.accounting(ref, pairs, up, ceilings, survey, metadata)
    assert a["stability_platform"] == {"n": 1, "by_status": {"no_match": 1}}
    assert a["outside_platform"]["n"] == 3
    assert a["outside_platform"]["no_match_by_what_mavedb_holds"] == {
        "no deposit under the gene's names": 1,
        "deposits of the gene, none with an ok ceiling": 1,
        "an ok-ceiling deposit of the gene, from other papers only": 0}
    assert a["bound_by_gene"] == {"loose_tier": 1, "strict_tier": 1,
                                  "strict_tier_experiments": ["GENE1_HUMAN_Smith_2020"]}
    assert a["strict_capable_deposits"] == {"n": 1, "by_method": {"se": 1}}
    no_ci = survey.assign(all_columns=["score;se", "score;se"])     # nothing to check se by
    assert pg.accounting(ref, pairs, up, ceilings, no_ci, metadata)["bound_by_gene"]["strict_tier"] == 0


def test_the_analysis_refuses_to_decide_with_score_tables_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(pg, "INPUTS", tmp_path)
    monkeypatch.setattr(pg, "missing_records", lambda: [])
    monkeypatch.setattr(pg, "load_reference", lambda: pd.DataFrame([_pg()]))
    monkeypatch.setattr(pg, "load_uniprot", lambda: UP)
    monkeypatch.setattr(pg, "load_deposits", lambda: pd.DataFrame([_dep("urn:mavedb:00000001-a-1")]))
    with pytest.raises(FileNotFoundError, match="1 score tables missing"):
        pg.run()


def test_a_tier_without_a_usable_experiment_is_reported_not_ranked():
    rho = pd.DataFrame({"m": [0.4]}, index=["e"])
    out = pg.analyse(rho, pd.Series([0.30], index=["e"]))
    assert "note" in out and out["n_experiments_with_ceiling"] == 1
    assert pg.analyse(rho.iloc[:0], pd.Series(dtype=float))["n_experiments_with_ceiling"] == 0

"""Guards for the MaveDB corrections of the current generation: the human flag judged on
each deposit's record, the designed-stability platform by experiment set, the survey and
metadata v2, the ceilings table v2 (the annotation step that reproduces v1, the error
column of the deposits with several), and the tables downstream. Pure functions on
synthetic records; the delivered files skip where results/ is not built."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas import mavedb_ceilings as mc  # noqa: E402
from atlas import mavedb_records as mr  # noqa: E402
from atlas import paths  # noqa: E402

RESULTS = REPO / "results"


def _built(*keys_or_names):
    for k in keys_or_names:
        p = paths.path(k) if k in paths.NAMES else RESULTS / k
        if not p.exists():
            pytest.skip(f"results/ not built in this checkout ({p.name})")


def _seq(code):
    return {"targetSequence": {"taxonomy": {"code": code}}}


def _acc(acc, assembly=None):
    return {"targetAccession": {"accession": acc, "assembly": assembly}}


# --------------------------------------------------------------------------
# the human flag
# --------------------------------------------------------------------------
def test_each_target_is_judged_on_its_record(tmp_path):
    assert mr.target_human(_seq(9606), tmp_path) == (True, "taxonomy")
    assert mr.target_human(_seq(1773), tmp_path) == (False, "taxonomy")
    assert mr.target_human(_acc("NC_000017.11", "GRCh38"), tmp_path) == (True, "assembly")
    assert mr.target_human(_acc("NC_000067.7", "GRCm39"), tmp_path) == (False, "assembly")
    assert mr.target_human(_acc("ENST00000460680.5"), tmp_path) == (True, "accession")
    assert mr.target_human(_acc("ENSMUST00000000001.5"), tmp_path) == (False, "accession")
    assert mr.target_human(_acc("NM_007294.4"), tmp_path) == (None, "undetermined")
    (tmp_path / "ncbi").mkdir()
    (tmp_path / "ncbi" / "NM_007294.4.json").write_text(json.dumps(
        {"result": {"uids": ["1"], "1": {"taxid": 9606}}}))
    assert mr.target_human(_acc("NM_007294.4"), tmp_path) == (True, "accession")
    assert mr.target_human({"name": "x"}, tmp_path) == (None, "undetermined")


def test_one_human_target_makes_a_human_deposit(tmp_path):
    rec = {"targetGenes": [_seq(562), _seq(9606)]}
    assert mr.deposit_human(rec, False, tmp_path) == (True, "taxonomy")
    assert mr.deposit_human({"targetGenes": [_seq(562)]}, True, tmp_path) == (False, "taxonomy")
    mixed = {"targetGenes": [_seq(562), _acc("XYZ123")]}
    assert mr.deposit_human(mixed, True, tmp_path) == (False, "undetermined")
    assert mr.deposit_human(None, True, tmp_path) == (True, "record_unavailable")
    cas9 = {"targetGenes": [{"name": "SpCas9 (Human Codon Optimized)", **_seq(1314)}]}
    assert mr.deposit_human(cas9, True, tmp_path) == (False, "taxonomy")


# --------------------------------------------------------------------------
# the platform
# --------------------------------------------------------------------------
def test_a_seed_marks_its_whole_experiment_set():
    survey = pd.DataFrame({"urn": ["urn:mavedb:00000218-0-1", "urn:mavedb:00000218-a-1",
                                   "urn:mavedb:00000218-a-2", "urn:mavedb:00000702-a-1",
                                   "urn:mavedb:00000005-a-1"]})
    titles = pd.Series(["X combined scores", "X trypsin digestion", "X chymotrypsin digestion",
                        "villin domain", "pin1 something"])
    columns = pd.Series(["score;dG", "", "", "score;se", "score;score_95CI_low"])
    t = mr.platform_table(survey, titles, columns).set_index("urn")
    assert t.loc["urn:mavedb:00000218-0-1", "platform"] == "designed_stability"
    assert t.loc["urn:mavedb:00000218-0-1", "platform_basis"] == "experiment set"
    assert t.loc["urn:mavedb:00000218-a-1", "platform_basis"] == "seed: title"
    assert t.loc["urn:mavedb:00000702-a-1", "platform"] == "other"      # a name is not a seed
    assert t.loc["urn:mavedb:00000005-a-1", "platform_basis"] == "seed: score_95CI_low"
    assert mr.experiment_set("urn:mavedb:00001262-a-3") == "00001262"


# --------------------------------------------------------------------------
# the ceilings table: the error column of the deposits with several
# --------------------------------------------------------------------------
def _sge(kind: str, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    lfc = rng.normal(size=n)
    se_a, se_b = rng.uniform(0.1, 0.3, n), rng.uniform(0.2, 0.5, n)
    if kind == "z":                 # BAP1's form: score is the LFC, z = LFC / SE_b
        return pd.DataFrame({"score": lfc, "SE_a": se_a, "SE_b": se_b,
                             "z_b": np.round(lfc / se_b, 6)})
    if kind == "stat":              # RAD51C's form: score = adj_lfc / SE_b
        return pd.DataFrame({"score": lfc / se_b, "adj_lfc": lfc, "SE_a": se_a, "SE_b": se_b})
    return pd.DataFrame({"score": lfc, "SE": se_a, "SE_rep1": se_b})


def test_the_error_column_is_chosen_by_the_relations():
    z = mc.choose_error_column(_sge("z"), ["SE_a", "SE_b"])
    assert (z["choice"], z["se"], z["via"]) == ("z_column", "SE_b", "z_b")
    t = mc.choose_error_column(_sge("stat"), ["SE_a", "SE_b"])
    assert (t["choice"], t["se"], t["via"]) == ("test_statistic", "SE_b", "adj_lfc")
    f = mc.choose_error_column(_sge("plain"), ["SE", "SE_rep1"])
    assert (f["choice"], f["se"]) == ("first_of_several", "SE")


def test_a_test_statistic_gives_no_ceiling():
    row, ch = mc.from_table_v2(_sge("stat"), "urn:x", "errors", "", ["SE_a", "SE_b"])
    assert row["status"] == "no_score_scale_error" and row["ceiling"] is None
    row, ch = mc.from_table_v2(_sge("z"), "urn:x", "errors", "", ["SE_a", "SE_b"])
    assert row["error_column"] == "SE_b" and row["method"] == "se"


def test_the_stratum_follows_the_platform_then_the_human_flag():
    assert mc.stratum_of("designed_stability", True) == "designed_stability"
    assert mc.stratum_of("other", True) == "human"
    assert mc.stratum_of("other", False) == "other_natural"


def test_the_annotation_step_reproduces_v1s_five_columns():
    _built("mavedb_ceilings_v1.tsv", "mavedb_survey_v1.tsv", "mavedb_metadata_v1.tsv")
    v1 = pd.read_csv(RESULTS / "mavedb_ceilings_v1.tsv", sep="\t")
    again = mc.annotate(v1[mc.COMPUTED], pd.read_csv(RESULTS / "mavedb_survey_v1.tsv", sep="\t"),
                        pd.read_csv(RESULTS / "mavedb_metadata_v1.tsv", sep="\t"))
    for c in mc.ANNOTATED:
        assert all(mc._same(a, b) for a, b in zip(again[c], v1[c])), c


# --------------------------------------------------------------------------
# the delivered v2 files
# --------------------------------------------------------------------------
def test_the_survey_v2_changes_only_the_human_flag():
    _built("mavedb_survey", "mavedb_survey_v1.tsv", "mavedb_metadata", "mavedb_metadata_v1.tsv")
    for v2k, v1n in (("mavedb_survey", "mavedb_survey_v1.tsv"),
                     ("mavedb_metadata", "mavedb_metadata_v1.tsv")):
        v1 = pd.read_csv(RESULTS / v1n, sep="\t", keep_default_na=False, dtype=str)
        v2 = pd.read_csv(paths.path(v2k), sep="\t", keep_default_na=False, dtype=str)
        assert list(v2.columns) == list(v1.columns) + ["human_basis", "platform", "platform_basis"]
        for c in v1.columns:
            if c != "human":
                assert v1[c].equals(v2[c]), (v2k, c)
        assert set(v2["human_basis"]) <= set(mr.BASES)


def test_the_platform_is_541_sets_of_three():
    _built("mavedb_survey")
    sv = pd.read_csv(paths.path("mavedb_survey"), sep="\t")
    plat = sv[sv["platform"] == "designed_stability"]
    sizes = plat["urn"].map(mr.experiment_set).value_counts()
    assert len(sizes) == 541 and set(sizes) == {3} and len(plat) == 1623
    for u in ("urn:mavedb:00000218-0-1", "urn:mavedb:00000462-0-1", "urn:mavedb:00000361-0-1"):
        assert sv.set_index("urn").loc[u, "platform"] == "designed_stability"
    for u in ("urn:mavedb:00000702-a-1", "urn:mavedb:00000960-a-1", "urn:mavedb:00001136-a-1",
              "urn:mavedb:00001137-a-1", "urn:mavedb:00001138-a-1"):
        assert sv.set_index("urn").loc[u, "platform"] == "other"


def test_the_ceilings_v2_carry_v1_but_where_revised():
    _built("mavedb_ceilings", "mavedb_ceilings_v1.tsv")
    v1 = pd.read_csv(RESULTS / "mavedb_ceilings_v1.tsv", sep="\t")
    v2 = pd.read_csv(paths.path("mavedb_ceilings"), sep="\t")
    assert list(v2.columns) == mc.COLUMNS_V2 and v2["urn"].equals(v1["urn"])
    rev = v2["revised"].fillna("")
    same = ~rev.str.contains("error_column")
    for c in mc.COMPUTED:
        assert all(mc._same(a, b) for a, b in zip(v1.loc[same, c], v2.loc[same, c])), c
    multi = v2[v2["error_column_choice"].notna()]
    assert len(multi) == 43
    bap1 = multi[multi["urn"].str.startswith("urn:mavedb:00000662-")]
    rad51c = multi[multi["urn"].str.startswith("urn:mavedb:00000673-")]
    assert len(bap1) == 18 and set(bap1["error_column"]) == {"SE_bind_continuous"}
    assert len(rad51c) == 10 and set(rad51c["status"]) == {"no_score_scale_error"}
    assert rad51c["ceiling"].isna().all()
    assert (v2["stratum"] == [mc.stratum_of(p, h) for p, h in zip(v2["platform"], v2["human"])]).all()


def test_the_choices_of_the_43_and_the_build_record():
    _built("mavedb_ceilings")
    v2 = pd.read_csv(paths.path("mavedb_ceilings"), sep="\t")
    assert v2["error_column_choice"].value_counts().to_dict() == {
        "z_column": 18, "first_of_several": 14, "test_statistic": 11}
    spop = v2.set_index("urn").loc["urn:mavedb:00001258-a-1"]
    assert spop["status"] == "no_score_scale_error" and "logFC / logFC_SE" in spop["note"]


def test_the_reaudit_v2_changes_only_the_human_flag():
    _built("mavedb_survey_reaudit", "mavedb_survey_reaudit_v1.tsv", "mavedb_survey")
    v1 = pd.read_csv(RESULTS / "mavedb_survey_reaudit_v1.tsv", sep="\t", keep_default_na=False, dtype=str)
    v2 = pd.read_csv(paths.path("mavedb_survey_reaudit"), sep="\t", keep_default_na=False, dtype=str)
    assert list(v1.columns) == list(v2.columns)
    for c in v1.columns:
        if c != "human":
            assert v1[c].equals(v2[c]), c
    sv = pd.read_csv(paths.path("mavedb_survey"), sep="\t").set_index("urn")["human"]
    assert (v2["human"] == v2["urn"].map(sv).astype(str)).all()


def test_the_accounting_reads_the_v2_tables():
    _built("mavedb_accounting", "mavedb_survey", "mavedb_ceilings")
    ac = pd.read_csv(paths.path("mavedb_accounting"), sep="\t").set_index("metric")
    sv = pd.read_csv(paths.path("mavedb_survey"), sep="\t")
    ce = pd.read_csv(paths.path("mavedb_ceilings"), sep="\t")
    assert ac.loc["human score sets", "count"] == int(sv["human"].sum())
    assert ac.loc["paper counting class: human ceiling candidates", "count"] == int((ce["stratum"] == "human").sum())

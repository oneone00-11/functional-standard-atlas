"""atlas.clinvar_status: the release's ClinVar rules, ported as written; an exact
chrom/pos/ref/alt match; the two arms; one row per variant; the release check; the copy
that stops on a wrong hash. Offline: a small VCF written here."""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas import clinvar_status as cs  # noqa: E402

VCF_LINES = [
    "##fileformat=VCFv4.1",
    "##fileDate=2026-06-15",
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    "17\t100\t11\tA\tG\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=reviewed_by_expert_panel",
    "17\t100\t12\tA\tT\t.\t.\tCLNREVSTAT=no_classification_for_the_single_variant",
    "17\t200\t13\tC\tCA\t.\t.\tCLNSIG=Conflicting_classifications_of_pathogenicity",
    "17\t300\t14\tG\tA\t.\t.\tCLNSIG=Likely_benign",
    "13\t300\t15\tG\tA\t.\t.\tCLNSIG=Uncertain_significance",
]


def write_vcf(path: Path, lines=VCF_LINES) -> Path:
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")
    return path


def matrix() -> pd.DataFrame:
    return pd.DataFrame({
        "variant_id": ["v1", "v2", "v3", "v4", "v5"],
        "gene": ["G", "G", "G", "G", "G"],
        "chrom": ["17", "17", "17", "17", "17"],
        "pos": [100, 100, 200, 300, 201],
        "ref": ["A", "A", "C", "G", "CA"],
        "alt": ["G", "T", "CA", "C", "C"],
        "hgvs_c": ["c.1+1A>G", "c.1+1A>T", "c.1+20dup", "c.30G>C", "c.2+5del"]})


def test_clnsig_collapses_to_five_classes():
    f = cs.classify_clnsig
    assert f("") == "Other"
    assert f("Pathogenic/Likely_pathogenic") == "P/LP"
    assert f("Benign/Likely_benign") == "B/LB"
    assert f("Conflicting_classifications_of_pathogenicity") == "Conflicting"
    assert f("Uncertain_significance") == "VUS"
    assert f("Pathogenic|risk_factor") == "P/LP"
    assert f("not_provided") == "Other"


def test_arms_and_the_strict_arm():
    assert cs.clinvar_arm(float("nan")) == "unrecorded"
    assert cs.clinvar_arm("P/LP") == cs.clinvar_arm("B/LB") == "classified"
    assert cs.clinvar_arm("VUS") == cs.clinvar_arm("Other") == "recorded_unclassified"
    assert cs.clinvar_arm_strict("Other", "") == "recorded_no_assertion"
    assert cs.clinvar_arm_strict("Other", "not_provided") == "recorded_unclassified"
    assert cs.clinvar_arm_strict("B/LB", "") == "classified"
    assert cs.clinvar_arm_strict(float("nan"), float("nan")) == "unrecorded"


def test_a_record_counts_only_at_the_exact_chrom_pos_ref_alt(tmp_path):
    vcf = write_vcf(tmp_path / "c.vcf.gz")
    t, file_date = cs.annotate(matrix(), vcf)
    assert file_date == "2026-06-15"
    assert t["variant_id"].tolist() == ["v1", "v2", "v3", "v4", "v5"]   # the matrix's order
    got = dict(zip(t["variant_id"], zip(t["arm"], t["arm_strict"])))
    assert got["v1"] == ("classified", "classified")
    assert got["v2"] == ("recorded_unclassified", "recorded_no_assertion")   # empty CLNSIG
    assert got["v3"] == ("recorded_unclassified", "recorded_unclassified")   # an insertion
    assert got["v4"] == ("unrecorded", "unrecorded")       # 17:300 G>A exists, G>C does not
    assert got["v5"] == ("unrecorded", "unrecorded")       # not normalised, so not matched
    assert t.loc[t["variant_id"] == "v1", "variation_id"].item() == "11"


def test_two_records_at_one_allele_stop_the_annotation(tmp_path):
    vcf = write_vcf(tmp_path / "c.vcf.gz", VCF_LINES + [
        "17\t100\t99\tA\tG\t.\t.\tCLNSIG=Benign"])
    with pytest.raises(SystemExit, match="two rows"):
        cs.annotate(matrix(), vcf)


def test_the_release_check_compares_both_arms_row_by_row(tmp_path):
    t, _ = cs.annotate(matrix(), write_vcf(tmp_path / "c.vcf.gz"))
    s = t[["variant_id", "variation_id", "clnsig_raw", "clnsig_class", "clnrevstat"]].copy()
    s["clinvar_arm"], s["clinvar_arm_strict"] = t["arm"], t["arm_strict"]
    assert cs.release_check(t, s)["holds"]
    s.loc[1, "clinvar_arm_strict"] = "recorded_unclassified"
    out = cs.release_check(t, s)
    assert not out["holds"] and out["arm_strict"]["differing"] == ["v2"]


def test_the_summary_counts_against_the_old_list(tmp_path):
    m = matrix()
    t, _ = cs.annotate(m, write_vcf(tmp_path / "c.vcf.gz"))
    s = cs.summary(t, m, {"v1", "v4"})
    old = s[(s["part"] == "old_list") & (s["gene"] == "all")].set_index("arm")["n"]
    assert old["old_list_now_unrecorded"] == 1          # v4
    assert old["recorded_not_in_old_list"] == 2         # v2, v3
    arms = s[s["part"] == "arms"]
    assert arms["n"].sum() == len(m)
    assert set(arms["stratum"]) == {"splice_1_2", "splice_11_50", "coding_or_utr",
                                    "splice_3_10"}


def _function_sources(path: Path) -> dict[str, str]:
    src = path.read_text()
    tree = ast.parse(src)
    return {n.name: ast.get_source_segment(src, n) for n in tree.body
            if isinstance(n, ast.FunctionDef)}


def test_the_rules_are_the_releases_as_written():
    base = cs.COMPANION_DIR
    build, config = base / cs.RULE_FILES[0], base / cs.RULE_FILES[1]
    if not (build.exists() and config.exists()):
        pytest.skip("the release's files are not here (python -m atlas.clinvar_status fetch)")
    ours = _function_sources(Path(cs.__file__))
    theirs = _function_sources(build) | {"classify_clnsig":
                                         _function_sources(config)["classify_clnsig"]}
    for name in ("classify_clnsig", "clinvar_arm", "clinvar_arm_strict"):
        assert ours[name] == theirs[name], name
    a, b = ours["annotate_clinvar"].split("\n"), theirs["annotate_clinvar"].split("\n")
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(a) == len(b) and diff == [("    classify = classify_clnsig",
                                          "    classify = _load_clnsig_classifier()")]


def test_the_copy_stops_unless_both_hashes_are_the_releases(tmp_path, monkeypatch):
    comp = tmp_path / "companion"
    src = comp / cs.VCF_RELEASE_PATH
    src.parent.mkdir(parents=True)
    write_vcf(src)
    md5 = hashlib.md5(src.read_bytes()).hexdigest()
    (src.parent / (src.name + ".md5")).write_text(f"{md5}  /somewhere/{src.name}\n")
    manifest = {"sources": {"clinvar_vcf": {"sha256": cs.VCF_SHA256}}}

    def git(*args):
        if args[0] == "rev-parse":
            return (cs.TAG_COMMIT + "\n").encode()
        if args[1].endswith("analysis_set_v1.manifest.json"):
            return json.dumps(manifest).encode()
        return b"x"

    monkeypatch.setattr(cs, "_git", git)
    monkeypatch.setattr(cs, "companion_repo", lambda: comp)
    monkeypatch.setattr(cs, "public_url", lambda size, md5: {"url": None})
    with pytest.raises(SystemExit, match="not the release's"):
        cs.fetch(dest=tmp_path / "refs" / cs.VCF_NAME, out=tmp_path / "raw")
    manifest["sources"]["clinvar_vcf"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="release manifest records"):
        cs.fetch(dest=tmp_path / "refs2" / cs.VCF_NAME, out=tmp_path / "raw")

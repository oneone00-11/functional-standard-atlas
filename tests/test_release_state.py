"""The manuscript's cited archive must still be the repository it describes.

The declared-count check compares the manuscript against a facts file and passes
when both agree, which is internal consistency and nothing more. It reported
clean while the manuscript cited a DOI for v2.3.2, quoted a test count measured
on a v2.4.0 candidate that was never published, and named a supplementary table
whose generator postdates the tag.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas import release_state as rd   # noqa: E402

MANUSCRIPT = Path.home() / "Desktop" / "functional-standard-atlas_manuscript-gb.docx"


def _needs_git():
    """A release archive is not a git checkout, so these checks cannot run from
    one. That is the point of the archive; skip rather than fail there."""
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout; this is a property of the repository")


def test_a_release_tag_exists_to_compare_against():
    _needs_git()
    assert rd.last_published_tag() is not None


def test_divergence_is_detected_while_the_repository_is_ahead_of_the_tag():
    _needs_git()
    """Regression for the state this check was written for: 19 commits past
    v2.3.2 while the manuscript cites that deposit."""
    if not MANUSCRIPT.exists():
        pytest.skip("manuscript not available in this checkout")
    r = rd.divergence(MANUSCRIPT)
    assert r["dois"], "the manuscript should cite at least one deposit"
    if r["commits_ahead"] == 0:
        pytest.skip("repository is at the published tag; nothing to detect")
    assert r["fatal"], r
    assert str(r["commits_ahead"]) in r["why"] and r["tag"] in r["why"]


def test_the_deliverables_added_since_the_tag_are_named():
    _needs_git()
    if not MANUSCRIPT.exists():
        pytest.skip("manuscript not available in this checkout")
    r = rd.divergence(MANUSCRIPT)
    if r["commits_ahead"] == 0:
        pytest.skip("repository is at the published tag")
    # the operator has to know what is missing from the deposit, not just that
    # something is
    assert r["added_since_tag"] or r["changed"]


def test_a_table_absent_from_the_deposit_is_reported():
    """Table S11 postdates the cited tag, so the deposit cannot carry it."""
    _needs_git()
    if not list((REPO / "results").glob("table_s*")):
        pytest.skip("results/ not built; nothing to compare against the deposit")
    if not MANUSCRIPT.exists():
        pytest.skip("manuscript not available in this checkout")
    tag = rd.last_published_tag()
    listing = rd._git("ls-tree", "-r", "--name-only", tag).splitlines()
    d = rd.deposit(MANUSCRIPT, listing)
    missing = " ".join(d["missing_from_deposit"])
    assert d["fatal"], "a table added after the tag should be reported missing"
    assert "s11" in missing.lower(), d


def test_a_complete_deposit_passes(tmp_path):
    """The check must clear once the deposit actually carries the artefacts."""
    if not MANUSCRIPT.exists():
        pytest.skip("manuscript not available in this checkout")
    listing = [f"results/{p.name}" for p in (REPO / "results").glob("table_s*")]
    if not listing:
        pytest.skip("results/ not built in this checkout")
    d = rd.deposit(MANUSCRIPT, listing)
    assert not d["fatal"], d


# --- counts declared outside the manuscript ----------------------------------

def test_declarations_outside_the_manuscript_agree_with_the_facts():
    """README, CITATION and the Zenodo description state counts and a version
    too. Nothing checked them, and the README carried 128/126/91/37 for three
    releases after those numbers stopped being true."""
    _needs_git()
    bad = rd.declarations()
    assert not bad, "stale declarations:\n  " + "\n  ".join(
        f"{x['file']}: {x['says']!r} expected {x['expected']}" for x in bad)


def test_a_stale_count_outside_the_manuscript_is_caught(tmp_path):
    """Regression for the four README numbers: the check must report them."""
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "pipeline_facts.json").write_text(json.dumps({
        "guardrail_tests_collected": 148,
        "guardrail_tests_passing_from_archive": 141,
        "guardrail_tests_passing_from_bare_clone": 110,
        "guardrail_tests_skipped_from_bare_clone": 38,
        "release_version": "2.4.0"}))
    (tmp_path / "README.md").write_text(
        "run the 91 code-only guardrails (37 skip without data)\n"
        "make test      # 128 collected. 126 pass from a clean extract\n")
    (tmp_path / "CITATION.cff").write_text('version: "2.3.2"\n')

    bad = rd.declarations(repo=tmp_path)
    said = {x["says"] for x in bad}
    for stale in ("91 code-only guardrails", "128 collected",
                  "126 pass from a clean extract"):
        assert any(stale in s for s in said), (stale, said)
    assert any("37 skip without data" in s for s in said), said
    assert any(x["key"] == "release_version" for x in bad), bad


def test_correct_declarations_pass(tmp_path):
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "pipeline_facts.json").write_text(json.dumps({
        "guardrail_tests_collected": 148,
        "guardrail_tests_passing_from_archive": 141,
        "guardrail_tests_passing_from_bare_clone": 110,
        "guardrail_tests_skipped_from_bare_clone": 38,
        "release_version": "2.4.0"}))
    (tmp_path / "README.md").write_text(
        "run the 110 code-only guardrails (38 skip without data)\n"
        "make test      # 148 collected. 141 pass from a clean extract\n")
    (tmp_path / "CITATION.cff").write_text('version: "2.4.0"\n')
    assert rd.declarations(repo=tmp_path) == []


def test_an_archive_never_contains_another_archive():
    """A release candidate is a build product, not an input. It reached a commit
    once, and the next build tried to package it into its own successor."""
    _needs_git()
    sys.path.insert(0, str(REPO / "src"))
    from atlas.package_release import manifest

    packed = [f for f in manifest() if f.startswith("dist/") or f.endswith(".zip")]
    assert not packed, f"the archive would contain build products: {packed}"

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

from atlas import release_drift as rd   # noqa: E402

MANUSCRIPT = Path.home() / "Desktop" / "functional-standard-atlas_manuscript-gb.docx"


def _needs_git():
    """A release archive is not a git checkout, so these checks cannot run from
    one. That is the point of the archive; skip rather than fail there."""
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout; drift is a property of the repository")


def test_a_release_tag_exists_to_compare_against():
    _needs_git()
    assert rd.last_published_tag() is not None


def test_drift_is_detected_while_the_repository_is_ahead_of_the_tag():
    _needs_git()
    """Regression for the state this check was written for: 19 commits past
    v2.3.2 while the manuscript cites that deposit."""
    if not MANUSCRIPT.exists():
        pytest.skip("manuscript not available in this checkout")
    r = rd.drift(MANUSCRIPT)
    assert r["dois"], "the manuscript should cite at least one deposit"
    if r["commits_ahead"] == 0:
        pytest.skip("repository is at the published tag; nothing to detect")
    assert r["fatal"], r
    assert str(r["commits_ahead"]) in r["why"] and r["tag"] in r["why"]


def test_the_deliverables_added_since_the_tag_are_named():
    _needs_git()
    if not MANUSCRIPT.exists():
        pytest.skip("manuscript not available in this checkout")
    r = rd.drift(MANUSCRIPT)
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

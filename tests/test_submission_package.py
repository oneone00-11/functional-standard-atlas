"""The package that goes to the journal, checked as one object.

The manuscript had gates; the package around it had none, and three defects
survived every check because each lived in the gap between two deliverables:
the guardrail-test count stated three times with three values, the minimum
detectable rho given as 0.107 in the manuscript and 0.197 in Note S6, and
Table S11 cited in the text while Additional file 1 carries twelve sheets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas import submission_package as sp   # noqa: E402

CONFIG = REPO / "config" / "submission_package.yaml"

# Both manuscripts ship a package. Only the atlas one was checked at first, so
# the companion package -- manuscript, supplement, cover letter -- had no
# inventory, no cross-file number check and no reference closure at all.
PACKAGES = ("atlas", "bib")

# The companion cover letter is rewritten per submission and the venue is
# undecided, so it is declared and absent on purpose.
KNOWN_ABSENT = {"BiB_cover_letter.pdf"}


def _spec():
    if not CONFIG.exists():
        pytest.skip("no submission-package manifest in this checkout")
    spec = sp.load(CONFIG)
    root = sp.package_root(spec)
    if not root.is_dir():
        pytest.skip("the submission package is not on this machine")
    return spec


def test_the_manifest_is_well_formed():
    spec = _spec()
    for name, pkg in spec["packages"].items():
        assert pkg.get("manuscript"), name
        for e in pkg["files"]:
            assert e.get("path") and e.get("role"), e
            # every deliverable either has a generator or admits it has none
            assert e.get("generated_by") or e.get("maintained_by_hand"), (
                f"{e['path']} declares neither a generator nor hand maintenance")


@pytest.mark.parametrize("pkg", PACKAGES)
def test_every_declared_file_exists(pkg):
    spec = _spec()
    missing = [m for m in sp.inventory(spec, pkg)["missing"]
               if Path(m).name not in KNOWN_ABSENT]
    assert not missing, missing


@pytest.mark.parametrize("pkg", PACKAGES)
def test_nothing_undeclared_sits_in_the_package_directories(pkg):
    """An old export of the related manuscript survived two submissions by
    sitting in the directory with nobody listing it."""
    spec = _spec()
    inv = sp.inventory(spec, pkg)
    assert not inv["undeclared"], (
        "undeclared files in the package:\n  " + "\n  ".join(inv["undeclared"]))


@pytest.mark.parametrize("pkg", PACKAGES)
def test_a_quantity_stated_twice_agrees(pkg):
    """Regression: the guardrail count is 144 in the manuscript, 102 in the
    cover letter and 83 in Note S12; the deep-intronic minimum detectable rho
    is 0.107 in the manuscript and 0.197 in Note S6."""
    spec = _spec()
    bad = sp.quantities(spec, pkg)
    assert not bad, "quantities stated inconsistently:\n  " + "\n  ".join(
        f"{q['quantity']}: {', '.join(q['values'])}" for q in bad)


@pytest.mark.parametrize("pkg", PACKAGES)
def test_every_cited_item_is_in_the_file_the_text_names(pkg):
    """Regression: Table S11 is cited as Additional file 1 and is not in it.
    Note S11 exists -- in Additional file 2 -- which is why a check that pooled
    items across the package saw nothing wrong."""
    spec = _spec()
    refs = sp.references(spec, pkg)
    assert not refs["cited_but_absent"], refs["cited_but_absent"]
    assert not refs["misdirected"], "\n  ".join(
        f"{m['kind']} {m['item']} cited as {m['cited_in']}, found in "
        f"{', '.join(m['actually_in']) or 'nowhere'}" for m in refs["misdirected"])


def test_panel_letters_are_not_treated_as_separate_items():
    assert sp._canonical("S17C") == "S17"      # panel C of Figure S17
    assert sp._canonical("S5b") == "S5B"       # a distinct table
    assert sp._canonical("S10") == "S10"

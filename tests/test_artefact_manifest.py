"""The watch for the delivered figures and tables.

The manuscript claims figures and tables were verified against the archived
release by content hash. Before this file nothing did that: `atlas.manifest`
hashes downloaded sources and the frozen matrix, `test_release_hygiene` scans
`git ls-files` (so the ~270 shipped `results/` files were never in scope), and
`test_figure_numbering` checks names rather than content. This test recomputes
every hash in `manifests/artefacts.json` and fails when a delivered artefact
drifts from it.

`results/` is gitignored, so on a clean clone there is nothing to hash and the
content tests skip -- but the manifest itself is tracked and is checked for
structure regardless, so a truncated or hand-edited manifest still fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas.artefact_manifest import MANIFEST, RESULTS, check, collect  # noqa: E402

MIN_MAIN_FIGURES = 6      # the manuscript's Figures 1-6
MIN_MAIN_TABLES = 2       # Table 1 (composition) and Table 2 (territory guide)


def _manifest() -> dict:
    if not MANIFEST.exists():
        pytest.fail(f"{MANIFEST.relative_to(REPO)} is missing; run "
                    f"`python -m atlas.artefact_manifest --write`")
    return json.loads(MANIFEST.read_text())


def test_manifest_is_tracked_and_well_formed():
    m = _manifest()
    assert m["files"], "manifest records no artefacts"
    assert m["n_files"] == len(m["files"])
    for rel, meta in m["files"].items():
        assert rel.startswith("results/"), rel
        assert len(meta["sha256"]) == 64, rel
        assert meta["bytes"] > 0, rel
        assert meta["role"] in m["by_role"], rel


def test_manifest_covers_the_main_figures_and_tables():
    """Coverage is the point: a manifest that quietly lost the main figures
    would still pass a pure hash check."""
    m = _manifest()
    assert m["by_role"]["main_figure"] >= MIN_MAIN_FIGURES, m["by_role"]
    assert m["by_role"]["main_table"] >= MIN_MAIN_TABLES, m["by_role"]
    assert m["by_role"]["supp_figure"] > 0 and m["by_role"]["supp_table"] > 0


def test_delivered_artefacts_match_their_recorded_hashes():
    if not RESULTS.is_dir() or not any(RESULTS.iterdir()):
        pytest.skip("results/ not built in this checkout")
    problems = check()
    assert not problems, "delivered artefacts drifted from the manifest:\n" + "\n".join(problems)


def test_nothing_delivered_is_left_unrecorded():
    """The failure mode this whole file exists for: a new artefact ships
    without anything watching it."""
    if not RESULTS.is_dir() or not any(RESULTS.iterdir()):
        pytest.skip("results/ not built in this checkout")
    recorded = set(_manifest()["files"])
    unrecorded = sorted(set(collect()) - recorded)
    assert not unrecorded, ("delivered but absent from manifests/artefacts.json:\n  "
                            + "\n  ".join(unrecorded))

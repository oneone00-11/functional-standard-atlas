"""Guardrails for the manuscript number checker.

The checker exists because the manuscript claims every reported value is read
programmatically from pipeline output, and nothing enforced it. These tests pin
the two behaviours that make it worth having:

  * a token with no counterpart anywhere in results/ must be reported, never
    silently skipped -- the whitelist is the only way past, and it is tracked;
  * a match that carries no information must not be counted as verification.
    A two-decimal number in a dense range matches hundreds of unrelated pipeline
    values; calling that "verified" is precisely how the complementarity range
    survived six rounds of checking.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas.check_manuscript_numbers import (  # noqa: E402
    CITATION, IDENTIFIER, TOKEN, WEAK_MATCH_MIN, WHITELIST, _tolerance,
    load_whitelist, pipeline_values,
)

RESULTS = REPO / "results"


def test_whitelist_is_tracked_and_every_entry_carries_a_reason():
    assert WHITELIST.exists(), "the whitelist must be committed, not implicit"
    _, _, spec = load_whitelist()
    assert spec, "whitelist is empty; it should at least document why"
    for name, group in spec.items():
        assert group.get("reason", "").strip(), f"{name} has no reason"
        assert group.get("values") or group.get("pattern"), f"{name} matches nothing"


def test_tolerance_follows_printed_precision():
    assert _tolerance("0.12") == pytest.approx(0.005, abs=1e-9)
    assert _tolerance("0.1234") == pytest.approx(0.00005, abs=1e-10)
    assert _tolerance("17") == pytest.approx(0.5, abs=1e-9)


def test_identifiers_and_citations_are_not_read_as_measurements():
    """A DOI and a citation number are not values the pipeline must explain."""
    text = ("available at https://doi.org/10.5281/zenodo.21828448 [61] and "
            "urn:mavedb:00000097-0-2, see [22, 23, 24]")
    stripped = CITATION.sub(" ", IDENTIFIER.sub(" ", text))
    assert TOKEN.search(stripped) is None, TOKEN.search(stripped).group(0)


def test_thousands_separated_tokens_are_read():
    m = TOKEN.search("across 64,178 variants")
    assert m and m.group(1) == "64,178"


def test_row_counts_enter_the_pool():
    """"2,452 of 2,803 score sets" are real pipeline quantities that appear
    nowhere inside the files themselves."""
    if not RESULTS.is_dir() or not (RESULTS / "mavedb_survey_v1.tsv").exists():
        pytest.skip("results/ not built in this checkout")
    pool = pipeline_values()
    for n in (2452.0, 2803.0):
        assert np.any(np.abs(pool - n) <= 0.5), f"{n:.0f} missing from the pool"


def test_generated_prose_is_excluded_from_the_pool():
    """Supplemental_Note.md is produced by atlas.supplement; a literal hardcoded
    in that generator would otherwise validate itself through it."""
    src = (REPO / "src" / "atlas" / "check_manuscript_numbers.py").read_text()
    fn = src.split("def pipeline_values")[1].split("\ndef ")[0]
    assert '".md"' not in fn, "generated prose must stay out of the pool"


def test_weak_matches_are_a_separate_class():
    """The limit of this tool, pinned so it is not forgotten: a low-precision
    number in a dense range matches by coincidence, so the checker alone could
    not have caught the complementarity range. That binding is what the
    analysis-claims manifest provides."""
    if not RESULTS.is_dir() or not any(RESULTS.iterdir()):
        pytest.skip("results/ not built in this checkout")
    pool = pipeline_values()
    n = int(np.sum(np.abs(pool - 0.12) <= 0.005))
    assert n >= WEAK_MATCH_MIN, (
        "0.12 no longer matches many pipeline values; the weak-match threshold "
        "may need revisiting")

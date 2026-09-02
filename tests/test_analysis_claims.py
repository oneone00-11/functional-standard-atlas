"""Every analysis the manuscript claims must have code and output behind it.

A value check asks whether a number is right. It cannot ask whether the analysis
a sentence describes exists at all, which is a different question.
The complementarity paragraph claimed a nineteen-predictor axis analysis and
cited a matrix that has carried nine models since the commit that introduced the
seven meta-predictors -- a sentence with no number wrong in it, and nothing to
catch it.

config/analysis_claims.yaml binds each such sentence to a script and an output.
These tests assert the bindings resolve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

MANIFEST = REPO / "config" / "analysis_claims.yaml"
VALID_STATUS = {"verified", "narrative", "gap"}


def _claims() -> list[dict]:
    assert MANIFEST.exists(), f"{MANIFEST.relative_to(REPO)} is missing"
    return yaml.safe_load(MANIFEST.read_text())["claims"]


def test_manifest_is_well_formed():
    claims = _claims()
    assert claims, "no claims recorded"
    for c in claims:
        assert c.get("para", "").startswith(("P", "T")), c
        assert c.get("claim", "").strip(), c
        assert c.get("status") in VALID_STATUS, c


def test_every_named_script_exists():
    """A claim pointing at a script that is not there is the failure mode."""
    missing = [f"{c['para']}: {c['script']}" for c in _claims()
               if c.get("script") and not (REPO / c["script"]).exists()]
    assert not missing, "claims name scripts that do not exist:\n  " + "\n  ".join(missing)


def test_every_named_output_exists():
    """`results/` is gitignored, so outputs are only checkable in a built
    checkout; tracked paths are checked either way."""
    tracked = [c for c in _claims()
               if c.get("output") and not str(c["output"]).startswith("results/")]
    missing = [f"{c['para']}: {c['output']}" for c in tracked
               if not (REPO / c["output"]).exists()]
    assert not missing, "claims name tracked outputs that do not exist:\n  " + "\n  ".join(missing)

    if not (REPO / "results").is_dir() or not any((REPO / "results").iterdir()):
        pytest.skip("results/ not built in this checkout")
    built = [c for c in _claims()
             if c.get("output") and str(c["output"]).startswith("results/")]
    missing = [f"{c['para']}: {c['output']}" for c in built
               if not (REPO / c["output"]).exists()]
    assert not missing, "claims name pipeline outputs that do not exist:\n  " + "\n  ".join(missing)


def test_non_verified_claims_explain_themselves():
    for c in _claims():
        if c["status"] != "verified":
            assert c.get("note", "").strip(), (
                f"{c['para']} is '{c['status']}' with no note explaining why")


def test_no_unresolved_gaps():
    """A `gap` row is a claim the manuscript makes that nothing supports.
    Failing here is the point: it should not be possible to ship one quietly."""
    gaps = [f"{c['para']}: {c['claim']}" for c in _claims() if c["status"] == "gap"]
    assert not gaps, "manuscript claims with nothing behind them:\n  " + "\n  ".join(gaps)


MANUSCRIPT = Path.home() / "Desktop" / "functional-standard-atlas_manuscript-gb.docx"


def test_every_claim_carries_an_anchor():
    """`para` is commentary. The anchor is what actually binds."""
    missing = [f"{c['para']}: {c['claim']}" for c in _claims() if not c.get("anchor")]
    assert not missing, "claims with no content anchor:\n  " + "\n  ".join(missing)


def test_every_anchor_hits_exactly_once():
    """Zero hits means the sentence was reworded and the row now describes
    nothing; more than one means the anchor is not distinctive and could bind to
    the wrong sentence. Both were invisible under paragraph numbers, which
    always resolve to *some* paragraph however far the text has moved."""
    if not MANUSCRIPT.exists():
        pytest.skip("manuscript not available in this checkout")
    from atlas.check_manuscript_numbers import _blocks

    blocks = _blocks(MANUSCRIPT)
    bad = []
    for c in _claims():
        anchor = c.get("anchor")
        if not anchor:
            continue
        n = sum(text.count(anchor) for _, text in blocks)
        if n != 1:
            bad.append(f"{c['para']}: {n} hits for {anchor!r} ({c['claim']})")
    assert not bad, "anchors that do not hit exactly once:\n  " + "\n  ".join(bad)


def test_paragraph_numbers_are_not_load_bearing():
    """Regression for the 2026-09-02 shift: with every `para` deliberately
    wrong, resolution must be unaffected, because nothing matches on it."""
    if not MANUSCRIPT.exists():
        pytest.skip("manuscript not available in this checkout")
    import atlas.check_manuscript_numbers as cmn

    before = cmn.resolve_anchors(MANUSCRIPT)
    shifted = yaml.safe_load(MANIFEST.read_text())
    for c in shifted["claims"]:
        if str(c.get("para", "")).startswith("P"):
            c["para"] = f"P{int(c['para'][1:]) + 23}"
    tmp = REPO / "config" / ".analysis_claims.shifted.yaml"
    tmp.write_text(yaml.safe_dump(shifted, allow_unicode=True))
    try:
        after = cmn.resolve_anchors(MANUSCRIPT, tmp)
    finally:
        tmp.unlink()
    assert before == after and before, (
        "paragraph numbers still influence resolution")

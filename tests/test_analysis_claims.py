"""Every analysis the manuscript claims must have code and output behind it.

This is the check that no earlier round performed. Six rounds asked whether a
number was right; none asked whether the analysis a sentence describes exists.
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

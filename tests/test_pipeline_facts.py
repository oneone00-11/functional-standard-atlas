"""`make facts` must not disarm the checks that read the facts file.

Only two of these numbers can be computed here: how many tests are collected,
and how many pass without the hygiene denylist. The rest are measured at release
time, by running the suite from a clean extract of the archive and from a bare
clone, and are carried through `--write` rather than recomputed.

They used to be dropped. write() rebuilt the file from a fixed key list written
before the bare-clone fields existed, so running the documented `make facts`
deleted five measured fields. Nothing would have failed: atlas.release_state
skips any key the facts file does not hold, so the README, the Zenodo
description and the supplement's Note S12 would have gone unchecked while every
gate reported green.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas import pipeline_facts as pf     # noqa: E402
from atlas import release_state as rs      # noqa: E402


def test_write_preserves_every_count_measured_at_release(tmp_path, monkeypatch):
    facts = tmp_path / "pipeline_facts.json"
    facts.write_text(json.dumps({
        "guardrail_tests_collected": 1,
        "guardrail_tests_passing": 0,
        "guardrail_tests_skipped_without_denylist": 1,
        "guardrail_tests_passing_from_archive": 145,
        "guardrail_tests_skipped_from_archive": 23,
        "archive_measured_for_version": "v2.4.0",
        "guardrail_tests_passing_from_bare_clone": 113,
        "guardrail_tests_skipped_from_bare_clone": 55,
        "bare_clone_measured_for": "the release tag",
        "release_version": "2.4.0",
    }))
    monkeypatch.setattr(pf, "FACTS", facts)
    monkeypatch.setattr(pf, "collected_tests", lambda: 200)

    after = pf.write()

    assert after["guardrail_tests_collected"] == 200
    assert after["guardrail_tests_passing"] == 199
    for key in pf.MEASURED_AT_RELEASE:
        assert key in after, f"`make facts` dropped {key}"
    assert after["guardrail_tests_passing_from_archive"] == 145
    assert after["guardrail_tests_skipped_from_bare_clone"] == 55
    assert after["release_version"] == "2.4.0"


def test_every_key_the_declared_count_check_reads_survives_a_rewrite():
    """The two modules have to agree on which keys exist, or the check degrades
    into silence exactly when someone regenerates the file."""
    written = set(pf.MEASURED_AT_RELEASE) | {
        "guardrail_tests_collected", "guardrail_tests_passing",
        "guardrail_tests_skipped_without_denylist",
        "tests_covering_the_analyses_introduced_here"}
    read = set(rs.DECLARED_COUNTS.values()) | {"release_version"}
    missing = sorted(read - written)
    assert not missing, (
        "atlas.release_state checks keys that `make facts` does not keep: "
        + ", ".join(missing))

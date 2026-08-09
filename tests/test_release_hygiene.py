"""Refuse to publish drafting material in a public artefact.

This repository is public and is cited from the manuscript, so every tracked
path reaches readers. Drafting material -- working notes, correspondence,
planning documents, scratch drafts -- is easy to leave behind and hard to
retract once archived, so this is a gate rather than a habit.

The terms to reject are deliberately NOT held here. A denylist naming what was
removed is itself a disclosure of what was removed, and this file is tracked.
They live in `.release-denylist` at the repository root, which is gitignored:
one term per line, `#` comments allowed, and an optional `path-prefix<TAB>term`
line under `[allow]` to exempt a legitimate use.

Without that file the scan cannot run and the test skips. Keep a copy outside
the repository; it is the operational half of this check.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DENYLIST = REPO / ".release-denylist"
SELF = "tests/test_release_hygiene.py"

TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".cff", ".toml",
                 ".json", ".tsv", ".csv", ".ini", ""}


def _load() -> tuple[list[str], list[tuple[str, str]]]:
    terms: list[str] = []
    allow: list[tuple[str, str]] = []
    section = "deny"
    for raw in DENYLIST.read_text().splitlines():
        line = raw.split("#", 1)[0].strip() if not raw.startswith("\t") else raw.rstrip()
        if not line.strip():
            continue
        if line.strip().lower() == "[allow]":
            section = "allow"
            continue
        if section == "deny":
            terms.append(line.strip())
        else:
            prefix, _, term = line.strip().partition("\t")
            allow.append((prefix.strip(), term.strip()))
    return terms, allow


def _tracked() -> list[str] | None:
    try:
        r = subprocess.run(["git", "ls-files"], cwd=REPO,
                           capture_output=True, text=True)
    except OSError:
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return [ln for ln in r.stdout.splitlines() if ln.strip() and ln.strip() != SELF]


def test_no_drafting_material_is_tracked():
    if not DENYLIST.exists():
        pytest.skip(f"{DENYLIST.name} not present; see this module's docstring")
    tracked = _tracked()
    if tracked is None:
        pytest.skip("not a git checkout; the gate applies to the repository")

    terms, allow = _load()
    assert terms, f"{DENYLIST.name} lists no terms"

    def ok(path: str, term: str) -> bool:
        return any(term == t and (pre == "" or path.startswith(pre)) for pre, t in allow)

    hits = []
    for rel in tracked:
        p = REPO / rel
        for term in terms:
            if term.lower() in rel.lower() and not ok(rel, term):
                hits.append(f"{rel}: path matches a rejected term")
        if p.suffix.lower() not in TEXT_SUFFIXES or not p.exists():
            continue
        if p.stat().st_size > 4_000_000:
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for term in terms:
            if term in text and not ok(rel, term):
                hits.append(f"{rel}: contents match a rejected term")

    assert not hits, ("drafting material is tracked and would ship publicly:\n  "
                      + "\n  ".join(sorted(set(hits))))

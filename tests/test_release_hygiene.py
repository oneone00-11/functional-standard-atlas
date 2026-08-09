"""Refuse to ship internal submission-process material in a public artefact.

The repository is public and is cited from the manuscript's Data Access section,
so anything tracked here reaches editors and reviewers. Two rounds of manual
scanning found leaks after the fact -- cover letters, a journal-positioning
comparison, an early draft carrying "[co-authors TBD]" against a manuscript
declared single-author. This test is the gate that makes a third round
unnecessary: it scans tracked paths and their contents for terms that belong to
the drafting process rather than to the science.

Adding a term here is cheap. Every entry in ALLOW must say why it is legitimate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Terms that must not appear in a tracked path name or its contents.
FORBIDDEN = [
    "AGENTS", "TO_VERIFY", "DRIFT", "RETARGET", "journal_positioning",
    "skeleton_v1", "co-authors TBD", "cover_letter", "title_options",
    "number_audit", "ref_conversion", "TODO(", "positioning",
    "Genome Biology", "AJHG", "APC", "影响因子", "接受率", "被拒",
    # target-journal and companion-submission leakage, added after the round-3
    # scan found both in files the original denylist did not reach
    "Genome Research", "variant-fm-benchmark",
]

# (path prefix, term) pairs that are legitimate. Each needs a stated reason.
ALLOW: list[tuple[str, str]] = [
    # "Genome Biology" is a journal name in the bibliography and in assay
    # provenance notes; citing it is normal scholarship, not a leak.
    ("results/references_v1", "Genome Biology"),
    ("results/table2_model_inventory", "Genome Biology"),
    ("src/atlas/references.py", "Genome Biology"),
    # APC is also a gene symbol and appears inside longer tokens (e.g. capture).
    ("", "APC"),
    # Journal names in the bibliography and in the model inventory are citations.
    ("results/references_v1", "Genome Research"),
    ("results/table2_model_inventory", "Genome Research"),
    ("src/atlas/references.py", "Genome Research"),
    # Per-model provenance notes legitimately record the cross-check against the
    # companion benchmark, which the manuscript's Concordance validation reports.
    ("models/", "variant-fm-benchmark"),
]

TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".cff", ".toml",
                 ".json", ".tsv", ".csv", ".ini", ""}


SELF = "tests/test_release_hygiene.py"  # the denylist necessarily contains every term


def _tracked() -> list[str] | None:
    """Tracked paths, or None when this is not a git checkout (e.g. the Zenodo
    archive, which ships the same tree without .git)."""
    try:
        r = subprocess.run(["git", "ls-files"], cwd=REPO,
                           capture_output=True, text=True)
    except OSError:
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return [ln for ln in r.stdout.splitlines() if ln.strip() and ln.strip() != SELF]


def _allowed(path: str, term: str) -> bool:
    return any(term == t and (pre == "" or path.startswith(pre)) for pre, t in ALLOW)


def _hits() -> list[str] | None:
    tracked = _tracked()
    if tracked is None:
        return None
    found = []
    for rel in tracked:
        p = REPO / rel
        for term in FORBIDDEN:
            if term.lower() in rel.lower() and not _allowed(rel, term):
                found.append(f"{rel}: path contains {term!r}")
        if p.suffix.lower() not in TEXT_SUFFIXES or not p.exists():
            continue
        if p.stat().st_size > 4_000_000:
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for term in FORBIDDEN:
            if term in text and not _allowed(rel, term):
                found.append(f"{rel}: contains {term!r}")
    return found


def test_no_submission_process_material_is_tracked():
    hits = _hits()
    if hits is None:
        pytest.skip("not a git checkout; the gate applies to the repository")
    assert not hits, (
        "internal submission-process material is tracked and would ship in the "
        "public archive:\n  " + "\n  ".join(hits))


def test_gate_actually_detects_a_known_leak(tmp_path):
    """The gate must fail on a real leak, not pass because the scan is broken."""
    assert any("co-authors TBD" in t for t in FORBIDDEN)
    assert not _allowed("paper/manuscript_v1.md", "co-authors TBD")

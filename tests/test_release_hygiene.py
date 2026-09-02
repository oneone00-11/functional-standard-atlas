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

SCOPE. This gate used to scan `git ls-files`, which is not what ships. A release
archive also carries `data/raw/`, `data/frozen/` and the ~270 files in
`results/` -- all gitignored, all published in v2.3.1 and v2.3.2 without ever
passing this check. It now scans `atlas.package_release.manifest()`, the
definition of what actually goes in the archive, and a companion test asserts
that definition covers every file in a candidate archive.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas.package_release import manifest, verify  # noqa: E402
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


def _shipped() -> list[str] | None:
    """Everything the release archive will contain -- not merely what git tracks."""
    try:
        files = manifest()
    except OSError:
        return None
    if not files:
        return None
    return [f for f in files if f != SELF]


def test_no_drafting_material_ships():
    if not DENYLIST.exists():
        pytest.skip(f"{DENYLIST.name} not present; see this module's docstring")
    tracked = _shipped()
    if tracked is None:
        pytest.skip("cannot determine the packaging output in this checkout")

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

    assert not hits, ("drafting material would ship publicly:\n  "
                      + "\n  ".join(sorted(set(hits))))


def test_the_scan_covers_the_whole_archive():
    """The defect this gate carried for months: scanning a subset of what ships.
    Coverage is asserted directly, so a future path that is packaged but not
    enumerated fails here rather than shipping unscanned."""
    files = manifest()
    if not files:
        # `manifest()` starts from `git ls-files`; an extracted archive or a
        # source tarball is not a git checkout, so there is nothing to enumerate
        # and nothing to assert. The gate applies where a release is built.
        pytest.skip("not a git checkout; the packaging manifest cannot be built")
    for top in ("data/raw", "data/frozen", "results"):
        base = REPO / top
        if not base.is_dir() or not any(base.rglob("*")):
            continue
        on_disk = {p.relative_to(REPO).as_posix() for p in base.rglob("*")
                   if p.is_file() and p.name != ".DS_Store"
                   and "_scratch" not in p.parts}
        missing = sorted(on_disk - set(files))
        assert not missing, (
            f"{top}/ ships but {len(missing)} of its files are outside the scan:\n  "
            + "\n  ".join(missing[:20]))


def test_development_output_never_ships():
    """CONVENTIONS.md rule 6. Three July smoke-test artefacts reached the v2.3.1
    and v2.3.2 archives, one of them carrying an OPEN DECISION note and a term
    the denylist rejects outside models/. The place is the rule: `_scratch` is
    excluded structurally, so the next one is excluded before it is written."""
    files = manifest()
    leaked = [f for f in files if "_scratch" in Path(f).parts]
    assert not leaked, "development output is in the packaging manifest:\n  " + "\n  ".join(leaked)

    scratch = REPO / "results" / "_scratch"
    if scratch.is_dir():
        assert any(scratch.iterdir()), "results/_scratch exists but is empty"


def test_manifest_accounts_for_a_reference_archive():
    """If an extracted archive is available, nothing in it may be undeclared."""
    root = REPO.parent / "_release_candidate"
    if not root.is_dir():
        pytest.skip("no extracted archive to check against")
    r = verify(root)
    assert not r["in_archive_only"], (
        "files present in the archive but not declared by package_release:\n  "
        + "\n  ".join(r["in_archive_only"][:20]))

"""Content hashes for the delivered figures and tables.

The manuscript states that "figures and tables were verified against the
archived release by content hash". Nothing performed that check: `manifest.py`
hashes downloaded sources and the frozen matrix, `test_release_hygiene` scans
`git ls-files`, and `test_figure_numbering` checks names. Every delivered
artefact in `results/` was outside all of them -- the same
delivered-product-decoupled-from-the-code defect already documented for Table
S9, Table 2 and MAIN_FIGS.

This module closes it. `results/` is gitignored, so the manifest lives in
`manifests/artefacts.json`, which is tracked; the guardrail
(`tests/test_artefact_manifest.py`) recomputes every hash and fails on drift.

    python -m atlas.artefact_manifest --write     # regenerate after a rebuild
    python -m atlas.artefact_manifest --check     # verify (what the test runs)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
MANIFEST = REPO / "manifests" / "artefacts.json"

# Delivered artefacts, by role. Anything matching is hashed; the roles exist so
# a reviewer can see at a glance that the main figures and tables are covered
# and not just the long tail of supplementary panels.
PATTERNS: dict[str, tuple[str, ...]] = {
    "main_figure":  ("fig_*.pdf", "fig_*.png"),
    "supp_figure":  ("supp_fig_*.pdf", "supp_fig_*.png",
                     "forest_*.png", "forest_*.pdf",
                     "concordance_*.png", "concordance_*.pdf"),
    "main_table":   ("table1_*.tsv", "table2_*.tsv", "tableS3_*.tsv"),
    "supp_table":   ("*_v1.tsv", "*_v1.json"),
}
# Not artefacts: intermediate score columns and the frozen matrix, which
# `atlas.manifest` already covers at the point of download/build.
EXCLUDE_SUBSTR = ("_scores", "score_matrix", "provenance", "summary")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(results: Path = RESULTS) -> dict[str, dict]:
    """Map relative path -> {role, sha256, bytes}, deterministically ordered."""
    seen: dict[str, dict] = {}
    for role, globs in PATTERNS.items():
        for pat in globs:
            for p in sorted(results.glob(pat)):
                rel = p.relative_to(REPO).as_posix()
                if rel in seen or any(s in p.name for s in EXCLUDE_SUBSTR):
                    continue
                seen[rel] = {"role": role, "sha256": sha256_file(p),
                             "bytes": p.stat().st_size}
    return dict(sorted(seen.items()))


def write(results: Path = RESULTS, manifest: Path = MANIFEST) -> dict:
    files = collect(results)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": ("SHA-256 of every delivered figure and table. Regenerate with "
                 "`python -m atlas.artefact_manifest --write` after a rebuild; "
                 "tests/test_artefact_manifest.py fails if a delivered artefact "
                 "drifts from this record."),
        "n_files": len(files),
        "by_role": {r: sum(1 for v in files.values() if v["role"] == r)
                    for r in PATTERNS},
        "files": files,
    }
    manifest.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")
    return payload


def check(results: Path = RESULTS, manifest: Path = MANIFEST) -> list[str]:
    """Return a list of human-readable problems; empty means the manifest holds."""
    if not manifest.exists():
        return [f"manifest missing: {manifest}"]
    recorded = json.loads(manifest.read_text())["files"]
    problems: list[str] = []
    for rel, meta in recorded.items():
        p = REPO / rel
        if not p.exists():
            problems.append(f"MISSING   {rel}")
            continue
        got = sha256_file(p)
        if got != meta["sha256"]:
            problems.append(f"CHANGED   {rel}\n            recorded {meta['sha256']}\n"
                            f"            on disk  {got}")
    for rel in collect(results):
        if rel not in recorded:
            problems.append(f"UNRECORDED {rel}  (delivered but not in the manifest)")
    return problems


def against_archive(archive_root: Path, manifest: Path = MANIFEST) -> dict:
    """Compare the manifest against an extracted release archive.

    This is the literal form of the manuscript's claim: the delivered figures
    and tables are checked, by content hash, against the archived release rather
    than only against the current working tree.
    """
    recorded = json.loads(manifest.read_text())["files"]
    same, differ, absent = [], [], []
    for rel, meta in recorded.items():
        a = archive_root / rel
        if not a.exists():
            absent.append(rel)
        elif sha256_file(a) == meta["sha256"]:
            same.append(rel)
        else:
            differ.append(rel)
    return {"same": same, "differ": differ, "absent": absent}


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--against", metavar="ARCHIVE_ROOT",
                   help="extracted release archive to compare against")
    a = ap.parse_args()
    if a.against:
        r = against_archive(Path(a.against))
        n = len(r["same"]) + len(r["differ"]) + len(r["absent"])
        print(f"[artefacts] {n} manifest entries vs {a.against}")
        print(f"  identical : {len(r['same'])}")
        print(f"  differ    : {len(r['differ'])}")
        print(f"  absent    : {len(r['absent'])}")
        for rel in r["differ"]:
            print(f"    DIFFERS {rel}")
        for rel in r["absent"]:
            print(f"    ABSENT  {rel}")
        sys.exit(1 if r["absent"] else 0)
    if a.write:
        payload = write()
        print(f"[artefacts] wrote {MANIFEST.relative_to(REPO)}: "
              f"{payload['n_files']} files {payload['by_role']}")
        return
    problems = check()
    if problems:
        print(f"[artefacts] {len(problems)} problem(s):")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    n = len(json.loads(MANIFEST.read_text())["files"])
    print(f"[artefacts] OK -- {n} delivered figures and tables match the manifest")


if __name__ == "__main__":
    main()

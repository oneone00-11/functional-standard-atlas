"""What actually ships in a release archive.

Until now nothing defined this. The archive was assembled by hand, so the
release-hygiene gate had nothing to scan and fell back to `git ls-files` --
which misses every file that is gitignored but shipped: `data/raw/`,
`data/frozen/` and the ~270 files in `results/`. Those were published in v2.3.1
and v2.3.2 without ever passing the check that exists to stop drafting material
reaching readers.

Defining the packaging output in code is the fix. The gate scans this list, a
test asserts the list covers every file in a candidate archive, and the archive
itself can be built from it.

    python -m atlas.package_release --list
    python -m atlas.package_release --build dist/functional-standard-atlas-v2.4.0.zip
    python -m atlas.package_release --verify <extracted archive root>
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Gitignored, but part of the published record: the frozen inputs the analysis
# is pinned to, the third-party deposits it was built from, and its outputs.
SHIPPED_UNTRACKED = ("data/raw", "data/frozen", "results")

# Never ships: caches, environments, the denylist (naming what was removed is
# itself a disclosure), and the per-run logs that are deliberately excluded.
# `_scratch` is where development output goes (CONVENTIONS.md): smoke tests,
# probes and one-off checks, none of which belong in a published archive.
EXCLUDE_PARTS = ("__pycache__", ".venv", ".pytest_cache", ".git", "cache",
                 "_scratch", "dist")
# `dist` is where release candidates are built. An archive must never contain a
# previous archive: a 47 MB zip reached a commit once, and the next build then
# tried to package it into its own successor and failed on the file it had just
# replaced. Excluded here as well as gitignored, so neither alone is load-bearing.
# `models/*/run_log.json` is gitignored and so never reaches `tracked()`; the
# per-model `run_log_<column>.json` files ARE tracked and did ship in v2.3.1 and
# v2.3.2, so no name rule is needed here -- trackedness already decides.
EXCLUDE_NAMES = (".release-denylist", ".DS_Store")
EXCLUDE_SUFFIXES = (".pyc",)


def _excluded(rel: str) -> bool:
    parts = Path(rel).parts
    if any(p in EXCLUDE_PARTS for p in parts):
        return True
    name = Path(rel).name
    return name in EXCLUDE_NAMES or Path(rel).suffix in EXCLUDE_SUFFIXES


def tracked(repo: Path = REPO) -> list[str]:
    r = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True)
    if r.returncode != 0:
        return []
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def manifest(repo: Path = REPO) -> list[str]:
    """Every path that belongs in the release archive, sorted and deduplicated."""
    out: set[str] = {p for p in tracked(repo) if not _excluded(p)}
    for top in SHIPPED_UNTRACKED:
        base = repo / top
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                rel = p.relative_to(repo).as_posix()
                if not _excluded(rel):
                    out.add(rel)
    return sorted(out)


def build(dest: Path, repo: Path = REPO, prefix: str = "atlas") -> tuple[Path, int]:
    files = manifest(repo)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            z.write(repo / rel, f"{prefix}/{rel}")
    return dest, len(files)


def verify(archive_root: Path, repo: Path = REPO) -> dict:
    """Compare an extracted archive against what the manifest says should be in it."""
    declared = set(manifest(repo))
    present = {p.relative_to(archive_root).as_posix()
               for p in archive_root.rglob("*") if p.is_file()}
    return {"declared": sorted(declared),
            "in_archive_only": sorted(present - declared),
            "declared_only": sorted(declared - present),
            "both": sorted(declared & present)}


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--build", metavar="ZIP")
    g.add_argument("--verify", metavar="ARCHIVE_ROOT")
    a = ap.parse_args()
    if a.list:
        files = manifest()
        for f in files:
            print(f)
        print(f"\n[package] {len(files)} files", file=sys.stderr)
        return
    if a.build:
        dest, n = build(Path(a.build))
        print(f"[package] built {dest} with {n} files")
        return
    r = verify(Path(a.verify))
    print(f"[package] declared {len(r['declared'])}, archive shares {len(r['both'])}")
    print(f"  in archive but not declared : {len(r['in_archive_only'])}")
    for f in r["in_archive_only"][:40]:
        print(f"     {f}")
    print(f"  declared but not in archive : {len(r['declared_only'])}")
    for f in r["declared_only"][:40]:
        print(f"     {f}")
    sys.exit(1 if r["in_archive_only"] else 0)


if __name__ == "__main__":
    main()

"""Does the manuscript's cited archive still match this repository?

The declared-count check compares the manuscript against manifests/pipeline_facts.json
and passes when both agree. That verifies internal consistency and nothing else. It
reported clean while the manuscript cited a DOI for v2.3.2, quoted a test count
measured on a v2.4.0 release candidate that was never published, and referenced a
supplementary table whose generator postdates the tag -- three statements that cannot
all be true of one deposit.

Two checks, because they need different things:

  divergence() offline. The repository has moved past the tag the manuscript cites.
               Fails when HEAD is ahead of the last published tag and the manuscript
               cites a DOI, and names the deliverables that changed.

  deposit()    at release time. Given the file listing of the cited deposit, every
               artefact the manuscript names must be in it. The listing is passed in
               rather than fetched here, so the check is testable offline and the
               network call stays at the call site.

Usage (PYTHONPATH=src):
    python -m atlas.release_state <manuscript.docx>
    python -m atlas.release_state <manuscript.docx> --deposit-files listing.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOI = re.compile(r"10\.5281/zenodo\.(\d+)")
# Supplementary tables the manuscript names, as they appear in results/
TABLE_TOKEN = re.compile(r"\bS(\d+[a-z]?)\b")


def _git(*args: str, repo: Path = REPO) -> str:
    return subprocess.run(("git", *args), cwd=repo, capture_output=True,
                          text=True).stdout.strip()


def last_published_tag(repo: Path = REPO) -> str | None:
    tags = [t for t in _git("tag", "--sort=-creatordate", repo=repo).splitlines()
            if re.search(r"v\d+\.\d+", t)]
    return tags[0] if tags else None


def cited_dois(docx_path: Path) -> list[str]:
    import docx

    d = docx.Document(str(docx_path))
    text = " ".join(p.text for p in d.paragraphs)
    return sorted(set(DOI.findall(text)))


def divergence(docx_path: Path, repo: Path = REPO) -> dict:
    """Commits and deliverables between the cited tag and HEAD."""
    from atlas.package_release import manifest

    tag = last_published_tag(repo)
    dois = cited_dois(docx_path)
    if tag is None:
        return {"tag": None, "dois": dois, "commits_ahead": 0,
                "changed": [], "fatal": bool(dois),
                "why": "the manuscript cites a deposit but the repository has no release tag"}
    ahead = _git("rev-list", "--count", f"{tag}..HEAD", repo=repo)
    ahead = int(ahead) if ahead.isdigit() else 0
    shipped = set(manifest(repo))
    in_tag = set(_git("ls-tree", "-r", "--name-only", tag, repo=repo).splitlines())
    # deliverables that ship today but were not in the tagged tree, plus tracked
    # files the tag carried that have since changed
    added = sorted(f for f in shipped if f not in in_tag and (repo / f).exists()
                   and not f.startswith(("data/", "results/")))
    changed = _git("diff", "--name-only", f"{tag}..HEAD", repo=repo).splitlines()
    changed = sorted(f for f in changed if f in shipped)
    return {"tag": tag, "dois": dois, "commits_ahead": ahead,
            "added_since_tag": added, "changed": changed,
            "fatal": bool(dois and ahead),
            "why": (f"the manuscript cites zenodo.{', '.join(dois)} but HEAD is "
                    f"{ahead} commits past {tag}") if dois and ahead else ""}


def deposit(docx_path: Path, listing: list[str], repo: Path = REPO) -> dict:
    """Every results/ artefact the manuscript names must be in the deposit."""
    import docx

    d = docx.Document(str(docx_path))
    text = " ".join(p.text for p in d.paragraphs)
    wanted = {t.name for t in (repo / "results").glob("table_s*")} if (repo / "results").is_dir() else set()
    named = {f"S{n}" for n in TABLE_TOKEN.findall(text)}
    have = {Path(f).name for f in listing}
    missing = sorted(w for w in wanted if w not in have)
    return {"tables_named_in_text": sorted(named), "deposit_files": len(listing),
            "missing_from_deposit": missing, "fatal": bool(missing)}


# ---------------------------------------------------------------------------
# Counts declared outside the manuscript
# ---------------------------------------------------------------------------
# The manuscript is not the only place that states how many tests pass, which
# version is current, or which deposit to cite. README.md, CITATION.cff and the
# Zenodo description say it too, and nothing checked them: the README carried
# 128/126/91/37 for three releases after those numbers stopped being true.
DECLARATION_FILES = ("README.md", "CITATION.cff", ".zenodo.json")

# Those three were not the whole of it either. The supplement's Note S12 states
# the suite size in prose, and it ships twice: as `results/Supplemental_Note.md`
# inside the archive, and as the string constant in `atlas.supplement` that
# generates it. Both said "83 guardrail tests" while the archive they describe
# runs 142, and no check looked at either -- the count was found by reading.
# So the scan covers the generators and the generated prose that ships with them.
SCANNED_SOURCE = "src"
SCANNED_PROSE = "results"


def _python_prose(path: Path) -> str:
    """The docstrings and string constants of a module, and nothing else.

    Deliberately not the raw file. This module's own patterns are literals that
    would match themselves, and the comments around them quote counts that are
    stale on purpose -- "the README carried 128/126/91/37" is documentation, not
    a declaration. What a reader can end up holding is the prose, so the prose
    is what is held to the facts.
    """
    try:
        tree = ast.parse(path.read_text(errors="ignore"))
    except (OSError, SyntaxError):
        return ""
    return "\n".join(n.value for n in ast.walk(tree)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str))


def declaration_texts(repo: Path = REPO) -> list[tuple[str, str]]:
    """(path, text) for every declaration this check reads."""
    out: list[tuple[str, str]] = []
    for rel in DECLARATION_FILES:
        f = repo / rel
        if f.exists():
            out.append((rel, f.read_text(errors="ignore")))
    for p in sorted((repo / SCANNED_SOURCE).rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        prose = _python_prose(p)
        if prose:
            out.append((p.relative_to(repo).as_posix(), prose))
    for p in sorted((repo / SCANNED_PROSE).rglob("*.md")):
        if "_scratch" in p.parts:      # development output; never ships
            continue
        out.append((p.relative_to(repo).as_posix(), p.read_text(errors="ignore")))
    return out


DECLARED_COUNTS = {
    r"(\d{2,4})\s+collected": "guardrail_tests_collected",
    r"(\d{2,4})\s+pass(?:es)? from a clean extract": "guardrail_tests_passing_from_archive",
    r"(\d{2,4})\s+guardrail tests": "guardrail_tests_passing_from_archive",
    r"(\d{2,4})\s+code-only guardrails": "guardrail_tests_passing_from_bare_clone",
    r"bare clone runs (\d{2,4})": "guardrail_tests_passing_from_bare_clone",
    r"\((\d{1,3}) skip without data\)": "guardrail_tests_skipped_from_bare_clone",
    r"skips (\d{1,3}) for want of data": "guardrail_tests_skipped_from_bare_clone",
    # Naming the checkout each figure belongs to is the point of the wording;
    # these two make the archive's pair checkable rather than merely stated.
    r"release archive runs (\d{2,4})": "guardrail_tests_passing_from_archive",
    r"release archive runs \d{2,4} and skips (\d{1,3})":
        "guardrail_tests_skipped_from_archive",
}


def declarations(repo: Path = REPO, docx_path: Path | None = None) -> list[dict]:
    """Every count, version and DOI stated outside the manuscript, checked.

    Compared against manifests/pipeline_facts.json for counts, against the
    version the archive was actually measured for, and -- when a manuscript is
    given -- against the deposit it cites, so the two cannot drift apart.
    """
    facts_path = repo / "manifests" / "pipeline_facts.json"
    facts = json.loads(facts_path.read_text()) if facts_path.exists() else {}
    cited = set(cited_dois(docx_path)) if docx_path else set()
    bad: list[dict] = []
    for rel, text in declaration_texts(repo):
        for pat, key in DECLARED_COUNTS.items():
            want = facts.get(key)
            if want is None:
                continue
            for m in re.finditer(pat, text):
                if int(m.group(1)) != int(want):
                    bad.append({"file": rel, "says": m.group(0), "phrase": m.group(0),
                                "expected": want, "key": key})
        want_v = str(facts.get("release_version") or "")
        for m in re.finditer(r'^version:\s*"?([\d.]+)"?', text, re.M):
            if want_v and m.group(1) != want_v:
                bad.append({"file": rel, "says": m.group(0).strip(),
                            "expected": want_v, "key": "release_version"})
        if cited:
            for m in re.finditer(r"10\.5281/zenodo\.(\d+)", text):
                if m.group(1) not in cited:
                    bad.append({"file": rel, "says": m.group(0),
                                "expected": "one of " + ", ".join(sorted(cited)),
                                "key": "deposit cited by the manuscript"})
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("docx")
    ap.add_argument("--deposit-files", help="JSON list of filenames in the cited deposit")
    a = ap.parse_args()
    doc = Path(a.docx)

    r = divergence(doc)
    print(f"[release] last published tag: {r['tag'] or '(none)'}; "
          f"manuscript cites: {', '.join('zenodo.'+d for d in r['dois']) or '(no DOI)'}")
    print(f"[release] HEAD is {r['commits_ahead']} commits past the tag")
    if r.get("added_since_tag"):
        print(f"  deliverables added since the tag ({len(r['added_since_tag'])}):")
        for f in r["added_since_tag"][:12]:
            print(f"    + {f}")
    if r.get("changed"):
        print(f"  shipped files changed since the tag: {len(r['changed'])}")
    bad = r["fatal"]
    if bad:
        print(f"  OUT OF SYNC: {r['why']}")

    decl = declarations(docx_path=doc)
    if decl:
        print(f"[release] declarations outside the manuscript: {len(decl)} stale")
        for x in decl:
            print(f"  STALE {x['file']}: {x['says']!r} -- expected {x['expected']} "
                  f"({x['key']})")
        bad = True
    else:
        print("[release] declarations in README, CITATION and the Zenodo description agree")

    if a.deposit_files:
        listing = json.loads(Path(a.deposit_files).read_text())
        d = deposit(doc, listing)
        print(f"[release] deposit carries {d['deposit_files']} files; "
              f"tables named in the text: {', '.join(d['tables_named_in_text'])}")
        for m in d["missing_from_deposit"]:
            print(f"  MISSING FROM DEPOSIT: {m}")
        bad = bad or d["fatal"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

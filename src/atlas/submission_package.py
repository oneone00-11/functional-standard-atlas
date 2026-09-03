"""Is the package that goes to the journal internally consistent?

The manuscript has been gated for a while. The package around it -- the
supplementary workbook, the supplementary PDF, the cover letter, the related
manuscript -- had nothing looking at it, and three defects survived every check
because each lived in the gap between two deliverables:

    the guardrail-test count appears three times with three values
    the minimum detectable rho is 0.107 in the manuscript, 0.197 in Note S6
    Table S11 is cited in the text and absent from Additional file 1

Three checks, matching the three ways a package comes apart:

    inventory()   P-1  the declared files exist, and nothing undeclared is
                       sitting in the package directories
    quantities()  P-2  a labelled number stated in two deliverables agrees
    references()  P-3  every Table/Note/Figure the text cites exists somewhere
                       in the package, and what exists but is never cited

Usage (PYTHONPATH=src):  python -m atlas.submission_package [--package atlas]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config" / "submission_package.yaml"

# Deliverables carry text in three shapes; each needs its own reader.
_SUFFIX_READERS: dict[str, str] = {".docx": "docx", ".xlsx": "xlsx", ".pdf": "pdf"}


def load(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text())


def package_root(spec: dict) -> Path:
    return Path(spec.get("root", "~/Desktop")).expanduser()


# ---------------------------------------------------------------------------
# reading the deliverables
# ---------------------------------------------------------------------------
def read_docx(p: Path) -> str:
    import docx

    d = docx.Document(str(p))
    parts = [q.text for q in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts.extend(c.text for c in row.cells)
    return "\n".join(parts)


def read_xlsx(p: Path) -> str:
    import openpyxl

    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    parts = []
    for name in wb.sheetnames:
        parts.append(f"[sheet {name}]")
        for row in wb[name].iter_rows(values_only=True):
            parts.append(" ".join("" if v is None else str(v) for v in row))
    return "\n".join(parts)


def sheet_names(p: Path) -> list[str]:
    import openpyxl

    return list(openpyxl.load_workbook(p, read_only=True).sheetnames)


def read_pdf(p: Path) -> str:
    import pypdf

    r = pypdf.PdfReader(str(p))
    return "\n".join((page.extract_text() or "") for page in r.pages)


def read(p: Path) -> str:
    kind = _SUFFIX_READERS.get(p.suffix.lower())
    if kind == "docx":
        return read_docx(p)
    if kind == "xlsx":
        return read_xlsx(p)
    if kind == "pdf":
        return read_pdf(p)
    return ""


def corpus(spec: dict, name: str) -> dict[str, str]:
    """Every declared deliverable, as text, keyed by its declared path."""
    root = package_root(spec)
    out: dict[str, str] = {}
    for entry in spec["packages"][name]["files"]:
        p = root / entry["path"]
        if entry.get("is_directory"):
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file() and f.suffix.lower() in _SUFFIX_READERS:
                        out[f"{entry['path']}/{f.name}"] = read(f)
            continue
        if p.exists():
            out[entry["path"]] = read(p)
    return out


# ---------------------------------------------------------------------------
# P-1  inventory
# ---------------------------------------------------------------------------
def inventory(spec: dict, name: str) -> dict:
    root = package_root(spec)
    declared_files, declared_dirs, missing = set(), set(), []
    for entry in spec["packages"][name]["files"]:
        p = root / entry["path"]
        if entry.get("is_directory"):
            declared_dirs.add(entry["path"])
            if not p.is_dir():
                missing.append(entry["path"])
            continue
        declared_files.add(entry["path"])
        if not p.exists():
            missing.append(entry["path"])

    # anything sitting in a package directory that nobody declared. An old
    # export of a related manuscript survived two submissions this way.
    undeclared = []
    for d in sorted({Path(f).parent.as_posix() for f in declared_files if "/" in f}):
        base = root / d
        if not base.is_dir():
            continue
        for f in sorted(base.iterdir()):
            if not f.is_file() or f.name.startswith("."):
                continue
            rel = f"{d}/{f.name}"
            if rel not in declared_files:
                undeclared.append(rel)
    return {"declared": sorted(declared_files | declared_dirs),
            "missing": sorted(missing), "undeclared": undeclared}


# ---------------------------------------------------------------------------
# P-2  the same quantity, stated twice
# ---------------------------------------------------------------------------
def quantities(spec: dict, name: str, texts: dict[str, str] | None = None) -> list[dict]:
    texts = corpus(spec, name) if texts is None else texts
    out = []
    for label, q in spec.get("quantities", {}).items():
        pat = re.compile(q["pattern"], re.I)
        found: dict[str, set[str]] = {}
        for rel, text in texts.items():
            vals = {m.group(1).lower() for m in pat.finditer(text)}
            if vals:
                found[rel] = vals
        distinct = {v for vs in found.values() for v in vs}
        if len(found) >= 2 and len(distinct) > 1:
            out.append({"quantity": label, "values": sorted(distinct),
                        "where": {k: sorted(v) for k, v in found.items()}})
    return out


# ---------------------------------------------------------------------------
# P-3  every reference lands inside the package
# ---------------------------------------------------------------------------
# "Additional file 1: Table S11" names both the item AND the file it should be
# in. Pooling provided items across the whole package hides the defect this
# check exists for: Note S11 does exist -- in Additional file 2 -- while Table
# S11, which the text points at in Additional file 1, does not.
CITED = re.compile(
    r"(?:Additional file (?P<file>\d)\s*:\s*)?"
    r"(?P<kind>Table|Note|Figure|Fig\.?)\s+(?P<item>S\d+[a-z]?)\b", re.I)
# which declared path each "Additional file N" refers to
FILE_ROLE = {"1": "Additional_file_1", "2": "Additional_file_2"}

# How a deliverable names the items it carries, which is not how the manuscript
# cites them. Additional file 2 lists its notes as "S11. AlphaGenome definition
# sweep" in the contents and its figures as "Supplemental_Fig_S16"; matching only
# "Note S11" reported eight items as missing that are present under another name.
PROVIDES = re.compile(
    r"(?:Table|Note|Figure|Fig\.?)\s+(S\d+[a-z]?)\b"
    r"|Supplemental_(?:Fig|Table|Note)_(S\d+[a-z]?)\b"
    r"|(?m:^\s*(S\d+[a-z]?)\.\s+[A-Z])", re.I)


def _canonical(item: str) -> str:
    """S17C is panel C of Figure S17, not a separate item; S5b and S10b are
    separate items. The manuscript's convention is the case of the suffix:
    uppercase marks a panel, lowercase a distinct table or note."""
    m = re.fullmatch(r"[Ss](\d+)([A-Za-z])", item)
    if m and m.group(2).isupper():
        return f"S{m.group(1)}"
    return item.upper()


def provided_items(text: str) -> set[str]:
    out = set()
    for m in PROVIDES.finditer(text):
        item = next((g for g in m.groups() if g), None)
        if item:
            out.add(_canonical(item))
    return out


def references(spec: dict, name: str, texts: dict[str, str] | None = None) -> dict:
    root = package_root(spec)
    pkg = spec["packages"][name]
    texts = corpus(spec, name) if texts is None else texts
    manuscript = pkg["manuscript"]
    cited: set[str] = set()
    targeted: list[tuple[str, str, str]] = []   # (item, kind, "Additional file N")
    for m in CITED.finditer(texts.get(manuscript, "")):
        item = _canonical(m.group("item"))
        cited.add(item)
        if m.group("file"):
            targeted.append((item, m.group("kind").title(), m.group("file")))

    # what the package actually provides: workbook sheet names, and any
    # "Table S9" / "Note S12" heading in a PDF's text layer
    provided: dict[str, set[str]] = {}
    for entry in pkg["files"]:
        p = root / entry["path"]
        keys: set[str] = set()
        if entry.get("is_directory") and p.is_dir():
            for f in sorted(p.glob("*.xlsx")):
                keys |= {s.upper() for s in sheet_names(f)}
                keys |= {m.group(1).upper() for m in re.finditer(r"_(S\d+[a-z]?)_", f.name)}
        elif p.suffix.lower() == ".xlsx" and p.exists():
            keys |= {s.upper() for s in sheet_names(p)}
        elif p.exists():
            keys |= provided_items(texts.get(entry["path"], ""))
        if keys:
            provided[entry["path"]] = keys

    everywhere = {k for ks in provided.values() for k in ks}

    # A citation that names its file must be satisfied by THAT file.
    misdirected, unverifiable = [], []
    for item, kind, fileno in sorted(set(targeted)):
        stem = FILE_ROLE.get(fileno)
        if stem is None:
            continue
        holders = [rel for rel, keys in provided.items() if stem in rel and item in keys]
        if holders:
            continue
        # A figure is an image. Its caption is often baked into the picture, so
        # absence from a PDF's text layer is not evidence of absence from the
        # PDF. Reported as unchecked rather than as missing -- a gate that cries
        # wolf on eight figures to catch one table gets switched off.
        if kind.lower().startswith("fig"):
            unverifiable.append({"item": item, "kind": kind,
                                 "cited_in": f"Additional file {fileno}"})
            continue
        elsewhere = sorted(rel for rel, keys in provided.items() if item in keys)
        misdirected.append({"item": item, "kind": kind,
                            "cited_in": f"Additional file {fileno}",
                            "actually_in": elsewhere})
    return {"cited": sorted(cited), "provided": {k: sorted(v) for k, v in provided.items()},
            "cited_but_absent": sorted(cited - everywhere),
            "misdirected": misdirected, "unverifiable": unverifiable,
            "present_but_never_cited": sorted(everywhere - cited)}


def freshness(spec: dict, name: str, texts: dict[str, str] | None = None) -> list[dict]:
    """Deliverables exported by hand from something else, checked against it.

    A hand export has no build step to keep it honest, so the only guarantee it
    can carry is a list of phrases that must and must not appear. The related
    manuscript sat three months behind its source, still carrying a claim the
    source had retracted, and nothing in the package could tell.
    """
    texts = corpus(spec, name) if texts is None else texts
    out = []
    for entry in spec["packages"][name]["files"]:
        text = texts.get(entry["path"])
        if text is None or not (entry.get("must_contain") or entry.get("must_not_contain")):
            continue
        absent = [m for m in entry.get("must_contain", []) if m not in text]
        present = [m for m in entry.get("must_not_contain", []) if m in text]
        if absent or present:
            out.append({"path": entry["path"], "exported_from": entry.get("exported_from"),
                        "missing": absent, "should_be_gone": present})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="atlas")
    ap.add_argument("--config", default=str(CONFIG))
    a = ap.parse_args()
    spec = load(Path(a.config))
    name = a.package

    inv = inventory(spec, name)
    print(f"[package] {name}: {len(inv['declared'])} declared entries")
    for m in inv["missing"]:
        print(f"  MISSING: {m}")
    for u in inv["undeclared"]:
        print(f"  UNDECLARED in the package directory: {u}")

    texts = corpus(spec, name)
    print(f"[package] read {len(texts)} deliverables")

    bad = quantities(spec, name, texts)
    if bad:
        print(f"[package] quantities stated inconsistently: {len(bad)}")
        for q in bad:
            print(f"  DISAGREEMENT  {q['quantity']}: {', '.join(q['values'])}")
            for f, v in q["where"].items():
                print(f"      {f}: {', '.join(v)}")
    else:
        print("[package] every cross-referenced quantity agrees")

    stale = freshness(spec, name, texts)
    for f in stale:
        print(f"  STALE EXPORT {f['path']}")
        if f["exported_from"]:
            print(f"      exported from {f['exported_from']}")
        for m in f["missing"]:
            print(f"      missing: {m!r}")
        for m in f["should_be_gone"]:
            print(f"      still present: {m!r}")

    refs = references(spec, name, texts)
    for r in refs["cited_but_absent"]:
        print(f"  CITED BUT ABSENT FROM THE PACKAGE: {r}")
    for m in refs["misdirected"]:
        where = ", ".join(m["actually_in"]) or "nowhere in the package"
        print(f"  CITED IN THE WRONG FILE: {m['kind']} {m['item']} is cited as "
              f"{m['cited_in']} but that file does not carry it; found in: {where}")
    if refs["unverifiable"]:
        items = ", ".join(f"{u['kind']} {u['item']}" for u in refs["unverifiable"])
        print(f"  not checkable from a text layer (images): {items}")
    if refs["present_but_never_cited"]:
        print(f"  present but never cited: {', '.join(refs['present_but_never_cited'])}")

    return 1 if (inv["missing"] or bad or stale or refs["cited_but_absent"]
                 or refs["misdirected"]) else 0


if __name__ == "__main__":
    sys.exit(main())

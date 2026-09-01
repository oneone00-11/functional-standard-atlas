"""Does every number in the manuscript come from pipeline output?

The manuscript states that "every value reported in the text, tables and figures
is read programmatically from pipeline output rather than transcribed". Nothing
enforced that, and one sentence -- the between-model correlation range in the
complementarity paragraph -- carried a figure with no source anywhere in
`results/`, which survived six rounds of review because every round asked
whether a number was *right* and none asked whether it had a *source*.

This is that missing step. Every decimal and integer token in the .docx (body
paragraphs, table cells and figure captions alike) is matched against every
numeric value appearing anywhere in `results/`, at a tolerance of half the last
printed digit, trying the value as printed and as a percentage. Tokens that
match nothing are NOT skipped: they are reported as `no-source` and the run
fails unless they are listed in config/manuscript_number_whitelist.yaml with a
reason.

    python -m atlas.check_manuscript_numbers path/to/manuscript.docx
    python -m atlas.check_manuscript_numbers path/to/manuscript.docx --verbose
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
WHITELIST = REPO / "config" / "manuscript_number_whitelist.yaml"

# Tokens are read as printed, so the tolerance can follow the printed precision.
TOKEN = re.compile(r"(?<![\w.])([-−+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?![\w])")
MAX_ABS = 1e7          # beyond this a token is an accession or an identifier
REF_HEADING = re.compile(r"^\s*references\s*$", re.I)
# Identifiers are not measurements: DOIs, URLs and MaveDB URNs carry digits that
# no pipeline output should be expected to explain.
IDENTIFIER = re.compile(r"(?:https?://\S+|doi:\S+|10\.\d{4,}/\S+|urn:\S+|"
                        r"\bzenodo\.\d+|\bNM_\d+(?:\.\d+)?)", re.I)
# Numbered citations point at the reference list, not at pipeline output.
CITATION = re.compile(r"\[\s*\d{1,2}(?:\s*[,–—-]\s*\d{1,2})*\s*\]")


def _norm(tok: str) -> float:
    return float(tok.replace(",", "").replace("−", "-").replace("+", ""))


def _tolerance(tok: str) -> float:
    frac = tok.split(".")[1] if "." in tok else ""
    return 0.5 * 10 ** (-len(frac)) + 1e-12


def pipeline_values(results: Path = RESULTS) -> np.ndarray:
    """Every numeric value appearing under results/, plus the row count of each
    table -- counts like "2,452 of 2,803 score sets" are genuine pipeline
    quantities that appear nowhere inside the files themselves.

    Generated prose (`Supplemental_Note.md`) is deliberately excluded. It is
    produced by `atlas.supplement`, and any number hardcoded in that generator
    would otherwise validate itself through it: the check would confirm that a
    transcribed literal matches the same transcribed literal.
    """
    vals: set[float] = set()
    for p in sorted(results.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".tsv", ".csv", ".json"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in re.finditer(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text):
            try:
                v = float(m.group(0))
            except ValueError:
                continue
            if abs(v) <= MAX_ABS:
                vals.add(v)
        if p.suffix.lower() in {".tsv", ".csv"}:
            n = sum(1 for _ in text.splitlines() if _.strip())
            vals.add(float(max(n - 1, 0)))          # data rows, header excluded
            vals.add(float(n))
    # Parquet outputs are binary, so only their shape is readable as a value --
    # and shape is often exactly what the manuscript quotes ("10,888 calls in
    # total" is the row count of the definition sweep).
    for p in sorted(results.rglob("*.parquet")):
        try:
            import pyarrow.parquet as pq
            md = pq.ParquetFile(p).metadata
        except Exception:
            continue
        vals.add(float(md.num_rows))
        vals.add(float(md.num_columns))
    return np.array(sorted(vals))


def manuscript_tokens(docx_path: Path) -> list[dict]:
    """Tokens from body paragraphs, table cells and captions; the reference
    list is excluded because its years and page ranges are not measurements."""
    import docx

    d = docx.Document(str(docx_path))
    units: list[tuple[str, str]] = []
    in_refs = False
    for i, p in enumerate(d.paragraphs):
        if REF_HEADING.match(p.text.strip()):
            in_refs = True
        if in_refs or not p.text.strip():
            continue
        units.append((f"P{i}", p.text))
    for ti, tb in enumerate(d.tables):
        for ri, row in enumerate(tb.rows):
            for ci, cell in enumerate(row.cells):
                if cell.text.strip():
                    units.append((f"T{ti + 1}r{ri}c{ci}", cell.text))

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for uid, raw in units:
        text = CITATION.sub(" ", IDENTIFIER.sub(" ", raw))
        for m in TOKEN.finditer(text):
            tok = m.group(1)
            try:
                val = _norm(tok)
            except ValueError:
                continue
            if abs(val) > MAX_ABS:
                continue
            key = (uid, tok)
            if key in seen:
                continue
            seen.add(key)
            lo = max(0, m.start() - 55)
            out.append({"uid": uid, "token": tok, "value": val,
                        "context": re.sub(r"\s+", " ", text[lo:m.end() + 35]).strip()})
    return out


def load_whitelist(path: Path = WHITELIST) -> tuple[set[str], list[re.Pattern], dict]:
    spec = yaml.safe_load(path.read_text()) if path.exists() else {}
    values: set[str] = set()
    patterns: list[re.Pattern] = []
    for group in spec.values():
        for v in group.get("values", []) or []:
            values.add(str(v))
        if group.get("pattern"):
            patterns.append(re.compile(group["pattern"]))
    return values, patterns, spec


# A token printed to few decimals sits in a dense part of the pool, so a match
# carries no information: "0.12" is within half a last-digit of hundreds of
# unrelated pipeline values. Such matches are reported separately as `weak`,
# because treating them as verification is exactly the mistake that let the
# complementarity range survive review.
WEAK_MATCH_MIN = 25          # distinct pool values inside the tolerance window


def classify(docx_path: Path, results: Path = RESULTS,
             whitelist: Path = WHITELIST) -> dict:
    pool = pipeline_values(results)
    wl_values, wl_patterns, _ = load_whitelist(whitelist)
    matched, weak, no_source, whitelisted = [], [], [], []
    for t in manuscript_tokens(docx_path):
        tol = _tolerance(t["token"])
        v = abs(t["value"])
        n_direct = int(np.sum(np.abs(pool - v) <= tol)) if pool.size else 0
        n_pct = int(np.sum(np.abs(pool - v / 100.0) <= tol / 100)) if pool.size else 0
        n_hits = n_direct + n_pct
        t["n_pool_matches"] = n_hits
        if n_hits:
            (weak if n_hits >= WEAK_MATCH_MIN else matched).append(t)
        elif t["token"].lstrip("+-−") in wl_values or any(
                p.match(t["token"].lstrip("+-−")) for p in wl_patterns):
            whitelisted.append(t)
        else:
            no_source.append(t)
    return {"matched": matched, "weak": weak, "whitelisted": whitelisted,
            "no_source": no_source, "n_pool": int(pool.size)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("docx")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", dest="as_json")
    a = ap.parse_args()
    r = classify(Path(a.docx))
    total = sum(len(r[k]) for k in ("matched", "weak", "whitelisted", "no_source"))
    print(f"[numbers] {Path(a.docx).name}: {total} tokens against "
          f"{r['n_pool']:,} pipeline values")
    print(f"  matched, specific          : {len(r['matched'])}")
    print(f"  matched, WEAK (ambiguous)  : {len(r['weak'])}")
    print(f"  whitelisted (see config/)  : {len(r['whitelisted'])}")
    print(f"  NO SOURCE                  : {len(r['no_source'])}")
    if a.verbose:
        for t in r["whitelisted"]:
            print(f"    whitelist {t['uid']:>10s}  {t['token']}")
        for t in r["weak"]:
            print(f"    weak      {t['uid']:>10s}  {t['token']:>10s}  "
                  f"({t['n_pool_matches']} pool values in tolerance)   …{t['context'][:88]}…")
    for t in r["no_source"]:
        print(f"    NO SOURCE {t['uid']:>10s}  {t['token']:>12s}   …{t['context']}…")
    if a.as_json:
        Path(a.as_json).write_text(json.dumps(r, indent=1, default=str))
    sys.exit(1 if r["no_source"] else 0)


if __name__ == "__main__":
    main()

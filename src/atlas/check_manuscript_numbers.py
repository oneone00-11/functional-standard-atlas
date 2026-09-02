"""Does every number in the manuscript come from pipeline output?

The manuscript states that "every value reported in the text, tables and figures
is read programmatically from pipeline output rather than transcribed". Nothing
enforced that, and one sentence -- the between-model correlation range in the
complementarity paragraph -- carried a figure with no source anywhere in
`results/`. A value check asks whether a number is *right*; it never asks
whether the number has a *source*, so a figure with no origin passes.

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
CLAIMS = REPO / "config" / "analysis_claims.yaml"
FACTS = REPO / "manifests" / "pipeline_facts.json"

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
    # Counts the manuscript cites that live in no results/ file -- the guardrail
    # suite size above all. Without them a stale count matches an unrelated
    # value by coincidence and is reported as verified.
    if FACTS.exists():
        for v in json.loads(FACTS.read_text()).values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.add(float(v))
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
# because treating them as verification is exactly the mistake that lets a
# sourceless number pass.
WEAK_MATCH_MIN = 25          # distinct pool values inside the tolerance window


# Counts the manuscript states about the pipeline itself. Pool membership is the
# wrong test for these: both 102 (the previous guardrail count) and 118 (the one
# after that) match exactly one unrelated pipeline value, so a stale count is
# reported as VERIFIED rather than missed. They are declared facts, so they are
# compared against the declared value.
DECLARED_COUNTS = {
    r"(\d{1,4})\s+guardrail tests": "guardrail_tests_passing_from_archive",
}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
# A claim may assert how many things an output covers. That is a statement about
# the output's SHAPE, and no value-membership check can test it: "the nineteen
# predictors span two axes" cites a matrix of nine, and every number printed in
# that sentence still corresponds to some cell. Cardinality is the check that
# catches it.
CARDINALITY = re.compile(
    r"\b(" + "|".join(NUMBER_WORDS) + r"|\d{1,3})\s+"
    r"(?:[a-z-]+\s+){0,3}?"                    # "nine broad-scope and splice-aware predictors"
    r"(predictors?|models?|tools?|genes?|strata|stratum|columns?|rows?)\b", re.I)


def paragraph_cardinalities(docx_path: Path) -> dict[str, list[tuple[int, str, str]]]:
    """paragraph id -> [(count, noun, phrase)] for every cardinality asserted."""
    import docx

    d = docx.Document(str(docx_path))
    out: dict[str, list[tuple[int, str, str]]] = {}
    in_refs = False
    for i, para in enumerate(d.paragraphs):
        if REF_HEADING.match(para.text.strip()):
            in_refs = True
        if in_refs or not para.text.strip():
            continue
        for m in CARDINALITY.finditer(para.text):
            tok = m.group(1).lower()
            n = NUMBER_WORDS.get(tok, None)
            if n is None:
                try:
                    n = int(tok)
                except ValueError:
                    continue
            out.setdefault(f"P{i}", []).append((n, m.group(2).lower(), m.group(0)))
    return out


def claim_cardinalities(path: Path = CLAIMS) -> dict[str, list[dict]]:
    """paragraph id -> the cardinality assertions recorded for it."""
    if not path.exists():
        return {}
    spec = yaml.safe_load(path.read_text()) or {}
    out: dict[str, list[dict]] = {}
    for c in spec.get("claims", []):
        if c.get("cardinality"):
            out.setdefault(c["para"], []).append(
                {"count": int(c["cardinality"]["count"]),
                 "noun": c["cardinality"]["noun"],
                 "output": c.get("output")})
    return out


def claim_scopes(path: Path = CLAIMS) -> dict[str, list[str]]:
    """paragraph id -> the outputs config/analysis_claims.yaml binds it to.

    A number inside a paragraph that declares its own source should be checked
    against THAT source, not against every value the pipeline has ever emitted.
    Checking against the whole pool is what let the complementarity range pass:
    0.12 matches 299 unrelated values, so it never reached `no source`.
    """
    if not path.exists():
        return {}
    spec = yaml.safe_load(path.read_text()) or {}
    scopes: dict[str, list[str]] = {}
    for c in spec.get("claims", []):
        out = c.get("output")
        if out and c.get("status") == "verified":
            scopes.setdefault(c["para"], []).append(out)
    return scopes


def scoped_values(outputs: list[str]) -> np.ndarray:
    """Every numeric value in the specific outputs a claim names."""
    vals: set[float] = set()
    for rel in outputs:
        p = REPO / rel
        if not p.exists():
            continue
        if p.suffix.lower() == ".parquet":
            try:
                import pyarrow.parquet as pq
                md = pq.ParquetFile(p).metadata
                vals.add(float(md.num_rows)); vals.add(float(md.num_columns))
                tbl = pq.read_table(p)
                for col in tbl.column_names:
                    for v in tbl.column(col).to_pylist()[:200000]:
                        if isinstance(v, (int, float)) and not isinstance(v, bool) \
                                and abs(v) <= MAX_ABS:
                            vals.add(float(v))
            except Exception:
                pass
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
            vals.add(float(max(n - 1, 0))); vals.add(float(n))
    return np.array(sorted(vals))


def _hits(pool: np.ndarray, v: float, tol: float) -> int:
    if not pool.size:
        return 0
    return (int(np.sum(np.abs(pool - v) <= tol))
            + int(np.sum(np.abs(pool - v / 100.0) <= tol / 100)))


def classify(docx_path: Path, results: Path = RESULTS,
             whitelist: Path = WHITELIST, claims: Path = CLAIMS) -> dict:
    pool = pipeline_values(results)
    wl_values, wl_patterns, _ = load_whitelist(whitelist)
    scopes = claim_scopes(claims)
    scoped_cache: dict[str, np.ndarray] = {}

    # Cardinality: does the count a paragraph asserts match what the claims
    # manifest records for the output it cites? No value check can ask this.
    asserted = paragraph_cardinalities(docx_path)
    recorded = claim_cardinalities(claims)
    cardinality_mismatch = []
    for para, entries in recorded.items():
        for e in entries:
            for n, noun, phrase in asserted.get(para, []):
                if noun.rstrip("s") != e["noun"].rstrip("s"):
                    continue
                if n != e["count"]:
                    cardinality_mismatch.append(
                        {"uid": para, "phrase": phrase, "asserted": n,
                         "recorded": e["count"], "noun": noun, "output": e["output"]})

    # Declared counts about the pipeline, compared to manifests/pipeline_facts.json.
    facts = json.loads(FACTS.read_text()) if FACTS.exists() else {}
    stale_counts = []
    if facts:
        import docx as _docx
        doc = _docx.Document(str(docx_path))
        for i, para in enumerate(doc.paragraphs):
            for pat, key in DECLARED_COUNTS.items():
                for m in re.finditer(pat, para.text, re.I):
                    stated = int(m.group(1))
                    truth = facts.get(key)
                    if truth is not None and stated != truth:
                        stale_counts.append({"uid": f"P{i}", "phrase": m.group(0),
                                             "stated": stated, "recorded": truth,
                                             "key": key})

    matched, weak, no_source, whitelisted, scoped_mismatch = [], [], [], [], []
    for t in manuscript_tokens(docx_path):
        tol = _tolerance(t["token"])
        v = abs(t["value"])
        bare = t["token"].lstrip("+-−")
        white = bare in wl_values or any(p.match(bare) for p in wl_patterns)

        outputs = scopes.get(t["uid"])
        if outputs and not white:
            key = "|".join(sorted(outputs))
            if key not in scoped_cache:
                scoped_cache[key] = scoped_values(outputs)
            n_scoped = _hits(scoped_cache[key], v, tol)
            t["scope"] = outputs
            t["n_scoped_matches"] = n_scoped
            if n_scoped:
                t["n_pool_matches"] = n_scoped
                matched.append(t)
                continue
            # Absent from the output its own paragraph declares. That alone is
            # not a fault: a paragraph carries several sentences and may state
            # atlas composition alongside an analysis result. It becomes a fault
            # when the value has no specific source anywhere either -- absent
            # from the declared output AND only ambiguously present in the pool
            # is the signature of a number with no real provenance, which is
            # exactly the complementarity range.
            n_pool = _hits(pool, v, tol)
            t["n_pool_matches"] = n_pool
            if n_pool and n_pool < WEAK_MATCH_MIN:
                matched.append(t)          # specific source, just not this one
                continue
            scoped_mismatch.append(t)
            continue

        n_hits = _hits(pool, v, tol)
        t["n_pool_matches"] = n_hits
        if n_hits:
            (weak if n_hits >= WEAK_MATCH_MIN else matched).append(t)
        elif white:
            whitelisted.append(t)
        else:
            no_source.append(t)
    return {"matched": matched, "weak": weak, "whitelisted": whitelisted,
            "no_source": no_source, "scoped_mismatch": scoped_mismatch,
            "cardinality_mismatch": cardinality_mismatch,
            "stale_counts": stale_counts,
            "n_pool": int(pool.size), "n_scoped_paragraphs": len(scopes)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("docx")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", dest="as_json")
    a = ap.parse_args()
    r = classify(Path(a.docx))
    keys = ("matched", "weak", "whitelisted", "no_source", "scoped_mismatch")
    total = sum(len(r[k]) for k in keys)
    print(f"[numbers] {Path(a.docx).name}: {total} tokens against "
          f"{r['n_pool']:,} pipeline values, "
          f"{r['n_scoped_paragraphs']} paragraphs scoped to a declared output")
    print(f"  matched, specific          : {len(r['matched'])}")
    print(f"  matched, WEAK (ambiguous)  : {len(r['weak'])}")
    print(f"  whitelisted (see config/)  : {len(r['whitelisted'])}")
    print(f"  NO SOURCE                  : {len(r['no_source'])}")
    print(f"  SCOPED MISMATCH            : {len(r['scoped_mismatch'])}")
    print(f"  CARDINALITY MISMATCH       : {len(r['cardinality_mismatch'])}")
    print(f"  STALE DECLARED COUNT       : {len(r['stale_counts'])}")
    for c in r["stale_counts"]:
        print(f"    STALE {c['uid']}: text says \"{c['phrase']}\" but "
              f"{c['key']} is {c['recorded']}")
    for c in r["cardinality_mismatch"]:
        print(f"    CARDINALITY {c['uid']}: text says \"{c['phrase']}\" but "
              f"{c['output']} carries {c['recorded']} {c['noun']}")
    for t in r["scoped_mismatch"]:
        print(f"    SCOPED MISMATCH {t['uid']:>8s}  {t['token']:>10s}  "
              f"absent from {', '.join(t['scope'])}\n"
              f"        …{t['context']}…")
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
    sys.exit(1 if (r["no_source"] or r["cardinality_mismatch"]
                   or r["stale_counts"]) else 0)


if __name__ == "__main__":
    main()

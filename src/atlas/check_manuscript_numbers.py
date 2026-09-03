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
IDENTIFIER = re.compile(r"(?:SHA-\d+|\b\d+\.\d+\.\d+\b|https?://\S+|doi:\S+|10\.\d{4,}/\S+|urn:\S+|"
                        r"\bzenodo\.\d+|\bNM_\d+(?:\.\d+)?)", re.I)
# Numbered citations point at the reference list, not at pipeline output.
CITATION = re.compile(r"\[\s*\d{1,2}(?:\s*[,–—-]\s*\d{1,2})*\s*\]")

# Territory names, not measurements. "11-50 bp", "splice +-1-2" and ">50 bp" are
# labels for strata; tokenising them yields 11, 50, 1, 2 -- one- and two-digit
# integers that match something in almost any output file, which is most of what
# the old global-fallback escape hatch was quietly absorbing. Removing them at
# the tokeniser is a correction, not an exemption: no measurement is skipped.
STRATUM = re.compile(
    r"±\s*\d{1,3}\s*[–—-]\s*\d{1,3}"          # +-1-2
    r"|\b\d{1,3}\s*[–—-]\s*\d{1,3}\s*bp\b"    # 11-50 bp
    r"|\b[><]\s*\d{1,3}\s*bp\b"                # >50 bp
    r"|\b\d{1,2}-mer\b")                        # 6-mer


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
        text = STRATUM.sub(" ", CITATION.sub(" ", IDENTIFIER.sub(" ", raw)))
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


def _blocks(docx_path: Path) -> list[tuple[str, str]]:
    """(id, text) for every paragraph and table cell, in document order."""
    import docx

    d = docx.Document(str(docx_path))
    out = [(f"P{i}", para.text) for i, para in enumerate(d.paragraphs)]
    for ti, t in enumerate(d.tables):
        for ri, row in enumerate(t.rows):
            for ci, cell in enumerate(row.cells):
                out.append((f"T{ti+1}r{ri+1}c{ci+1}", cell.text))
    return out


def resolve_anchors(docx_path: Path, path: Path = CLAIMS) -> dict[int, str]:
    """Row index in analysis_claims.yaml -> the block id its anchor lands in.

    Claims bind to the manuscript by a unique substring of the sentence that
    makes them, not by paragraph number. Inserting one paragraph in the
    Discussion shifted every later paragraph by one and silently misdirected
    23 bindings; the gate only showed it because a number happened to change.
    The `para:` field is retained as a reading aid and is never matched on.

    A row whose anchor hits zero blocks, or more than one, is omitted here and
    reported by tests/test_analysis_claims.py rather than guessed at.
    """
    if not path.exists():
        return {}
    spec = yaml.safe_load(path.read_text()) or {}
    blocks = _blocks(docx_path)
    resolved: dict[int, str] = {}
    for idx, c in enumerate(spec.get("claims", [])):
        anchor = c.get("anchor")
        if not anchor:
            continue
        hits = [uid for uid, text in blocks if text.count(anchor) == 1]
        if len(hits) == 1 and sum(t.count(anchor) for _, t in blocks) == 1:
            resolved[idx] = hits[0]
    return resolved


def claim_cardinalities(docx_path: Path, path: Path = CLAIMS) -> dict[str, list[dict]]:
    """block id -> the cardinality assertions recorded for it, by anchor."""
    if not path.exists():
        return {}
    spec = yaml.safe_load(path.read_text()) or {}
    where = resolve_anchors(docx_path, path)
    out: dict[str, list[dict]] = {}
    for idx, c in enumerate(spec.get("claims", [])):
        if c.get("cardinality") and idx in where:
            out.setdefault(where[idx], []).append(
                {"count": int(c["cardinality"]["count"]),
                 "noun": c["cardinality"]["noun"],
                 "output": c.get("output")})
    return out


def claim_derivations(docx_path: Path, path: Path = CLAIMS) -> dict[str, list[dict]]:
    """block id -> values the paragraph states that no output stores directly.

    A summary of stored values -- a max, a ratio, a count of rows meeting a
    condition -- has no literal source, and widening the scope cannot give it
    one. Each is declared individually with the arithmetic that produces it.
    Deliberately not a paragraph-level exemption: a row licenses one number.
    """
    if not path.exists():
        return {}
    spec = yaml.safe_load(path.read_text()) or {}
    where = resolve_anchors(docx_path, path)
    out: dict[str, list[dict]] = {}
    for idx, c in enumerate(spec.get("claims", [])):
        if c.get("derived") and idx in where:
            out.setdefault(where[idx], []).extend(c["derived"])
    return out


def claim_scopes(docx_path: Path, path: Path = CLAIMS) -> dict[str, list[str]]:
    """block id -> the outputs config/analysis_claims.yaml binds it to, by anchor.

    A number inside a paragraph that declares its own source should be checked
    against THAT source, not against every value the pipeline has ever emitted.
    Checking against the whole pool is what let the complementarity range pass:
    0.12 matches 299 unrelated values, so it never reached `no source`.
    """
    if not path.exists():
        return {}
    spec = yaml.safe_load(path.read_text()) or {}
    where = resolve_anchors(docx_path, path)
    scopes: dict[str, list[str]] = {}
    for idx, c in enumerate(spec.get("claims", [])):
        out = c.get("output")
        if out and c.get("status") == "verified" and idx in where:
            # a claim may name several outputs; all of them scope the block
            scopes.setdefault(where[idx], []).extend(
                out if isinstance(out, list) else [out])
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
    """Matches for a token, tried as printed and as a percentage, both signs.

    Only the token's magnitude reaches here, so a stored -0.73 would never match
    a printed -0.73 if the negative form were not searched.
    """
    if not pool.size:
        return 0
    n = 0
    for cand in (v, -v):
        n += int(np.sum(np.abs(pool - cand) <= tol))
        n += int(np.sum(np.abs(pool - cand / 100.0) <= tol / 100))
    return n


def classify(docx_path: Path, results: Path = RESULTS,
             whitelist: Path = WHITELIST, claims: Path = CLAIMS) -> dict:
    pool = pipeline_values(results)
    wl_values, wl_patterns, _ = load_whitelist(whitelist)
    scopes = claim_scopes(docx_path, claims)
    derivations = claim_derivations(docx_path, claims)
    scoped_cache: dict[str, np.ndarray] = {}

    # Cardinality: does the count a paragraph asserts match what the claims
    # manifest records for the output it cites? No value check can ask this.
    asserted = paragraph_cardinalities(docx_path)
    recorded = claim_cardinalities(docx_path, claims)
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
            if not n_scoped:
                for d in derivations.get(t["uid"], []):
                    if abs(abs(float(d["value"])) - v) <= tol:
                        n_scoped = 1
                        t["derived_from"] = d["how"]
                        break
            t["scope"] = outputs
            t["n_scoped_matches"] = n_scoped
            if n_scoped:
                t["n_pool_matches"] = n_scoped
                matched.append(t)
                continue
            # Absent from the output its own paragraph declares. This is a
            # fault, full stop. The rule used to fall back to the global pool
            # and pass anything matching exactly one value anywhere, on the
            # reasoning that a specific source somewhere is good enough. It is
            # not: in the companion manuscript 74.5 was absent from the table
            # its paragraph declares, matched one unrelated value, and passed --
            # while being the wrong number for its own sentence. A paragraph
            # that declares a source asserts that its numbers come from it.
            # Widen the declaration in config/analysis_claims.yaml, or declare
            # the value as derived; never exempt the paragraph.
            t["n_pool_matches"] = _hits(pool, v, tol)
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
    # A scoped mismatch is now fatal. It used to be printed and tolerated,
    # which is how a number absent from its own paragraph's declared source
    # reached a published draft of the companion manuscript.
    sys.exit(1 if (r["no_source"] or r["cardinality_mismatch"]
                   or r["stale_counts"] or r["scoped_mismatch"]) else 0)


if __name__ == "__main__":
    main()

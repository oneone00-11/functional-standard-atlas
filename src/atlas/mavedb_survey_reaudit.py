"""The MaveDB survey's audit, finished: a new sample on the current classification, each
deposit in it adjudicated by fixed rules and its evidence kept.

v5's audit (atlas.mavedb_survey_audit, results/mavedb_survey_audit_v1.tsv) sampled 131
deposits before the survey's error-column pattern was widened, and adjudicated none of
them; of its 60 deposits then classed "none", 51 are now classed as carrying an error
column. Nine in ten of the current 2,430 such deposits come from the names the widening
added, which that sample could not reach. Its files are left as they are.

Strata. A deposit counts as carrying an error column when a column name matches
atlas.mavedb_survey.RE_ERROR. That pattern is four alternatives; ERROR_PARTS splits it at
its top level (so that the four together are the pattern itself) and a deposit's stratum
is the first part, in the order below, that one of its error columns matches:

  conventional     se, sd, variance, ci_lower, ... (the names before the widening)
  interval_prefix  a CI bound with its level prefixed (score_95CI_low)
  sigma            sigma, raw_sigma, sigma1_uncorr
  err              a bare err token

and "none" for a deposit with neither replicate nor error columns. "replicates" (22) is
not sampled, as the survey checked it numerically; nor is "replicates_unverified" (29),
which is not counted.

Sample (SEED): 30 deposits each from conventional, interval_prefix and sigma, all 12 of
err, and 40 of none: 142.

Evidence, in this order: the MaveDB record (score-sets/{urn}: methodText, abstractText,
the column descriptions where there are any, the primary publication; the experiment's
record when the score set names no publication); the primary publication where it is
openly readable (PMC, bioRxiv, medRxiv); the current score table, checked numerically:
the share of rows that are numbers in each matched column; spread columns (SE, SD,
variance, sigma, err) never negative; interval bounds holding lower <= score <= upper in at
least 99% of the rows where all three are numbers (and the same against the column the
bound's stem names, where the table has one). Every response is cached in
data/raw/mavedb_audit_v2/ with the time it was fetched.

Verdicts are written by hand in config/mavedb_reaudit.yaml, by these rules, each with its
evidence (a source and a quotation of at most 25 words):

  each matched column of the four error strata
    uncertainty_of_score   the score's own standard error, standard deviation or interval
    uncertainty_of_other   the error of another quantity (raw_sigma of raw_score)
    not_uncertainty        the name matches but the column is not an error
    undetermined           the evidence does not decide it
  and a deposit is judged right when one of its columns is uncertainty_of_score;
  each deposit of none
    missed_uncertainty     another column is the score's error (named)
    missed_replicates      replicate columns the pattern did not match (mean checked)
    none                   neither
    undetermined
  Deposits with the same column names and the same publication may share one reading
  of the columns (a family); the numeric checks are made deposit by deposit. Where the
  current table's columns differ from the survey's record, that is noted and the current
  table is adjudicated. Each verdict also records what it rests on (basis): text, a
  quotation describes the column; relation, a numeric relation between the columns pins
  it (recomputed here: relations()); header, the column's name and the numeric checks
  only. A quotation's source is the score set's or the experiment's record
  (mavedb:score_set:<field>, mavedb:experiment:<field>), a PMC full text (pmc:<PMCID>,
  cached by `literature`) or the table's header (table:header); `assemble` stops on a
  quotation that is not in its source.

Summary: each stratum's proportion judged right, with its Wilson 95% interval (and the
same with the verdicts that rest on the header alone left out); their average weighted
by the strata's sizes (228, 1,608, 582, 12), with the strata's Wilson bounds weighted
the same way as its interval; the number of deposits permitting an estimate this implies
(22 + 2,430 x proportion; and for the human deposits, their replicates plus each
stratum's human count x its proportion); the proportion of none that was missed (a
missed error column, or missed replicates whose mean reproduces the score: the survey's
own rule for replicates) and the count it implies; v5's counts beside them.

Subcommands (from the repository root, PYTHONPATH=src):
  sample      data/raw/mavedb_audit_v2/sample.tsv
  fetch       the records, the experiments and the score tables of the sample (a table
              the API cannot serve is recorded as unavailable, the record kept)
  literature  the PMC full texts the verdicts quote (Europe PMC, else NCBI)
  assemble    results/mavedb_survey_reaudit_v1.tsv (one row per deposit) and
              results/mavedb_survey_reaudit_v1.json (the summary)
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.mavedb_ceilings import platform_series
from atlas.mavedb_survey import (API, PAUSE, RE_ERROR, RE_REPLICATE, SCORE_COL, TIMEOUT,
                                 _reconciling_subset)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
SURVEY = RESULTS / "mavedb_survey_v1.tsv"
METADATA = RESULTS / "mavedb_metadata_v1.tsv"
CACHE = REPO / "data" / "raw" / "mavedb_audit_v2"
SAMPLE = CACHE / "sample.tsv"
VERDICTS = REPO / "config" / "mavedb_reaudit.yaml"
OUT_TSV = RESULTS / "mavedb_survey_reaudit_v1.tsv"
OUT_JSON = RESULTS / "mavedb_survey_reaudit_v1.json"

SEED = 20261124
STRATA = ("conventional", "interval_prefix", "sigma", "err")
SIZES = {"conventional": 228, "interval_prefix": 1608, "sigma": 582, "err": 12}
DRAW = {"conventional": 30, "interval_prefix": 30, "sigma": 30, "err": 12, "none": 40}
V5 = {"permits": 2452, "of": 2803, "human_permits": 991, "human_of": 1202}
COLUMN_VERDICTS = ("uncertainty_of_score", "uncertainty_of_other", "not_uncertainty",
                   "undetermined")
NONE_VERDICTS = ("missed_uncertainty", "missed_replicates", "none", "undetermined")
MAX_QUOTE_WORDS = 25
BASES = ("text", "relation", "header")   # what a verdict rests on; see resolve()
STANDARD = {"accession", "hgvs_nt", "hgvs_splice", "hgvs_pro"}
Z = 1.959963984540054


# ==========================================================================
# strata
# ==========================================================================
def split_alternation(pattern: str) -> list[str]:
    """The top-level alternatives of a regular expression."""
    parts, depth, cur, i = [], 0, "", 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            cur += pattern[i:i + 2]
            i += 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    return parts + [cur]


def error_parts(pattern: str = RE_ERROR.pattern) -> dict[str, re.Pattern]:
    """RE_ERROR's four alternatives, named; together they are RE_ERROR itself."""
    alts = split_alternation(pattern)
    if "|".join(alts) != pattern or len(alts) != 4:
        raise SystemExit(f"RE_ERROR is not the four alternatives this audit splits: {alts}")
    named = {}
    for a in alts:
        key = ("sigma" if a.startswith("(^|_)sigma") else
               "err" if a.startswith("(^|_)err") else
               "interval_prefix" if a.startswith("\\d*ci") else
               "conventional" if a.startswith("(^|_)(se|") else None)
        if key is None or key in named:
            raise SystemExit(f"an alternative of RE_ERROR this audit does not know: {a}")
        named[key] = re.compile(a, re.I)
    return {k: named[k] for k in STRATA}


ERROR_PARTS = error_parts()


def error_stratum(columns: list[str]) -> str | None:
    """The first part, in STRATA's order, that one of the columns matches."""
    for k in STRATA:
        if any(ERROR_PARTS[k].search(c) for c in columns):
            return k
    return None


def _split(cell) -> list[str]:
    return [c for c in str(cell).split(";") if c and c != "nan"]


def survey_strata(survey: pd.DataFrame) -> pd.Series:
    """errors -> its error stratum; none -> none; the rest keep the survey's class."""
    def one(r):
        if r["permits"] == "errors":
            return error_stratum(_split(r["error_columns"]))
        return r["permits"]
    return survey.apply(one, axis=1)


def platforms(survey: pd.DataFrame, metadata: pd.DataFrame) -> pd.Series:
    """atlas.mavedb_ceilings.platform_series on the survey's titles and the deposits'
    target names (the current rule)."""
    t = survey[["urn", "title"]].merge(metadata[["urn", "target_genes"]], on="urn", how="left")
    return pd.Series(platform_series(t["title"], t["target_genes"]).to_numpy(),
                     index=survey.index)


def draw(survey: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """The sample: DRAW[stratum] deposits of each, in the survey's order within each."""
    s = survey.copy()
    s["stratum"] = survey_strata(s)
    got = {k: int((s["stratum"] == k).sum()) for k in STRATA}
    if got != SIZES:
        raise SystemExit(f"the strata are {got}, not {SIZES}: stopping")
    rng = np.random.default_rng(seed)
    parts = []
    for k in (*STRATA, "none"):
        pool = s.index[s["stratum"] == k]
        n = DRAW[k]
        pick = pool if n >= len(pool) else np.sort(rng.choice(pool, n, replace=False))
        parts.append(s.loc[pick].assign(draw_order=range(len(pick))))
    out = pd.concat(parts)
    return out[["urn", "stratum", "draw_order", "permits", "human", "title", "n_variants",
                "all_columns", "error_columns", "replicate_columns"]].reset_index(drop=True)


def write_sample(survey_path: Path = SURVEY, metadata_path: Path = METADATA,
                 out: Path = SAMPLE) -> pd.DataFrame:
    survey = pd.read_csv(survey_path, sep="\t")
    sample = draw(survey)
    plat = platforms(survey, pd.read_csv(metadata_path, sep="\t")).set_axis(survey["urn"])
    sample["platform"] = sample["urn"].map(plat)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, sep="\t", index=False)
    return sample


# ==========================================================================
# fetching
# ==========================================================================
def _safe(urn: str) -> str:
    return urn.replace(":", "_")


def _get(url: str, session=None, attempts: int = 4):
    import requests
    s = session or requests
    for k in range(attempts):
        try:
            r = s.get(url, timeout=TIMEOUT, headers={"User-Agent": "atlas-reaudit/1.0"})
            r.raise_for_status()
            return r
        except Exception:                                       # noqa: BLE001
            if k == attempts - 1:
                raise
            time.sleep(5 * 2 ** k)


def fetch(sample: pd.DataFrame, cache: Path = CACHE, force: bool = False) -> dict:
    """Each sampled deposit's record, its experiment's record and its score table."""
    import requests
    session = requests.Session()
    log_path = cache / "fetch_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for d in ("records", "experiments", "scores"):
        (cache / d).mkdir(parents=True, exist_ok=True)
    fail_path = cache / "fetch_failures.json"
    failures = json.loads(fail_path.read_text()) if fail_path.exists() else []
    for urn in sample["urn"]:
        rec_p = cache / "records" / f"{_safe(urn)}.json"
        sc_p = cache / "scores" / f"{_safe(urn)}.csv"
        if not force and rec_p.exists() and sc_p.exists() and urn in log:
            continue
        now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")   # noqa: E731
        try:
            rec = _get(f"{API}/score-sets/{urn}", session).json()
            exp_urn = (rec.get("experiment") or {}).get("urn")
            exp = _get(f"{API}/experiments/{exp_urn}", session).json() if exp_urn else None
        except Exception as e:                                  # noqa: BLE001
            failures.append({"urn": urn, "utc": now(), "error": f"{type(e).__name__}: {e}"[:200]})
            fail_path.write_text(json.dumps(failures, indent=1) + "\n")
            print(f"  {urn}: {failures[-1]['error']} (run again to retry)", flush=True)
            continue
        rec_p.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
        if exp is not None:
            (cache / "experiments" / f"{_safe(exp_urn)}.json").write_text(
                json.dumps(exp, indent=1, ensure_ascii=False) + "\n")
        entry = {"fetched_utc": now(), "experiment": exp_urn}
        try:
            sc_p.write_bytes(_get(f"{API}/score-sets/{urn}/scores", session).content)
            entry["scores"] = "ok"
        except Exception as e:                                  # noqa: BLE001
            entry |= {"scores": "unavailable", "scores_error": f"{type(e).__name__}: {e}"[:200]}
            failures.append({"urn": urn, "utc": now(), "error": entry["scores_error"]})
            fail_path.write_text(json.dumps(failures, indent=1) + "\n")
            print(f"  {urn}: score table unavailable ({entry['scores_error'][:60]}); the "
                  "record is kept; run again to retry", flush=True)
        log[urn] = entry
        log_path.write_text(json.dumps(log, indent=1) + "\n")
        time.sleep(PAUSE)
    return log


EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def fetch_literature(pmcids: list[str], cache: Path = CACHE) -> dict:
    """Full texts from PMC (Europe PMC's fullTextXML), for the quotations."""
    import requests
    d = cache / "literature"
    d.mkdir(parents=True, exist_ok=True)
    log_path = d / "fetch_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for pmcid in pmcids:
        if (d / f"{pmcid}.xml").exists() and pmcid in log:
            continue
        urls = (f"{EUROPEPMC}/{pmcid}/fullTextXML",
                f"{EUTILS}/efetch.fcgi?db=pmc&id={pmcid.removeprefix('PMC')}&rettype=xml")
        for url in urls:
            try:
                r = _get(url, attempts=2)
            except Exception as e:                              # noqa: BLE001
                log.setdefault("_failures", []).append(
                    {"pmcid": pmcid, "url": url, "error": f"{type(e).__name__}: {e}"[:160]})
                continue
            if b"<body" not in r.content:
                log.setdefault("_failures", []).append(
                    {"pmcid": pmcid, "url": url, "error": "no article body in the response"})
                continue
            (d / f"{pmcid}.xml").write_bytes(r.content)
            log[pmcid] = {"fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "url": url, "bytes": len(r.content)}
            break
        log_path.write_text(json.dumps(log, indent=1) + "\n")
        time.sleep(PAUSE)
    return log


def _norm(text: str) -> str:
    """Whitespace collapsed, typographic quotes and dashes made plain."""
    for a, b in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'),
                 ("\u2010", "-"), ("\u2011", "-"), ("\u2013", "-"), ("\u2014", "-"),
                 ("\u2212", "-"), ("\u00a0", " ")):
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip()


def source_text(source: str, urn: str, cache: Path = CACHE) -> str:
    """The text a quotation's source names:
      mavedb:score_set:<field>    the score set's record (methodText, abstractText, title)
      mavedb:experiment:<field>   its experiment's record
      pmc:<PMCID>                 the cached full text
      table:header                the current score table's header"""
    kind, _, rest = source.partition(":")
    if kind == "mavedb":
        which, _, field = rest.partition(":")
        rec = record(urn, cache) if which == "score_set" else (experiment(urn, cache) or {})
        return _norm(str(rec.get(field) or ""))
    if kind == "pmc":
        pmcid = rest.split()[0]
        xml = (cache / "literature" / f"{pmcid}.xml").read_text(errors="replace")
        return _norm(re.sub(r"<[^>]+>", " ", xml))
    if kind == "table" and rest == "header":
        return _norm(",".join(scores(urn, cache).columns))
    raise SystemExit(f"unknown evidence source {source!r}")


def quote_found(quote: str, source: str, urn: str, cache: Path = CACHE) -> bool:
    return _norm(quote) in source_text(source, urn, cache)


def record(urn: str, cache: Path = CACHE) -> dict:
    return json.loads((cache / "records" / f"{_safe(urn)}.json").read_text())


def experiment(urn: str, cache: Path = CACHE) -> dict | None:
    exp_urn = (record(urn, cache).get("experiment") or {}).get("urn")
    p = cache / "experiments" / f"{_safe(exp_urn)}.json" if exp_urn else None
    return json.loads(p.read_text()) if p and p.exists() else None


def scores(urn: str, cache: Path = CACHE) -> pd.DataFrame:
    """The score table; when it could not be fetched, an empty table with the columns the
    record lists (datasetColumns), so that the record can still be adjudicated."""
    p = cache / "scores" / f"{_safe(urn)}.csv"
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    dc = record(urn, cache).get("datasetColumns") or {}
    cols = ["accession", "hgvs_nt", "hgvs_splice", "hgvs_pro"] + list(dc.get("scoreColumns") or [])
    return pd.DataFrame(columns=cols)


def table_available(urn: str, cache: Path = CACHE) -> bool:
    return (cache / "scores" / f"{_safe(urn)}.csv").exists()


def publications(rec: dict) -> list[dict]:
    """Primary publications: identifier, database, url, DOI and title as recorded."""
    out = []
    for p in rec.get("primaryPublicationIdentifiers") or []:
        out.append({k: p.get(k) for k in ("identifier", "dbName", "url", "doi", "title")})
    return out


# ==========================================================================
# numeric checks
# ==========================================================================
RE_LOWER = re.compile(r"(low(er)?|lo)($|_)", re.I)
RE_UPPER = re.compile(r"(high(er)?|upper|hi)($|_)", re.I)


def column_kind(col: str) -> str:
    """spread (SE, SD, variance, sigma, err), lower or upper (an interval bound)."""
    is_bound = re.search(r"ci", col, re.I) and (RE_LOWER.search(col) or RE_UPPER.search(col))
    if is_bound:
        return "lower" if RE_LOWER.search(col) else "upper"
    return "spread"


def _stem(col: str) -> str:
    """score_95CI_low -> score; dG_95CI_high -> dG; ci_lower -> ''."""
    m = re.match(r"^(.*?)[_.]?\d*ci[_.]?(low(er)?|lo|high(er)?|upper|hi)($|_.*)$", col, re.I)
    if m:
        return m.group(1)
    m = re.match(r"^(.*?)[_.]?(lower|upper)[_.]?ci($|_.*)$", col, re.I)
    return m.group(1) if m else ""


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def numeric_checks(df: pd.DataFrame, columns: list[str]) -> dict:
    """The checks of the module docstring, for the matched columns."""
    out: dict = {"rows": int(len(df)), "columns": {}}
    score = _num(df, SCORE_COL) if SCORE_COL in df.columns else None
    for c in columns:
        x = _num(df, c)
        rec = {"kind": column_kind(c), "numeric_share": round(float(x.notna().mean()), 6)
               if len(df) else None}
        if rec["kind"] == "spread":
            rec["negative"] = int((x < 0).sum())
            rec["non_negative"] = bool((x.dropna() >= 0).all()) and bool(x.notna().any())
        out["columns"][c] = rec
    lows = [c for c in columns if out["columns"][c]["kind"] == "lower"]
    ups = [c for c in columns if out["columns"][c]["kind"] == "upper"]
    pairs = []
    for lo in lows:
        mate = [u for u in ups if _stem(u) == _stem(lo)]
        if mate:
            pairs.append((lo, mate[0]))
    out["intervals"] = []
    for lo, hi in pairs:
        a, b = _num(df, lo), _num(df, hi)
        targets = [("score", score)] if score is not None else []
        stem = _stem(lo)
        if stem and stem != SCORE_COL and stem in df.columns:
            targets.append((stem, _num(df, stem)))
        for name, y in targets:
            both = a.notna() & b.notna() & y.notna()
            n = int(both.sum())
            inside = int(((a[both] <= y[both]) & (y[both] <= b[both])).sum())
            out["intervals"].append({"lower": lo, "upper": hi, "against": name, "rows": n,
                                     "inside": inside,
                                     "share_inside": round(inside / n, 6) if n else None,
                                     "holds": bool(n and inside / n >= 0.99)})
    return out


def relations(df: pd.DataFrame, specs: list[dict], rtol: float = 1e-6) -> list[dict]:
    """Relations between a table's columns that a verdict rests on, recomputed here:

      equal       {"equal": [column, expression]}  the column equals a pandas expression
                  of the others (backquote odd names), row by row
      row_mean    {"row_mean": [column, [columns]]}  the column is the rows' mean
      row_std     {"row_std": [column, [columns], ddof]}  the column is the rows' SD
      ratio       {"ratio": [numerator, denominator]}  their ratio is one constant
      linear      {"linear": [y, x]}  y is a linear function of x
      halfwidth   {"halfwidth": [upper, centre, scale]}  (upper - centre) / scale, its
                  range (1.96 for a 95% interval built from an SE)

    Each result gives the rows compared and the share that hold to rtol; a relation may
    carry its own rtol (for a deposit that rounds its values)."""
    x = df.apply(pd.to_numeric, errors="coerce")
    out = []
    for spec in specs:
        kind = next(k for k in spec if k != "rtol")
        args, tol = spec[kind], float(spec.get("rtol", rtol))
        rec = {"relation": kind, "args": args, "rtol": tol}
        if kind == "halfwidth":
            v = ((x[args[0]] - x[args[1]]) / x[args[2]]).replace([np.inf, -np.inf], np.nan).dropna()
            rec |= {"rows": int(len(v)), "min": float(v.min()), "median": float(v.median()),
                    "max": float(v.max())}
            out.append(rec)
            continue
        if kind == "equal":
            lhs, rhs = x[args[0]], x.eval(args[1])
        elif kind == "row_mean":
            lhs, rhs = x[args[0]], x[args[1]].mean(axis=1)
        elif kind == "row_std":
            lhs, rhs = x[args[0]], x[args[1]].std(axis=1, ddof=args[2])
        elif kind == "ratio":
            r = (x[args[0]] / x[args[1]]).replace([np.inf, -np.inf], np.nan)
            lhs, rhs = r, pd.Series(float(r.median()), index=r.index)
            rec["constant"] = float(r.median())
        elif kind == "linear":
            m = x[args[0]].notna() & x[args[1]].notna()
            slope, icept = np.polyfit(x.loc[m, args[1]], x.loc[m, args[0]], 1)
            lhs, rhs = x[args[0]], slope * x[args[1]] + icept
            rec |= {"slope": float(slope), "intercept": float(icept)}
        else:
            raise SystemExit(f"unknown relation {kind}")
        both = lhs.notna() & rhs.notna()
        ok = np.isclose(lhs[both], rhs[both], rtol=tol, atol=1e-9)
        rec |= {"rows": int(both.sum()), "share_holding": round(float(ok.mean()), 6)
                if both.any() else None}
        out.append(rec)
    return out


def replicate_check(df: pd.DataFrame, columns: list[str]) -> dict:
    """For a none deposit judged missed_replicates: does the mean of some subset of the
    named columns reproduce the score (atlas.mavedb_survey's reconciliation)?"""
    sub = _reconciling_subset(df, columns)
    return {"columns": columns, "reconciling_subset": sub, "reproduces_score": bool(sub)}


# ==========================================================================
# verdicts and summary
# ==========================================================================
def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, centre - half), 12), round(min(1.0, centre + half), 12))


def load_verdicts(path: Path = VERDICTS) -> dict:
    import yaml
    return yaml.safe_load(path.read_text())


def _words(s: str) -> int:
    return len(str(s).split())


def resolve(urn: str, stratum: str, matched: list[str], v: dict) -> dict:
    """One deposit's verdicts and evidence, its family's where it names one."""
    d = (v.get("deposits") or {}).get(urn)
    if d is None:
        raise SystemExit(f"{urn}: no verdict in {VERDICTS.name}")
    fam = (v.get("families") or {}).get(d.get("family")) if d.get("family") else None
    if d.get("family") and fam is None:
        raise SystemExit(f"{urn}: family {d['family']} is not defined")
    evidence = list((fam or {}).get("evidence") or []) + list(d.get("evidence") or [])
    if not evidence:
        raise SystemExit(f"{urn}: no evidence")
    for e in evidence:
        if not e.get("source") or not e.get("quote"):
            raise SystemExit(f"{urn}: evidence without a source or a quotation")
        if _words(e["quote"]) > MAX_QUOTE_WORDS:
            raise SystemExit(f"{urn}: a quotation of {_words(e['quote'])} words")
    basis = d.get("basis") or (fam or {}).get("basis")
    if basis not in BASES:
        raise SystemExit(f"{urn}: basis {basis!r} is not one of {BASES}")
    out = {"family": d.get("family") or "", "evidence": evidence, "basis": basis,
           "note": " ".join(x for x in ((fam or {}).get("note"), d.get("note")) if x),
           "relations": list((fam or {}).get("relations") or []) + list(d.get("relations") or [])}
    if stratum == "none":
        verdict = d.get("verdict")
        if verdict not in NONE_VERDICTS:
            raise SystemExit(f"{urn}: verdict {verdict!r} is not one of {NONE_VERDICTS}")
        out |= {"verdict": verdict, "missed_columns": list(d.get("missed_columns") or [])}
        if verdict in ("missed_uncertainty", "missed_replicates") and not out["missed_columns"]:
            raise SystemExit(f"{urn}: {verdict} without the columns")
        out["column_verdicts"] = {}
        return out
    cols = dict((fam or {}).get("columns") or {}) | dict(d.get("columns") or {})
    missing = [c for c in matched if c not in cols]
    bad = {c: x for c, x in cols.items() if x not in COLUMN_VERDICTS}
    if missing or bad:
        raise SystemExit(f"{urn}: matched columns without a verdict {missing}; bad {bad}")
    out["column_verdicts"] = {c: cols[c] for c in matched}
    out["verdict"] = ("right" if "uncertainty_of_score" in out["column_verdicts"].values()
                      else "wrong")
    return out


def summarise(rows: pd.DataFrame, survey: pd.DataFrame) -> dict:
    s = survey.copy()
    s["stratum"] = survey_strata(s)
    human = s["human"].astype(bool)
    out = {"strata": {}, "sizes": SIZES, "seed": SEED, "draw": DRAW}
    for k in STRATA:
        r = rows[rows["stratum"] == k]
        n, right = len(r), int((r["verdict"] == "right").sum())
        lo, hi = wilson(right, n)
        und = int(r["column_verdicts"].map(lambda d: "undetermined" in json.loads(d).values()
                                           and "uncertainty_of_score" not in
                                           json.loads(d).values()).sum())
        firm = int(((r["verdict"] == "right") & (r["basis"] != "header")).sum())
        out["strata"][k] = {"size": SIZES[k], "human_size": int(((s["stratum"] == k) & human).sum()),
                            "sampled": n, "right": right, "proportion": right / n if n else None,
                            "wilson_95": [lo, hi], "undetermined_only": und,
                            "right_not_on_header_alone": firm,
                            "proportion_not_on_header_alone": firm / n if n else None,
                            "wilson_95_not_on_header_alone": list(wilson(firm, n))}
    w = {k: SIZES[k] / sum(SIZES.values()) for k in STRATA}
    st = out["strata"]
    est = sum(w[k] * st[k]["proportion"] for k in STRATA)
    lo = sum(w[k] * st[k]["wilson_95"][0] for k in STRATA)
    hi = sum(w[k] * st[k]["wilson_95"][1] for k in STRATA)
    n_err = sum(SIZES.values())
    rep = int((s["permits"] == "replicates").sum())
    rep_h = int(((s["permits"] == "replicates") & human).sum())
    out["weighted"] = {"weights": SIZES, "proportion": est, "interval": [lo, hi],
                       "interval_method": "each stratum's Wilson 95% bounds, weighted by the "
                                          "strata's sizes (conservative)"}
    implied = {"all": {"replicates": rep, "errors": n_err,
                       "permits": rep + n_err * est,
                       "range": [rep + n_err * lo, rep + n_err * hi]}}
    h_sizes = {k: st[k]["human_size"] for k in STRATA}
    implied["human"] = {"replicates": rep_h, "errors": sum(h_sizes.values()),
                        "permits": rep_h + sum(h_sizes[k] * st[k]["proportion"] for k in STRATA),
                        "range": [rep_h + sum(h_sizes[k] * st[k]["wilson_95"][0] for k in STRATA),
                                  rep_h + sum(h_sizes[k] * st[k]["wilson_95"][1] for k in STRATA)],
                        "by_stratum": h_sizes}
    out["implied_permits"] = implied
    nr = rows[rows["stratum"] == "none"]
    counted = (nr["verdict"] == "missed_uncertainty") | (
        (nr["verdict"] == "missed_replicates") & nr["replicates_reproduce_score"].fillna(False).astype(bool))
    missed = int(counted.sum())
    n_none = int((s["stratum"] == "none").sum())
    n_none_h = int(((s["stratum"] == "none") & human).sum())
    mlo, mhi = wilson(missed, len(nr))
    out["none"] = {"size": n_none, "human_size": n_none_h, "sampled": int(len(nr)),
                   "missed": missed, "missed_rule": "missed_uncertainty, or missed_replicates "
                   "whose mean reproduces the score (the survey's own rule for replicates)",
                   "missed_not_on_header_alone": int((counted & (nr["basis"] != "header")).sum()),
                   "survey_fetch_failed": int(nr["survey_note"].fillna("").str.contains("fetch").sum()),
                   "by_verdict": nr["verdict"].value_counts().to_dict(),
                   "proportion": missed / len(nr) if len(nr) else None,
                   "wilson_95": [mlo, mhi],
                   "implied_missed": {"all": [n_none * missed / len(nr), n_none * mlo, n_none * mhi],
                                      "human": [n_none_h * missed / len(nr), n_none_h * mlo,
                                                n_none_h * mhi]}}
    out["v5"] = V5
    return out


def assemble(cache: Path = CACHE) -> dict:
    sample = pd.read_csv(SAMPLE, sep="\t")
    survey = pd.read_csv(SURVEY, sep="\t")
    v = load_verdicts()
    log = json.loads((cache / "fetch_log.json").read_text())
    survey_note = survey.set_index("urn")["note"].fillna("").to_dict()
    rows = []
    for r in sample.itertuples():
        df = scores(r.urn, cache)
        cols = list(df.columns)
        matched = [c for c in cols if RE_ERROR.search(c) and c != SCORE_COL]
        reps = [c for c in cols if RE_REPLICATE.search(c)]
        then = _split(r.error_columns)
        res = resolve(r.urn, r.stratum, matched, v)
        checks = numeric_checks(df, matched)
        if not table_available(r.urn, cache):
            checks["table"] = ("unavailable: " + log[r.urn].get("scores_error", "")
                               + "; columns from the record's datasetColumns")
        if res.get("relations"):
            checks["relations"] = relations(df, res["relations"])
        if r.stratum == "none" and res["verdict"] == "missed_replicates":
            checks["replicates"] = replicate_check(df, res["missed_columns"])
        if r.stratum == "none" and res["verdict"] == "missed_uncertainty":
            checks["missed"] = numeric_checks(df, res["missed_columns"])
        missing = [e for e in res["evidence"] if not quote_found(e["quote"], e["source"], r.urn, cache)]
        if missing:
            raise SystemExit(f"{r.urn}: quotation not in its source: {missing[0]}")
        rec = record(r.urn, cache)
        rep = checks.get("replicates", {})
        rows.append({
            "urn": r.urn, "stratum": r.stratum, "human": bool(r.human), "platform": r.platform,
            "title": rec.get("title"), "matched_columns": ";".join(matched),
            "survey_error_columns": ";".join(then),
            "columns_changed": (";".join(cols) != str(r.all_columns)),
            "table_available": table_available(r.urn, cache),
            "current_columns": ";".join(cols), "replicate_columns_now": ";".join(reps),
            "survey_note": survey_note.get(r.urn, ""),
            "family": res["family"], "verdict": res["verdict"], "basis": res["basis"],
            "replicates_reproduce_score": rep.get("reproduces_score") if rep else None,
            "column_verdicts": json.dumps(res["column_verdicts"]),
            "missed_columns": ";".join(res.get("missed_columns") or []),
            "evidence": json.dumps(res["evidence"], ensure_ascii=False),
            "numeric_check": json.dumps(checks), "note": res["note"],
            "publications": json.dumps(publications(rec)),
            "fetched_utc": log[r.urn]["fetched_utc"]})
    t = pd.DataFrame(rows)
    summ = summarise(t, survey)
    summ |= {"rows": int(len(t)), "fetched": sorted({x[:10] for x in t["fetched_utc"]}),
             "verdicts_file": str(VERDICTS.relative_to(REPO)),
             "columns_changed_since_survey": int(t["columns_changed"].sum())}
    RESULTS.mkdir(exist_ok=True)
    t.to_csv(OUT_TSV, sep="\t", index=False)
    OUT_JSON.write_text(json.dumps(summ, indent=1, ensure_ascii=False, default=str) + "\n")
    return summ


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sample", "fetch", "literature", "assemble"])
    args = ap.parse_args(argv)
    if args.command == "sample":
        s = write_sample()
        print(s["stratum"].value_counts().to_dict())
    elif args.command == "fetch":
        sample = pd.read_csv(SAMPLE, sep="\t")
        log = fetch(sample)
        left = int((~sample["urn"].isin(log)).sum())
        print(f"{len(log)} deposits fetched; {left} left")
    elif args.command == "literature":
        want = sorted(load_verdicts().get("literature") or {})
        log = fetch_literature(want)
        got = [p for p in want if p in log]
        print(f"{len(got)} of {len(want)} full texts cached; missing: "
              f"{[p for p in want if p not in log]}")
    else:
        summ = assemble()
        print(json.dumps({k: summ[k] for k in ("strata", "weighted", "implied_permits", "none")},
                         indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

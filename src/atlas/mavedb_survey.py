"""Survey MaveDB for deposits that permit a measurement-reliability estimate.

The paper's central prescription — report a per-stratum reliability estimate, or
state that the assay does not permit one — is only actionable to the extent that
deposits publish what the estimate needs. This module measures that extent
across the whole resource rather than asserting it from our own seven assays.

A deposit is counted as permitting an estimate if its score table carries either

  replicates  two or more per-variant score columns whose mean reproduces the
              deposited score (the BRCA1 case), or
  errors      a per-variant standard error, standard deviation or confidence
              interval from which one can be derived (the BARD1/PALB2 case).

For replicates the check is a reconciliation: a deposit may publish replicates
of more than one readout, so every subset of two or more replicate columns is
tried and the deposit counts only if some subset's mean reproduces the deposited
score across the table. Deposits carrying replicate columns that no subset
reconciles are reported separately — publishing replicates one cannot tie to the
score is its own failure mode, and a common one. Error columns are accepted on
name alone; the CI/SE reconciliation the primary analysis requires is stricter,
so those counts are upper bounds on what is usable. That is the honest direction
for this claim to err in.

Usage (with PYTHONPATH=src):

    python -m atlas.mavedb_survey --out results/mavedb_survey_v1.tsv
    python -m atlas.mavedb_survey --out results/mavedb_survey_v1.tsv --limit 50

Listing endpoint: POST https://api.mavedb.org/api/v1/score-sets/search
Scores endpoint:  GET  https://api.mavedb.org/api/v1/score-sets/{urn}/scores
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://api.mavedb.org/api/v1"
TIMEOUT = 120
PAUSE = 0.3          # be polite to the API

# Column-name patterns. Deliberately generous: a false positive is caught by the
# replicate check or reported as "errors" and re-examined by hand; a false
# negative would silently understate the resource.
RE_REPLICATE = re.compile(
    r"^(score[_.]?)?(rep(licate)?[_.]?\d+|r\d+)$|_rep(licate)?[_.]?\d+$"
    # rep1_score / rep_score — the replicate index leads rather than trails
    r"|^rep(licate)?[_.]?\d*[_.]?score$", re.I)
RE_ERROR = re.compile(
    r"(^|_)(se|sem|std[_.]?err|stderr|standard[_.]?error|sd|std[_.]?dev|"
    r"variance|ci[_.]?(lower|upper|lo|hi)|lower[_.]?ci|upper[_.]?ci)($|_)"
    # added after the audit of 60 deposits classified as permitting nothing, each
    # justified by a column name actually seen there (see mavedb_survey_audit):
    #   sigma / raw_sigma / sigma1_uncorr   a standard deviation by another name
    #   fitting_err                          a bare `err` token
    #   score_95CI_low / dG_95CI_high        a CI bound with the level prefixed,
    #                                        which the original `(^|_)ci` could
    #                                        not reach, and low/high spelt out
    r"|(^|_)sigma\d*($|_)"
    r"|(^|_)err($|_)"
    r"|\d*ci[_.]?(low|high|lower|upper|lo|hi)($|_)", re.I)

SCORE_COL = "score"


def _search(body: dict) -> dict:
    r = requests.post(f"{API}/score-sets/search", json=body, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def search_score_sets(limit: int | None = None) -> list[dict]:
    """Return metadata for every published score set.

    The search endpoint takes no offset or limit: it answers with at most 100
    records plus ``numScoreSets``, the true total. Paging parameters are
    accepted and ignored (start=0 and start=100 return the same first record),
    so the resource is enumerated by partitioning instead. ``filter-options``
    lists every target gene name with its count; no bucket exceeds the 100-record
    cap, so one query per gene name reaches every deposit. Score sets carrying
    several target genes appear in more than one bucket and are de-duplicated by
    URN; the union is checked against ``numScoreSets`` by the caller.
    """
    opts = requests.post(f"{API}/score-sets/search/filter-options",
                         json={"published": True}, timeout=TIMEOUT)
    opts.raise_for_status()
    names = [o["value"] for o in opts.json().get("targetGeneNames", [])]
    total = _search({"published": True}).get("numScoreSets")

    seen: dict[str, dict] = {}
    for i, name in enumerate(names, 1):
        try:
            batch = _search({"published": True, "targets": [name]}).get("scoreSets", [])
        except Exception as exc:                  # noqa: BLE001 — one bad bucket
            print(f"  bucket {name!r} failed: {type(exc).__name__}", file=sys.stderr)
            continue
        if len(batch) >= 100:
            print(f"  WARNING bucket {name!r} hit the 100 cap; may be truncated",
                  file=sys.stderr)
        for m in batch:
            urn = m.get("urn")
            if urn and urn not in seen:
                seen[urn] = m
        if limit and len(seen) >= limit:
            break
        if i % 100 == 0:
            print(f"  enumerating: {i}/{len(names)} buckets, {len(seen):,} unique",
                  file=sys.stderr)
        time.sleep(PAUSE)

    out = list(seen.values())
    print(f"  enumerated {len(out):,} unique score sets "
          f"(API reports {total:,})", file=sys.stderr)
    if limit:
        return out[:limit]
    return out


MAX_REP_COLS = 6   # 2**6 subsets is cheap; beyond this, report unverified


def _reconciling_subset(df, reps, score_col=SCORE_COL):
    """Return the replicate columns whose mean reproduces the deposited score.

    A deposit may publish replicates of more than one readout — BRCA1 carries
    two fitness and two RNA replicates — in which case the deposited score is
    the mean of one subset, not of every replicate column. Testing only the
    full set would reject exactly the deposits whose error model is usable, so
    every subset of size >= 2 is tried, largest first. Agreement must hold to
    floating-point tolerance across the whole table, which makes a spurious
    match implausible.
    """
    if not (2 <= len(reps) <= MAX_REP_COLS) or score_col not in df.columns:
        return None
    sub = df[reps + [score_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 50:
        return None
    for k in range(len(reps), 1, -1):
        for combo in combinations(reps, k):
            if np.isclose(sub[list(combo)].mean(axis=1), sub[score_col],
                          rtol=1e-6, atol=1e-8).mean() > 0.99:
                return list(combo)
    return None


def classify(urn: str) -> dict:
    """Fetch one score table and decide what it permits."""
    row = {"urn": urn, "n_variants": None, "n_columns": None,
           "all_columns": "",
           "replicate_columns": "", "error_columns": "",
           "reconciling_replicates": "",
           "replicates_reproduce_score": None, "permits": "none", "note": ""}
    try:
        r = requests.get(f"{API}/score-sets/{urn}/scores", timeout=TIMEOUT)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as exc:                      # noqa: BLE001 — recorded, not raised
        row["note"] = f"fetch/parse failed: {type(exc).__name__}"
        return row

    cols = list(df.columns)
    row["n_variants"], row["n_columns"] = len(df), len(cols)
    # the full header, so later questions about column naming are answerable
    # from the output rather than by re-querying the API
    row["all_columns"] = ";".join(cols)

    reps = [c for c in cols if RE_REPLICATE.search(c)]
    errs = [c for c in cols if RE_ERROR.search(c) and c != SCORE_COL]
    row["replicate_columns"] = ";".join(reps)
    row["error_columns"] = ";".join(errs)

    reconciling = _reconciling_subset(df, reps)
    row["reconciling_replicates"] = ";".join(reconciling) if reconciling else ""
    row["replicates_reproduce_score"] = bool(reconciling) if len(reps) >= 2 else None

    if row["replicates_reproduce_score"]:
        row["permits"] = "replicates"
    elif errs:
        row["permits"] = "errors"
    elif len(reps) >= 2:
        row["permits"] = "replicates_unverified"
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/mavedb_survey_v1.tsv")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N score sets (for a smoke test)")
    ap.add_argument("--human-only", action="store_true",
                    help="restrict to score sets whose target is Homo sapiens")
    args = ap.parse_args(argv)

    meta = search_score_sets(limit=args.limit)
    print(f"score sets returned: {len(meta):,}", file=sys.stderr)

    def is_human(m: dict) -> bool:
        blob = json.dumps(m.get("targetGenes", m.get("target_genes", []))).lower()
        return "homo sapiens" in blob or "human" in blob

    if args.human_only:
        meta = [m for m in meta if is_human(m)]
        print(f"human score sets: {len(meta):,}", file=sys.stderr)

    rows = []
    for i, m in enumerate(meta, 1):
        urn = m.get("urn")
        if not urn:
            continue
        rec = classify(urn)
        rec["title"] = (m.get("title") or "")[:120]
        rec["human"] = is_human(m)
        rows.append(rec)
        if i % 25 == 0:
            print(f"  {i}/{len(meta)}", file=sys.stderr)
        time.sleep(PAUSE)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)

    # deposit metadata, from the records the search already returned — no extra
    # API calls. Downstream analyses need publication dates and target genes.
    meta = pd.DataFrame([{
        "urn": m.get("urn"),
        "published_date": m.get("publishedDate"),
        "creation_date": m.get("creationDate"),
        "num_variants": m.get("numVariants"),
        "target_genes": ";".join(
            g.get("name", "") for g in (m.get("targetGenes") or []) if isinstance(g, dict)),
        "human": is_human(m),
    } for m in meta])
    meta.to_csv(out.parent / "mavedb_metadata_v1.tsv", sep="\t", index=False)
    print(f"wrote {out.parent / 'mavedb_metadata_v1.tsv'}  ({len(meta):,} rows)")

    n = len(df)
    counts = df["permits"].value_counts().to_dict()
    permits = counts.get("replicates", 0) + counts.get("errors", 0)
    unrec = counts.get("replicates_unverified", 0)
    neither = counts.get("none", 0)
    nh = int(df["human"].sum())
    hu = int(((df["permits"].isin(["replicates", "errors"])) & df["human"]).sum())

    print(f"\nwrote {out}  ({n:,} score sets surveyed)")
    print(f"  permits an estimate                      {permits:>6,}  ({permits / n:5.1%})")
    print(f"    of which reconciled replicates         {counts.get('replicates', 0):>6,}")
    print(f"    of which a per-variant error column    {counts.get('errors', 0):>6,}")
    print(f"  replicates present, not reconcilable     {unrec:>6,}  ({unrec / n:5.1%})")
    print(f"  neither                                  {neither:>6,}  ({neither / n:5.1%})")
    if nh:
        print(f"\n  of {nh:,} human score sets, {hu:,} ({hu / nh:.1%}) permit an estimate")

    print("\nSentence for the paper (check against the numbers above):")
    print(f'  "Across {n:,} published MaveDB score sets, only {permits:,} publish either '
          f'replicate scores that reconcile with the deposited score or a per-variant error '
          f'estimate, and a further {unrec:,} publish replicates that do not reconcile."')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit the column-name classification behind the MaveDB survey.

The survey's headline figure rests on matching column names against two regular
expressions. A name the patterns do not recognise deflates it; a name matched
for the wrong reason inflates it. This module samples both sides and dumps the
full header of each sampled deposit, so the figure can be defended rather than
asserted. Headers are read by streaming the first line of each score table, so
the audit costs a fraction of the survey.

Usage (with PYTHONPATH=src):

    python -m atlas.mavedb_survey_audit \
        --survey results/mavedb_survey_v1.tsv \
        --out results/mavedb_survey_audit_v1.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from atlas.mavedb_survey import API, PAUSE, RE_ERROR, RE_REPLICATE, TIMEOUT

SEED = 20260820
N_NONE = 60          # false-negative check
N_ERRORS = 25        # false-positive check


def header(urn: str) -> list[str]:
    """First line of the score table, without downloading the rest."""
    with requests.get(f"{API}/score-sets/{urn}/scores",
                      timeout=TIMEOUT, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line:
                return next(csv.reader([line]))
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey", default="results/mavedb_survey_v1.tsv")
    ap.add_argument("--out", default="results/mavedb_survey_audit_v1.tsv")
    args = ap.parse_args(argv)

    d = pd.read_csv(args.survey, sep="\t")

    picks = pd.concat([
        d[d.permits == "none"].sample(min(N_NONE, (d.permits == "none").sum()),
                                      random_state=SEED),
        d[d.permits == "errors"].sample(min(N_ERRORS, (d.permits == "errors").sum()),
                                        random_state=SEED),
        d[d.permits == "replicates"],              # only 18 — check all
        d[d.permits == "replicates_unverified"],   # only 28 — check all
    ])
    print(f"auditing {len(picks)} deposits", file=sys.stderr)

    rows = []
    for i, (_, r) in enumerate(picks.iterrows(), 1):
        try:
            cols = header(r.urn)
            note = ""
        except Exception as exc:                   # noqa: BLE001
            cols, note = [], f"{type(exc).__name__}"
        rows.append({
            "urn": r.urn,
            "permits": r.permits,
            "matched_replicate": r.get("replicate_columns", ""),
            "matched_error": r.get("error_columns", ""),
            "reconciling": r.get("reconciling_replicates", ""),
            "n_columns": len(cols),
            "all_columns": ";".join(cols),
            "note": note,
        })
        if i % 25 == 0:
            print(f"  {i}/{len(picks)}", file=sys.stderr)
        time.sleep(PAUSE)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)

    # Anything in a `none` deposit that neither pattern matched but that a human
    # might read as a replicate or an error column.
    print(f"\nwrote {out}")
    print("\n--- columns seen in deposits classified `none` ---")
    seen: dict[str, int] = {}
    for r in rows:
        if r["permits"] != "none":
            continue
        for c in r["all_columns"].split(";"):
            if c and not RE_REPLICATE.search(c) and not RE_ERROR.search(c):
                seen[c] = seen.get(c, 0) + 1
    for c, n in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>3}  {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

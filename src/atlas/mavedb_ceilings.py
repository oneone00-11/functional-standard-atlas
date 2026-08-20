"""Compute the attenuation ceiling of every MaveDB deposit that permits one.

The survey (``atlas.mavedb_survey``) establishes which deposits publish what a
reliability estimate needs. This module computes the estimate itself, so the
paper can describe the ceiling *distribution* across the resource rather than
generalising from seven assays.

Estimators are the ones the primary analysis uses (``atlas.reliability``):

  replicates  mean pairwise Spearman correlation among the reconciling replicate
              columns, raised to the reliability of their mean by Spearman-Brown
  errors      rel = 1 - mean(SE^2) / var(score)

Error columns come in four shapes. They are tried in this order and the rule
used is recorded, because they are not equally trustworthy:

  1  se / sem / standard_error / SE_*      used directly
  2  a CI pair (lower, upper)              SE = (upper - lower) / (2 * 1.96)
  3  sd / sigma / std_dev                  used as if it were an SE
  4  variance                              SE = sqrt(variance)

Rule 3 is flagged: a deposited SD may be the spread across replicates rather
than the standard error of their mean, and the two differ by sqrt(n). Treating
it as an SE understates reliability, so the resulting ceiling is a lower bound.

**Reliability is deliberately not clipped.** ``rel = 1 - mean(SE^2)/var`` can
only leave [0, 1] from below, and it does so when the error column is not on the
scale of the score — which is a property of the deposit worth counting, not an
error to hide. Out-of-range values are classified, not squashed:

  ok            0 < rel < 0.99
  degenerate    rel <= 0      error column not on the score's scale
  suspicious    rel >= 0.99   SE vanishing against var; almost always the same
  insufficient  fewer than 50 usable rows, or var(score) == 0

Usage (with PYTHONPATH=src):

    python -m atlas.mavedb_ceilings --survey results/mavedb_survey_v1.tsv \
        --out results/mavedb_ceilings_v1.tsv [--limit 30]
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from atlas.mavedb_survey import API, PAUSE, TIMEOUT
from atlas.reliability import spearman_brown

SCORE = "score"
MIN_N = 50
CI_Z = 1.959963984540054          # two-sided 95%

RE_SE = re.compile(r"(^|_)(se|sem|std[_.]?err|stderr|standard[_.]?error)($|_)", re.I)
RE_SD = re.compile(r"(^|_)(sd|sigma\d*|std[_.]?dev)($|_)", re.I)
RE_VAR = re.compile(r"(^|_)variance($|_)", re.I)
RE_LO = re.compile(r"(low(er)?|lo)($|_)", re.I)
RE_HI = re.compile(r"(high(er)?|upper|hi)($|_)", re.I)
RE_CI = re.compile(r"ci", re.I)


# Rocklin/Tsuboyama-lineage protease-resistance stability data on de novo designed
# mini-proteins. Identified from target names (HHH_rd*, EEHEE_rd*, EHEE_rd*, TrROS,
# villin/pin1-style scaffolds) and titles containing "trypsin"/"chymotrypsin"/
# "digestion". These are not human variant effect measurements, and their published
# interval is a curve-fitting CI on dG rather than a reproducibility interval, so
# they are reported separately and never pooled into the main distribution.
RE_DESIGNED = re.compile(
    r"HHH_rd|EEHEE_rd|EHEE_rd|HEEH_rd|TrROS|villin|pin1|rd\d_\d{4}", re.I)
RE_PROTEASE = re.compile(r"trypsin|chymotrypsin|digestion", re.I)


RE_ASSAY_SUFFIX = re.compile(
    r"\s*(chymotrypsin|trypsin)\s*digestion\s*$|\s*combined\s+scores?\s*$", re.I)


def platform_of(title: str, targets: str = "") -> str:
    """`designed_stability` for the protease-resistance mini-protein platform."""
    blob = f"{title} {targets}"
    if RE_DESIGNED.search(blob) or RE_PROTEASE.search(blob):
        return "designed_stability"
    return "other"


def platform_series(titles: pd.Series, targets: pd.Series) -> pd.Series:
    """Platform per deposit, propagated across each protein's deposit group.

    The platform deposits three score sets per protein: trypsin, chymotrypsin
    and a combined score. Only the first two name a protease, so classifying on
    title alone leaves the combined deposit — the one carrying the fitting CI
    whose ceiling is in question — filed as an ordinary deposit. Grouping on the
    target stem and marking the whole group keeps the three together.
    """
    stem = titles.fillna("").map(lambda t: RE_ASSAY_SUFFIX.sub("", str(t)).strip())
    direct = pd.Series([platform_of(str(t), str(g))
                        for t, g in zip(titles.fillna(""), targets.fillna(""))],
                       index=titles.index)
    flagged = set(stem[direct == "designed_stability"])
    return stem.map(lambda k: "designed_stability" if k in flagged else "other")


def _ci_pairs(cols: list[str]) -> list[tuple[str, str]]:
    """CI bounds that share a stem, e.g. score_95CI_low / score_95CI_high."""
    lo = [c for c in cols if RE_CI.search(c) and RE_LO.search(c)]
    hi = [c for c in cols if RE_CI.search(c) and RE_HI.search(c)]
    out = []
    for a in lo:
        stem = RE_LO.sub("", a)
        for b in hi:
            if RE_HI.sub("", b) == stem:
                out.append((a, b))
                break
    return out


def se_from(df: pd.DataFrame) -> tuple[pd.Series | None, str, str]:
    """Return (per-variant SE, column(s) used, rule name), by priority."""
    cols = [c for c in df.columns if c != SCORE]
    for c in cols:
        if RE_SE.search(c):
            return pd.to_numeric(df[c], errors="coerce"), c, "se"
    for a, b in _ci_pairs(cols):
        lo = pd.to_numeric(df[a], errors="coerce")
        hi = pd.to_numeric(df[b], errors="coerce")
        return (hi - lo).abs() / (2 * CI_Z), f"{a}+{b}", "ci"
    for c in cols:
        if RE_SD.search(c):
            return pd.to_numeric(df[c], errors="coerce"), c, "sd_as_se"
    for c in cols:
        if RE_VAR.search(c):
            return np.sqrt(pd.to_numeric(df[c], errors="coerce").clip(lower=0)), c, "variance"
    return None, "", ""


def reliability_from_replicates(df: pd.DataFrame, reps: list[str]) -> tuple[float, int]:
    """Spearman-Brown reliability of the mean of `reps`."""
    sub = df[reps].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < MIN_N:
        return float("nan"), len(sub)
    rs = [sub[a].corr(sub[b], method="spearman") for a, b in combinations(reps, 2)]
    rs = [r for r in rs if np.isfinite(r)]
    if not rs:
        return float("nan"), len(sub)
    return spearman_brown(float(np.mean(rs)), k=len(reps)), len(sub)


def classify_rel(rel: float, n: int, var: float) -> str:
    if n < MIN_N or not np.isfinite(var) or var <= 0 or not np.isfinite(rel):
        return "insufficient"
    if rel <= 0:
        return "degenerate"
    if rel >= 0.99:
        return "suspicious"
    return "ok"


def one(urn: str, permits: str, reconciling: str) -> dict:
    row = {"urn": urn, "permits": permits, "method": "", "error_column": "",
           "n_variants": None, "var_score": None, "mean_se2": None,
           "reliability_raw": None, "ceiling": None, "status": "insufficient",
           "note": ""}
    try:
        r = requests.get(f"{API}/score-sets/{urn}/scores", timeout=TIMEOUT)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    except Exception as exc:                       # noqa: BLE001 — recorded
        row["note"] = f"fetch/parse failed: {type(exc).__name__}"
        return row

    if SCORE not in df.columns:
        row["note"] = "no score column"
        return row
    score = pd.to_numeric(df[SCORE], errors="coerce")
    var = float(score.var())
    row["var_score"] = var

    reps = [c for c in (reconciling or "").split(";") if c and c in df.columns]
    if permits == "replicates" and len(reps) >= 2:
        rel, n = reliability_from_replicates(df, reps)
        row.update(method="replicates", error_column=";".join(reps),
                   n_variants=n, reliability_raw=rel)
    else:
        se, col, rule = se_from(df)
        if se is None:
            row["note"] = "no usable error column"
            return row
        sub = pd.concat([score.rename(SCORE), se.rename("se")], axis=1).dropna()
        n = len(sub)
        row["n_variants"] = n
        if n >= MIN_N and var > 0:
            mse = float((sub["se"] ** 2).mean())
            row["mean_se2"] = mse
            row["reliability_raw"] = 1 - mse / float(sub[SCORE].var())
        row.update(method=rule, error_column=col)
        if rule == "sd_as_se":
            row["note"] = "SD used as SE; ceiling is a lower bound"

    rel = row["reliability_raw"]
    row["status"] = classify_rel(rel if rel is not None else float("nan"),
                                 row["n_variants"] or 0, var)
    if row["status"] == "ok":
        row["ceiling"] = float(np.sqrt(rel))
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey", default="results/mavedb_survey_v1.tsv")
    ap.add_argument("--out", default="results/mavedb_ceilings_v1.tsv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    s = pd.read_csv(args.survey, sep="\t").fillna("")
    cand = s[s["permits"].isin(["replicates", "errors"])]
    if args.limit:
        cand = cand.head(args.limit)
    print(f"candidates: {len(cand):,}", file=sys.stderr)

    rows = []
    for i, (_, r) in enumerate(cand.iterrows(), 1):
        rows.append(one(r["urn"], r["permits"], r.get("reconciling_replicates", "")))
        if i % 50 == 0:
            print(f"  {i}/{len(cand)}", file=sys.stderr)
        time.sleep(PAUSE)

    d = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, sep="\t", index=False)

    n = len(d)
    print(f"\nwrote {out}  ({n:,} deposits)")
    for st in ("ok", "degenerate", "suspicious", "insufficient"):
        k = int((d["status"] == st).sum())
        print(f"  {st:<14} {k:>6,}  ({k / n:5.1%})")
    ok = d[d["status"] == "ok"]["ceiling"]
    if len(ok):
        q = ok.quantile([0, .25, .5, .75, 1]).round(3).tolist()
        print(f"\n  ceiling five-number: min {q[0]}  Q1 {q[1]}  median {q[2]}  "
              f"Q3 {q[3]}  max {q[4]}")
        below = int((ok < 0.45).sum())
        print(f"  below the 0.45 correction bound: {below:,} / {len(ok):,} "
              f"({below / len(ok):.1%})")
    print("\n  rules used:", d[d.status == "ok"]["method"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

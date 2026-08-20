"""Measure reproducibility on the designed-stability platform from two proteases.

The ceilings computed for that platform come from a curve-fitting confidence
interval on dG, which measures how well a parameter is determined from one
melting curve rather than how far a repeat experiment would move the score. That
objection is semantic, and this module turns it into a measurement.

The platform deposits, for each designed protein, a trypsin score set and a
chymotrypsin score set alongside the combined score. The two proteases are
independent experiments on the same variants, so their agreement bounds the
reliability of a single measurement, and Spearman-Brown raises that to the
reliability of their mean — the quantity the combined score's ceiling claims.

**This is a lower bound, not an estimate.** Two proteases are not technical
replicates: they differ in cleavage specificity, so their disagreement carries
real methodological difference as well as noise. The paper uses the same device
elsewhere (BAP1's nested time points, VHL's second selection condition), where a
comparison that is not a clean replicate still bounds reliability in a known
direction. A bound is enough here: if it sits far below the ceiling the fitting
CI implies, the CI is not measuring reproducibility.

Usage (with PYTHONPATH=src):
    python -m atlas.protease_agreement --out results/protease_agreement_v1.tsv [--limit 20]
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from atlas.mavedb_survey import API, PAUSE, TIMEOUT
from atlas.reliability import spearman_brown

MIN_OVERLAP = 50
SCORE = "score"
KEYS = ["hgvs_pro", "hgvs_nt"]

RE_TRYP = re.compile(r"(?<!chymo)trypsin", re.I)
RE_CHYMO = re.compile(r"chymotrypsin", re.I)
RE_COMB = re.compile(r"combined", re.I)
RE_SUFFIX = re.compile(r"\s*(chymotrypsin|trypsin)\s*digestion\s*$|\s*combined\s+scores?\s*$", re.I)


def target_of(title: str) -> str:
    """Strip the assay suffix so the three deposits of one protein share a key."""
    return RE_SUFFIX.sub("", str(title)).strip()


def kind(title: str) -> str:
    t = str(title)
    if RE_COMB.search(t):
        return "combined"
    if RE_CHYMO.search(t):
        return "chymo"
    if RE_TRYP.search(t):
        return "trypsin"
    return "other"


def fetch(urn: str) -> pd.DataFrame:
    r = requests.get(f"{API}/score-sets/{urn}/scores", timeout=TIMEOUT)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), low_memory=False)


def agreement(urn_a: str, urn_b: str) -> tuple[float, int, str]:
    a, b = fetch(urn_a), fetch(urn_b)
    for key in KEYS:
        if key in a.columns and key in b.columns:
            x = a[[key, SCORE]].dropna().rename(columns={SCORE: "a"})
            y = b[[key, SCORE]].dropna().rename(columns={SCORE: "b"})
            m = x.merge(y, on=key).apply(pd.to_numeric, errors="coerce", axis=0) \
                 if False else x.merge(y, on=key)
            m["a"] = pd.to_numeric(m["a"], errors="coerce")
            m["b"] = pd.to_numeric(m["b"], errors="coerce")
            m = m.dropna(subset=["a", "b"])
            if len(m) >= MIN_OVERLAP:
                return float(m["a"].corr(m["b"], method="spearman")), len(m), key
    return float("nan"), 0, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ceilings", default="results/mavedb_ceilings_v1.tsv")
    ap.add_argument("--out", default="results/protease_agreement_v1.tsv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    d = pd.read_csv(args.ceilings, sep="\t")
    d = d[d.platform == "designed_stability"].copy()
    d["target"] = d.title.map(target_of)
    d["kind"] = d.title.map(kind)

    groups = []
    for tgt, g in d.groupby("target"):
        k = dict(zip(g["kind"], g["urn"]))
        if "trypsin" in k and "chymo" in k:
            groups.append((tgt, k.get("trypsin"), k.get("chymo"), k.get("combined")))
    if args.limit:
        groups = groups[:args.limit]
    print(f"targets with both proteases: {len(groups):,}", file=sys.stderr)

    ci_ceiling = dict(zip(d.urn, d.reliability_raw))
    rows = []
    for i, (tgt, ut, uc, ucomb) in enumerate(groups, 1):
        row = {"target": tgt, "urn_trypsin": ut, "urn_chymo": uc,
               "urn_combined": ucomb or "", "n_overlap": 0,
               "rho_between_proteases": None, "rel_sb": None,
               "ceiling_empirical": None, "ceiling_from_ci": None,
               "delta": None, "key": "", "note": ""}
        try:
            rho, n, key = agreement(ut, uc)
            row.update(rho_between_proteases=rho, n_overlap=n, key=key)
            if np.isfinite(rho) and n >= MIN_OVERLAP:
                rel = spearman_brown(rho, k=2)
                row["rel_sb"] = rel
                if rel > 0:
                    row["ceiling_empirical"] = float(np.sqrt(rel))
                if ucomb and ucomb in ci_ceiling:
                    r2 = ci_ceiling[ucomb]
                    if pd.notna(r2) and r2 > 0:
                        row["ceiling_from_ci"] = float(np.sqrt(r2))
                        if row["ceiling_empirical"] is not None:
                            row["delta"] = row["ceiling_from_ci"] - row["ceiling_empirical"]
        except Exception as exc:                    # noqa: BLE001
            row["note"] = f"{type(exc).__name__}"
        rows.append(row)
        if i % 25 == 0:
            print(f"  {i}/{len(groups)}", file=sys.stderr)
        time.sleep(PAUSE)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows); t.to_csv(out, sep="\t", index=False)
    print(f"\nwrote {out}  ({len(t):,} targets)")

    ok = t[t.rho_between_proteases.notna() & (t.n_overlap >= MIN_OVERLAP)]
    print(f"  paired with >= {MIN_OVERLAP} shared variants: {len(ok):,}")
    if len(ok):
        print(f"  mean overlap: {ok.n_overlap.mean():,.0f} variants")
        q = ok.rho_between_proteases.quantile([0, .25, .5, .75, 1]).round(3).tolist()
        print(f"  rho between proteases: min {q[0]}  Q1 {q[1]}  median {q[2]}  Q3 {q[3]}  max {q[4]}")
        e = ok.ceiling_empirical.dropna()
        if len(e):
            print(f"  ceiling_empirical (Spearman-Brown, LOWER BOUND): median {e.median():.3f}")
        dd = ok.delta.dropna()
        if len(dd):
            print(f"  ceiling_from_ci: median {ok.ceiling_from_ci.dropna().median():.3f}")
            print(f"  delta (ci - empirical): median {dd.median():+.3f}  "
                  f"positive in {int((dd > 0).sum())}/{len(dd)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
